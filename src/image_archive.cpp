#include "uf/image_archive.hpp"
#include "uf/json_compat.hpp"
#include "uf/python_json.hpp"
#include "uf/process.hpp"
#include "uf/python_path.hpp"
#include "uf/utf8.hpp"

#include <openssl/evp.h>

#include <array>
#include <cerrno>
#include <fcntl.h>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <unistd.h>
#include <utility>

namespace uf {
namespace {

using ImageDigest = std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)>;

ImageDigest image_digest() {
    ImageDigest context(EVP_MD_CTX_new(), EVP_MD_CTX_free);
    if (!context || EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1) {
        throw std::runtime_error("Cannot hash UF intra image payload");
    }
    return context;
}

std::string image_digest_hex(EVP_MD_CTX* context) {
    std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
    unsigned int size = 0;
    if (EVP_DigestFinal_ex(context, digest.data(), &size) != 1 || size != 32) {
        throw std::runtime_error("Cannot finish UF intra image payload hash");
    }
    std::ostringstream result;
    result << std::hex << std::setfill('0');
    for (unsigned int i = 0; i < size; ++i) {
        result << std::setw(2) << static_cast<unsigned>(digest[i]);
    }
    return result.str();
}

FileDescriptor open_image_file(const std::filesystem::path& path) {
    if (path.native().find('\0') != std::string::npos)
        throw std::invalid_argument("embedded null byte");
    for (;;) {
        ProcessInterrupt::check();
        const int descriptor = open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (descriptor >= 0) return FileDescriptor(descriptor);
        const int error = errno;
        if (error == EINTR) continue;
        throw std::system_error(error, std::generic_category(),
            "Cannot open UF intra image: " + path.string());
    }
}

std::string read_image_bytes(FileDescriptor& input, std::size_t amount,
                             const std::filesystem::path& path) {
    std::string bytes(amount, '\0');
    std::size_t offset = 0;
    while (offset < amount) {
        ProcessInterrupt::check();
        const auto count = read(input.get(), bytes.data() + offset, amount - offset);
        if (count < 0) {
            const int error = errno;
            if (error == EINTR) continue;
            throw std::system_error(error, std::generic_category(),
                "Cannot read UF intra image: " + path.string());
        }
        if (count == 0) break;
        offset += static_cast<std::size_t>(count);
    }
    bytes.resize(offset);
    return bytes;
}

}  // namespace

IntraImageArchive read_intra_image(const std::filesystem::path& path,
                                    bool verify_payload, bool retain_payload) {
    IntraImageArchive result;
    ProcessInterrupt::check();
    result.path = python_path_absolute(path);
    auto input = open_image_file(result.path);
    const auto close_input = [&] {
        const int descriptor = input.release();
        if (close(descriptor) != 0) {
            const int error = errno;
            throw std::system_error(error, std::generic_category(),
                "Cannot close UF intra image: " + result.path.string());
        }
    };
    try {
        if (read_image_bytes(input, 8, result.path) != std::string("UFIMAGE\x01", 8))
            throw std::invalid_argument("Not a UF intra image (RT and UF bitstreams use different networks)");
        const auto size_bytes = read_image_bytes(input, 4, result.path);
        if (size_bytes.size() != 4) throw std::invalid_argument("Truncated UF image header");
        const auto byte = [&](std::size_t index) {
            return static_cast<std::uint32_t>(static_cast<unsigned char>(size_bytes[index]));
        };
        const auto length = (byte(0) << 24u) | (byte(1) << 16u) | (byte(2) << 8u) | byte(3);
        if (!length || length > 1024u * 1024u)
            throw std::invalid_argument("Invalid UF image header length");
        const auto header = read_image_bytes(input, length, result.path);
        const auto text = decode_python_json_bytes(header);
        // The byte decoder consumes one encoding BOM; a remaining U+FEFF is
        // document content and json.loads(bytes) does not consume it again.
        if (text.compare(0, 3, "\xef\xbb\xbf") == 0)
            throw std::invalid_argument("Expecting value");
        auto ordered_metadata = parse_python_ordered_json(text);
        result.metadata = ordered_metadata;
        const auto& data = result.metadata;
        if (python_json_dict_get(data, "format") != "dcvc-uf-image" ||
            !json_truthy(python_json_dict_get(data.at("pipeline"), "image_only")))
            throw std::invalid_argument("Invalid UF image metadata");
        if (!archive_completion_record(data))
            throw std::invalid_argument("Intra image has no recognized completion record");
        if (!json_python_equal(data.at("video").at("frames"), nlohmann::json(1)) ||
            !json_python_equal(data.at("video").at("packets"), nlohmann::json(1)))
            throw std::invalid_argument("Intra image must contain exactly one frame");
        result.recorded_pipeline = std::move(ordered_metadata.at("pipeline"));
        ordered_metadata.clear();
        // stream.read(length) can return a shorter yet valid JSON header at EOF;
        // Python's stream.tell() then reflects the bytes actually consumed.
        result.payload_offset = 12u + header.size();
        if (verify_payload) {
            auto digest = image_digest();
            std::uint64_t payload_size = 0;
            for (;;) {
                const auto chunk = read_image_bytes(input, 1024u * 1024u, result.path);
                if (chunk.empty()) break;
                if (EVP_DigestUpdate(digest.get(), chunk.data(), chunk.size()) != 1)
                    throw std::runtime_error("Cannot hash UF intra image payload");
                if (chunk.size() > std::numeric_limits<std::uint64_t>::max() - payload_size)
                    throw std::overflow_error("UF image payload exceeds native UInt64 size");
                payload_size += chunk.size();
                if (retain_payload)
                    result.payload.insert(result.payload.end(), chunk.begin(), chunk.end());
            }
            const auto actual_hash = image_digest_hex(digest.get());
            if (!json_python_equal(nlohmann::json(actual_hash), data.at("payload_sha256")) ||
                !json_file_size_equal(payload_size, data.at("payload_size")))
                throw std::invalid_argument("UF image payload is incomplete or modified");
        }
        close_input();
        return result;
    } catch (...) {
        // Closing the context may replace its active read/validation failure;
        // release first and never retry a close on the same descriptor.
        if (input.get() >= 0) close_input();
        throw;
    }
}

}  // namespace uf
