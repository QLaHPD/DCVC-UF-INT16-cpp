#include "uf/archive.hpp"
#include "uf/json_compat.hpp"
#include "uf/process.hpp"
#include "uf/python_json.hpp"
#include "uf/python_numeric.hpp"
#include "uf/python_path.hpp"
#include "uf/utf8.hpp"

#include <openssl/evp.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <fcntl.h>
#include <sys/stat.h>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <unistd.h>
#include <vector>

namespace uf {
namespace {

bool safe_basename(const std::string& value) {
    if (value.empty() || value == "." || value == "..") return false;
    const std::filesystem::path path(value);
    return path.filename() == path && !path.has_parent_path();
}

// re.fullmatch(r'(.+)_\d+x\d+_qI\d+_qP\d+\.bin', name): parse the
// suffix from the right so a greedy stem can contain earlier similar suffixes.
// Regex \d uses the reference Unicode decimal tables, not ASCII/isdigit.
// Python's default dot accepts every character except LF, including CR and
// surrogateescaped filename bytes. Keep the original stem bytes unchanged.
std::string_view flat_video_stem(std::string_view name) {
    constexpr std::string_view extension = ".bin";
    if (name.size() < extension.size() ||
        name.substr(name.size() - extension.size()) != extension) return {};
    name.remove_suffix(extension.size());
    constexpr std::array<std::string_view, 2> markers{{"_qP", "_qI"}};
    for (const auto marker : markers) {
        const auto position = name.rfind(marker);
        if (position == std::string_view::npos ||
            !python_decimal_digits(name.substr(position + marker.size()))) return {};
        name = name.substr(0, position);
    }
    const auto separator = name.rfind('_');
    if (separator == std::string_view::npos) return {};
    const auto dimensions = name.substr(separator + 1);
    const auto x = dimensions.find('x');
    if (x == std::string_view::npos ||
        !python_decimal_digits(dimensions.substr(0, x)) ||
        !python_decimal_digits(dimensions.substr(x + 1))) return {};
    const auto stem = name.substr(0, separator);
    if (stem.find('\n') != std::string_view::npos) return {};
    return stem;
}

using JsonSet = std::vector<nlohmann::json>;

void insert_scalar(JsonSet& values, const nlohmann::json& value) {
    if (value.is_array() || value.is_object())
        throw std::invalid_argument("unhashable type: '" +
            std::string(json_compat_detail::python_type_name(value)) + "'");
    if (std::none_of(values.begin(), values.end(), [&](const auto& saved) {
            return json_python_equal(saved, value);
        })) values.push_back(value);
}

// set(dict) consumes keys, set(list) consumes values, and set(str) consumes
// Unicode characters. Parsed strings have valid UTF-8. Scalar JSON receivers
// are not iterable; list/dict elements are unhashable. Native JSON numeric and
// NaN identity limits still apply. No Python object/hash runtime is required.
JsonSet iterated_set(const nlohmann::json& value) {
    JsonSet result;
    if (value.is_object()) {
        for (auto item = value.begin(); item != value.end(); ++item)
            insert_scalar(result, item.key());
    } else if (value.is_array()) {
        for (const auto& item : value) insert_scalar(result, item);
    } else if (value.is_string()) {
        const auto& text = value.get_ref<const std::string&>();
        for (std::size_t offset = 0; offset < text.size();) {
            const auto first = static_cast<unsigned char>(text[offset]);
            const std::size_t length = first < 0x80 ? 1 : first < 0xe0 ? 2 : first < 0xf0 ? 3 : 4;
            insert_scalar(result, text.substr(offset, length));
            offset += length;
        }
    } else {
        throw std::invalid_argument("'" + std::string(json_compat_detail::python_type_name(value)) +
                                    "' object is not iterable");
    }
    return result;
}

bool equal_sets(const JsonSet& first, const JsonSet& second) {
    return first.size() == second.size() &&
        std::all_of(first.begin(), first.end(), [&](const auto& value) {
            return std::any_of(second.begin(), second.end(), [&](const auto& other) {
                return json_python_equal(value, other);
            });
        });
}

void close_archive_descriptor(FileDescriptor& file, const std::filesystem::path& path) {
    const int owned = file.release();
    if (close(owned) != 0) {
        const int error = errno;
        throw std::system_error(error, std::generic_category(),
                                "Cannot close UF completed artifact: " + path.string());
    }
}

std::array<std::uint8_t, 32> parse_identity(const std::string& value) {
    if (value.size() != 64) throw std::runtime_error("Invalid prepared-model identity");
    std::array<std::uint8_t, 32> bytes{};
    for (std::size_t i = 0; i != bytes.size(); ++i) {
        const auto digit = [](char c) -> int {
            if (c >= '0' && c <= '9') return c - '0';
            if (c >= 'a' && c <= 'f') return c - 'a' + 10;
            if (c >= 'A' && c <= 'F') return c - 'A' + 10;
            throw std::runtime_error("Invalid prepared-model identity hex");
        };
        bytes[i] = static_cast<std::uint8_t>(digit(value[2 * i]) * 16 + digit(value[2 * i + 1]));
    }
    return bytes;
}

}  // namespace

std::filesystem::path manifest_path(const std::filesystem::path& input) {
    const auto path = python_path_absolute(input);
    if (python_path_is_directory(path)) return path / "manifest.json";
    const auto filename = path.filename().string();
    if (filename.size() >= 8 && filename.compare(filename.size() - 8, 8, ".uf.json") == 0) {
        return path;
    }
    if (filename.empty()) throw std::invalid_argument("Archive path has an empty name");
    // pathlib 3.10 treats a trailing dot as no suffix, so 'a.' becomes
    // 'a..uf.json'. A leading dot also does not begin a suffix on its own.
    const auto dot = filename.rfind('.');
    const bool has_suffix = dot != std::string::npos && dot > 0 && dot + 1 < filename.size();
    auto candidate = path.parent_path() /
        ((has_suffix ? filename.substr(0, dot) : filename) + ".uf.json");
    if (python_path_exists(candidate)) return candidate;
    const auto stem = flat_video_stem(filename);
    if (!stem.empty()) {
        candidate = path.parent_path() / (std::string(stem) + ".uf.json");
        if (python_path_exists(candidate)) return candidate;
    }
    return path.parent_path() / "manifest.json";
}

std::filesystem::path artifact_path(const Archive& archive, const std::string& logical_name) {
    static const nlohmann::json empty_files = nlohmann::json::object();
    const nlohmann::json fallback(logical_name);
    const auto& files = python_json_dict_get(archive.metadata, "files", empty_files);
    const auto& selected = python_json_dict_get(files, logical_name, fallback);
    if (!json_truthy(selected)) throw std::invalid_argument("Unsafe artifact path");
    const auto actual = selected.get<std::string>();
    if (!safe_basename(actual)) throw std::invalid_argument("Unsafe artifact path: " + actual);
    return archive.root / python_path_from_text(actual);
}

std::string sha256_file(const std::filesystem::path& path) {
    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(EVP_MD_CTX_new(), EVP_MD_CTX_free);
    if (!context || EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1) {
        throw std::runtime_error("OpenSSL SHA-256 initialization failed");
    }
    const auto filename = path.string();
    if (filename.find('\0') != std::string::npos)
        throw std::invalid_argument("embedded null byte");
    int descriptor;
    for (;;) {
        ProcessInterrupt::check();
        descriptor = open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (descriptor >= 0) break;
        const int error = errno;
        if (error == EINTR) continue;
        throw std::system_error(error, std::generic_category(), "Cannot hash file: " + filename);
    }
    FileDescriptor input(descriptor);
    const auto close_input = [&] {
        const int owned = input.release();
        if (close(owned) != 0) {
            const int error = errno;
            throw std::system_error(error, std::generic_category(), "Cannot close hashed file: " + filename);
        }
    };
    try {
        std::vector<char> buffer(4 * 1024 * 1024);
        for (;;) {
            ProcessInterrupt::check();
            const auto count = read(input.get(), buffer.data(), buffer.size());
            if (count < 0) {
                const int error = errno;
                if (error == EINTR) continue;
                throw std::system_error(error, std::generic_category(), "Cannot read hashed file: " + filename);
            }
            if (count == 0) break;
            if (EVP_DigestUpdate(context.get(), buffer.data(), static_cast<std::size_t>(count)) != 1)
                throw std::runtime_error("OpenSSL SHA-256 update failed");
        }
        // The maintained helper leaves its with block before hexdigest().
        close_input();
    } catch (...) {
        if (input.get() >= 0) close_input();
        throw;
    }
    std::array<unsigned char, EVP_MAX_MD_SIZE> hash{};
    unsigned int size = 0;
    if (EVP_DigestFinal_ex(context.get(), hash.data(), &size) != 1 || size != 32) {
        throw std::runtime_error("OpenSSL SHA-256 finalization failed");
    }
    std::ostringstream text;
    text << std::hex << std::setfill('0');
    for (unsigned int i = 0; i != size; ++i) text << std::setw(2) << static_cast<unsigned>(hash[i]);
    return text.str();
}

void copy_completed_archive_file(const std::filesystem::path& source,
                             const std::filesystem::path& target,
                             const std::function<void()>& check) {
    ProcessInterrupt::check();
    if (check) check();
    const auto open_file = [&](const std::filesystem::path& path, int flags) {
        if (path.native().find('\0') != std::string::npos)
            throw std::invalid_argument("embedded null byte");
        for (;;) {
            ProcessInterrupt::check();
            const int descriptor = open(path.c_str(), flags | O_CLOEXEC, 0666);
            if (descriptor >= 0) return FileDescriptor(descriptor);
            const int error = errno;
            if (error == EINTR) continue;
            throw std::system_error(error, std::generic_category(),
                                    "Cannot open UF completed artifact: " + path.string());
        }
    };
    auto input = open_file(source, O_RDONLY);
    try {
        struct stat status{};
        // FileIO initialization tolerates fstat errors except EBADF, and
        // rejects directories before entering the writer context.
        const int status_result = fstat(input.get(), &status);
        const int status_error = errno;
        if (status_result < 0 && status_error == EBADF)
            throw std::system_error(status_error, std::generic_category(),
                                    "Cannot stat UF completed artifact: " + source.string());
        if (status_result == 0 && S_ISDIR(status.st_mode))
            throw std::system_error(EISDIR, std::generic_category(),
                                    "Cannot read UF completed artifact: " + source.string());
        auto output = open_file(target, O_WRONLY | O_CREAT | O_EXCL);
        try {
            std::vector<char> buffer(1024 * 1024);
            for (;;) {
                // BufferedReader.read(n) fills the chunk unless EOF intervenes.
                std::size_t received = 0;
                while (received < buffer.size()) {
                    ProcessInterrupt::check();
                    const auto count = read(input.get(), buffer.data() + received,
                                            buffer.size() - received);
                    if (count < 0) {
                        const int error = errno;
                        if (error == EINTR) continue;
                        throw std::system_error(error, std::generic_category(),
                                                "Cannot read UF completed artifact: " + source.string());
                    }
                    if (count == 0) break;
                    received += static_cast<std::size_t>(count);
                }
                if (received == 0) break;
                if (check) check();
                std::size_t written = 0;
                while (written < received) {
                    ProcessInterrupt::check();
                    const auto count = write(output.get(), buffer.data() + written, received - written);
                    if (count < 0) {
                        const int error = errno;
                        if (error == EINTR) continue;
                        throw std::system_error(error, std::generic_category(),
                                                "Cannot write UF completed artifact: " + target.string());
                    }
                    if (count == 0)
                        throw std::system_error(EIO, std::generic_category(), "Cannot write UF completed artifact");
                    written += static_cast<std::size_t>(count);
                }
            }
            for (;;) {
                ProcessInterrupt::check();
                if (fsync(output.get()) == 0) break;
                const int error = errno;
                if (error == EINTR) continue;
                throw std::system_error(error, std::generic_category(),
                                        "Cannot sync UF completed artifact: " + target.string());
            }
            close_archive_descriptor(output, target);
        } catch (...) {
            // Multiple with entries unwind writer first, then reader. Either
            // checked close may replace the active copy/read/write/sync error.
            if (output.get() >= 0) close_archive_descriptor(output, target);
            throw;
        }
        close_archive_descriptor(input, source);
    } catch (...) {
        if (input.get() >= 0) close_archive_descriptor(input, source);
        throw;
    }
}

Archive load_archive(const std::filesystem::path& input, bool verify_artifacts,
                     const nlohmann::json* expected) {
    Archive result;
    result.manifest = manifest_path(input);
    if (python_path_is_symlink(result.manifest)) {
        throw std::invalid_argument("Archive manifest must not be a symlink");
    }
    result.root = result.manifest.parent_path();
    result.metadata = read_python_json_file(
        result.manifest, python_process_text_encoding(), false);
    const auto& data = result.metadata;
    if (python_json_dict_get(data, "format") != "dcvc-uf-archive" ||
        !json_python_equal(python_json_dict_get(data, "version"), nlohmann::json(1))) {
        throw std::invalid_argument("Not a supported UF archive: " + result.root.string());
    }
    const bool flat = python_json_dict_get(data, "layout") == "flat";
    if (flat) {
        const auto& stem_value = python_json_dict_get(data, "archive_stem");
        if (!stem_value.is_string() || !safe_basename(stem_value.get_ref<const std::string&>()))
            throw std::invalid_argument("Invalid flat archive name");
        const auto& stem = stem_value.get_ref<const std::string&>();
        if (python_path_text(result.manifest.filename()) != stem + ".uf.json")
            throw std::invalid_argument("Flat manifest name does not match archive");
        static const nlohmann::json empty_object = nlohmann::json::object();
        const auto files_set = iterated_set(python_json_dict_get(data, "files", empty_object));
        const auto artifacts_set = iterated_set(python_json_dict_get(data, "artifacts", empty_object));
        if (!equal_sets(files_set, artifacts_set))
            throw std::invalid_argument("Flat artifact map is incomplete");
        const auto& files = data.at("files");
        if (!files.is_object()) json_compat_detail::missing_attribute(files, "values");
        JsonSet distinct;
        for (const auto& name : files) insert_scalar(distinct, name);
        if (distinct.size() != files.size())
            throw std::invalid_argument("Flat artifacts must have distinct paths");
    }
    const auto input_path = python_path_normalize(input);
    if (flat && input_path.extension() == ".bin" &&
        input_path.filename() != artifact_path(result, "video.bin").filename()) {
        throw std::invalid_argument("Input bitstream does not belong to this manifest");
    }
    if (expected && !expected->is_null() && !json_python_equal(data.at("pipeline"), *expected))
        throw std::invalid_argument("Existing archive uses different encoding settings/models: " + result.root.string());
    if (!archive_completion_record(data))
        throw std::invalid_argument("Archive has no recognized completion record: " + result.root.string());
    const auto& artifacts = data.at("artifacts");
    if (!artifacts.is_object()) json_compat_detail::missing_attribute(artifacts, "items");
    for (auto it = artifacts.begin(); it != artifacts.end(); ++it) {
        // load_bundle checks logical names separately from artifact_path's
        // selected stored path. The snapshot caller only checks stored paths.
        if (it.key() == "." || it.key().find('/') != std::string::npos)
            throw std::invalid_argument("Unsafe artifact name: " + it.key());
        const auto path = artifact_path(result, it.key());
        if (!verify_artifacts) continue;
        const auto& info = it.value();
        if (python_path_is_symlink(path) || !python_path_is_file(path) ||
            !json_python_equal(nlohmann::json(python_path_size(path)), info.at("size")) ||
            !json_python_equal(nlohmann::json(sha256_file(path)), info.at("sha256"))) {
            throw std::invalid_argument("Archive artifact missing or changed: " + path.string());
        }
    }
    return result;
}

std::filesystem::path memory_stage_root() {
    const auto mounts = read_python_text_file("/proc/mounts", python_process_text_encoding());
    for (const auto& line : python_splitlines(mounts)) {
        const auto fields = python_split_whitespace(line);
        if (fields.size() >= 3 && fields[1] == "/dev/shm" && fields[2] == "tmpfs")
            return "/dev/shm";
    }
    throw std::runtime_error("Encoding requires /dev/shm mounted as tmpfs for RAM staging");
}

Sps archive_video_dimensions(const nlohmann::json& video) {
    const auto dimension = [](const nlohmann::json& value) {
        const auto size = python_json_exact_uint64(value);
        if (size > std::numeric_limits<std::uint32_t>::max())
            throw std::overflow_error("UF archive dimension exceeds native UInt32 range");
        return static_cast<std::uint32_t>(size);
    };
    return {0, dimension(video.at("width")), dimension(video.at("height"))};
}

ScanResult scan_archive_bitstream(const Archive& archive) {
    const auto& data = archive.metadata;
    if (!data.contains("video") || !data.contains("pipeline")) {
        throw std::runtime_error("UF archive has no video or pipeline metadata");
    }
    const auto& video = data["video"];
    const auto& pipeline = data["pipeline"];
    const auto result = scan_bitstream(artifact_path(archive, "video.bin"),
                                       pipeline.at("variant").get<std::string>(),
                                       python_json_exact_uint64(video.at("frames")));
    if (!json_python_equal(nlohmann::json(result.dimensions.width), video.at("width")) ||
        !json_python_equal(nlohmann::json(result.dimensions.height), video.at("height")) ||
        !json_python_equal(nlohmann::json(result.packets), video.at("packets"))) {
        throw std::runtime_error("UF packet structure differs from archive metadata");
    }
    const auto runtime = pipeline.value("runtime", std::string("fp16"));
    if (runtime == "int16") {
        if (!result.preamble.integer) throw std::runtime_error("Missing UF16 preamble");
        const auto& integer = pipeline.at("integer");
        if (integer.at("arithmetic") != "uf-int16-q9-w13-acc64-rna-sat-v1" ||
            result.preamble.image_identity != parse_identity(integer.at("image").get<std::string>()) ||
            result.preamble.video_identity != parse_identity(integer.at("video").get<std::string>())) {
            throw std::runtime_error("UF16 prepared-model identity mismatch");
        }
    } else if (runtime == "fp16") {
        if (result.preamble.integer) throw std::runtime_error("Unexpected UF16 preamble");
    } else {
        throw std::runtime_error("Unsupported UF archive runtime");
    }
    return result;
}

}  // namespace uf
