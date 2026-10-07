// Copyright 2020 InterDigital Communications, Inc.
// Licensed under Apache-2.0; see licenses/Apache-2.0.txt.
// Native UF adaptation with C++ stream handling and shared SPS ownership.
// Path normalization/error policy follows CPython 3.10 pathlib; see
// licenses/CPython-3.10.20.txt. Installed libc declarations were read only.
#include "uf/bitstream.hpp"
#include "uf/python_path.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <fstream>
#include <limits>
#include <streambuf>
#include <stdexcept>
#include <system_error>
#include <sys/stat.h>
#include <utility>

namespace uf {
namespace {

std::uint8_t read_byte(std::istream& input) {
    const int value = input.get();
    if (value == std::char_traits<char>::eof()) {
        throw std::runtime_error("Truncated UF bitstream");
    }
    return static_cast<std::uint8_t>(value);
}

void write_byte(std::ostream& output, std::uint8_t value) {
    output.put(static_cast<char>(value));
    if (!output) throw std::runtime_error("Failed to write UF bitstream");
}

void read_exact(std::istream& input, std::uint8_t* destination, std::size_t size) {
    input.read(reinterpret_cast<char*>(destination), static_cast<std::streamsize>(size));
    if (input.gcount() != static_cast<std::streamsize>(size)) {
        throw std::runtime_error("Truncated UF bitstream payload");
    }
}

bool check_payload_available(std::istream& input, std::size_t size) {
    // Probe the buffer directly: tellg/seekg failures can set stream flags and
    // prevent subsequent reads from a pipe or another non-seekable buffer.
    auto* buffer = input.rdbuf();
    const auto start = buffer->pubseekoff(0, std::ios::cur, std::ios::in);
    if (start == std::streampos(-1)) return false;
    const auto end = buffer->pubseekoff(0, std::ios::end, std::ios::in);
    // A buffer may expose its current position without supporting movement.
    // A failed end probe supplies no bound; continue with sequential reads.
    if (end == std::streampos(-1)) return false;
    if (buffer->pubseekpos(start, std::ios::in) == std::streampos(-1)) {
        throw std::runtime_error("Cannot restore UF bitstream payload position");
    }
    if (end < start || static_cast<std::uint64_t>(end - start) < size) {
        throw std::runtime_error("Truncated UF bitstream payload");
    }
    return true;
}

}  // namespace

std::int64_t filesize(const std::filesystem::path& filepath) {
    const auto& original = filepath.native();
    const auto invalid_file = [&]() {
        return std::invalid_argument("Invalid file \"" + original + "\".");
    };
    if (original.find('\0') != std::string::npos) throw invalid_file();
    const auto path = python_path_normalize(filepath);
    struct stat info{};
    if (::stat(path.c_str(), &info) != 0) {
        const int error = errno;
        // Path.is_file treats these failures as a non-file, but propagates
        // other operating-system failures such as permission errors.
        if (error == ENOENT || error == ENOTDIR || error == EBADF || error == ELOOP) {
            throw invalid_file();
        }
        throw std::system_error(error, std::generic_category(), "Cannot inspect UF file: " + original);
    }
    if (!S_ISREG(info.st_mode)) throw invalid_file();
    // Keep the separate size stat: Python's helper performs a second lookup
    // and propagates failures even if the earlier regular-file check passed.
    if (::stat(path.c_str(), &info) != 0) {
        const int error = errno;
        throw std::system_error(error, std::generic_category(), "Cannot inspect UF file size: " + original);
    }
    return static_cast<std::int64_t>(info.st_size);
}

std::vector<std::uint8_t> read_bytes(std::istream& input, std::size_t count) {
    std::vector<std::uint8_t> result;
    if (count > result.max_size() ||
        count > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
        throw std::length_error("UF byte read exceeds the native buffer size limit");
    }
    if (input.bad() || (input.fail() && !input.eof())) {
        throw std::runtime_error("Failed to read UF bytes");
    }
    if (!count) return result;
    if (check_payload_available(input, count)) {
        result.resize(count);
        read_exact(input, result.data(), count);
        return result;
    }
    // Keep only the temporary read block bounded; a successful call still
    // returns all bytes. Grow from received data on non-seekable inputs.
    std::array<std::uint8_t, 64 * 1024> block{};
    auto remaining = count;
    while (remaining) {
        const auto size = std::min(remaining, block.size());
        read_exact(input, block.data(), size);
        result.insert(result.end(), block.begin(), block.begin() + size);
        remaining -= size;
    }
    return result;
}

std::vector<std::uint8_t> read_uchars(std::istream& input, std::size_t count) {
    return read_bytes(input, count);
}

std::size_t write_uchars(std::ostream& output, const std::vector<std::uint8_t>& values) {
    if (values.size() > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
        throw std::length_error("UF byte write exceeds the native stream size limit");
    }
    const auto* data = values.empty() ? "" : reinterpret_cast<const char*>(values.data());
    output.write(data, static_cast<std::streamsize>(values.size()));
    if (!output) throw std::runtime_error("Failed to write UF bytes");
    return values.size();
}

std::size_t write_bytes(std::ostream& output, const std::vector<std::uint8_t>& values) {
    if (values.empty()) return 0;
    return write_uchars(output, values);
}

void SpsHelper::add_sps_by_id(const Sps& sps) {
    for (auto& existing : spss_) {
        if (existing->id == sps.id) {
            existing = std::make_shared<Sps>(sps);
            return;
        }
    }
    spss_.push_back(std::make_shared<Sps>(sps));
}

std::shared_ptr<Sps> SpsHelper::get_sps_by_id(std::uint8_t id) const {
    for (const auto& sps : spss_) {
        if (sps->id == id) return sps;
    }
    return {};
}

std::pair<std::uint8_t, bool> SpsHelper::get_sps_id(const Sps& target) {
    int largest_id = -1;
    for (const auto& sps : spss_) {
        if (sps->height == target.height && sps->width == target.width) {
            return {sps->id, false};
        }
        largest_id = std::max(largest_id, static_cast<int>(sps->id));
    }
    if (largest_id >= 15) throw std::out_of_range("UF SPS registry has no unused ID");
    auto sps = std::make_shared<Sps>(target);
    sps->id = static_cast<std::uint8_t>(largest_id + 1);
    const auto id = sps->id;
    spss_.push_back(std::move(sps));
    return {id, true};
}

std::optional<std::uint8_t> read_optional_byte(std::istream& input) {
    const int value = input.get();
    if (value == std::char_traits<char>::eof()) {
        if (input.bad() || !input.eof()) throw std::runtime_error("Failed to read UF bitstream");
        return std::nullopt;
    }
    return static_cast<std::uint8_t>(value);
}

std::optional<NalHeader> read_optional_header(std::istream& input) {
    const auto byte = read_optional_byte(input);
    if (!byte) return std::nullopt;
    const auto type = *byte >> 4u;
    if (type > 2) throw std::runtime_error("Unsupported UF NAL type");
    return NalHeader{static_cast<NalType>(type),
                     static_cast<std::uint8_t>(*byte & 0x0fu)};
}

NalHeader read_header(std::istream& input) {
    const auto header = read_optional_header(input);
    if (!header) throw std::runtime_error("Truncated UF NAL header");
    return *header;
}

std::uint32_t read_uint_adaptive(std::istream& input) {
    const std::uint32_t high = read_byte(input);
    if (!(high & 0x80u)) return high;
    const std::uint32_t next = read_byte(input);
    if (!(high & 0x40u)) return ((high & 0x3fu) << 8u) | next;
    const std::uint32_t third = read_byte(input);
    const std::uint32_t low = read_byte(input);
    return ((high & 0x3fu) << 24u) | (next << 16u) | (third << 8u) | low;
}

std::size_t write_uint_adaptive(std::ostream& output, std::uint32_t value) {
    if (value < (1u << 7u)) {
        write_byte(output, static_cast<std::uint8_t>(value));
        return 1;
    } else if (value < (1u << 14u)) {
        write_byte(output, static_cast<std::uint8_t>(0x80u | (value >> 8u)));
        write_byte(output, static_cast<std::uint8_t>(value));
        return 2;
    } else if (value < (1u << 30u)) {
        write_byte(output, static_cast<std::uint8_t>(0xc0u | (value >> 24u)));
        write_byte(output, static_cast<std::uint8_t>(value >> 16u));
        write_byte(output, static_cast<std::uint8_t>(value >> 8u));
        write_byte(output, static_cast<std::uint8_t>(value));
        return 4;
    } else {
        throw std::out_of_range("UF adaptive integer exceeds 30 bits");
    }
}

Preamble read_preamble(std::istream& input) {
    Preamble result;
    const auto first = input.peek();
    if (first == std::char_traits<char>::eof()) {
        if (input.bad() || !input.eof()) throw std::runtime_error("Failed to read UF bitstream");
        return result;
    }
    // All valid FP16 record headers have a NAL type of 0, 1 or 2. UF16 starts
    // with 'U' (NAL type 5), so a single-byte peek distinguishes valid streams
    // without consuming/replaying FP16 bytes or imposing a four-byte minimum.
    if (first != 'U') return result;
    std::array<std::uint8_t, 4> prefix{};
    read_exact(input, prefix.data(), prefix.size());
    if (prefix != std::array<std::uint8_t, 4>{'U', 'F', '1', '6'}) {
        throw std::runtime_error("Unsupported UF preamble signature");
    }
    const std::array<std::uint8_t, 4> version{1, 0, 0, 0};
    std::array<std::uint8_t, 4> actual{};
    read_exact(input, actual.data(), actual.size());
    if (actual != version) throw std::runtime_error("Unsupported UF16 preamble version");
    result.integer = true;
    read_exact(input, result.image_identity.data(), result.image_identity.size());
    read_exact(input, result.video_identity.data(), result.video_identity.size());
    return result;
}

void write_preamble(std::ostream& output, const Preamble& value) {
    if (!value.integer) return;
    const std::array<std::uint8_t, 8> magic{'U', 'F', '1', '6', 1, 0, 0, 0};
    output.write(reinterpret_cast<const char*>(magic.data()), magic.size());
    output.write(reinterpret_cast<const char*>(value.image_identity.data()),
                 value.image_identity.size());
    output.write(reinterpret_cast<const char*>(value.video_identity.data()),
                 value.video_identity.size());
    if (!output) throw std::runtime_error("Failed to write UF16 preamble");
}

Sps read_sps(std::istream& input, std::uint8_t id) {
    Sps result{id, 0, 0};
    result.height = read_uint_adaptive(input);
    result.width = read_uint_adaptive(input);
    if (!result.width || !result.height) throw std::runtime_error("Invalid UF dimensions");
    return result;
}

std::size_t write_sps(std::ostream& output, const Sps& value) {
    if (value.id >= 16 || !value.width || !value.height) {
        throw std::invalid_argument("Invalid UF SPS");
    }
    write_byte(output, value.id);
    const auto height_bytes = write_uint_adaptive(output, value.height);
    const auto width_bytes = write_uint_adaptive(output, value.width);
    return 1 + height_bytes + width_bytes;
}

Packet read_packet(std::istream& input, NalType type, std::uint8_t sps_id) {
    if (type != NalType::intra && type != NalType::predicted) {
        throw std::invalid_argument("Expected an I or P packet");
    }
    Packet result;
    result.type = type;
    result.sps_id = sps_id;
    result.qp = read_byte(input);
    const auto flags = read_byte(input);
    result.entropy_parallel = flags >> 1u;
    result.reset_feature_memory = (flags & 1u) != 0;
    const auto size = read_uint_adaptive(input);
    result.payload = read_bytes(input, size);
    return result;
}

std::size_t write_packet(std::ostream& output, const Packet& value) {
    if (value.sps_id >= 16 || value.entropy_parallel >= 128 ||
        (value.type != NalType::intra && value.type != NalType::predicted) ||
        value.payload.size() >= (1u << 30u)) {
        throw std::invalid_argument("Invalid UF packet");
    }
    write_byte(output, (static_cast<std::uint8_t>(value.type) << 4u) | value.sps_id);
    write_byte(output, value.qp);
    write_byte(output, (value.entropy_parallel << 1u) | value.reset_feature_memory);
    const auto length_bytes = write_uint_adaptive(output, static_cast<std::uint32_t>(value.payload.size()));
    const auto payload_bytes = write_bytes(output, value.payload);
    if (!output) throw std::runtime_error("Failed to write UF packet");
    return 3 + length_bytes + payload_bytes;
}

ScanResult scan_bitstream(const std::filesystem::path& path, std::string variant,
                          std::optional<std::uint64_t> expected_frames) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot open UF bitstream: " + path.string());
    return scan_bitstream(input, std::move(variant), expected_frames);
}

ScanResult scan_bitstream(std::istream& input, std::string variant,
                          std::optional<std::uint64_t> expected_frames) {
    if (variant != "ld" && variant != "hts" && variant != "htl") {
        throw std::invalid_argument("UF model variant must be ld, hts or htl");
    }
    ScanResult result;
    result.preamble = read_preamble(input);
    std::array<std::optional<Sps>, 16> sps_table{};
    const std::uint64_t chunk = variant == "ld" ? 1 : 8;
    bool first_packet = true;
    while (const auto header = read_optional_header(input)) {
        const auto type = header->type;
        const auto sps_id = header->sps_id;
        if (type == NalType::sps) {
            sps_table[sps_id] = read_sps(input, sps_id);
            continue;
        }
        if (!sps_table[sps_id]) throw std::runtime_error("UF packet uses an unknown SPS");
        if (first_packet && type != NalType::intra) {
            throw std::runtime_error("UF bitstream must start with an I packet");
        }
        first_packet = false;
        const auto& sps = *sps_table[sps_id];
        if (result.packets &&
            (result.dimensions.width != sps.width || result.dimensions.height != sps.height)) {
            throw std::runtime_error("UF dimensions changed within the stream");
        }
        result.dimensions = sps;
        if (expected_frames && result.maximum_frames >= *expected_frames) {
            throw std::runtime_error("Unexpected extra UF packet");
        }
        const auto packet = read_packet(input, type, sps_id);
        if (!packet.entropy_parallel) throw std::runtime_error("Invalid UF entropy parallelism");
        ++result.packets;
        if (type == NalType::intra) ++result.intra_packets;
        else ++result.predicted_packets;
        result.maximum_frames += type == NalType::intra ? 1 : chunk;
    }
    if (!result.packets) throw std::runtime_error("UF bitstream has no coded frames");
    if (expected_frames && result.maximum_frames < *expected_frames) {
        throw std::runtime_error("UF bitstream has too few coded frames");
    }
    return result;
}

}  // namespace uf
