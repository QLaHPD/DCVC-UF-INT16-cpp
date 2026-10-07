// Copyright 2020 InterDigital Communications, Inc.
// Licensed under Apache-2.0; see licenses/Apache-2.0.txt.
// Native UF stream/SPS interfaces adapted from src/utils/stream_helper.py.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <istream>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

namespace uf {

enum class NalType : std::uint8_t { sps = 0, intra = 1, predicted = 2 };

struct NalHeader {
    NalType type{};
    std::uint8_t sps_id{};
};

struct Sps {
    std::uint8_t id{};
    std::uint32_t width{};
    std::uint32_t height{};
};

// Typed counterpart of Python SPSHelper. Records are copied when registered
// and shared when looked up, so a lookup remains alive after ID replacement.
class SpsHelper {
public:
    void add_sps_by_id(const Sps& sps);
    std::shared_ptr<Sps> get_sps_by_id(std::uint8_t id) const;
    // Ignore target.id, reuse the first matching dimensions, or allocate one
    // above the largest registered ID. New IDs are limited to the wire's 0..15.
    std::pair<std::uint8_t, bool> get_sps_id(const Sps& target);

private:
    std::vector<std::shared_ptr<Sps>> spss_;
};

struct Packet {
    NalType type{};
    std::uint8_t sps_id{};
    std::uint8_t qp{};
    std::uint8_t entropy_parallel{};
    bool reset_feature_memory{};
    std::vector<std::uint8_t> payload;
};

struct Preamble {
    bool integer{};
    std::array<std::uint8_t, 32> image_identity{};
    std::array<std::uint8_t, 32> video_identity{};
};

struct ScanResult {
    Preamble preamble;
    Sps dimensions;
    std::uint64_t packets{};
    std::uint64_t intra_packets{};
    std::uint64_t predicted_packets{};
    std::uint64_t maximum_frames{};
};

// POSIX counterpart of Python filesize: follow symlinks, require a regular
// file on the first stat, then return the second stat's signed byte count.
// Invalid paths/non-files throw invalid_argument; other stat failures throw
// system_error. Dot/separator normalization preserves '..' for symlinks.
std::int64_t filesize(const std::filesystem::path& filepath);
// Typed counterparts of UF's default byte/string and unsigned-char formats.
// Reads require the requested count; both return byte vectors. Empty byte
// writes are skipped, while write_uchars performs even a zero-length write.
// Counts are nonnegative native sizes; custom Python struct format strings
// and arbitrary Python iterable/value conversion are not exposed here.
std::vector<std::uint8_t> read_bytes(std::istream& input, std::size_t count);
std::vector<std::uint8_t> read_uchars(std::istream& input, std::size_t count);
std::size_t write_bytes(std::ostream& output, const std::vector<std::uint8_t>& values);
std::size_t write_uchars(std::ostream& output, const std::vector<std::uint8_t>& values);
std::uint32_t read_uint_adaptive(std::istream& input);
// Adaptive integer, SPS and packet writers return serialized byte counts,
// as the maintained Python helpers do.
std::size_t write_uint_adaptive(std::ostream& output, std::uint32_t value);
std::optional<std::uint8_t> read_optional_byte(std::istream& input);
// UF supports only SPS/I/P NAL types. Invalid type nibbles throw; optional
// reading returns empty only at clean EOF, while read_header requires a byte.
std::optional<NalHeader> read_optional_header(std::istream& input);
NalHeader read_header(std::istream& input);
// Detect a valid UF16 preamble without seeking or consuming FP16 NAL headers.
// A prefix beginning with 'U' must be the UF16 signature/version/identities.
Preamble read_preamble(std::istream& input);
void write_preamble(std::ostream& output, const Preamble& value);
Sps read_sps(std::istream& input, std::uint8_t id);
std::size_t write_sps(std::ostream& output, const Sps& value);
// Non-seekable payload reads grow from received chunks. Seekable buffers
// reject lengths beyond their remaining data, then allocate the exact size.
Packet read_packet(std::istream& input, NalType type, std::uint8_t sps_id);
std::size_t write_packet(std::ostream& output, const Packet& value);
ScanResult scan_bitstream(const std::filesystem::path& path, std::string variant,
                          std::optional<std::uint64_t> expected_frames = std::nullopt);
ScanResult scan_bitstream(std::istream& input, std::string variant,
                          std::optional<std::uint64_t> expected_frames = std::nullopt);

}  // namespace uf
