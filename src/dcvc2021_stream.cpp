// Copyright 2020 InterDigital Communications, Inc.
// Licensed under Apache-2.0; see licenses/Apache-2.0.txt.
// Native adaptation of the original DCVC stream_helper.py.
#include "uf/dcvc2021.hpp"
#include "uf/process.hpp"

#include <algorithm>
#include <array>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace uf::dcvc2021 {
namespace {
void uint32(std::ostream& output, std::uint32_t value) {
    const std::array<char, 4> bytes{static_cast<char>(value >> 24), static_cast<char>(value >> 16),
        static_cast<char>(value >> 8), static_cast<char>(value)};
    output.write(bytes.data(), bytes.size());
    if (!output) throw std::runtime_error("Cannot write DCVC stream header");
}
std::uint32_t uint32(std::istream& input) {
    std::array<unsigned char, 4> bytes{};
    input.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
    if (!input) throw std::runtime_error("Truncated DCVC stream header");
    return (std::uint32_t(bytes[0]) << 24) | (std::uint32_t(bytes[1]) << 16) |
           (std::uint32_t(bytes[2]) << 8) | bytes[3];
}
std::uint32_t length(const std::string& value) {
    if (value.size() > std::numeric_limits<std::uint32_t>::max()) throw std::out_of_range("DCVC payload exceeds uint32");
    return static_cast<std::uint32_t>(value.size());
}
void bytes(std::ostream& output, const std::string& value) {
    for (std::size_t position = 0; position < value.size();) {
        ProcessInterrupt::check();
        const auto count = std::min<std::size_t>(65536, value.size() - position);
        output.write(value.data() + position, static_cast<std::streamsize>(count));
        if (!output) throw std::runtime_error("Cannot write DCVC stream payload");
        position += count;
    }
}
std::string bytes(std::istream& input, std::uint32_t count) {
    std::string result(count, '\0');
    for (std::size_t position = 0; position < result.size();) {
        ProcessInterrupt::check();
        const auto chunk = std::min<std::size_t>(65536, result.size() - position);
        input.read(result.data() + position, static_cast<std::streamsize>(chunk));
        if (!input) throw std::runtime_error("Truncated DCVC stream payload");
        position += chunk;
    }
    return result;
}
void check_payload(std::istream& input, std::uint64_t count) {
    const auto start = input.tellg();
    input.seekg(0, std::ios::end);
    const auto end = input.tellg();
    input.seekg(start);
    if (!input || start < std::streampos(0) || end < start ||
        count > static_cast<std::uint64_t>(end - start)) {
        throw std::runtime_error("Truncated DCVC stream payload");
    }
    // The reference reader intentionally leaves trailing bytes unread.
}
void finish(std::ostream& output) {
    output.flush();
    if (!output) throw std::runtime_error("Cannot finish DCVC stream");
}
}  // namespace

void write_p(const std::filesystem::path& output, const PFrameStream& stream) {
    std::ofstream file(output, std::ios::binary | std::ios::trunc);
    if (!file) throw std::runtime_error("Cannot open DCVC P-frame output");
    const std::array<std::uint32_t, 6> header{stream.height, stream.width, length(stream.mv_y_string),
        length(stream.mv_z_string), length(stream.y_string), length(stream.z_string)};
    for (auto value : header) uint32(file, value);
    bytes(file, stream.mv_y_string); bytes(file, stream.mv_z_string);
    bytes(file, stream.y_string); bytes(file, stream.z_string);
    finish(file);
}
PFrameStream read_p(const std::filesystem::path& input) {
    std::ifstream file(input, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open DCVC P-frame input");
    std::array<std::uint32_t, 6> header{};
    for (auto& value : header) value = uint32(file);
    check_payload(file, std::uint64_t(header[2]) + header[3] + header[4] + header[5]);
    PFrameStream result;
    result.height = header[0]; result.width = header[1];
    result.mv_y_string = bytes(file, header[2]); result.mv_z_string = bytes(file, header[3]);
    result.y_string = bytes(file, header[4]); result.z_string = bytes(file, header[5]);
    return result;
}
void write_i(const std::filesystem::path& output, const IFrameStream& stream) {
    std::ofstream file(output, std::ios::binary | std::ios::trunc);
    if (!file) throw std::runtime_error("Cannot open DCVC I-frame output");
    const std::array<std::uint32_t, 4> header{stream.height, stream.width,
        length(stream.y_string), length(stream.z_string)};
    for (auto value : header) uint32(file, value);
    bytes(file, stream.y_string); bytes(file, stream.z_string);
    finish(file);
}
IFrameStream read_i(const std::filesystem::path& input) {
    std::ifstream file(input, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open DCVC I-frame input");
    std::array<std::uint32_t, 4> header{};
    for (auto& value : header) value = uint32(file);
    check_payload(file, std::uint64_t(header[2]) + header[3]);
    IFrameStream result;
    result.height = header[0]; result.width = header[1];
    result.y_string = bytes(file, header[2]); result.z_string = bytes(file, header[3]);
    return result;
}
}  // namespace uf::dcvc2021
