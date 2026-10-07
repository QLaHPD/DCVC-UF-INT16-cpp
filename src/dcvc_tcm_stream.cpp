// Copyright 2020 InterDigital Communications, Inc., Apache-2.0.
// DCVC-TCM stream_helper.py: a length-prefixed combined P-frame payload.
#include "uf/dcvc_tcm.hpp"
#include "uf/process.hpp"
#include <algorithm>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace uf::tcm {
torch::Tensor pad(const torch::Tensor& input, std::int64_t multiple) {
    const auto shape = dcvc2021::downsampled_shape(input.size(2), input.size(3), multiple);
    if (shape.first > std::numeric_limits<std::int64_t>::max() / multiple ||
        shape.second > std::numeric_limits<std::int64_t>::max() / multiple)
        throw std::overflow_error("TCM padded image dimensions exceed int64");
    return torch::nn::functional::pad(input, torch::nn::functional::PadFuncOptions(
        {0, shape.second * multiple - input.size(3), 0, shape.first * multiple - input.size(2)})
        .mode(torch::kConstant).value(0));
}
torch::Tensor crop(const torch::Tensor& input, std::int64_t height, std::int64_t width) {
    if (height <= 0 || width <= 0) throw std::invalid_argument("Invalid TCM crop dimensions");
    return torch::nn::functional::pad(input, torch::nn::functional::PadFuncOptions(
        {0, width - input.size(3), 0, height - input.size(2)}).mode(torch::kConstant).value(0));
}
void write_p(const std::filesystem::path& path, const std::string& stream) {
    if (stream.size() > std::numeric_limits<std::uint32_t>::max()) throw std::out_of_range("TCM P-frame payload exceeds uint32");
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("Cannot open TCM P-frame output");
    const auto length = static_cast<std::uint32_t>(stream.size());
    const std::array<char, 4> bytes{static_cast<char>(length >> 24), static_cast<char>(length >> 16),
        static_cast<char>(length >> 8), static_cast<char>(length)};
    output.write(bytes.data(), bytes.size());
    for (std::size_t position = 0; position < stream.size();) {
        ProcessInterrupt::check();
        const auto count = std::min<std::size_t>(65536, stream.size() - position);
        output.write(stream.data() + position, static_cast<std::streamsize>(count));
        if (!output) throw std::runtime_error("Cannot write TCM P-frame payload");
        position += count;
    }
    output.flush();
    if (!output) throw std::runtime_error("Cannot finish TCM P-frame output");
}
std::string read_p(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot open TCM P-frame input");
    std::array<unsigned char, 4> header{};
    input.read(reinterpret_cast<char*>(header.data()), header.size());
    if (!input) throw std::runtime_error("Truncated TCM P-frame header");
    const auto length = (std::uint32_t(header[0]) << 24) | (std::uint32_t(header[1]) << 16) |
        (std::uint32_t(header[2]) << 8) | header[3];
    const auto begin = input.tellg();
    input.seekg(0, std::ios::end); const auto end = input.tellg(); input.seekg(begin);
    if (!input || begin < std::streampos(0) || end < begin || length > static_cast<std::uint64_t>(end - begin))
        throw std::runtime_error("Truncated TCM P-frame payload");
    std::string stream(length, 0);
    for (std::size_t position = 0; position < stream.size();) {
        ProcessInterrupt::check();
        const auto count = std::min<std::size_t>(65536, stream.size() - position);
        input.read(stream.data() + position, static_cast<std::streamsize>(count));
        if (!input) throw std::runtime_error("Cannot read TCM P-frame payload");
        position += count;
    }
    return stream;  // The reference reader leaves trailing bytes unread.
}
}  // namespace uf::tcm
