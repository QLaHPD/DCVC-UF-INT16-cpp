// Copyright 2020 InterDigital Communications, Inc., Apache-2.0.
#include "uf/dcvc_dc.hpp"
#include "uf/process.hpp"
#include "uf/torch_pickle.hpp"
#include <algorithm>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace uf::dc {
namespace {
void integer(std::ostream& output, std::uint32_t value, int width) {
    for (int index = width - 1; index >= 0; --index) output.put(static_cast<char>(value >> (8 * index)));
    if (!output) throw std::runtime_error("Cannot write DC stream header");
}
std::uint32_t integer(std::istream& input, int width) {
    std::uint32_t result = 0;
    for (int index = 0; index < width; ++index) {
        const auto byte = input.get();
        if (byte == std::char_traits<char>::eof()) throw std::runtime_error("Truncated DC stream header");
        result = (result << 8) | static_cast<unsigned char>(byte);
    }
    return result;
}
std::uint8_t flag(bool in_checkpoint, std::uint8_t index) {
    if (index > 63) throw std::out_of_range("DC quality index exceeds six bits");
    return static_cast<std::uint8_t>((in_checkpoint ? 128 : 0) + (index << 1));
}
void payload(std::ostream& output, const std::string& value) {
    if (value.size() > std::numeric_limits<std::uint32_t>::max()) throw std::out_of_range("DC payload exceeds uint32");
    integer(output, static_cast<std::uint32_t>(value.size()), 4);
    for (std::size_t offset = 0; offset < value.size();) {
        ProcessInterrupt::check();
        const auto count = std::min<std::size_t>(65536, value.size() - offset);
        output.write(value.data() + offset, static_cast<std::streamsize>(count)); offset += count;
        if (!output) throw std::runtime_error("Cannot write DC stream payload");
    }
    output.flush();
    if (!output) throw std::runtime_error("Cannot finish DC stream");
}
std::string payload(std::istream& input) {
    const auto count = integer(input, 4);
    const auto first = input.tellg();
    input.seekg(0, std::ios::end); const auto last = input.tellg(); input.seekg(first);
    if (!input || first < std::streampos(0) || last < first || count > static_cast<std::uint64_t>(last - first))
        throw std::runtime_error("Truncated DC stream payload");
    std::string result(count, '\0');
    for (std::size_t offset = 0; offset < result.size();) {
        ProcessInterrupt::check();
        const auto size = std::min<std::size_t>(65536, result.size() - offset);
        input.read(result.data() + offset, static_cast<std::streamsize>(size)); offset += size;
        if (!input) throw std::runtime_error("Truncated DC stream payload");
    }
    return result;
}
}  // namespace
StateDict get_state_dict(const std::filesystem::path& path) {
    auto value = load_torch_pickle_cpu(path);
    const auto unwrap = [&](const char* name) {
        if (!value.isGenericDict()) throw std::runtime_error("DC checkpoint must be a dictionary");
        const auto dictionary = value.toGenericDict();
        if (const auto found = dictionary.find(name); found != dictionary.end()) value = found->value();
    };
    unwrap("state_dict"); unwrap("net");
    if (!value.isGenericDict()) throw std::runtime_error("DC checkpoint state must be a dictionary");
    StateDict result;
    std::vector<std::string> keys;
    for (const auto& entry : value.toGenericDict()) {
        if (!entry.key().isString() || !entry.value().isTensor()) throw std::runtime_error("Invalid DC state dictionary entry");
        const auto key = entry.key().toStringRef();
        keys.push_back(key); result[key] = entry.value().toTensor();
    }
    for (const auto& key : keys) if (key.rfind("module.", 0) == 0) {
        const auto tensor = result.at(key);
        result.erase(key); result[key.substr(7)] = tensor;
    }
    return result;
}
torch::Tensor pad(const torch::Tensor& input, std::int64_t multiple) {
    const auto size = dcvc2021::downsampled_shape(input.size(2), input.size(3), multiple);
    if (size.first > std::numeric_limits<std::int64_t>::max() / multiple ||
        size.second > std::numeric_limits<std::int64_t>::max() / multiple)
        throw std::overflow_error("DC padded size exceeds int64");
    return torch::nn::functional::pad(input, torch::nn::functional::PadFuncOptions(
        {0, size.second * multiple - input.size(3), 0, size.first * multiple - input.size(2)}).mode(torch::kConstant).value(0));
}
torch::Tensor crop(const torch::Tensor& input, std::int64_t height, std::int64_t width) {
    return torch::nn::functional::pad(input, torch::nn::functional::PadFuncOptions(
        {0, width - input.size(3), 0, height - input.size(2)}).mode(torch::kConstant).value(0));
}
void write_i(const std::filesystem::path& path, const IFrameStream& stream) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("Cannot open DC I-frame output");
    integer(output, stream.height, 4); integer(output, stream.width, 4);
    integer(output, flag(stream.q_in_ckpt, stream.q_index), 1); payload(output, stream.bit_stream);
}
IFrameStream read_i(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot open DC I-frame input");
    IFrameStream result;
    result.height = integer(input, 4); result.width = integer(input, 4);
    const auto quality = integer(input, 1);
    result.q_in_ckpt = (quality >> 7) > 0; result.q_index = static_cast<std::uint8_t>((quality & 0x7f) >> 1);
    result.bit_stream = payload(input);
    return result;
}
void write_p(const std::filesystem::path& path, const PFrameStream& stream) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("Cannot open DC P-frame output");
    integer(output, flag(stream.q_in_ckpt, stream.q_index), 1); integer(output, stream.frame_idx, 1);
    payload(output, stream.bit_stream);
}
PFrameStream read_p(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot open DC P-frame input");
    PFrameStream result;
    const auto quality = integer(input, 1);
    result.q_in_ckpt = (quality >> 7) > 0; result.q_index = static_cast<std::uint8_t>((quality & 0x7f) >> 1);
    result.frame_idx = static_cast<std::uint8_t>(integer(input, 1)); result.bit_stream = payload(input);
    return result;
}
}  // namespace uf::dc
