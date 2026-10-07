// Copyright 2020 InterDigital Communications, Inc., Apache-2.0.
// Native DCVC-HEM stream_helper.py and checkpoint normalization.
#include "dcvc_hem_internal.hpp"
#include "uf/process.hpp"
#include "uf/torch_pickle.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace uf::hem {
namespace {
void integer(std::ostream& output, std::uint32_t value, int width) {
    for (int i = width - 1; i >= 0; --i) output.put(static_cast<char>(value >> (8 * i)));
    if (!output) throw std::runtime_error("Cannot write HEM stream header");
}
std::uint32_t integer(std::istream& input, int width) {
    std::uint32_t value = 0;
    for (int i = 0; i < width; ++i) { const auto byte = input.get(); if (byte == std::char_traits<char>::eof())
        throw std::runtime_error("Truncated HEM stream header"); value = (value << 8) | static_cast<unsigned char>(byte); }
    return value;
}
void payload(std::ostream& output, const std::string& value) {
    if (value.size() > std::numeric_limits<std::uint32_t>::max()) throw std::out_of_range("HEM payload exceeds uint32");
    integer(output, static_cast<std::uint32_t>(value.size()), 4);
    for (std::size_t offset = 0; offset < value.size();) {
        ProcessInterrupt::check(); const auto count = std::min<std::size_t>(65536, value.size() - offset);
        output.write(value.data() + offset, static_cast<std::streamsize>(count)); offset += count;
        if (!output) throw std::runtime_error("Cannot write HEM stream payload");
    }
    output.flush(); if (!output) throw std::runtime_error("Cannot finish HEM stream");
}
std::string payload(std::istream& input) {
    const auto count = integer(input, 4); const auto first = input.tellg();
    input.seekg(0, std::ios::end); const auto last = input.tellg(); input.seekg(first);
    if (!input || first < std::streampos(0) || last < first || count > static_cast<std::uint64_t>(last - first))
        throw std::runtime_error("Truncated HEM stream payload");
    std::string result(count, '\0');
    for (std::size_t offset = 0; offset < result.size();) {
        ProcessInterrupt::check(); const auto chunk = std::min<std::size_t>(65536, result.size() - offset);
        input.read(result.data() + offset, static_cast<std::streamsize>(chunk)); offset += chunk;
        if (!input) throw std::runtime_error("Truncated HEM stream payload");
    }
    return result;  // The source leaves trailing bytes unread.
}
RoundedQ rounded(double integer_value) {
    if (!std::isfinite(integer_value) || integer_value < 0 || integer_value > 65535)
        throw std::invalid_argument("Invalid HEM quality index");
    const auto index = static_cast<std::uint16_t>(integer_value);
    return {index / 100.0, index};
}
}  // namespace
torch::Tensor detail::scalar(double value) {
    return torch::tensor(value, torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat64));
}
std::uint32_t detail::dimension(std::int64_t value) {
    if (value < 0 || value > std::numeric_limits<std::uint32_t>::max()) throw std::out_of_range("HEM dimension exceeds uint32");
    return static_cast<std::uint32_t>(value);
}
RoundedQ get_rounded_q(double scale) {
    const auto value = std::clamp(scale, 0.01, 655.0) * 100;
    const auto floor = std::floor(value), fraction = value - floor;
    return rounded(fraction > 0.5 || (fraction == 0.5 && std::fmod(floor, 2) != 0) ? floor + 1 : floor);
}
RoundedQ get_rounded_q(const torch::Tensor& scale) {
    if (!scale.defined() || scale.numel() != 1) throw std::invalid_argument("HEM quality scale must be scalar");
    // NumPy 1.26.1's _wrapfunc falls back to ndarray.clip because Tensor.clip
    // does not accept out=None. Its legacy scalar promotion uses Float64 for
    // zero-dimensional inputs plus the Python floating bounds.
    if (!scale.device().is_cpu() || scale.requires_grad() || scale.scalar_type() == torch::kBFloat16)
        throw std::invalid_argument("HEM NumPy-compatible quality rounding requires a CPU tensor convertible to NumPy");
    if (scale.dim() == 0 || (!scale.is_floating_point() && !scale.is_complex()))
        return get_rounded_q(scale.item<double>());
    return rounded((scale.clamp(0.01, 655.0) * 100).round().item<double>());
}
StateDict get_state_dict(const std::filesystem::path& path) {
    auto value = load_torch_pickle_cpu(path);
    const auto unwrap = [&](const char* name) {
        if (!value.isGenericDict()) throw std::runtime_error("HEM checkpoint must be a dictionary");
        const auto dictionary = value.toGenericDict();
        if (const auto found = dictionary.find(name); found != dictionary.end()) value = found->value();
    };
    unwrap("state_dict"); unwrap("net");
    if (!value.isGenericDict()) throw std::runtime_error("HEM checkpoint state must be a dictionary");
    StateDict result; std::vector<std::string> keys;
    for (const auto& entry : value.toGenericDict()) {
        if (!entry.key().isString() || !entry.value().isTensor()) throw std::runtime_error("Invalid HEM state dictionary entry");
        const auto key = entry.key().toStringRef(); keys.push_back(key); result[key] = entry.value().toTensor();
    }
    // consume_prefix_in_state_dict_if_present mutates the dictionary in a
    // second pass. A prefixed key overwrites its existing unprefixed key.
    for (const auto& key : keys) if (key.rfind("module.", 0) == 0) {
        const auto tensor = result.at(key); result.erase(key); result[key.substr(7)] = tensor;
    }
    return result;
}
torch::Tensor pad(const torch::Tensor& input, std::int64_t multiple) {
    const auto size = dcvc2021::downsampled_shape(input.size(2), input.size(3), multiple);
    if (size.first > std::numeric_limits<std::int64_t>::max() / multiple || size.second > std::numeric_limits<std::int64_t>::max() / multiple)
        throw std::overflow_error("HEM padded size exceeds int64");
    return torch::nn::functional::pad(input, torch::nn::functional::PadFuncOptions(
        {0, size.second * multiple - input.size(3), 0, size.first * multiple - input.size(2)}).mode(torch::kConstant).value(0));
}
torch::Tensor crop(const torch::Tensor& input, std::int64_t height, std::int64_t width) {
    return torch::nn::functional::pad(input, torch::nn::functional::PadFuncOptions(
        {0, width - input.size(3), 0, height - input.size(2)}).mode(torch::kConstant).value(0));
}
void write_i(const std::filesystem::path& path, const IFrameStream& stream) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc); if (!output) throw std::runtime_error("Cannot open HEM I-frame output");
    integer(output, stream.height, 4); integer(output, stream.width, 4); integer(output, stream.q_index, 2); payload(output, stream.bit_stream);
}
IFrameStream read_i(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary); if (!input) throw std::runtime_error("Cannot open HEM I-frame input");
    IFrameStream result; result.height = integer(input, 4); result.width = integer(input, 4);
    result.q_index = static_cast<std::uint16_t>(integer(input, 2)); result.bit_stream = payload(input); return result;
}
void write_p(const std::filesystem::path& path, const PFrameStream& stream) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc); if (!output) throw std::runtime_error("Cannot open HEM P-frame output");
    integer(output, stream.mv_y_q_index, 2); integer(output, stream.y_q_index, 2); payload(output, stream.bit_stream);
}
PFrameStream read_p(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary); if (!input) throw std::runtime_error("Cannot open HEM P-frame input");
    PFrameStream result; result.mv_y_q_index = static_cast<std::uint16_t>(integer(input, 2));
    result.y_q_index = static_cast<std::uint16_t>(integer(input, 2)); result.bit_stream = payload(input); return result;
}
}  // namespace uf::hem
