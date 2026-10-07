// Copyright 2020 InterDigital Communications, Inc., Apache-2.0.
// Native adaptation of the original DCVC family's entropy_models.py.
#include "uf/dcvc2021_image.hpp"
#include "uf/process.hpp"
#include "dcvc2021_ndtri.hpp"
#include "dcvc2021_image_internal.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace uf::dcvc2021::image {
namespace {
std::vector<std::int32_t> integers(const torch::Tensor& input) {
    const auto cpu = input.to(torch::kCPU, torch::kInt32).contiguous().view({-1});
    if (!cpu.numel()) return {};
    return {cpu.data_ptr<std::int32_t>(), cpu.data_ptr<std::int32_t>() + cpu.numel()};
}
torch::Tensor floating_values(const std::vector<double>& values) {
    auto result = torch::empty({static_cast<std::int64_t>(values.size())});
    if (!values.empty()) result.copy_(torch::from_blob(const_cast<double*>(values.data()),
        {static_cast<std::int64_t>(values.size())}, torch::TensorOptions().dtype(torch::kFloat64)));
    return result;
}
torch::Tensor decoded_values(const std::vector<std::int32_t>& values) {
    auto integer = torch::empty({static_cast<std::int64_t>(values.size())},
                                torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt32));
    std::copy(values.begin(), values.end(), integer.data_ptr<std::int32_t>());
    auto result = torch::empty(integer.sizes(), torch::TensorOptions().device(torch::kCPU));
    result.copy_(integer);  // Python Tensor(list), including its default dtype.
    return result;
}
torch::Tensor indexes(std::int64_t batch, std::int64_t channels, std::int64_t height, std::int64_t width) {
    return torch::arange(channels, torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt32))
        .view({1, channels, 1, 1}).repeat({batch, 1, height, width});
}
torch::Tensor cumulative(const torch::Tensor& input) { return 0.5 * torch::erfc(-std::pow(2.0, -0.5) * input); }
}  // namespace

EntropyModelImpl::EntropyModelImpl(double likelihood_bound) {
    if (likelihood_bound > 0) likelihood_lower_bound =
        register_module("likelihood_lower_bound", LowerBound(likelihood_bound));
    const auto options = torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt32);
    offset = register_buffer("_offset", torch::empty({0}, options));
    quantized = register_buffer("_quantized_cdf", torch::empty({0}, options));
    lengths = register_buffer("_cdf_length", torch::empty({0}, options));
}
void EntropyModelImpl::to(torch::Device device, torch::Dtype dtype, bool non_blocking) {
    detail::move_module(*this, device, dtype, non_blocking);
}
void EntropyModelImpl::to(torch::Dtype dtype, bool non_blocking) {
    detail::move_module(*this, std::nullopt, dtype, non_blocking);
}
void EntropyModelImpl::to(torch::Device device, bool non_blocking) {
    detail::move_module(*this, device, std::nullopt, non_blocking);
}

torch::Tensor EntropyModelImpl::quantize(const torch::Tensor& input, const std::string& mode,
                                        const torch::Tensor& means) {
    if (mode != "dequantize" && mode != "symbols") throw std::invalid_argument("Invalid quantization mode: " + mode);
    auto output = input.clone();
    if (means.defined()) output.sub_(means);
    output = output.round();
    if (mode == "symbols") return output.to(torch::kInt32);
    if (means.defined()) output.add_(means);
    return output;
}
torch::Tensor EntropyModelImpl::dequantize(const torch::Tensor& input, const torch::Tensor& means) {
    if (!means.defined()) return input.to(torch::kFloat32);
    auto output = input.to(means.options());
    output.add_(means);
    return output;
}

CdfTables EntropyModelImpl::tables() const {
    if (!quantized.numel() || !offset.numel() || !lengths.numel()) throw std::runtime_error("Uninitialized CDFs; call update() first");
    if (quantized.dim() != 2 || offset.dim() != 1 || lengths.dim() != 1 ||
        quantized.size(0) != offset.numel() || offset.numel() != lengths.numel()) {
        throw std::runtime_error("Invalid image entropy CDF shape");
    }
    const auto flat = integers(quantized), sizes = integers(lengths);
    CdfTables result;
    result.offsets = integers(offset);
    for (std::size_t i = 0; i < sizes.size(); ++i) {
        if (sizes[i] < 3 || sizes[i] > quantized.size(1)) throw std::runtime_error("Invalid image entropy CDF length");
        const auto first = flat.begin() + i * quantized.size(1);
        result.rows.emplace_back(first, first + sizes[i]);
    }
    result.validate();
    return result;
}

void EntropyModelImpl::set_cdf(const torch::Tensor& pmf, const torch::Tensor& tails,
                               const torch::Tensor& pmf_lengths, const torch::Tensor& offsets) {
    const auto probabilities = pmf.to(torch::kCPU, torch::kFloat32).contiguous();
    const auto tail = tails.to(torch::kCPU, torch::kFloat32).contiguous();
    const auto sizes = integers(pmf_lengths);
    auto cdf = torch::zeros({static_cast<std::int64_t>(sizes.size()), pmf.size(1) + 2},
                             torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt32));
    for (std::size_t i = 0; i < sizes.size(); ++i) {
        ProcessInterrupt::check();
        if (sizes[i] < 1 || sizes[i] > pmf.size(1)) throw std::runtime_error("Invalid image entropy PMF length");
        const auto* first = probabilities.data_ptr<float>() + i * probabilities.size(1);
        std::vector<float> values(first, first + sizes[i]);
        values.push_back(tail.data_ptr<float>()[i]);
        const auto row = quantized_cdf(values);
        std::copy(row.begin(), row.end(), cdf.data_ptr<std::int32_t>() + i * cdf.size(1));
    }
    // set_data keeps the registered buffer handles while changing shape and
    // device. The reference creates CDFs on CPU even for a GPU bottleneck.
    quantized.set_data(cdf);
    offset.set_data(offsets.detach());
    lengths.set_data((pmf_lengths + 2).detach());
}

std::vector<std::string> EntropyModelImpl::compress(const torch::Tensor& input, const torch::Tensor& indices,
                                                   const torch::Tensor& means) const {
    const auto symbols = quantize(input, "symbols", means);
    if (input.dim() != 4 || input.sizes() != indices.sizes()) throw std::invalid_argument("Invalid image entropy input/index shape");
    const auto cdf = tables();
    std::vector<std::string> strings;
    for (std::int64_t i = 0; i < input.size(0); ++i) strings.push_back(
        rans_encode(integers(symbols.select(0, i)), integers(indices.select(0, i)), cdf));
    return strings;
}

torch::Tensor EntropyModelImpl::decompress(const std::vector<std::string>& strings, const torch::Tensor& indices,
                                          const torch::Tensor& means) const {
    if (indices.dim() != 4 || static_cast<std::int64_t>(strings.size()) != indices.size(0)) {
        throw std::invalid_argument("Invalid image entropy strings/index shape");
    }
    const auto cdf = tables();
    if (means.defined() && (means.dim() != 4 || means.size(0) != indices.size(0) ||
        means.size(1) != indices.size(1) || (means.sizes() != indices.sizes() && (means.size(2) != 1 || means.size(3) != 1)))) {
        throw std::invalid_argument("Invalid image entropy means/index shape");
    }
    auto output = torch::empty(indices.sizes(), quantized.options());
    RansDecoder decoder;
    for (std::size_t i = 0; i < strings.size(); ++i) {
        decoder.set_stream(strings[i]);
        const auto values = decoder.decode(integers(indices.select(0, i)), cdf);
        // Preserve Tensor(list) -> integer-buffer assignment before adding
        // medians/means, including reference rounding of large integers.
        output.select(0, i).copy_(decoded_values(values).reshape(output.select(0, i).sizes()));
    }
    return dequantize(output, means);
}

EntropyBottleneckImpl::EntropyBottleneckImpl(std::int64_t channels, double tail_mass, double init_scale,
    std::vector<std::int64_t> filters, double likelihood_bound)
    : EntropyModelImpl(likelihood_bound), channels_(channels) {
    biases = register_module("_biases", torch::nn::ParameterList());
    factors = register_module("_factors", torch::nn::ParameterList());
    matrices = register_module("_matrices", torch::nn::ParameterList());
    std::vector<std::int64_t> sizes{1};
    sizes.insert(sizes.end(), filters.begin(), filters.end());
    sizes.push_back(1);
    const auto scale = std::pow(init_scale, 1.0 / (filters.size() + 1));
    for (std::size_t i = 0; i + 1 < sizes.size(); ++i) {
        const auto initial = std::log(std::expm1(1 / scale / sizes[i + 1]));
        auto matrix = torch::empty({channels, sizes[i + 1], sizes[i]}).fill_(initial);
        matrices->append(matrix.set_requires_grad(true));
        auto bias = torch::empty({channels, sizes[i + 1], 1}).uniform_(-0.5, 0.5);
        biases->append(bias.set_requires_grad(true));
        if (i < filters.size()) {
            auto factor = torch::zeros({channels, sizes[i + 1], 1});
            factors->append(factor.set_requires_grad(true));
        }
    }
    quantiles = register_parameter("quantiles", floating_values({-init_scale, 0, init_scale}).view({1, 1, 3}).repeat({channels, 1, 1}));
    const auto desired = std::log(2 / tail_mass - 1);
    target = register_buffer("target", floating_values({-desired, 0, desired}));
}

torch::Tensor EntropyBottleneckImpl::medians() const { return quantiles.slice(2, 1, 2); }

torch::Tensor EntropyBottleneckImpl::logits_cumulative(const torch::Tensor& input, bool stop_gradient) {
    auto logits = input;
    for (std::size_t i = 0; i < matrices->size(); ++i) {
        auto matrix = matrices->at(i), bias = biases->at(i);
        if (stop_gradient) { matrix = matrix.detach(); bias = bias.detach(); }
        logits = torch::matmul(torch::softplus(matrix), logits);
        logits.add_(bias);
        if (i < factors->size()) {
            auto factor = factors->at(i);
            if (stop_gradient) factor = factor.detach();
            logits.add_(factor.tanh() * logits.tanh());
        }
    }
    return logits;
}
torch::Tensor EntropyBottleneckImpl::likelihood(const torch::Tensor& input) {
    const auto lower = logits_cumulative(input - 0.5, false), upper = logits_cumulative(input + 0.5, false);
    const auto sign = -(lower + upper).sign().detach();
    return ((sign * upper).sigmoid() - (sign * lower).sigmoid()).abs();
}
std::pair<torch::Tensor, torch::Tensor> EntropyBottleneckImpl::forward(const torch::Tensor& input) {
    const auto channels_first = input.permute({1, 2, 3, 0}).contiguous();
    const auto shape = channels_first.sizes().vec();
    const auto output = quantize(channels_first.reshape({channels_first.size(0), 1, -1}), "dequantize", medians());
    auto probability = likelihood(output);
    if (likelihood_lower_bound) probability = likelihood_lower_bound(probability);
    return {output.reshape(shape).permute({3, 0, 1, 2}).contiguous(),
            probability.reshape(shape).permute({3, 0, 1, 2}).contiguous()};
}
void EntropyBottleneckImpl::update(bool force) {
    if (offset.numel() && !force) return;
    torch::NoGradGuard no_grad;
    const auto middle = quantiles.select(1, 0).select(1, 1);
    const auto minima = (middle - quantiles.select(1, 0).select(1, 0)).ceil().to(torch::kInt32).clamp_min(0);
    const auto maxima = (quantiles.select(1, 0).select(1, 2) - middle).ceil().to(torch::kInt32).clamp_min(0);
    const auto pmf_lengths = maxima + minima + 1;
    const auto maximum = pmf_lengths.max().item<std::int64_t>();
    const auto samples = torch::arange(maximum, torch::TensorOptions().device(middle.device())).view({1, 1, maximum}) +
                         (middle - minima).view({channels_, 1, 1});
    const auto lower = logits_cumulative(samples - 0.5, true), upper = logits_cumulative(samples + 0.5, true);
    const auto sign = -(lower + upper).sign();
    const auto pmf = ((sign * upper).sigmoid() - (sign * lower).sigmoid()).abs().select(1, 0);
    const auto tails = lower.select(1, 0).slice(1, 0, 1).sigmoid() +
                       (-upper.select(1, 0).slice(1, maximum - 1)).sigmoid();
    set_cdf(pmf, tails, pmf_lengths, -minima);
}
std::vector<std::string> EntropyBottleneckImpl::compress(const torch::Tensor& input) const {
    return EntropyModelImpl::compress(input, indexes(input.size(0), input.size(1), input.size(2), input.size(3)),
                                      medians().detach().view({1, -1, 1, 1}));
}
torch::Tensor EntropyBottleneckImpl::decompress(const std::vector<std::string>& strings,
    std::pair<std::int64_t, std::int64_t> size) const {
    const auto channels = quantized.size(0);
    return EntropyModelImpl::decompress(strings, indexes(strings.size(), channels, size.first, size.second),
                                        medians().detach().view({1, -1, 1, 1}));
}

GaussianConditionalImpl::GaussianConditionalImpl(torch::Tensor scales, double bound, double tail_mass, double likelihood_bound)
    : EntropyModelImpl(likelihood_bound), tail_mass_(tail_mass) {
    if (!(bound > 0)) throw std::invalid_argument("Invalid Gaussian scale bound");
    auto prepared = torch::empty({0});
    if (scales.defined()) {
        if (scales.dim() != 1 || !scales.numel() || !(scales > 0).all().item<bool>() ||
            (scales.slice(0, 1) < scales.slice(0, 0, scales.numel() - 1)).any().item<bool>()) {
            throw std::invalid_argument("Invalid Gaussian scale table");
        }
        prepared = torch::empty(scales.sizes());
        prepared.copy_(scales.to(torch::kCPU));
    }
    scale_table = register_buffer("scale_table", prepared);
    scale_bound = register_buffer("scale_bound", floating_values({bound}));
    lower_bound_scale = register_module("lower_bound_scale", LowerBound(bound));
}
void GaussianConditionalImpl::update_scale_table(const torch::Tensor& values, bool force) {
    if (offset.numel() && !force) return;
    if (values.dim() != 1) throw std::invalid_argument("Invalid Gaussian scale-table dimensions");
    auto prepared = torch::empty(values.sizes());
    prepared.copy_(values.to(torch::kCPU));
    scale_table.set_data(prepared);
    update();
}
void GaussianConditionalImpl::update() {
    const auto multiplier = -detail::normal_quantile(tail_mass_ / 2);
    const auto center = (scale_table * multiplier).ceil().to(torch::kInt32);
    const auto pmf_lengths = 2 * center + 1;
    const auto maximum = pmf_lengths.max().item<std::int64_t>();
    const auto samples = (torch::arange(maximum, torch::TensorOptions().device(center.device())).to(torch::kInt32)
                           .unsqueeze(0) - center.unsqueeze(1)).abs().to(torch::kFloat32);
    const auto scales = scale_table.unsqueeze(1).to(torch::kFloat32);
    const auto upper = cumulative((0.5 - samples) / scales), lower = cumulative((-0.5 - samples) / scales);
    set_cdf(upper - lower, 2 * lower.slice(1, 0, 1), pmf_lengths, -center);
}
torch::Tensor GaussianConditionalImpl::likelihood(const torch::Tensor& input, const torch::Tensor& scales,
                                                   const torch::Tensor& means) {
    const auto values = (means.defined() ? input - means : input).abs();
    const auto bounded = lower_bound_scale(scales);
    return cumulative((0.5 - values) / bounded) - cumulative((-0.5 - values) / bounded);
}
std::pair<torch::Tensor, torch::Tensor> GaussianConditionalImpl::forward(
    const torch::Tensor& input, const torch::Tensor& scales, const torch::Tensor& means) {
    const auto output = quantize(input, "dequantize", means);
    auto probability = likelihood(output, scales, means);
    if (likelihood_lower_bound) probability = likelihood_lower_bound(probability);
    return {output, probability};
}
torch::Tensor GaussianConditionalImpl::build_indexes(const torch::Tensor& scales) {
    const auto bounded = lower_bound_scale(scales);
    auto result = torch::full_like(bounded, scale_table.numel() - 1).to(torch::kInt32);
    for (std::int64_t i = 0; i + 1 < scale_table.numel(); ++i) result -=
        (bounded <= scale_table.select(0, i)).to(torch::kInt32);
    return result;
}
torch::Tensor default_scale_table() { return torch::linspace(std::log(0.11), std::log(256.0), 64).exp(); }
}  // namespace uf::dcvc2021::image
