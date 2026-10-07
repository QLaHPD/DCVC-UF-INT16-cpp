// Copyright 2020 InterDigital Communications, Inc., Apache-2.0.
// Native adaptation of DCVC-TCM/src/entropy_models/entropy_models.py.
#include "uf/dcvc_tcm_image.hpp"
#include "dcvc2021_ndtri.hpp"
#include <cmath>
#include <stdexcept>

namespace uf::tcm::image {
namespace {
torch::Tensor cumulative(const torch::Tensor& input) {
    return 0.5 * torch::erfc(-std::pow(2.0, -0.5) * input);
}
void require_coder(bool initialized) {
    if (!initialized) throw std::runtime_error("TCM image entropy coder is uninitialized; call update(force=true)");
}
}  // namespace

torch::Tensor d_quant(const torch::Tensor& input, const torch::Tensor& means) {
    const auto residual = input - means;
    const auto noise = (residual.round() - residual).clone().detach();
    return residual + noise + means;
}
EntropyBottleneckImpl::EntropyBottleneckImpl(std::int64_t channels)
    : dcvc2021::image::EntropyBottleneckImpl(channels) {}
void EntropyBottleneckImpl::update(bool force) {
    // A checkpoint with populated offsets skips coder initialization too.
    if (offset.numel() && !force) return;
    coder_initialized_ = true;
    dcvc2021::image::EntropyBottleneckImpl::update(force);
}
std::vector<std::string> EntropyBottleneckImpl::compress(const torch::Tensor& input) const {
    require_coder(coder_initialized_);
    return dcvc2021::image::EntropyBottleneckImpl::compress(input);
}
torch::Tensor EntropyBottleneckImpl::decompress(const std::vector<std::string>& strings,
    std::pair<std::int64_t, std::int64_t> shape) const {
    require_coder(coder_initialized_);
    return dcvc2021::image::EntropyBottleneckImpl::decompress(strings, shape);
}

GaussianConditionalImpl::GaussianConditionalImpl(double bound, double tail_mass, double likelihood_bound)
    : dcvc2021::image::EntropyModelImpl(likelihood_bound), tail_mass_(tail_mass),
      log_min_(std::log(0.11)), log_step_((std::log(256.0) - log_min_) / 63) {
    scale_table = torch::linspace(log_min_, std::log(256.0), 64).exp();
    scale_bound = register_buffer("scale_bound", torch::empty({1}).fill_(bound));
    if (!(bound > 0)) throw std::invalid_argument("Invalid TCM Gaussian scale bound");
    lower_bound_scale = register_module("lower_bound_scale", dcvc2021::image::LowerBound(bound));
}
void GaussianConditionalImpl::update() {
    coder_initialized_ = true;
    const auto multiplier = -dcvc2021::image::detail::normal_quantile(tail_mass_ / 2);
    const auto center = (scale_table * multiplier).ceil().to(torch::kInt32);
    const auto pmf_lengths = 2 * center + 1;
    const auto maximum = pmf_lengths.max().item<std::int64_t>();
    const auto samples = (torch::arange(maximum, torch::TensorOptions().device(center.device()))
        .to(torch::kInt32).unsqueeze(0) - center.unsqueeze(1)).abs().to(torch::kFloat32);
    const auto scales = scale_table.unsqueeze(1).to(torch::kFloat32);
    const auto upper = cumulative((0.5 - samples) / scales);
    const auto lower = cumulative((-0.5 - samples) / scales);
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
torch::Tensor GaussianConditionalImpl::build_indexes(const torch::Tensor& scales) const {
    const auto bounded = torch::maximum(scales, torch::zeros_like(scales) + 1e-5);
    return ((bounded.log() - log_min_) / log_step_ + 1).clamp_(0, 63).to(torch::kInt32);
}
std::vector<std::string> GaussianConditionalImpl::compress(const torch::Tensor& input,
    const torch::Tensor& indexes, const torch::Tensor& means) const {
    require_coder(coder_initialized_);
    return dcvc2021::image::EntropyModelImpl::compress(input, indexes, means);
}
torch::Tensor GaussianConditionalImpl::decompress(const std::vector<std::string>& strings,
    const torch::Tensor& indexes, const torch::Tensor& means) const {
    require_coder(coder_initialized_);
    return dcvc2021::image::EntropyModelImpl::decompress(strings, indexes, means);
}
}  // namespace uf::tcm::image
