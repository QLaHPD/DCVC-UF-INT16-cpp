// Copyright 2020 InterDigital Communications, Inc., Apache-2.0.
// DCVC-TCM additions to the shared image residual/GDN layers.
#include "uf/dcvc_tcm_image.hpp"
#include <stdexcept>

namespace uf::tcm::image {
GDN1Impl::GDN1Impl(std::int64_t channels, bool inverse, double beta_min, double gamma_init)
    : dcvc2021::image::GDNImpl(channels, inverse, beta_min, gamma_init), inverse_(inverse) {}
torch::Tensor GDN1Impl::forward(const torch::Tensor& input) {
    if (input.dim() != 4) throw std::invalid_argument("TCM GDN1 input must be four dimensional");
    const auto b = beta_reparam(beta);
    const auto g = gamma_reparam(gamma).reshape({input.size(1), input.size(1), 1, 1});
    auto norm = torch::conv2d(input.abs(), g, b);
    if (!inverse_) norm = 1.0 / norm;
    return input * norm;
}
torch::nn::Conv2d conv3x3(std::int64_t in, std::int64_t out, std::int64_t stride) {
    return dcvc2021::image::conv(in, out, 3, stride);
}
torch::nn::Conv2d conv1x1(std::int64_t in, std::int64_t out, std::int64_t stride) {
    return dcvc2021::image::conv(in, out, 1, stride);
}
torch::nn::Sequential subpel_conv3x3(std::int64_t in, std::int64_t out, std::int64_t factor) {
    return dcvc2021::image::subpel(in, out, factor);
}
ResidualBlockImpl::ResidualBlockImpl(std::int64_t in, std::int64_t out, double slope) {
    conv1 = register_module("conv1", conv3x3(in, out));
    leaky_relu = register_module("leaky_relu", torch::nn::LeakyReLU(
        torch::nn::LeakyReLUOptions().inplace(true).negative_slope(slope)));
    conv2 = register_module("conv2", conv3x3(out, out));
}
torch::Tensor ResidualBlockImpl::forward(const torch::Tensor& input) {
    return leaky_relu(conv2(leaky_relu(conv1(input)))) + input;
}
ResidualBlockType2Impl::ResidualBlockType2Impl(std::int64_t in, std::int64_t out) {
    conv1 = register_module("conv1", conv3x3(in, out));
    relu = register_module("relu", torch::nn::ReLU(torch::nn::ReLUOptions().inplace(false)));
    conv2 = register_module("conv2", conv3x3(out, out));
    if (in != out) throw std::invalid_argument("TCM ResidualBlockType2 requires equal channel counts");
}
torch::Tensor ResidualBlockType2Impl::forward(const torch::Tensor& input) {
    return conv2(relu(conv1(relu(input)))) + input;
}
}  // namespace uf::tcm::image
