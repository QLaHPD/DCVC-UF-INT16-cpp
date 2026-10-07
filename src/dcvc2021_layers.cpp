// Copyright (c) Microsoft Corporation. Licensed under the MIT License.
// video_net.py / stream_helper.py in the original DCVC family.
// MaskedConv2d adaptation: Copyright 2020 InterDigital Communications, Inc.
// Licensed under Apache-2.0; see licenses/Apache-2.0.txt.
#include "uf/dcvc2021.hpp"

#include <torch/csrc/autograd/custom_function.h>

#include <cmath>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <tuple>

namespace uf::dcvc2021 {
namespace {
class LowerBound : public torch::autograd::Function<LowerBound> {
public:
    static torch::Tensor forward(torch::autograd::AutogradContext* context,
                                 const torch::Tensor& input, double bound) {
        auto floor = torch::ones_like(input) * bound;
        context->save_for_backward({input, floor});
        return torch::maximum(input, floor);
    }
    static torch::autograd::variable_list backward(torch::autograd::AutogradContext* context,
                                                   torch::autograd::variable_list gradients) {
        const auto saved = context->get_saved_variables();
        return {((saved.at(0) >= saved.at(1)) | (gradients.at(0) < 0))
                    .to(gradients.at(0).scalar_type()) * gradients.at(0), torch::Tensor{}};
    }
};

torch::Tensor double_size(const torch::Tensor& input) {
    return torch::nn::functional::interpolate(input,
        torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<std::int64_t>{input.size(2) * 2, input.size(3) * 2})
            .mode(torch::kBilinear).align_corners(false));
}
}  // namespace

std::pair<std::int64_t, std::int64_t> downsampled_shape(
    std::int64_t height, std::int64_t width, std::int64_t divisor) {
    if (height <= 0 || width <= 0 || divisor <= 0 ||
        height > std::numeric_limits<std::int64_t>::max() - divisor + 1 ||
        width > std::numeric_limits<std::int64_t>::max() - divisor + 1) {
        throw std::invalid_argument("Invalid DCVC image dimensions");
    }
    return {(height + divisor - 1) / divisor, (width + divisor - 1) / divisor};
}

torch::Tensor pad(const torch::Tensor& input, std::int64_t multiple) {
    const auto size = downsampled_shape(input.size(2), input.size(3), multiple);
    if (size.first > std::numeric_limits<std::int64_t>::max() / multiple ||
        size.second > std::numeric_limits<std::int64_t>::max() / multiple) {
        throw std::overflow_error("DCVC padded size exceeds int64");
    }
    const auto dh = size.first * multiple - input.size(2), dw = size.second * multiple - input.size(3);
    return torch::nn::functional::pad(input, torch::nn::functional::PadFuncOptions(
        {dw / 2, dw - dw / 2, dh / 2, dh - dh / 2}).mode(torch::kConstant).value(0));
}

torch::Tensor crop(const torch::Tensor& input, std::int64_t height, std::int64_t width) {
    if (height <= 0 || width <= 0) throw std::invalid_argument("Invalid DCVC crop dimensions");
    // Python // floors negative values too (crop can also add padding).
    const auto half = [](std::int64_t value) { return value / 2 - (value < 0 && value % 2 ? 1 : 0); };
    const auto dh = input.size(2) - height, dw = input.size(3) - width;
    const auto top = half(dh), left = half(dw);
    return torch::nn::functional::pad(input, torch::nn::functional::PadFuncOptions(
        {-left, -(dw - left), -top, -(dh - top)}).mode(torch::kConstant).value(0));
}

torch::Tensor flow_warp(const torch::Tensor& image, const torch::Tensor& flow) {
    // The Python grids are made by CPU linspace before transfer to CUDA.
    // Retain grids by device/flow shape, including their original dtype.
    using GridKey = std::tuple<int, int, std::int64_t, std::int64_t, std::int64_t, std::int64_t>;
    static std::map<GridKey, torch::Tensor> grids;
    static std::mutex grid_mutex;
    const GridKey key{static_cast<int>(image.device().type()), image.device().index(),
                      flow.size(0), flow.size(1), flow.size(2), flow.size(3)};
    torch::Tensor base;
    {
        std::lock_guard<std::mutex> lock(grid_mutex);
        auto found = grids.find(key);
        if (found == grids.end()) {
            const auto cpu = torch::TensorOptions().device(torch::kCPU);
            const auto horizontal = torch::linspace(-1.0, 1.0, flow.size(3), cpu)
                .view({1, 1, 1, flow.size(3)}).expand({flow.size(0), 1, flow.size(2), flow.size(3)});
            const auto vertical = torch::linspace(-1.0, 1.0, flow.size(2), cpu)
                .view({1, 1, flow.size(2), 1}).expand({flow.size(0), 1, flow.size(2), flow.size(3)});
            found = grids.emplace(key, torch::cat({horizontal, vertical}, 1).to(image.device())).first;
        }
        base = found->second;
    }
    const auto normalized = torch::cat({
        flow.slice(1, 0, 1) / ((image.size(3) - 1.0) / 2.0),
        flow.slice(1, 1, 2) / ((image.size(2) - 1.0) / 2.0)}, 1);
    const auto grid = (base + normalized).permute({0, 2, 3, 1});
    return torch::nn::functional::grid_sample(image, grid,
        torch::nn::functional::GridSampleFuncOptions().mode(torch::kBilinear)
            .padding_mode(torch::kBorder).align_corners(true));
}

GDNImpl::GDNImpl(std::int64_t channels, bool inverse, double beta_min,
                 double gamma_init, double reparam_offset) : inverse_(inverse) {
    gamma_bound_ = reparam_offset;
    pedestal_ = gamma_bound_ * gamma_bound_;
    beta_bound_ = std::sqrt(beta_min + pedestal_);
    beta = register_parameter("beta", torch::sqrt(torch::ones({channels}) + pedestal_));
    gamma = register_parameter("gamma", torch::sqrt(gamma_init * torch::eye(channels) + pedestal_));
}

torch::Tensor GDNImpl::forward(torch::Tensor input) {
    const auto original = input.sizes().vec();
    if (input.dim() == 5) input = input.view({original[0], original[1], original[2] * original[3], original[4]});
    const auto bounded_beta = LowerBound::apply(beta, beta_bound_).pow(2) - pedestal_;
    const auto bounded_gamma = (LowerBound::apply(gamma, gamma_bound_).pow(2) - pedestal_)
        .view({input.size(1), input.size(1), 1, 1});
    const auto norm = torch::conv2d(input.pow(2), bounded_gamma, bounded_beta).sqrt();
    auto result = inverse_ ? input * norm : input / norm;
    if (original.size() == 5) result = result.view(original);
    return result;
}

ResBlockImpl::ResBlockImpl(std::int64_t in, std::int64_t out, std::int64_t kernel, std::int64_t stride) {
    relu1 = register_module("relu1", torch::nn::ReLU());
    conv1 = register_module("conv1", torch::nn::Conv2d(
        torch::nn::Conv2dOptions(in, out, kernel).stride(stride).padding(kernel / 2)));
    torch::nn::init::xavier_uniform_(conv1->weight);
    torch::nn::init::constant_(conv1->bias, 0);
    relu2 = register_module("relu2", torch::nn::ReLU());
    conv2 = register_module("conv2", torch::nn::Conv2d(
        torch::nn::Conv2dOptions(out, out, kernel).stride(stride).padding(kernel / 2)));
    torch::nn::init::xavier_uniform_(conv2->weight);
    torch::nn::init::constant_(conv2->bias, 0);
    if (in != out) {
        adapt_conv = register_module("adapt_conv", torch::nn::Conv2d(torch::nn::Conv2dOptions(in, out, 1)));
        torch::nn::init::xavier_uniform_(adapt_conv->weight);
        torch::nn::init::constant_(adapt_conv->bias, 0);
    }
}

torch::Tensor ResBlockImpl::forward(const torch::Tensor& input) {
    const auto residual = conv2(relu2(conv1(relu1(input))));
    return (adapt_conv ? adapt_conv(input) : input) + residual;
}

LeakyResBlockImpl::LeakyResBlockImpl(std::int64_t channels) {
    conv = register_module("conv", torch::nn::Sequential());
    conv->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, channels, 3).padding(1)));
    conv->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.1).inplace(true)));
    conv->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, channels, 3).padding(1)));
    conv->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.1).inplace(true)));
}

torch::Tensor LeakyResBlockImpl::forward(const torch::Tensor& input) { return input + conv->forward(input); }

MaskedConv2dImpl::MaskedConv2dImpl(std::int64_t channels)
    : torch::nn::Conv2dImpl(torch::nn::Conv2dOptions(channels, channels * 2, 5).padding(2)) {
    mask = register_buffer("mask", torch::ones_like(weight));
    mask.slice(2, 2, 3).slice(3, 2).zero_();
    mask.slice(2, 3).zero_();
}

torch::Tensor MaskedConv2dImpl::forward(const torch::Tensor& input) {
    // Python writes weight.data in place; it does not multiply the weight
    // inside the gradient graph. Its AR coding path then reads this weight.
    weight.variable_data().mul_(mask);
    return torch::nn::Conv2dImpl::forward(input);
}

MEBasicImpl::MEBasicImpl() {
    const auto convolution = [](std::int64_t in, std::int64_t out) {
        return torch::nn::Conv2d(torch::nn::Conv2dOptions(in, out, 7).padding(3));
    };
    conv1 = register_module("conv1", convolution(8, 32));
    relu1 = register_module("relu1", torch::nn::ReLU());
    conv2 = register_module("conv2", convolution(32, 64));
    relu2 = register_module("relu2", torch::nn::ReLU());
    conv3 = register_module("conv3", convolution(64, 32));
    relu3 = register_module("relu3", torch::nn::ReLU());
    conv4 = register_module("conv4", convolution(32, 16));
    relu4 = register_module("relu4", torch::nn::ReLU());
    conv5 = register_module("conv5", convolution(16, 2));
}

torch::Tensor MEBasicImpl::forward(const torch::Tensor& input) {
    return conv5(relu4(conv4(relu3(conv3(relu2(conv2(relu1(conv1(input)))))))));
}

SpynetImpl::SpynetImpl() {
    moduleBasic = register_module("moduleBasic", torch::nn::ModuleList());
    for (int i = 0; i < 4; ++i) moduleBasic->push_back(MEBasic());
}

torch::Tensor SpynetImpl::forward(const torch::Tensor& current, const torch::Tensor& reference) {
    std::vector<torch::Tensor> first{current}, second{reference};
    for (int i = 0; i < 3; ++i) {
        first.push_back(torch::avg_pool2d(first.back(), {2, 2}, {2, 2}));
        second.push_back(torch::avg_pool2d(second.back(), {2, 2}, {2, 2}));
    }
    auto flow = torch::zeros({current.size(0), 2, second.back().size(2) / 2, second.back().size(3) / 2},
                              torch::TensorOptions().device(current.device()).dtype(torch::kFloat32));
    for (int level = 0; level < 4; ++level) {
        const auto up = double_size(flow) * 2.0;
        flow = up + moduleBasic->at<MEBasicImpl>(level).forward(
            torch::cat({first.at(3 - level), flow_warp(second.at(3 - level), up), up}, 1));
    }
    return flow;
}
}  // namespace uf::dcvc2021
