// Copyright 2020 InterDigital Communications, Inc., Apache-2.0.
// Native adaptation of the original DCVC GDN/parametrizer/residual layers.
#include "uf/dcvc2021_image.hpp"
#include "dcvc2021_image_internal.hpp"
#include <torch/csrc/autograd/custom_function.h>
#include <cmath>
#include <stdexcept>

namespace uf::dcvc2021::image {
void detail::move_module(torch::nn::Module& module, std::optional<torch::Device> device,
                         std::optional<torch::Dtype> dtype, bool non_blocking) {
    if (dtype && !c10::isFloatingType(*dtype) && !c10::isComplexType(*dtype)) {
        throw std::invalid_argument("Module dtype must be floating or complex");
    }
    const auto convert = [&](const torch::Tensor& tensor) {
        const auto scalar = dtype && (tensor.is_floating_point() || tensor.is_complex()) ? *dtype : tensor.scalar_type();
        return tensor.to(device.value_or(tensor.device()), scalar, non_blocking);
    };
    torch::NoGradGuard no_grad;
    // LibTorch's default Module::to(dtype) also casts integer buffers.
    // Python only casts floating/complex tensors, and also converts grads.
    module.apply([&](torch::nn::Module& current) {
        for (auto& item : current.named_parameters(false)) {
            auto gradient = item.value().grad();
            item.value().set_data(convert(item.value()));
            if (gradient.defined()) gradient.set_data(convert(gradient));
        }
        for (auto& item : current.named_buffers(false)) {
            if (item.value().defined()) item.value().set_data(convert(item.value()));
        }
    });
}
namespace {
class BoundFunction : public torch::autograd::Function<BoundFunction> {
public:
    static torch::Tensor forward(torch::autograd::AutogradContext* context,
                                 const torch::Tensor& input, const torch::Tensor& bound) {
        context->save_for_backward({input, bound});
        return torch::maximum(input, bound);
    }
    static torch::autograd::variable_list backward(torch::autograd::AutogradContext* context,
                                                   torch::autograd::variable_list gradients) {
        const auto saved = context->get_saved_variables();
        const auto pass = ((saved.at(0) >= saved.at(1)) | (gradients.at(0) < 0)).to(gradients.at(0).scalar_type());
        return {pass * gradients.at(0), torch::Tensor{}};
    }
};
torch::nn::LeakyReLU leaky() {
    return torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().inplace(true));
}
}  // namespace

LowerBoundImpl::LowerBoundImpl(double value) {
    bound = register_buffer("bound", torch::empty({1}).fill_(value));
}
torch::Tensor LowerBoundImpl::forward(const torch::Tensor& input) { return BoundFunction::apply(input, bound); }

NonNegativeParametrizerImpl::NonNegativeParametrizerImpl(double minimum, double offset) {
    pedestal = register_buffer("pedestal", torch::empty({1}).fill_(offset * offset));
    lower_bound = register_module("lower_bound", LowerBound(std::sqrt(minimum + offset * offset)));
}
torch::Tensor NonNegativeParametrizerImpl::init(const torch::Tensor& input) {
    return torch::maximum(input + pedestal, pedestal).sqrt();
}
torch::Tensor NonNegativeParametrizerImpl::forward(const torch::Tensor& input) {
    return lower_bound(input).pow(2) - pedestal;
}
GDNImpl::GDNImpl(std::int64_t channels, bool inverse, double beta_min, double gamma_init) : inverse_(inverse) {
    beta_reparam = register_module("beta_reparam", NonNegativeParametrizer(beta_min));
    beta = register_parameter("beta", beta_reparam->init(torch::ones({channels})));
    gamma_reparam = register_module("gamma_reparam", NonNegativeParametrizer());
    gamma = register_parameter("gamma", gamma_reparam->init(gamma_init * torch::eye(channels)));
}
torch::Tensor GDNImpl::forward(const torch::Tensor& input) {
    const auto beta_value = beta_reparam(beta);
    const auto gamma_value = gamma_reparam(gamma).reshape({input.size(1), input.size(1), 1, 1});
    auto norm = torch::conv2d(input.pow(2), gamma_value, beta_value);
    norm = inverse_ ? norm.sqrt() : norm.rsqrt();
    return input * norm;
}

torch::nn::Conv2d conv(std::int64_t in, std::int64_t out, std::int64_t kernel, std::int64_t stride) {
    return torch::nn::Conv2d(torch::nn::Conv2dOptions(in, out, kernel).stride(stride).padding(kernel / 2));
}
torch::nn::ConvTranspose2d deconv(std::int64_t in, std::int64_t out, std::int64_t kernel, std::int64_t stride) {
    return torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(in, out, kernel)
        .stride(stride).padding(kernel / 2).output_padding(stride - 1));
}
torch::nn::Sequential subpel(std::int64_t in, std::int64_t out, std::int64_t factor) {
    torch::nn::Sequential result;
    result->push_back(conv(in, out * factor * factor, 3, 1));
    result->push_back(torch::nn::PixelShuffle(torch::nn::PixelShuffleOptions(factor)));
    return result;
}
ResidualBlockImpl::ResidualBlockImpl(std::int64_t in, std::int64_t out) {
    conv1 = register_module("conv1", conv(in, out, 3, 1));
    leaky_relu = register_module("leaky_relu", leaky());
    conv2 = register_module("conv2", conv(out, out, 3, 1));
}
torch::Tensor ResidualBlockImpl::forward(const torch::Tensor& input) {
    return leaky_relu(conv2(leaky_relu(conv1(input)))) + input;
}
ResidualBlockWithStrideImpl::ResidualBlockWithStrideImpl(std::int64_t in, std::int64_t out, std::int64_t stride) {
    conv1 = register_module("conv1", conv(in, out, 3, stride));
    leaky_relu = register_module("leaky_relu", leaky());
    conv2 = register_module("conv2", conv(out, out, 3, 1));
    gdn = register_module("gdn", GDN(out));
    if (stride != 1) downsample = register_module("downsample", conv(in, out, 1, stride));
}
torch::Tensor ResidualBlockWithStrideImpl::forward(const torch::Tensor& input) {
    auto output = gdn(conv2(leaky_relu(conv1(input))));
    output.add_(downsample ? downsample(input) : input);
    return output;
}
ResidualBlockUpsampleImpl::ResidualBlockUpsampleImpl(std::int64_t in, std::int64_t out, std::int64_t factor) {
    subpel_conv = register_module("subpel_conv", subpel(in, out, factor));
    leaky_relu = register_module("leaky_relu", leaky());
    conv = register_module("conv", image::conv(out, out, 3, 1));
    igdn = register_module("igdn", GDN(out, true));
    upsample = register_module("upsample", subpel(in, out, factor));
}
torch::Tensor ResidualBlockUpsampleImpl::forward(const torch::Tensor& input) {
    auto output = igdn(conv(leaky_relu(subpel_conv->forward(input))));
    output.add_(upsample->forward(input));
    return output;
}
}  // namespace uf::dcvc2021::image
