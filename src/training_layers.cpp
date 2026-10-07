// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#include "uf/training_layers.hpp"

#include <ATen/TensorIndexing.h>
#include <torch/csrc/autograd/custom_function.h>

#include <cmath>
#include <stdexcept>

namespace uf::training {
namespace {

class LowerBound : public torch::autograd::Function<LowerBound> {
public:
    static torch::Tensor forward(torch::autograd::AutogradContext* context,
                                 const torch::Tensor& input, double bound) {
        context->save_for_backward({input});
        context->saved_data["bound"] = bound;
        return input.clamp_min(bound);
    }
    static torch::autograd::variable_list backward(torch::autograd::AutogradContext* context,
                                                  torch::autograd::variable_list gradients) {
        const auto input = context->get_saved_variables().at(0);
        const auto bound = context->saved_data.at("bound").toDouble();
        const auto pass = (input >= bound) | (gradients.at(0) < 0);
        return {pass * gradients.at(0), torch::Tensor{}};
    }
};

class Quantize : public torch::autograd::Function<Quantize> {
public:
    static torch::Tensor forward(torch::autograd::AutogradContext*, const torch::Tensor& input) {
        return input.round();
    }
    static torch::autograd::variable_list backward(torch::autograd::AutogradContext*,
                                                  torch::autograd::variable_list gradients) {
        return {gradients.at(0)};
    }
};

constexpr double kr = 0.2126, kg = 0.7152, kb = 0.0722;

}  // namespace

torch::Tensor lower_bound(const torch::Tensor& input, double bound) {
    return LowerBound::apply(input, bound);
}

torch::Tensor quantize(const torch::Tensor& input) {
    return Quantize::apply(input);
}

torch::Tensor rgb_to_ycbcr(const torch::Tensor& rgb, bool is_bgr) {
    const auto channels = rgb.chunk(3, -3);
    const auto& r = channels.at(is_bgr ? 2 : 0);
    const auto& g = channels.at(1);
    const auto& b = channels.at(is_bgr ? 0 : 2);
    const auto y = kr * r + kg * g + kb * b;
    const auto cb = 0.5 * (b - y) / (1 - kb) + 0.5;
    const auto cr = 0.5 * (r - y) / (1 - kr) + 0.5;
    return torch::cat({y, cb, cr}, -3).clamp(0.0, 1.0);
}

torch::Tensor ycbcr_to_rgb(const torch::Tensor& ycbcr, bool is_bgr, bool clamp) {
    const auto dtype = ycbcr.scalar_type();
    const auto channels = ycbcr.to(torch::kFloat32).chunk(3, -3);
    const auto& y = channels.at(0);
    const auto& cb = channels.at(1);
    const auto& cr = channels.at(2);
    const auto r = y + (2 - 2 * kr) * (cr - 0.5);
    const auto b = y + (2 - 2 * kb) * (cb - 0.5);
    const auto g = (y - kr * r - kb * b) / kg;
    auto rgb = is_bgr ? torch::cat({b, g, r}, -3) : torch::cat({r, g, b}, -3);
    if (clamp) rgb = rgb.clamp(0.0, 1.0);
    return rgb.to(dtype);
}

std::pair<torch::Tensor, torch::Tensor> yuv444_to_420(const torch::Tensor& yuv) {
    using at::indexing::None;
    using at::indexing::Slice;
    const auto y = yuv.index({Slice(), Slice(None, 1), Slice(), Slice()});
    const auto uv = yuv.index({Slice(), Slice(1, None), Slice(), Slice()});
    return {y, torch::avg_pool2d(uv, {2, 2}, {2, 2})};
}

torch::Tensor bit_estimator_z_probability(const torch::Tensor& input,
    const torch::Tensor& h, const torch::Tensor& b, const torch::Tensor& a) {
    using at::indexing::None;
    using at::indexing::Slice;
    auto value = input;
    for (std::int64_t layer = 0; layer < 4; ++layer) {
        // Python inserts None immediately after the layer slice, retaining
        // any trailing parameter axes instead of appending at the end.
        value = value * torch::softplus(h.index({Slice(), Slice(), Slice(layer, layer + 1), None})) +
                        b.index({Slice(), Slice(), Slice(layer, layer + 1), None});
        if (layer != 3) value = value + value.tanh() *
            a.index({Slice(), Slice(), Slice(layer, layer + 1), None}).tanh();
    }
    return value.sigmoid();
}

torch::Tensor bit_estimator_z_mass(const torch::Tensor& input,
    const torch::Tensor& h, const torch::Tensor& b, const torch::Tensor& a) {
    const auto dtype = input.scalar_type();
    const auto value = input.to(torch::kFloat32);
    const auto hf = h.to(torch::kFloat32), bf = b.to(torch::kFloat32), af = a.to(torch::kFloat32);
    const auto lower = bit_estimator_z_probability(value - 0.5, hf, bf, af);
    const auto upper = bit_estimator_z_probability(value + 0.5, hf, bf, af);
    return (upper - lower).to(dtype);
}

torch::Tensor gaussian_probability(const torch::Tensor& values, const torch::Tensor& scales) {
    const auto dtype = values.scalar_type();
    const auto value = values.to(torch::kFloat32).abs();
    const auto scale = lower_bound(scales.to(torch::kFloat32), 0.11);
    const auto factor = -std::pow(2.0, -0.5);
    const auto upper = torch::erfc(factor * ((0.5 - value) / scale));
    const auto lower = torch::erfc(factor * ((-0.5 - value) / scale));
    return (0.5 * (upper - lower)).clamp_min(1e-9).to(dtype);
}

std::pair<torch::Tensor, torch::Tensor> mse_yuv_rgb(const torch::Tensor& input,
                                                 const torch::Tensor& reconstruction) {
    const auto options = torch::nn::functional::MSELossFuncOptions().reduction(torch::kNone);
    const auto yuv = torch::nn::functional::mse_loss(input, reconstruction, options).sum({2, 3});
    const auto rgb = torch::nn::functional::mse_loss(ycbcr_to_rgb(input, false, false),
        ycbcr_to_rgb(reconstruction, false, false), options).sum({1, 2, 3});
    return {yuv, rgb};
}

torch::Tensor mse_weighted_average(const torch::Tensor& mse_yuv,
                                   const torch::Tensor& mse_rgb, std::int64_t pixels) {
    const auto dtype = mse_yuv.scalar_type();
    const auto components = mse_yuv.to(torch::kFloat32) / pixels;
    const auto y = components.select(1, 0), u = components.select(1, 1), v = components.select(1, 2);
    const auto yuv = torch::exp(0.0833 * (10 * y.clamp_min(1e-6).log() +
                                         u.clamp_min(1e-6).log() + v.clamp_min(1e-6).log())) * 3;
    const auto rgb = mse_rgb.to(torch::kFloat32) / pixels;
    return (yuv * 0.8 + rgb * 0.2).to(dtype);
}

torch::Tensor mse_eight_frame_sum(const std::vector<torch::Tensor>& mse,
                                 const std::vector<double>& weights) {
    // Python indexes the required entries and ignores trailing ones. at()
    // keeps short inputs checked without imposing an exact-length guard.
    return (mse.at(0) + mse.at(2) + mse.at(4) + mse.at(6)) * weights.at(1) +
           (mse.at(1) + mse.at(3) + mse.at(5)) * weights.at(2) + mse.at(7) * weights.at(0);
}

torch::Tensor WSiLUImpl::forward(const torch::Tensor& input) {
    return (4.0 * input).sigmoid() * input;
}

WSiLUChunkAddImpl::WSiLUChunkAddImpl() { wsilu = register_module("wsilu", WSiLU()); }

torch::Tensor WSiLUChunkAddImpl::forward(const torch::Tensor& input) {
    using at::indexing::None;
    using at::indexing::Slice;
    const auto value = wsilu->forward(input);
    // Keep the four explicit axes in Python's x[:, offset::4, :, :].
    return value.index({Slice(), Slice(0, None, 4), Slice(), Slice()}) +
           value.index({Slice(), Slice(1, None, 4), Slice(), Slice()}) +
           value.index({Slice(), Slice(2, None, 4), Slice(), Slice()}) +
           value.index({Slice(), Slice(3, None, 4), Slice(), Slice()});
}

DepthConvBlockImpl::DepthConvBlockImpl(std::int64_t in, std::int64_t out,
                                      bool dcb2, bool use_shortcut, bool force_adaptor)
    : shortcut(use_shortcut) {
    if (dcb2 && shortcut) throw std::invalid_argument("DCB2 does not use the extra shortcut");
    if (in != out || force_adaptor) adaptor = register_module("adaptor",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(in, out, 1)));
    const auto ratio = dcb2 ? 2 : 1;
    // Separate statements preserve Python's construction/RNG order.
    dc = register_module("dc", torch::nn::Sequential());
    dc->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(out, out / ratio, 1)));
    dc->push_back(WSiLU());
    dc->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(out / ratio, out / ratio, 3)
                                   .padding(1).groups(out / ratio)));
    dc->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(out / ratio, out, 1)));
    ffn = register_module("ffn", torch::nn::Sequential());
    ffn->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(out, out * 4 / ratio, 1)));
    ffn->push_back(WSiLUChunkAdd());
    ffn->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(out / ratio, out, 1)));
}

torch::Tensor DepthConvBlockImpl::forward(const torch::Tensor& input) {
    const auto value = adaptor ? adaptor->forward(input) : input;
    auto result = dc->forward(value) + value;
    result = ffn->forward(result) + result;
    if (shortcut) result = result + value;
    return result;
}

SubpelConv2xImpl::SubpelConv2xImpl(std::int64_t in, std::int64_t out,
                                  std::int64_t kernel, std::int64_t padding, bool force_bias) {
    conv = register_module("conv", torch::nn::Sequential());
    conv->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(in, out * 4, kernel)
                                     .padding(padding).bias(kernel > 1 || force_bias)));
    conv->push_back(torch::nn::PixelShuffle(torch::nn::PixelShuffleOptions(2)));
}

torch::Tensor SubpelConv2xImpl::forward(const torch::Tensor& input) { return conv->forward(input); }

ResidualBlockUpsampleImpl::ResidualBlockUpsampleImpl(std::int64_t in, std::int64_t out,
                                                    bool dcb2, bool shortcut, bool force_bias) {
    up = register_module("up", SubpelConv2x(in, out, 1, 0, force_bias));
    conv = register_module("conv", DepthConvBlock(out, out, dcb2, shortcut));
}

torch::Tensor ResidualBlockUpsampleImpl::forward(const torch::Tensor& input) {
    return conv->forward(up->forward(input));
}

ResidualBlockWithStride2Impl::ResidualBlockWithStride2Impl(std::int64_t in, std::int64_t out,
                                                          bool dcb2, bool shortcut) {
    down = register_module("down", torch::nn::Conv2d(torch::nn::Conv2dOptions(in * 4, out, 1)));
    conv = register_module("conv", DepthConvBlock(out, out, dcb2, shortcut));
}

torch::Tensor ResidualBlockWithStride2Impl::forward(const torch::Tensor& input) {
    return conv->forward(down->forward(torch::pixel_unshuffle(input, 2)));
}

}  // namespace uf::training
