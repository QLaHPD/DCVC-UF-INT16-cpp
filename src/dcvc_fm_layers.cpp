// Layers: Copyright 2020 InterDigital Communications, Inc., Apache-2.0.
// Video networks: Microsoft Corporation, MIT.
#include "uf/dcvc_fm.hpp"
#include <algorithm>
#include <limits>
#include <mutex>
#include <stdexcept>

namespace uf::fm {
namespace {
torch::nn::LeakyReLU leaky(double slope, bool inplace) {
    return torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(slope).inplace(inplace));
}
torch::nn::Sequential subpel(std::int64_t in, std::int64_t out, std::int64_t factor, std::int64_t kernel) {
    torch::nn::Sequential result;
    result->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(in, out * factor * factor, kernel).padding(kernel / 2)));
    result->push_back(torch::nn::PixelShuffle(torch::nn::PixelShuffleOptions(factor)));
    return result;
}
}  // namespace
torch::nn::Conv2d conv3x3(std::int64_t in, std::int64_t out, std::int64_t stride) {
    return torch::nn::Conv2d(torch::nn::Conv2dOptions(in, out, 3).stride(stride).padding(1));
}
torch::nn::Conv2d conv1x1(std::int64_t in, std::int64_t out, std::int64_t stride) {
    return torch::nn::Conv2d(torch::nn::Conv2dOptions(in, out, 1).stride(stride));
}
torch::nn::Sequential subpel_conv3x3(std::int64_t in, std::int64_t out, std::int64_t factor) {
    return subpel(in, out, factor, 3);
}
torch::nn::Sequential subpel_conv1x1(std::int64_t in, std::int64_t out, std::int64_t factor) {
    return subpel(in, out, factor, 1);
}
ResidualBlockWithStrideImpl::ResidualBlockWithStrideImpl(std::int64_t in, std::int64_t out,
                                                       std::int64_t stride, bool inplace) {
    conv1 = register_module("conv1", conv3x3(in, out, stride));
    leaky_relu = register_module("leaky_relu", leaky(0.01, inplace));
    conv2 = register_module("conv2", conv3x3(out, out));
    leaky_relu2 = register_module("leaky_relu2", leaky(0.1, inplace));
    if (stride != 1) downsample = register_module("downsample", conv1x1(in, out, stride));
}
torch::Tensor ResidualBlockWithStrideImpl::forward(const torch::Tensor& input) {
    auto result = leaky_relu2(conv2(leaky_relu(conv1(input))));
    const auto identity = downsample ? downsample(input) : input;
    return result + identity;
}
ResidualBlockUpsampleImpl::ResidualBlockUpsampleImpl(std::int64_t in, std::int64_t out,
                                                   std::int64_t factor, bool inplace) {
    subpel_conv = register_module("subpel_conv", subpel_conv1x1(in, out, factor));
    leaky_relu = register_module("leaky_relu", leaky(0.01, inplace));
    conv = register_module("conv", conv3x3(out, out));
    leaky_relu2 = register_module("leaky_relu2", leaky(0.1, inplace));
    upsample = register_module("upsample", subpel_conv1x1(in, out, factor));
}
torch::Tensor ResidualBlockUpsampleImpl::forward(const torch::Tensor& input) {
    const auto result = leaky_relu2(conv(leaky_relu(subpel_conv->forward(input))));
    return result + upsample->forward(input);
}
ResidualBlockImpl::ResidualBlockImpl(std::int64_t in, std::int64_t out, double slope, bool inplace) {
    conv1 = register_module("conv1", conv3x3(in, out));
    leaky_relu = register_module("leaky_relu", leaky(slope, inplace));
    conv2 = register_module("conv2", conv3x3(out, out));
    if (in != out) adaptor = register_module("adaptor", conv1x1(in, out));
}
torch::Tensor ResidualBlockImpl::forward(const torch::Tensor& input) {
    const auto identity = adaptor ? adaptor(input) : input;
    return leaky_relu(conv2(leaky_relu(conv1(input)))) + identity;
}
ResidualBlockWithStride2Impl::ResidualBlockWithStride2Impl(std::int64_t in, std::int64_t out, bool inplace) {
    down = register_module("down", torch::nn::Conv2d(torch::nn::Conv2dOptions(in, out, 2).stride(2)));
    conv = register_module("conv", torch::nn::Sequential());
    conv->push_back(conv3x3(out, out)); conv->push_back(leaky(0.01, inplace));
    conv->push_back(conv1x1(out, out)); conv->push_back(leaky(0.01, inplace));
}
torch::Tensor ResidualBlockWithStride2Impl::forward(const torch::Tensor& input) {
    const auto x = down(input);
    return conv->forward(x) + x;
}
DepthConvImpl::DepthConvImpl(std::int64_t in, std::int64_t out, double slope, bool inplace) {
    conv1 = register_module("conv1", torch::nn::Sequential());
    conv1->push_back(conv1x1(in, in)); conv1->push_back(leaky(slope, inplace));
    depth_conv = register_module("depth_conv",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(in, in, 3).padding(1).groups(in)));
    conv2 = register_module("conv2", conv1x1(in, out));
    if (in != out) adaptor = register_module("adaptor", conv1x1(in, out));
}
DepthConv2Impl::DepthConv2Impl(std::int64_t in, std::int64_t out, double slope, bool inplace) {
    conv1 = register_module("conv1", torch::nn::Sequential());
    conv1->push_back(conv1x1(in, out)); conv1->push_back(leaky(slope, inplace));
    conv1->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(out, out, 3).padding(1).groups(out)));
    conv2 = register_module("conv2", conv1x1(in, out));
    out_conv = register_module("out_conv", conv1x1(out, out));
    if (in != out) adaptor = register_module("adaptor", conv1x1(in, out));
}
torch::Tensor DepthConv2Impl::forward(const torch::Tensor& input) {
    const auto identity = adaptor ? adaptor(input) : input;
    const auto x1 = conv1->forward(input), x2 = conv2(input);
    return identity + out_conv(x1 * x2);
}
torch::Tensor DepthConvImpl::forward(const torch::Tensor& input) {
    const auto identity = adaptor ? adaptor(input) : input;
    return conv2(depth_conv(conv1->forward(input))) + identity;
}
ConvFFNImpl::ConvFFNImpl(std::int64_t channels, double slope, bool inplace) {
    if (channels < 1 || channels > std::numeric_limits<std::int64_t>::max() / 4)
        throw std::invalid_argument("Invalid FM ConvFFN channel count");
    const auto internal = std::max(std::min(channels * 4, std::int64_t{1024}), channels * 2);
    conv = register_module("conv", torch::nn::Sequential());
    conv->push_back(conv1x1(channels, internal)); conv->push_back(leaky(slope, inplace));
    conv->push_back(conv1x1(internal, channels)); conv->push_back(leaky(slope, inplace));
}
torch::Tensor ConvFFNImpl::forward(const torch::Tensor& input) { return input + conv->forward(input); }
ConvFFN2Impl::ConvFFN2Impl(std::int64_t channels, double, bool inplace) {
    conv = register_module("conv", conv1x1(channels, channels * 4));
    conv_out = register_module("conv_out", conv1x1(channels * 2, channels));
    relu = register_module("relu", leaky(0.1, inplace));  // Source overwrites the slope argument.
}
torch::Tensor ConvFFN2Impl::forward(const torch::Tensor& input) {
    const auto parts = conv(input).chunk(2, 1);
    return input + conv_out(parts.at(0) * relu(parts.at(1)));
}
ConvFFN3Impl::ConvFFN3Impl(std::int64_t channels, bool inplace) {
    conv = register_module("conv", conv1x1(channels, channels * 4));
    conv_out = register_module("conv_out", conv1x1(channels * 2, channels));
    relu1 = register_module("relu1", leaky(0.1, inplace));
    relu2 = register_module("relu2", leaky(0.01, inplace));
}
torch::Tensor ConvFFN3Impl::forward(const torch::Tensor& input) {
    const auto parts = conv(input).chunk(2, 1);
    return input + conv_out(relu1(parts.at(0)) + relu2(parts.at(1)));
}
DepthConvBlockImpl::DepthConvBlockImpl(std::int64_t in, std::int64_t out, double depth_slope,
                                     double ffn_slope, bool inplace, int variant) {
    block = register_module("block", torch::nn::Sequential());
    if (variant == 3) block->push_back(DepthConv2(in, out, depth_slope, inplace));
    else block->push_back(DepthConv(in, out, depth_slope, inplace));
    if (variant == 1) block->push_back(ConvFFN(out, ffn_slope, inplace));
    else if (variant == 4) block->push_back(ConvFFN3(out, inplace));
    else block->push_back(ConvFFN2(out, ffn_slope, inplace));
}
torch::Tensor DepthConvBlockImpl::forward(const torch::Tensor& input) { return block->forward(input); }
DepthConvBlock2Impl::DepthConvBlock2Impl(std::int64_t in, std::int64_t out, double ds, double fs, bool inplace)
    : DepthConvBlockImpl(in, out, ds, fs, inplace, 2) {}
DepthConvBlock3Impl::DepthConvBlock3Impl(std::int64_t in, std::int64_t out, double ds, double fs, bool inplace)
    : DepthConvBlockImpl(in, out, ds, fs, inplace, 3) {}
DepthConvBlock4Impl::DepthConvBlock4Impl(std::int64_t in, std::int64_t out, double ds, bool inplace)
    : DepthConvBlockImpl(in, out, ds, 0.1, inplace, 4) {}
ResBlockImpl::ResBlockImpl(std::int64_t channels, double slope, bool end, bool bottleneck, bool inplace) {
    first_layer = register_module("first_layer", leaky(slope, false));
    conv1 = register_module("conv1", conv3x3(channels, bottleneck ? channels / 2 : channels));
    relu = register_module("relu", leaky(slope, inplace));
    conv2 = register_module("conv2", conv3x3(bottleneck ? channels / 2 : channels, channels));
    if (end) last_layer = torch::nn::AnyModule(register_module("last_layer", relu));
    else last_layer = torch::nn::AnyModule(register_module("last_layer", torch::nn::Identity()));
}
torch::Tensor ResBlockImpl::forward(const torch::Tensor& input) {
    return input + last_layer.forward<torch::Tensor>(conv2(relu(conv1(first_layer(input)))));
}
torch::Tensor bilinear_up(const torch::Tensor& input) {
    return torch::nn::functional::interpolate(input, torch::nn::functional::InterpolateFuncOptions()
        .size(std::vector<std::int64_t>{input.size(2) * 2, input.size(3) * 2}).mode(torch::kBilinear).align_corners(false));
}
torch::Tensor bilinear_down(const torch::Tensor& input) {
    return torch::nn::functional::interpolate(input, torch::nn::functional::InterpolateFuncOptions()
        .size(std::vector<std::int64_t>{input.size(2) / 2, input.size(3) / 2}).mode(torch::kBilinear).align_corners(false));
}
MEBasicImpl::MEBasicImpl(int complexity) : by_pass(complexity < 0) {
    relu = register_module("relu", torch::nn::ReLU());
    if (complexity != 0 && complexity != 3) return;
    const auto kernel = complexity == 0 ? 7 : 5;
    const auto make_conv = [kernel](std::int64_t in, std::int64_t out) {
        return torch::nn::Conv2d(torch::nn::Conv2dOptions(in, out, kernel).stride(1).padding(kernel / 2));
    };
    conv1 = register_module("conv1", make_conv(8, 32)); conv2 = register_module("conv2", make_conv(32, 64));
    conv3 = register_module("conv3", make_conv(64, 32)); conv4 = register_module("conv4", make_conv(32, 16));
    conv5 = register_module("conv5", make_conv(16, 2));
}
torch::Tensor MEBasicImpl::forward(const torch::Tensor& input) {
    if (by_pass) return input.slice(1, input.size(1) - 2);
    if (!conv1) throw std::logic_error("FM MEBasic has no convolutions for this complexity level");
    return conv5(relu(conv4(relu(conv3(relu(conv2(relu(conv1(input)))))))));
}
ME_SpynetImpl::ME_SpynetImpl() {
    me_8x = register_module("me_8x", MEBasic(0)); me_4x = register_module("me_4x", MEBasic(0));
    me_2x = register_module("me_2x", MEBasic(3)); me_1x = register_module("me_1x", MEBasic(3));
}
torch::Tensor ME_SpynetImpl::forward(const torch::Tensor& current, const torch::Tensor& reference) {
    std::vector<torch::Tensor> first{current}, second{reference};
    for (int i = 0; i < 3; ++i) {
        first.push_back(torch::nn::functional::avg_pool2d(first.back(), torch::nn::functional::AvgPool2dFuncOptions(2).stride(2)));
        second.push_back(torch::nn::functional::avg_pool2d(second.back(), torch::nn::functional::AvgPool2dFuncOptions(2).stride(2)));
    }
    auto flow = torch::zeros({current.size(0), 2, first.back().size(2), first.back().size(3)}, current.options());
    flow = me_8x(torch::cat({first[3], second[3], flow}, 1));
    const std::array<MEBasic, 3> modules{me_4x, me_2x, me_1x};
    for (int i = 0; i < 3; ++i) {
        const auto up = bilinear_up(flow) * 2.0;
        flow = up + modules[i].ptr()->forward(torch::cat({first[2 - i], block_mc_func(second[2 - i], up), up}, 1));
    }
    return flow;
}
UNetImpl::UNetImpl(std::int64_t in, std::int64_t out, bool inplace, bool second) {
    if (second) max_pool = register_module("max_pool", torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)));
    const auto make_block = [&](std::int64_t a, std::int64_t b) {
        return DepthConvBlock(a, b, 0.01, 0.1, inplace, second ? 4 : 2);
    };
    conv1 = register_module("conv1", make_block(in, 32));
    if (!second) down1 = register_module("down1", torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 32, 2).stride(2)));
    conv2 = register_module("conv2", make_block(32, 64));
    if (!second) down2 = register_module("down2", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 2).stride(2)));
    conv3 = register_module("conv3", make_block(64, 128));
    context_refine = register_module("context_refine", torch::nn::Sequential());
    for (int i = 0; i < 4; ++i) context_refine->push_back(make_block(128, 128));
    up3 = register_module("up3", subpel_conv1x1(128, 64, 2));
    up_conv3 = register_module("up_conv3", make_block(128, 64));
    up2 = register_module("up2", subpel_conv1x1(64, 32, 2));
    up_conv2 = register_module("up_conv2", make_block(64, out));
}
torch::Tensor UNetImpl::forward(const torch::Tensor& input) {
    const auto first = conv1(input);
    const auto second = conv2(down1 ? down1(first) : max_pool(first));
    const auto third = context_refine->forward(conv3(down2 ? down2(second) : max_pool(second)));
    const auto up = up_conv3(torch::cat({second, up3->forward(third)}, 1));
    return up_conv2(torch::cat({first, up2->forward(up)}, 1));
}
UNet2Impl::UNet2Impl(std::int64_t in, std::int64_t out, bool inplace) : UNetImpl(in, out, inplace, true) {}
}  // namespace uf::fm
