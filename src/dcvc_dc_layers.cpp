// Layers: Copyright 2020 InterDigital Communications, Inc., Apache-2.0.
// Video networks: Microsoft Corporation, MIT.
#include "uf/dcvc_dc.hpp"
#include <algorithm>
#include <limits>
#include <mutex>
#include <stdexcept>

namespace uf::dc {
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
int cache_slot(torch::Device device) {
    const auto index = device.is_cpu() ? 8 : static_cast<int>(device.index());
    if (index < 0 || index > 8) throw std::out_of_range("DC warp cache device index");
    return index;
}
std::array<std::map<std::vector<std::int64_t>, torch::Tensor>, 9> grids;
std::mutex grid_mutex;
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
DepthConvImpl::DepthConvImpl(std::int64_t in, std::int64_t out, std::int64_t kernel,
                           std::int64_t stride, double slope, bool inplace) {
    conv1 = register_module("conv1", torch::nn::Sequential());
    conv1->push_back(conv1x1(in, in, stride)); conv1->push_back(leaky(slope, inplace));
    depth_conv = register_module("depth_conv",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(in, in, kernel).padding(kernel / 2).groups(in)));
    conv2 = register_module("conv2", conv1x1(in, out));
    if (stride != 1) {
        if (stride != 2) throw std::invalid_argument("DC DepthConv adaptor requires stride 2");
        adaptor = register_module("adaptor", torch::nn::Conv2d(torch::nn::Conv2dOptions(in, out, 2).stride(2)));
    } else if (in != out) adaptor = register_module("adaptor", conv1x1(in, out));
}
torch::Tensor DepthConvImpl::forward(const torch::Tensor& input) {
    const auto identity = adaptor ? adaptor(input) : input;
    return conv2(depth_conv(conv1->forward(input))) + identity;
}
ConvFFNImpl::ConvFFNImpl(std::int64_t channels, double slope, bool inplace) {
    if (channels < 1 || channels > std::numeric_limits<std::int64_t>::max() / 4)
        throw std::invalid_argument("Invalid DC ConvFFN channel count");
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
DepthConvBlockImpl::DepthConvBlockImpl(std::int64_t in, std::int64_t out, std::int64_t kernel,
                                     std::int64_t stride, double depth_slope, double ffn_slope,
                                     bool inplace, bool gated) {
    block = register_module("block", torch::nn::Sequential());
    block->push_back(DepthConv(in, out, kernel, stride, depth_slope, inplace));
    if (gated) block->push_back(ConvFFN2(out, ffn_slope, inplace));
    else block->push_back(ConvFFN(out, ffn_slope, inplace));
}
torch::Tensor DepthConvBlockImpl::forward(const torch::Tensor& input) { return block->forward(input); }
DepthConvBlock2Impl::DepthConvBlock2Impl(std::int64_t in, std::int64_t out, std::int64_t kernel,
                                       std::int64_t stride, double depth_slope, double ffn_slope, bool inplace)
    : DepthConvBlockImpl(in, out, kernel, stride, depth_slope, ffn_slope, inplace, true) {}
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
torch::Tensor flow_warp(const torch::Tensor& feature, const torch::Tensor& flow) {
    if (feature.dim() != 4 || flow.dim() != 4 || flow.size(1) != 2)
        throw std::invalid_argument("DC warp requires NCHW features and two-channel flow");
    torch::Tensor grid;
    {
        std::lock_guard<std::mutex> lock(grid_mutex);
        auto& cache = grids.at(cache_slot(flow.device()));
        const auto key = flow.sizes().vec();
        if (!cache.count(key)) {
            const auto options = torch::TensorOptions().device(flow.device()).dtype(torch::kFloat32);
            const auto horizontal = torch::linspace(-1, 1, flow.size(3), options)
                .view({1, 1, 1, flow.size(3)}).expand({flow.size(0), -1, flow.size(2), -1});
            const auto vertical = torch::linspace(-1, 1, flow.size(2), options)
                .view({1, 1, flow.size(2), 1}).expand({flow.size(0), -1, -1, flow.size(3)});
            cache[key] = torch::cat({horizontal, vertical}, 1);
        }
        grid = grids.at(cache_slot(feature.device())).at(key);
    }
    const auto normalized = torch::cat({flow.slice(1, 0, 1) / ((feature.size(3) - 1.0) / 2.0),
                                       flow.slice(1, 1, 2) / ((feature.size(2) - 1.0) / 2.0)}, 1);
    return torch::nn::functional::grid_sample(feature, (grid + normalized).permute({0, 2, 3, 1}),
        torch::nn::functional::GridSampleFuncOptions().mode(torch::kBilinear).padding_mode(torch::kBorder).align_corners(true));
}
torch::Tensor bilinear_up(const torch::Tensor& input) {
    return torch::nn::functional::interpolate(input, torch::nn::functional::InterpolateFuncOptions()
        .size(std::vector<std::int64_t>{input.size(2) * 2, input.size(3) * 2}).mode(torch::kBilinear).align_corners(false));
}
torch::Tensor bilinear_down(const torch::Tensor& input) {
    return torch::nn::functional::interpolate(input, torch::nn::functional::InterpolateFuncOptions()
        .size(std::vector<std::int64_t>{input.size(2) / 2, input.size(3) / 2}).mode(torch::kBilinear).align_corners(false));
}
MEBasicImpl::MEBasicImpl() {
    relu = register_module("relu", torch::nn::ReLU());
    const auto conv = [](std::int64_t in, std::int64_t out) {
        return torch::nn::Conv2d(torch::nn::Conv2dOptions(in, out, 7).stride(1).padding(3));
    };
    conv1 = register_module("conv1", conv(8, 32)); conv2 = register_module("conv2", conv(32, 64));
    conv3 = register_module("conv3", conv(64, 32)); conv4 = register_module("conv4", conv(32, 16));
    conv5 = register_module("conv5", conv(16, 2));
}
torch::Tensor MEBasicImpl::forward(const torch::Tensor& input) {
    return conv5(relu(conv4(relu(conv3(relu(conv2(relu(conv1(input)))))))));
}
SpynetImpl::SpynetImpl() {
    moduleBasic = register_module("moduleBasic", torch::nn::ModuleList());
    for (int index = 0; index < 4; ++index) moduleBasic->push_back(MEBasic());
}
torch::Tensor SpynetImpl::forward(const torch::Tensor& current, const torch::Tensor& reference) {
    std::vector<torch::Tensor> first{current}, second{reference};
    for (int index = 0; index < 3; ++index) {
        first.push_back(torch::nn::functional::avg_pool2d(first.back(), torch::nn::functional::AvgPool2dFuncOptions(2).stride(2)));
        second.push_back(torch::nn::functional::avg_pool2d(second.back(), torch::nn::functional::AvgPool2dFuncOptions(2).stride(2)));
    }
    auto flow = torch::zeros({current.size(0), 2, second.back().size(2) / 2, second.back().size(3) / 2}, current.options());
    for (int level = 0; level < 4; ++level) {
        const auto up = bilinear_up(flow) * 2.0;
        const auto index = 3 - level;
        flow = up + moduleBasic->at<MEBasicImpl>(level).forward(
            torch::cat({first[index], flow_warp(second[index], up), up}, 1));
    }
    return flow;
}
UNetImpl::UNetImpl(std::int64_t in, std::int64_t out, bool inplace, bool gated) {
    max_pool = register_module("max_pool", torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)));
    const auto block = [&](std::int64_t first, std::int64_t second) {
        return DepthConvBlock(first, second, 3, 1, 0.01, 0.1, inplace, gated);
    };
    conv1 = register_module("conv1", block(in, 32)); conv2 = register_module("conv2", block(32, 64));
    conv3 = register_module("conv3", block(64, 128));
    context_refine = register_module("context_refine", torch::nn::Sequential());
    for (int index = 0; index < 4; ++index) context_refine->push_back(block(128, 128));
    up3 = register_module("up3", subpel_conv1x1(128, 64, 2));
    up_conv3 = register_module("up_conv3", block(128, 64));
    up2 = register_module("up2", subpel_conv1x1(64, 32, 2));
    up_conv2 = register_module("up_conv2", block(64, out));
}
torch::Tensor UNetImpl::forward(const torch::Tensor& input) {
    const auto first = conv1(input), second = conv2(max_pool(first));
    const auto third = context_refine->forward(conv3(max_pool(second)));
    const auto up = up_conv3(torch::cat({second, up3->forward(third)}, 1));
    return up_conv2(torch::cat({first, up2->forward(up)}, 1));
}
UNet2Impl::UNet2Impl(std::int64_t in, std::int64_t out, bool inplace) : UNetImpl(in, out, inplace, true) {}
std::pair<torch::nn::Sequential, torch::nn::Sequential> get_hyper_enc_dec_models(
    std::int64_t y, std::int64_t z, bool reduced, bool inplace) {
    torch::nn::Sequential enc, dec;
    enc->push_back(conv3x3(y, z)); enc->push_back(leaky(0.01, inplace));
    if (!reduced) { enc->push_back(conv3x3(z, z)); enc->push_back(leaky(0.01, inplace)); }
    enc->push_back(conv3x3(z, z, 2)); enc->push_back(leaky(0.01, inplace));
    if (!reduced) { enc->push_back(conv3x3(z, z)); enc->push_back(leaky(0.01, inplace)); }
    enc->push_back(conv3x3(z, z, 2));
    dec->push_back(conv3x3(z, y)); dec->push_back(leaky(0.01, inplace));
    dec->push_back(subpel_conv1x1(y, y, 2)); dec->push_back(leaky(0.01, inplace));
    dec->push_back(conv3x3(y, y)); dec->push_back(leaky(0.01, inplace));
    dec->push_back(subpel_conv1x1(y, y, 2)); dec->push_back(leaky(0.01, inplace));
    dec->push_back(conv3x3(y, y));
    return {enc, dec};
}
}  // namespace uf::dc
