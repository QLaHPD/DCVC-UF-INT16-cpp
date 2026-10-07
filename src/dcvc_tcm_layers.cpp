// Copyright (c) Microsoft Corporation. Licensed under the MIT License.
// video_net.py and video_net_dmc.py in DCVC-family/DCVC-TCM.
#include "dcvc_tcm_internal.hpp"
#include "uf/process.hpp"
#include <mutex>
#include <stdexcept>

namespace uf::tcm::detail {
torch::nn::Conv2d conv(std::int64_t in, std::int64_t out, std::int64_t kernel, std::int64_t stride) {
    return torch::nn::Conv2d(torch::nn::Conv2dOptions(in, out, kernel).stride(stride).padding(kernel / 2));
}
torch::nn::ConvTranspose2d deconv(std::int64_t in, std::int64_t out, std::int64_t stride) {
    return torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(in, out, 3)
        .stride(stride).padding(1).output_padding(stride == 2 ? 1 : 0));
}
torch::nn::Sequential subpel(std::int64_t in, std::int64_t out) {
    torch::nn::Sequential result;
    result->push_back(conv(in, out * 4));
    result->push_back(torch::nn::PixelShuffle(torch::nn::PixelShuffleOptions(2)));
    return result;
}
torch::nn::LeakyReLU leaky(double slope) {
    return torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(slope).inplace(false));
}
}  // namespace uf::tcm::detail

namespace uf::tcm {
using detail::conv;
using detail::subpel;
torch::Tensor bilinear_up(const torch::Tensor& input) {
    return torch::nn::functional::interpolate(input, torch::nn::functional::InterpolateFuncOptions()
        .size(std::vector<std::int64_t>{input.size(2) * 2, input.size(3) * 2}).mode(torch::kBilinear).align_corners(false));
}
torch::Tensor bilinear_down(const torch::Tensor& input) {
    return torch::nn::functional::interpolate(input, torch::nn::functional::InterpolateFuncOptions()
        .size(std::vector<std::int64_t>{input.size(2) / 2, input.size(3) / 2}).mode(torch::kBilinear).align_corners(false));
}
torch::Tensor flow_warp(const torch::Tensor& image, const torch::Tensor& flow) {
    // TCM creates linspace on the image's device/dtype. Its nine Python cache
    // slots key only on flow shape; the first dtype for that key is retained.
    // CPU's -1 index and CUDA index 8 address the same final Python slot.
    using Key = std::array<std::int64_t, 4>;
    static std::array<std::map<Key, torch::Tensor>, 9> grids;
    static std::mutex mutex;
    if (image.dim() != 4 || flow.dim() != 4) throw std::invalid_argument("TCM flow warp requires NCHW tensors");
    int slot = 8;
    if (!image.device().is_cpu()) {
        if (!image.device().is_cuda() || image.device().index() < 0 || image.device().index() > 8)
            throw std::out_of_range("TCM's flow cache supports CPU and CUDA devices 0 through 8");
        slot = image.device().index();
    }
    const Key key{flow.size(0), flow.size(1), flow.size(2), flow.size(3)};
    torch::Tensor base;
    {
        std::lock_guard<std::mutex> lock(mutex);
        auto& cache = grids.at(slot);
        auto found = cache.find(key);
        if (found == cache.end()) {
            const auto options = torch::TensorOptions().device(image.device()).dtype(image.scalar_type());
            const auto horizontal = torch::linspace(-1.0, 1.0, flow.size(3), options)
                .view({1, 1, 1, flow.size(3)}).expand({flow.size(0), 1, flow.size(2), flow.size(3)});
            const auto vertical = torch::linspace(-1.0, 1.0, flow.size(2), options)
                .view({1, 1, flow.size(2), 1}).expand({flow.size(0), 1, flow.size(2), flow.size(3)});
            found = cache.emplace(key, torch::cat({horizontal, vertical}, 1)).first;
        }
        base = found->second;
    }
    const auto normalized = torch::cat({
        flow.slice(1, 0, 1) / ((image.size(3) - 1.0) / 2.0),
        flow.slice(1, 1, 2) / ((image.size(2) - 1.0) / 2.0)}, 1);
    return torch::nn::functional::grid_sample(image, (base + normalized).permute({0, 2, 3, 1}),
        torch::nn::functional::GridSampleFuncOptions().mode(torch::kBilinear).padding_mode(torch::kBorder).align_corners(true));
}
ResBlockImpl::ResBlockImpl(std::int64_t channels, double slope, bool start, bool end, bool bottleneck)
    : start_(start), end_(end) {
    leaky_relu = register_module("leaky_relu", detail::leaky(slope));
    conv1 = register_module("conv1", conv(channels, bottleneck ? channels / 2 : channels));
    conv2 = register_module("conv2", conv(bottleneck ? channels / 2 : channels, channels));
    if (start_) register_module("first_layer", leaky_relu);
    else register_module("first_layer", torch::nn::Identity());
    if (end_) register_module("last_layer", leaky_relu);
    else register_module("last_layer", torch::nn::Identity());
}
torch::Tensor ResBlockImpl::forward(const torch::Tensor& input) {
    auto result = start_ ? leaky_relu(input) : input;
    result = conv2(leaky_relu(conv1(result)));
    if (end_) result = leaky_relu(result);
    return input + result;
}
MEBasicImpl::MEBasicImpl() {
    relu = register_module("relu", torch::nn::ReLU());
    conv1 = register_module("conv1", conv(8, 32, 7));
    conv2 = register_module("conv2", conv(32, 64, 7));
    conv3 = register_module("conv3", conv(64, 32, 7));
    conv4 = register_module("conv4", conv(32, 16, 7));
    conv5 = register_module("conv5", conv(16, 2, 7));
}
torch::Tensor MEBasicImpl::forward(const torch::Tensor& input) {
    auto value = relu(conv1(input)); value = relu(conv2(value));
    value = relu(conv3(value)); value = relu(conv4(value)); return conv5(value);
}
SpynetImpl::SpynetImpl() {
    moduleBasic = register_module("moduleBasic", torch::nn::ModuleList());
    for (int level = 0; level < 4; ++level) moduleBasic->push_back(MEBasic());
}
torch::Tensor SpynetImpl::forward(const torch::Tensor& current, const torch::Tensor& reference) {
    std::array<torch::Tensor, 4> x{current}, y{reference};
    for (std::size_t i = 1; i < 4; ++i) {
        x[i] = torch::avg_pool2d(x[i - 1], {2, 2}, {2, 2});
        y[i] = torch::avg_pool2d(y[i - 1], {2, 2}, {2, 2});
    }
    auto motion = torch::zeros({current.size(0), 2, y[3].size(2) / 2, y[3].size(3) / 2}, current.options());
    for (std::size_t i = 0; i < 4; ++i) {
        ProcessInterrupt::check();
        auto up = bilinear_up(motion) * 2.0;
        const auto level = 3 - i;
        motion = up + moduleBasic->at<MEBasicImpl>(i).forward(
            torch::cat({x[level], flow_warp(y[level], up), up}, 1));
    }
    return motion;
}
FeatureExtractorImpl::FeatureExtractorImpl(std::int64_t channels) {
    conv1 = register_module("conv1", conv(channels, channels));
    res_block1 = register_module("res_block1", ResBlock(channels));
    conv2 = register_module("conv2", conv(channels, channels, 3, 2));
    res_block2 = register_module("res_block2", ResBlock(channels));
    conv3 = register_module("conv3", conv(channels, channels, 3, 2));
    res_block3 = register_module("res_block3", ResBlock(channels));
}
Contexts FeatureExtractorImpl::forward(const torch::Tensor& input) {
    const auto one = res_block1(conv1(input));
    const auto two = res_block2(conv2(one));
    return {one, two, res_block3(conv3(two))};
}
MultiScaleContextFusionImpl::MultiScaleContextFusionImpl(std::int64_t in, std::int64_t out) {
    conv3_up = register_module("conv3_up", subpel(in, out));
    res_block3_up = register_module("res_block3_up", ResBlock(out));
    conv3_out = register_module("conv3_out", conv(out, out));
    res_block3_out = register_module("res_block3_out", ResBlock(out));
    conv2_up = register_module("conv2_up", subpel(out * 2, out));
    res_block2_up = register_module("res_block2_up", ResBlock(out));
    conv2_out = register_module("conv2_out", conv(out * 2, out));
    res_block2_out = register_module("res_block2_out", ResBlock(out));
    conv1_out = register_module("conv1_out", conv(out * 2, out));
    res_block1_out = register_module("res_block1_out", ResBlock(out));
}
Contexts MultiScaleContextFusionImpl::forward(const Contexts& input) {
    const auto up3 = res_block3_up(conv3_up->forward(input[2]));
    const auto out3 = res_block3_out(conv3_out(input[2]));
    const auto combined2 = torch::cat({up3, input[1]}, 1);
    const auto up2 = res_block2_up(conv2_up->forward(combined2));
    const auto out2 = res_block2_out(conv2_out(combined2));
    const auto out1 = res_block1_out(conv1_out(torch::cat({up2, input[0]}, 1)));
    return {input[0] + out1, input[1] + out2, input[2] + out3};
}
ContextualEncoderImpl::ContextualEncoderImpl(std::int64_t n, std::int64_t m) {
    conv1 = register_module("conv1", conv(n + 3, n, 3, 2));
    gdn1 = register_module("gdn1", GDN(n));
    res1 = register_module("res1", ResBlock(n * 2, 0.1, false, true, true));
    conv2 = register_module("conv2", conv(n * 2, n, 3, 2));
    gdn2 = register_module("gdn2", GDN(n));
    res2 = register_module("res2", ResBlock(n * 2, 0.1, false, true, true));
    conv3 = register_module("conv3", conv(n * 2, n, 3, 2));
    gdn3 = register_module("gdn3", GDN(n));
    conv4 = register_module("conv4", conv(n, m, 3, 2));
}
torch::Tensor ContextualEncoderImpl::forward(const torch::Tensor& input, const Contexts& contexts) {
    auto feature = gdn1(conv1(torch::cat({input, contexts[0]}, 1)));
    feature = res1(torch::cat({feature, contexts[1]}, 1));
    feature = gdn2(conv2(feature));
    feature = res2(torch::cat({feature, contexts[2]}, 1));
    return conv4(gdn3(conv3(feature)));
}
ContextualDecoderImpl::ContextualDecoderImpl(std::int64_t n, std::int64_t m) {
    up1 = register_module("up1", subpel(m, n));
    gdn1 = register_module("gdn1", GDN(n, true));
    up2 = register_module("up2", subpel(n, n));
    gdn2 = register_module("gdn2", GDN(n, true));
    res1 = register_module("res1", ResBlock(n * 2, 0.1, false, true, true));
    up3 = register_module("up3", subpel(n * 2, n));
    gdn3 = register_module("gdn3", GDN(n, true));
    res2 = register_module("res2", ResBlock(n * 2, 0.1, false, true, true));
    up4 = register_module("up4", subpel(n * 2, 32));
}
torch::Tensor ContextualDecoderImpl::forward(const torch::Tensor& input, const Contexts& contexts) {
    auto feature = gdn1(up1->forward(input));
    feature = gdn2(up2->forward(feature));
    feature = res1(torch::cat({feature, contexts[2]}, 1));
    feature = gdn3(up3->forward(feature));
    feature = res2(torch::cat({feature, contexts[1]}, 1));
    return up4->forward(feature);
}
TemporalPriorEncoderImpl::TemporalPriorEncoderImpl(std::int64_t n, std::int64_t m) {
    conv1 = register_module("conv1", conv(n, n, 3, 2));
    gdn1 = register_module("gdn1", GDN(n));
    conv2 = register_module("conv2", conv(n * 2, m, 3, 2));
    gdn2 = register_module("gdn2", GDN(m));
    conv3 = register_module("conv3", conv(m + n, m * 3 / 2, 3, 2));
    gdn3 = register_module("gdn3", GDN(m * 3 / 2));
    conv4 = register_module("conv4", conv(m * 3 / 2, m * 2, 3, 2));
}
torch::Tensor TemporalPriorEncoderImpl::forward(const Contexts& contexts) {
    auto feature = gdn1(conv1(contexts[0]));
    feature = gdn2(conv2(torch::cat({feature, contexts[1]}, 1)));
    feature = gdn3(conv3(torch::cat({feature, contexts[2]}, 1)));
    return conv4(feature);
}
ReconGenerationImpl::ReconGenerationImpl(std::int64_t context, std::int64_t residual, std::int64_t channels) {
    feature_conv = register_module("feature_conv", torch::nn::Sequential());
    feature_conv->push_back(conv(context + residual, channels));
    feature_conv->push_back(ResBlock(channels)); feature_conv->push_back(ResBlock(channels));
    recon_conv = register_module("recon_conv", conv(channels, 3));
}
std::pair<torch::Tensor, torch::Tensor> ReconGenerationImpl::forward(
    const torch::Tensor& first, const torch::Tensor& second) {
    const auto feature = feature_conv->forward(torch::cat({first, second}, 1));
    return {feature, recon_conv(feature)};
}
}  // namespace uf::tcm
