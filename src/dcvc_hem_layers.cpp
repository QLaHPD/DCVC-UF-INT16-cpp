// Microsoft video/context/UNet layers: MIT.
// InterDigital image residual/subpixel layers: Apache-2.0.
#include "dcvc_hem_internal.hpp"
#include <torch/csrc/autograd/custom_function.h>
#include <map>
#include <mutex>
#include <stdexcept>

namespace uf::hem {
namespace {
class BoundFunction : public torch::autograd::Function<BoundFunction> {
public:
    static torch::Tensor forward(torch::autograd::AutogradContext* context, const torch::Tensor& input, double bound) {
        const auto floor = torch::ones_like(input) * bound;
        context->save_for_backward({input, floor}); return torch::maximum(input, floor);
    }
    static torch::autograd::variable_list backward(torch::autograd::AutogradContext* context,
                                                 torch::autograd::variable_list gradients) {
        const auto saved = context->get_saved_variables();
        return {((saved.at(0) >= saved.at(1)) | (gradients.at(0) < 0)).to(gradients.at(0).scalar_type()) * gradients.at(0),
                torch::Tensor{}};
    }
};
torch::nn::Sequential subpel(std::int64_t in, std::int64_t out, std::int64_t factor, std::int64_t kernel) {
    torch::nn::Sequential result;
    result->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(in, out * factor * factor, kernel).padding(kernel / 2)));
    result->push_back(torch::nn::PixelShuffle(torch::nn::PixelShuffleOptions(factor))); return result;
}
}  // namespace
torch::nn::LeakyReLU detail::leaky(double slope) {
    return torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(slope));
}
torch::Tensor lower_bound(const torch::Tensor& input, double bound) { return BoundFunction::apply(input, bound); }
torch::nn::Conv2d conv3x3(std::int64_t in, std::int64_t out, std::int64_t stride) {
    return torch::nn::Conv2d(torch::nn::Conv2dOptions(in, out, 3).padding(1).stride(stride));
}
torch::nn::Conv2d conv1x1(std::int64_t in, std::int64_t out, std::int64_t stride) {
    return torch::nn::Conv2d(torch::nn::Conv2dOptions(in, out, 1).stride(stride));
}
torch::nn::Sequential subpel_conv3x3(std::int64_t in, std::int64_t out, std::int64_t factor) { return subpel(in, out, factor, 3); }
torch::nn::Sequential subpel_conv1x1(std::int64_t in, std::int64_t out, std::int64_t factor) { return subpel(in, out, factor, 1); }
torch::Tensor bilinear_up(const torch::Tensor& input) {
    return torch::nn::functional::interpolate(input, torch::nn::functional::InterpolateFuncOptions()
        .size(std::vector<std::int64_t>{input.size(2) * 2, input.size(3) * 2}).mode(torch::kBilinear).align_corners(false));
}
torch::Tensor bilinear_down(const torch::Tensor& input) {
    return torch::nn::functional::interpolate(input, torch::nn::functional::InterpolateFuncOptions()
        .size(std::vector<std::int64_t>{input.size(2) / 2, input.size(3) / 2}).mode(torch::kBilinear).align_corners(false));
}
torch::Tensor flow_warp(const torch::Tensor& input, const torch::Tensor& flow) {
    using Key = std::array<std::int64_t, 4>;
    static std::array<std::map<Key, torch::Tensor>, 9> grids;
    static std::mutex mutex;
    if (flow.dim() != 4 || input.dim() != 4 || flow.size(1) != 2) throw std::invalid_argument("Invalid HEM warp shape");
    std::size_t slot = 8;
    if (!input.device().is_cpu()) {
        if (!input.device().is_cuda() || input.device().index() < 0 || input.device().index() > 8)
            throw std::invalid_argument("HEM warp cache supports CPU and CUDA indexes 0 through 8");
        slot = input.device().index();
    }
    const Key key{flow.size(0), flow.size(1), flow.size(2), flow.size(3)};
    torch::Tensor grid;
    {
        std::lock_guard<std::mutex> lock(mutex);
        auto& cache = grids.at(slot); auto found = cache.find(key);
        if (found == cache.end()) {
            const auto options = torch::TensorOptions().device(input.device()).dtype(input.scalar_type());
            const auto horizontal = torch::linspace(-1.0, 1.0, flow.size(3), options).view({1, 1, 1, flow.size(3)})
                .expand({flow.size(0), 1, flow.size(2), flow.size(3)});
            const auto vertical = torch::linspace(-1.0, 1.0, flow.size(2), options).view({1, 1, flow.size(2), 1})
                .expand({flow.size(0), 1, flow.size(2), flow.size(3)});
            found = cache.emplace(key, torch::cat({horizontal, vertical}, 1)).first;
        }
        grid = found->second;
    }
    const auto normalized = torch::cat({flow.slice(1, 0, 1) / ((input.size(3) - 1.0) / 2.0),
        flow.slice(1, 1, 2) / ((input.size(2) - 1.0) / 2.0)}, 1);
    return torch::nn::functional::grid_sample(input, (grid + normalized).permute({0, 2, 3, 1}),
        torch::nn::functional::GridSampleFuncOptions().mode(torch::kBilinear).padding_mode(torch::kBorder).align_corners(true));
}
ResidualBlockImpl::ResidualBlockImpl(std::int64_t in, std::int64_t out, double slope) {
    conv1 = register_module("conv1", conv3x3(in, out));
    leaky_relu = register_module("leaky_relu", detail::leaky(slope));
    conv2 = register_module("conv2", conv3x3(out, out));
}
torch::Tensor ResidualBlockImpl::forward(const torch::Tensor& input) { return leaky_relu(conv2(leaky_relu(conv1(input)))) + input; }
ResidualBlockWithStrideImpl::ResidualBlockWithStrideImpl(std::int64_t in, std::int64_t out, std::int64_t stride) {
    conv1 = register_module("conv1", conv3x3(in, out, stride));
    leaky_relu = register_module("leaky_relu", detail::leaky());
    conv2 = register_module("conv2", conv3x3(out, out));
    leaky_relu2 = register_module("leaky_relu2", detail::leaky(0.1));
    if (stride != 1) downsample = register_module("downsample", conv1x1(in, out, stride));
}
torch::Tensor ResidualBlockWithStrideImpl::forward(const torch::Tensor& input) {
    auto out = leaky_relu2(conv2(leaky_relu(conv1(input)))); out.add_(downsample ? downsample(input) : input); return out;
}
ResidualBlockUpsampleImpl::ResidualBlockUpsampleImpl(std::int64_t in, std::int64_t out, std::int64_t factor) {
    subpel_conv = register_module("subpel_conv", subpel_conv1x1(in, out, factor));
    leaky_relu = register_module("leaky_relu", detail::leaky()); conv = register_module("conv", conv3x3(out, out));
    leaky_relu2 = register_module("leaky_relu2", detail::leaky(0.1));
    upsample = register_module("upsample", subpel_conv1x1(in, out, factor));
}
torch::Tensor ResidualBlockUpsampleImpl::forward(const torch::Tensor& input) {
    auto out = leaky_relu2(conv(leaky_relu(subpel_conv->forward(input)))); out.add_(upsample->forward(input)); return out;
}
ResBlockImpl::ResBlockImpl(std::int64_t channels, double slope, bool start, bool end, bool bottleneck)
    : start_(start), end_(end) {
    if (slope < 0.0001) relu = torch::nn::AnyModule(torch::nn::ReLU());
    else relu = torch::nn::AnyModule(detail::leaky(slope));
    register_module("relu", relu.ptr());
    conv1 = register_module("conv1", conv3x3(channels, bottleneck ? channels / 2 : channels));
    conv2 = register_module("conv2", conv3x3(bottleneck ? channels / 2 : channels, channels));
    if (start_) register_module("first_layer", relu.ptr()); else register_module("first_layer", torch::nn::Identity());
    if (end_) register_module("last_layer", relu.ptr()); else register_module("last_layer", torch::nn::Identity());
}
torch::Tensor ResBlockImpl::forward(const torch::Tensor& input) {
    auto out = start_ ? relu.forward<torch::Tensor>(input) : input;
    out = conv2(relu.forward<torch::Tensor>(conv1(out)));
    if (end_) out = relu.forward<torch::Tensor>(out); return input + out;
}
MEBasicImpl::MEBasicImpl() {
    relu = register_module("relu", torch::nn::ReLU());
    const auto conv = [](std::int64_t in, std::int64_t out) { return torch::nn::Conv2d(torch::nn::Conv2dOptions(in, out, 7).padding(3)); };
    conv1 = register_module("conv1", conv(8, 32)); conv2 = register_module("conv2", conv(32, 64));
    conv3 = register_module("conv3", conv(64, 32)); conv4 = register_module("conv4", conv(32, 16));
    conv5 = register_module("conv5", conv(16, 2));
}
torch::Tensor MEBasicImpl::forward(const torch::Tensor& input) { return conv5(relu(conv4(relu(conv3(relu(conv2(relu(conv1(input))))))))); }
SpynetImpl::SpynetImpl() {
    moduleBasic = register_module("moduleBasic", torch::nn::ModuleList());
    for (int i = 0; i < 4; ++i) moduleBasic->push_back(MEBasic());
}
torch::Tensor SpynetImpl::forward(const torch::Tensor& current, const torch::Tensor& reference) {
    std::array<torch::Tensor, 4> a, b; a[0] = current; b[0] = reference;
    for (int i = 1; i < 4; ++i) { a[i] = torch::avg_pool2d(a[i - 1], {2, 2}, {2, 2}); b[i] = torch::avg_pool2d(b[i - 1], {2, 2}, {2, 2}); }
    auto flow = torch::zeros({current.size(0), 2, b[3].size(2) / 2, b[3].size(3) / 2}, current.options());
    for (int i = 0; i < 4; ++i) {
        const auto up = bilinear_up(flow) * 2.0;
        flow = up + moduleBasic->at<MEBasicImpl>(i).forward(torch::cat({a[3 - i], flow_warp(b[3 - i], up), up}, 1));
    }
    return flow;
}
SELayerImpl::SELayerImpl(std::int64_t channels, std::int64_t reduction) {
    fc = register_module("fc", torch::nn::Sequential());
    fc->push_back(torch::nn::Linear(torch::nn::LinearOptions(channels, channels / reduction).bias(false)));
    fc->push_back(torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)));
    fc->push_back(torch::nn::Linear(torch::nn::LinearOptions(channels / reduction, channels).bias(false)));
    fc->push_back(torch::nn::Sigmoid());
}
torch::Tensor SELayerImpl::forward(const torch::Tensor& input) {
    const auto y = fc->forward(input.mean({-1, -2})); return input * y.unsqueeze(-1).unsqueeze(-1);
}
ConvBlockResidualImpl::ConvBlockResidualImpl(std::int64_t in, std::int64_t out, bool se_layer) {
    conv = register_module("conv", torch::nn::Sequential());
    conv->push_back(conv3x3(in, out)); conv->push_back(detail::leaky()); conv->push_back(conv3x3(out, out));
    if (se_layer) conv->push_back(SELayer(out)); else conv->push_back(torch::nn::Identity());
    up_dim = register_module("up_dim", conv1x1(in, out));
}
torch::Tensor ConvBlockResidualImpl::forward(const torch::Tensor& input) {
    const auto first = conv->forward(input), second = up_dim(input); return second + first;
}
UNetImpl::UNetImpl(std::int64_t in, std::int64_t out) {
    max_pool = register_module("max_pool", torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)));
    conv1 = register_module("conv1", ConvBlockResidual(in, 32)); conv2 = register_module("conv2", ConvBlockResidual(32, 64));
    conv3 = register_module("conv3", ConvBlockResidual(64, 128));
    context_refine = register_module("context_refine", torch::nn::Sequential());
    for (int i = 0; i < 4; ++i) context_refine->push_back(ResBlock(128, 0));
    up3 = register_module("up3", subpel_conv1x1(128, 64, 2)); up_conv3 = register_module("up_conv3", ConvBlockResidual(128, 64));
    up2 = register_module("up2", subpel_conv1x1(64, 32, 2)); up_conv2 = register_module("up_conv2", ConvBlockResidual(64, out));
}
torch::Tensor UNetImpl::forward(const torch::Tensor& input) {
    const auto x1 = conv1(input), x2 = conv2(max_pool(x1)), x3 = context_refine->forward(conv3(max_pool(x2)));
    const auto d3 = up_conv3(torch::cat({x2, up3->forward(x3)}, 1));
    return up_conv2(torch::cat({x1, up2->forward(d3)}, 1));
}
std::pair<torch::nn::Sequential, torch::nn::Sequential> get_enc_dec_models(std::int64_t in, std::int64_t out, std::int64_t channels) {
    torch::nn::Sequential enc, dec;
    for (int i = 0; i < 3; ++i) { enc->push_back(ResidualBlockWithStride(i ? channels : in, channels, 2)); enc->push_back(ResidualBlock(channels, channels)); }
    enc->push_back(conv3x3(channels, channels, 2));
    for (int i = 0; i < 3; ++i) { dec->push_back(ResidualBlock(channels, channels)); dec->push_back(ResidualBlockUpsample(channels, channels, 2)); }
    dec->push_back(ResidualBlock(channels, channels)); dec->push_back(subpel_conv1x1(channels, out, 2)); return {enc, dec};
}
std::pair<torch::nn::Sequential, torch::nn::Sequential> get_hyper_enc_dec_models(std::int64_t y, std::int64_t z) {
    torch::nn::Sequential enc, dec;
    for (int i = 0; i < 5; ++i) { enc->push_back(conv3x3(i ? z : y, z, i == 2 || i == 4 ? 2 : 1)); if (i != 4) enc->push_back(detail::leaky()); }
    dec->push_back(conv3x3(z, y)); dec->push_back(detail::leaky()); dec->push_back(subpel_conv1x1(y, y, 2)); dec->push_back(detail::leaky());
    dec->push_back(conv3x3(y, y * 3 / 2)); dec->push_back(detail::leaky()); dec->push_back(subpel_conv1x1(y * 3 / 2, y * 3 / 2, 2));
    dec->push_back(detail::leaky()); dec->push_back(conv3x3(y * 3 / 2, y * 2)); return {enc, dec};
}
torch::nn::Sequential detail::prior(std::int64_t in, std::int64_t middle, std::int64_t out) {
    torch::nn::Sequential result;
    result->push_back(conv3x3(in, middle)); result->push_back(leaky(0.2));
    result->push_back(conv3x3(middle, middle)); result->push_back(leaky(0.2)); result->push_back(conv3x3(middle, out)); return result;
}
FeatureExtractorImpl::FeatureExtractorImpl(std::int64_t channels) {
    conv1 = register_module("conv1", conv3x3(channels, channels)); res_block1 = register_module("res_block1", ResBlock(channels));
    conv2 = register_module("conv2", conv3x3(channels, channels, 2)); res_block2 = register_module("res_block2", ResBlock(channels));
    conv3 = register_module("conv3", conv3x3(channels, channels, 2)); res_block3 = register_module("res_block3", ResBlock(channels));
}
Contexts FeatureExtractorImpl::forward(const torch::Tensor& input) {
    const auto a = res_block1(conv1(input)), b = res_block2(conv2(a)), c = res_block3(conv3(b)); return {a, b, c};
}
MultiScaleContextFusionImpl::MultiScaleContextFusionImpl(std::int64_t in, std::int64_t out) {
    conv3_up = register_module("conv3_up", subpel_conv3x3(in, out, 2)); res_block3_up = register_module("res_block3_up", ResBlock(out));
    conv3_out = register_module("conv3_out", conv3x3(out, out)); res_block3_out = register_module("res_block3_out", ResBlock(out));
    conv2_up = register_module("conv2_up", subpel_conv3x3(out * 2, out, 2)); res_block2_up = register_module("res_block2_up", ResBlock(out));
    conv2_out = register_module("conv2_out", conv3x3(out * 2, out)); res_block2_out = register_module("res_block2_out", ResBlock(out));
    conv1_out = register_module("conv1_out", conv3x3(out * 2, out)); res_block1_out = register_module("res_block1_out", ResBlock(out));
}
Contexts MultiScaleContextFusionImpl::forward(const Contexts& c) {
    const auto up3 = res_block3_up(conv3_up->forward(c[2])), out3 = res_block3_out(conv3_out(c[2]));
    const auto up2 = res_block2_up(conv2_up->forward(torch::cat({up3, c[1]}, 1)));
    const auto out2 = res_block2_out(conv2_out(torch::cat({up3, c[1]}, 1)));
    const auto out1 = res_block1_out(conv1_out(torch::cat({up2, c[0]}, 1)));
    return {c[0] + out1, c[1] + out2, c[2] + out3};
}
ContextualEncoderImpl::ContextualEncoderImpl(std::int64_t n, std::int64_t m) {
    conv1 = register_module("conv1", conv3x3(n + 3, n, 2)); res1 = register_module("res1", ResBlock(n * 2, 0.1, true, true, true));
    conv2 = register_module("conv2", conv3x3(n * 2, n, 2)); res2 = register_module("res2", ResBlock(n * 2, 0.1, true, true, true));
    conv3 = register_module("conv3", conv3x3(n * 2, n, 2)); conv4 = register_module("conv4", conv3x3(n, m, 2));
}
torch::Tensor ContextualEncoderImpl::forward(const torch::Tensor& input, const Contexts& c) {
    auto f = conv1(torch::cat({input, c[0]}, 1)); f = conv2(res1(torch::cat({f, c[1]}, 1)));
    f = conv3(res2(torch::cat({f, c[2]}, 1))); return conv4(f);
}
ContextualDecoderImpl::ContextualDecoderImpl(std::int64_t n, std::int64_t m) {
    up1 = register_module("up1", subpel_conv3x3(m, n, 2)); up2 = register_module("up2", subpel_conv3x3(n, n, 2));
    res1 = register_module("res1", ResBlock(n * 2, 0.1, true, true, true)); up3 = register_module("up3", subpel_conv3x3(n * 2, n, 2));
    res2 = register_module("res2", ResBlock(n * 2, 0.1, true, true, true)); up4 = register_module("up4", subpel_conv3x3(n * 2, 32, 2));
}
torch::Tensor ContextualDecoderImpl::forward(const torch::Tensor& input, const Contexts& c) {
    auto f = up2->forward(up1->forward(input)); f = up3->forward(res1(torch::cat({f, c[2]}, 1)));
    return up4->forward(res2(torch::cat({f, c[1]}, 1)));
}
ReconGenerationImpl::ReconGenerationImpl(std::int64_t context, std::int64_t residual, std::int64_t channels) {
    first_conv = register_module("first_conv", conv3x3(context + residual, channels));
    unet_1 = register_module("unet_1", UNet(channels)); unet_2 = register_module("unet_2", UNet(channels));
    recon_conv = register_module("recon_conv", conv3x3(channels, 3));
}
std::pair<torch::Tensor, torch::Tensor> ReconGenerationImpl::forward(const torch::Tensor& first, const torch::Tensor& second) {
    const auto feature = unet_2(unet_1(first_conv(torch::cat({first, second}, 1)))); return {feature, recon_conv(feature)};
}
}  // namespace uf::hem
