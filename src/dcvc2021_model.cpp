// Copyright (c) Microsoft Corporation. Licensed under the MIT License.
// Native adaptation of DCVC-family/DCVC/src/models/DCVC_net.py.
#include "uf/dcvc2021.hpp"
#include "uf/process.hpp"
#include "uf/torch_pickle.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace uf::dcvc2021 {
namespace {
torch::nn::Conv2d convolution(std::int64_t in, std::int64_t out, std::int64_t kernel,
                              std::int64_t stride = 1, std::int64_t padding = 0) {
    return torch::nn::Conv2d(torch::nn::Conv2dOptions(in, out, kernel).stride(stride).padding(padding));
}
torch::nn::ConvTranspose2d deconvolution(std::int64_t in, std::int64_t out,
                                        std::int64_t kernel, std::int64_t stride = 1,
                                        std::int64_t padding = 0, std::int64_t output_padding = 0) {
    return torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(in, out, kernel)
        .stride(stride).padding(padding).output_padding(output_padding));
}
torch::nn::LeakyReLU leaky(double slope = 0.01, bool inplace = true) {
    return torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(slope).inplace(inplace));
}
torch::nn::Sequential subpel(std::int64_t in, std::int64_t out) {
    torch::nn::Sequential result;
    result->push_back(convolution(in, out * 4, 3, 1, 1));
    result->push_back(torch::nn::PixelShuffle(torch::nn::PixelShuffleOptions(2)));
    return result;
}
torch::nn::Sequential hyper_encoder(std::int64_t channels) {
    torch::nn::Sequential result;
    result->push_back(convolution(channels, 64, 3, 1, 1));
    result->push_back(leaky());
    result->push_back(convolution(64, 64, 5, 2, 2));
    result->push_back(leaky());
    result->push_back(convolution(64, 64, 5, 2, 2));
    return result;
}
torch::nn::Sequential entropy_network(std::int64_t channels) {
    torch::nn::Sequential result;
    result->push_back(convolution(channels * 12 / 3, channels * 10 / 3, 1));
    result->push_back(leaky());
    result->push_back(convolution(channels * 10 / 3, channels * 8 / 3, 1));
    result->push_back(leaky());
    result->push_back(convolution(channels * 8 / 3, channels * 6 / 3, 1));
    return result;
}
void check_images(const torch::Tensor& reference, const torch::Tensor& input, bool coding = false) {
    if (reference.dim() != 4 || reference.size(1) != 3 || reference.sizes() != input.sizes() ||
        reference.device() != input.device() || reference.scalar_type() != input.scalar_type() ||
        (coding && reference.size(0) != 1)) {
        throw std::invalid_argument("DCVC requires matching RGB NCHW frames; entropy coding requires batch size one");
    }
}
torch::Tensor probability_bits(const torch::Tensor& probability) {
    return (-1.0 * (probability + 1e-5).log() / std::log(2.0)).clamp(0, 50).sum();
}
torch::Tensor feature_bits(const torch::Tensor& feature, const torch::Tensor& mean, const torch::Tensor& scale) {
    const auto quantized = (feature - mean).round() + mean;
    const auto values = quantized - mean;
    const auto sigma = scale.clamp(1e-5, 1e10);
    return probability_bits(laplace_cdf(values + 0.5, sigma) - laplace_cdf(values - 0.5, sigma));
}
std::uint32_t stream_dimension(std::int64_t dimension) {
    if (dimension <= 0 || dimension > std::numeric_limits<std::uint32_t>::max()) {
        throw std::out_of_range("DCVC stream dimension exceeds uint32");
    }
    return static_cast<std::uint32_t>(dimension);
}
}  // namespace

VideoModelImpl::VideoModelImpl() {
    bitEstimator_z = register_module("bitEstimator_z", BitEstimator(64));
    bitEstimator_z_mv = register_module("bitEstimator_z_mv", BitEstimator(64));
    feature_extract = register_module("feature_extract", torch::nn::Sequential());
    feature_extract->push_back(convolution(3, 64, 3, 1, 1));
    feature_extract->push_back(ResBlock(64, 64));
    context_refine = register_module("context_refine", torch::nn::Sequential());
    context_refine->push_back(ResBlock(64, 64));
    context_refine->push_back(convolution(64, 64, 3, 1, 1));

    mvEncoder = register_module("mvEncoder", torch::nn::Sequential());
    for (int i = 0; i < 4; ++i) {
        mvEncoder->push_back(convolution(i ? 128 : 2, 128, 3, 2, 1));
        if (i != 3) mvEncoder->push_back(GDN(128));
    }
    mvDecoder_part1 = register_module("mvDecoder_part1", torch::nn::Sequential());
    for (int i = 0; i < 4; ++i) {
        mvDecoder_part1->push_back(deconvolution(128, i == 3 ? 2 : 128, 3, 2, 1, 1));
        if (i != 3) mvDecoder_part1->push_back(GDN(128, true));
    }
    mvDecoder_part2 = register_module("mvDecoder_part2", torch::nn::Sequential());
    for (int i = 0; i < 6; ++i) {
        mvDecoder_part2->push_back(convolution(i ? 64 : 5, 64, 3, 1, 1));
        mvDecoder_part2->push_back(leaky(0.1, false));
    }
    mvDecoder_part2->push_back(convolution(64, 2, 3, 1, 1));

    contextualEncoder = register_module("contextualEncoder", torch::nn::Sequential());
    for (int i = 0; i < 4; ++i) {
        contextualEncoder->push_back(convolution(i ? 64 : 67, i == 3 ? 96 : 64, 5, 2, 2));
        if (i != 3) contextualEncoder->push_back(GDN(64));
        if (i < 2) contextualEncoder->push_back(LeakyResBlock(64));
    }
    contextualDecoder_part1 = register_module("contextualDecoder_part1", torch::nn::Sequential());
    for (int i = 0; i < 4; ++i) {
        contextualDecoder_part1->push_back(subpel(i ? 64 : 96, 64));
        if (i != 3) contextualDecoder_part1->push_back(GDN(64, true));
        if (i == 1 || i == 2) contextualDecoder_part1->push_back(LeakyResBlock(64));
    }
    contextualDecoder_part2 = register_module("contextualDecoder_part2", torch::nn::Sequential());
    contextualDecoder_part2->push_back(convolution(128, 64, 3, 1, 1));
    contextualDecoder_part2->push_back(ResBlock(64, 64));
    contextualDecoder_part2->push_back(ResBlock(64, 64));
    contextualDecoder_part2->push_back(convolution(64, 3, 3, 1, 1));

    priorEncoder = register_module("priorEncoder", hyper_encoder(96));
    priorDecoder = register_module("priorDecoder", torch::nn::Sequential());
    priorDecoder->push_back(deconvolution(64, 96, 5, 2, 2, 1));
    priorDecoder->push_back(leaky());
    priorDecoder->push_back(deconvolution(96, 96, 5, 2, 2, 1));
    priorDecoder->push_back(leaky());
    priorDecoder->push_back(deconvolution(96, 96, 3, 1, 1));
    mvpriorEncoder = register_module("mvpriorEncoder", hyper_encoder(128));
    mvpriorDecoder = register_module("mvpriorDecoder", torch::nn::Sequential());
    mvpriorDecoder->push_back(deconvolution(64, 64, 5, 2, 2, 1));
    mvpriorDecoder->push_back(leaky());
    mvpriorDecoder->push_back(deconvolution(64, 96, 5, 2, 2, 1));
    mvpriorDecoder->push_back(leaky());
    mvpriorDecoder->push_back(deconvolution(96, 256, 3, 1, 1));

    entropy_parameters = register_module("entropy_parameters", entropy_network(96));
    auto_regressive = register_module("auto_regressive", MaskedConv2d(96));
    auto_regressive_mv = register_module("auto_regressive_mv", MaskedConv2d(128));
    entropy_parameters_mv = register_module("entropy_parameters_mv", entropy_network(128));
    temporalPriorEncoder = register_module("temporalPriorEncoder", torch::nn::Sequential());
    for (int i = 0; i < 4; ++i) {
        temporalPriorEncoder->push_back(convolution(64, i == 3 ? 96 : 64, 5, 2, 2));
        if (i != 3) temporalPriorEncoder->push_back(GDN(64));
    }
    opticFlow = register_module("opticFlow", Spynet());
}

torch::Tensor VideoModelImpl::motion_compensation(const torch::Tensor& reference, const torch::Tensor& motion) {
    return context_refine->forward(flow_warp(feature_extract->forward(reference), motion));
}
torch::Tensor VideoModelImpl::refine_motion(const torch::Tensor& reference, const torch::Tensor& motion) {
    return mvDecoder_part2->forward(torch::cat({motion, reference}, 1)) + motion;
}

void VideoModelImpl::update(bool force) {
    bitEstimator_z_mv->update(force);
    bitEstimator_z->update(force);
    gaussian_encoder.update(force);
}

ForwardResult VideoModelImpl::forward(const torch::Tensor& reference, const torch::Tensor& input) {
    check_images(reference, input);
    const auto motion = mvEncoder->forward(opticFlow(input, reference));
    const auto z_mv = mvpriorEncoder->forward(motion).round();
    const auto params_mv = mvpriorDecoder->forward(z_mv);
    const auto quantized_motion = motion.round();
    const auto motion_parameters = entropy_parameters_mv->forward(torch::cat({
        params_mv, auto_regressive_mv(quantized_motion)}, 1)).chunk(2, 1);
    const auto context = motion_compensation(reference,
        refine_motion(reference, mvDecoder_part1->forward(quantized_motion)));
    const auto temporal = temporalPriorEncoder->forward(context);
    const auto feature = contextualEncoder->forward(torch::cat({input, context}, 1));
    const auto z = priorEncoder->forward(feature).round();
    const auto params = priorDecoder->forward(z);
    const auto quantized_feature = feature.round();
    const auto feature_parameters = entropy_parameters->forward(torch::cat({
        temporal, params, auto_regressive(quantized_feature)}, 1)).chunk(2, 1);
    ForwardResult result;
    result.recon_image = contextualDecoder_part2->forward(torch::cat({
        contextualDecoder_part1->forward(quantized_feature), context}, 1));
    result.context = context;
    const auto pixels = input.size(0) * input.size(2) * input.size(3);
    result.bpp_y = feature_bits(feature, feature_parameters.at(0), feature_parameters.at(1)) / pixels;
    result.bpp_mv_y = feature_bits(motion, motion_parameters.at(0), motion_parameters.at(1)) / pixels;
    result.bpp_z = probability_bits(bitEstimator_z(z + 0.5) - bitEstimator_z(z - 0.5)) / pixels;
    result.bpp_mv_z = probability_bits(bitEstimator_z_mv(z_mv + 0.5) - bitEstimator_z_mv(z_mv - 0.5)) / pixels;
    result.bpp = result.bpp_y + result.bpp_z + result.bpp_mv_y + result.bpp_mv_z;
    return result;
}

std::pair<std::string, torch::Tensor> VideoModelImpl::compress_ar(
    const torch::Tensor& latent, MaskedConv2d& prediction, const torch::Tensor& params,
    torch::nn::Sequential& entropy) {
    const auto height = latent.size(2), width = latent.size(3);
    auto decoded = torch::nn::functional::pad(latent, torch::nn::functional::PadFuncOptions({2, 2, 2, 2}));
    auto symbols = torch::zeros_like(latent), scales = torch::zeros_like(latent);
    for (std::int64_t h = 0; h < height; ++h) {
        ProcessInterrupt::check();
        for (std::int64_t w = 0; w < width; ++w) {
            const auto tile = decoded.slice(0, 0, 1).slice(2, h, h + 5).slice(3, w, w + 5);
            // Python's compression path reads the stored weight directly;
            // only MaskedConv2d.forward mutates it with the registered mask.
            const auto context = torch::conv2d(tile, prediction->weight, prediction->bias);
            const auto p = params.slice(0, 0, 1).slice(2, h, h + 1).slice(3, w, w + 1);
            const auto gaussian = entropy->forward(torch::cat({p, context}, 1)).chunk(2, 1);
            const auto q = (tile.slice(2, 2, 3).slice(3, 2, 3) - gaussian.at(0)).round();
            decoded.slice(0, 0, 1).slice(2, h + 2, h + 3).slice(3, w + 2, w + 3).copy_(q + gaussian.at(0));
            symbols.slice(0, 0, 1).slice(2, h, h + 1).slice(3, w, w + 1).copy_(q);
            scales.slice(0, 0, 1).slice(2, h, h + 1).slice(3, w, w + 1).copy_(gaussian.at(1));
        }
    }
    auto encoded = gaussian_encoder.compress(symbols.permute({0, 2, 3, 1}), scales.permute({0, 2, 3, 1}));
    return {std::move(encoded), decoded.slice(2, 2, height + 2).slice(3, 2, width + 2)};
}

torch::Tensor VideoModelImpl::decompress_ar(const std::string& encoded, std::int64_t channels,
    std::int64_t height, std::int64_t width, MaskedConv2d& prediction,
    const torch::Tensor& params, torch::nn::Sequential& entropy) {
    const auto size = downsampled_shape(height, width, 16);
    auto decoded = torch::zeros({1, channels, size.first + 4, size.second + 4},
                                torch::TensorOptions().device(params.device()));
    gaussian_encoder.set_stream(encoded);
    for (std::int64_t h = 0; h < size.first; ++h) {
        ProcessInterrupt::check();
        for (std::int64_t w = 0; w < size.second; ++w) {
            const auto tile = decoded.slice(0, 0, 1).slice(2, h, h + 5).slice(3, w, w + 5);
            const auto context = torch::conv2d(tile, prediction->weight, prediction->bias);
            const auto p = params.slice(0, 0, 1).slice(2, h, h + 1).slice(3, w, w + 1);
            const auto gaussian = entropy->forward(torch::cat({p, context}, 1)).chunk(2, 1);
            const auto value = gaussian_encoder.decode_stream(gaussian.at(1)).to(bitEstimator_z->f1->h.device()) + gaussian.at(0);
            decoded.slice(0, 0, 1).slice(2, h + 2, h + 3).slice(3, w + 2, w + 3).copy_(value);
        }
    }
    return decoded.slice(2, 2, size.first + 2).slice(3, 2, size.second + 2);
}

CompressedFrame VideoModelImpl::compress(const torch::Tensor& reference, const torch::Tensor& input) {
    check_images(reference, input, true);
    const auto device = input.device();
    const auto motion = mvEncoder->forward(opticFlow(input, reference));
    const auto z_mv = mvpriorEncoder->forward(motion).round();
    CompressedFrame result;
    result.mv_z_string = bitEstimator_z_mv->compress(z_mv);
    const auto z_mv_hat = bitEstimator_z_mv->decompress(result.mv_z_string, z_mv.size(2), z_mv.size(3)).to(device);
    const auto params_mv = mvpriorDecoder->forward(z_mv_hat);
    auto motion_result = compress_ar(motion, auto_regressive_mv, params_mv, entropy_parameters_mv);
    result.mv_y_string = std::move(motion_result.first);
    const auto context = motion_compensation(reference,
        refine_motion(reference, mvDecoder_part1->forward(motion_result.second)));
    const auto temporal = temporalPriorEncoder->forward(context);
    const auto feature = contextualEncoder->forward(torch::cat({input, context}, 1));
    const auto z = priorEncoder->forward(feature).round();
    result.z_string = bitEstimator_z->compress(z);
    const auto z_hat = bitEstimator_z->decompress(result.z_string, z.size(2), z.size(3)).to(device);
    const auto params = priorDecoder->forward(z_hat);
    auto image_result = compress_ar(feature, auto_regressive, torch::cat({temporal, params}, 1), entropy_parameters);
    result.y_string = std::move(image_result.first);
    result.recon_image = contextualDecoder_part2->forward(torch::cat({
        contextualDecoder_part1->forward(image_result.second), context}, 1));
    const auto pixels = input.size(0) * input.size(2) * input.size(3);
    result.bpp_y = static_cast<double>(result.y_string.size()) * 8 / pixels;
    result.bpp_z = static_cast<double>(result.z_string.size()) * 8 / pixels;
    result.bpp_mv_y = static_cast<double>(result.mv_y_string.size()) * 8 / pixels;
    result.bpp_mv_z = static_cast<double>(result.mv_z_string.size()) * 8 / pixels;
    result.bpp = result.bpp_y + result.bpp_z + result.bpp_mv_y + result.bpp_mv_z;
    return result;
}

torch::Tensor VideoModelImpl::decompress(const torch::Tensor& reference, const PFrameStream& encoded) {
    if (reference.dim() != 4 || reference.size(0) != 1 || reference.size(1) != 3) {
        throw std::invalid_argument("DCVC P-frame reference must have RGB NCHW shape and batch size one");
    }
    const auto device = bitEstimator_z->f1->h.device();
    const auto z_size = downsampled_shape(encoded.height, encoded.width, 64);
    const auto z_mv = bitEstimator_z_mv->decompress(encoded.mv_z_string, z_size.first, z_size.second).to(device);
    const auto params_mv = mvpriorDecoder->forward(z_mv);
    const auto motion = decompress_ar(encoded.mv_y_string, out_channel_mv, encoded.height, encoded.width,
                                      auto_regressive_mv, params_mv, entropy_parameters_mv);
    const auto context = motion_compensation(reference,
        refine_motion(reference, mvDecoder_part1->forward(motion)));
    const auto temporal = temporalPriorEncoder->forward(context);
    const auto z = bitEstimator_z->decompress(encoded.z_string, z_size.first, z_size.second).to(device);
    const auto params = priorDecoder->forward(z);
    const auto feature = decompress_ar(encoded.y_string, out_channel_M, encoded.height, encoded.width,
        auto_regressive, torch::cat({temporal, params}, 1), entropy_parameters);
    return contextualDecoder_part2->forward(torch::cat({
        contextualDecoder_part1->forward(feature), context}, 1)).clamp(0, 1);
}

CompressedFrame VideoModelImpl::encode(const torch::Tensor& reference, const torch::Tensor& input,
                                       const std::filesystem::path& output) {
    auto result = compress(reference, input);
    write_p(output, {stream_dimension(reference.size(2)), stream_dimension(reference.size(3)),
        result.mv_y_string, result.mv_z_string, result.y_string, result.z_string});
    return result;
}
torch::Tensor VideoModelImpl::decode(const torch::Tensor& reference, const std::filesystem::path& input) {
    return decompress(reference, read_p(input));
}
CompressedFrame VideoModelImpl::encode_decode(const torch::Tensor& reference, const torch::Tensor& input,
                                               const std::filesystem::path& output) {
    auto result = encode(reference, input, output);
    result.recon_image = decode(reference, output);
    return result;
}

void VideoModelImpl::load_dict(const std::map<std::string, torch::Tensor>& source) {
    std::map<std::string, torch::Tensor> state;
    for (const auto& item : source) {
        const auto name = item.first.rfind("module.", 0) == 0 ? item.first.substr(7) : item.first;
        state[name] = item.second;
    }
    load_normalized_dict(state);
}

void VideoModelImpl::load_normalized_dict(const std::map<std::string, torch::Tensor>& state) {
    std::map<std::string, torch::Tensor> targets;
    for (const auto& item : named_parameters(true)) targets.emplace(item.key(), item.value());
    for (const auto& item : named_buffers(true)) targets.emplace(item.key(), item.value());
    for (const auto& item : targets) {
        const auto it = state.find(item.first);
        if (it == state.end()) throw std::runtime_error("DCVC checkpoint is missing: " + item.first);
        if (it->second.sizes() != item.second.sizes()) throw std::runtime_error("DCVC checkpoint shape differs: " + item.first);
    }
    for (const auto& item : state) {
        if (!targets.count(item.first)) throw std::runtime_error("Unexpected DCVC checkpoint parameter: " + item.first);
    }
    torch::NoGradGuard no_grad;
    for (auto& item : targets) item.second.copy_(state.at(item.first));
    // As in Python, CDFs are separate cached state. A later load_dict does
    // not rebuild them; update(force=true) explicitly refreshes the tables.
}

void VideoModelImpl::load_checkpoint(const std::filesystem::path& checkpoint) {
    const auto value = load_torch_pickle_cpu(checkpoint);
    if (!value.isGenericDict()) throw std::runtime_error("DCVC checkpoint must be a state dictionary");
    std::map<std::string, torch::Tensor> state;
    for (const auto& entry : value.toGenericDict()) {
        if (!entry.key().isString() || !entry.value().isTensor()) throw std::runtime_error("Invalid DCVC checkpoint entry");
        auto name = entry.key().toStringRef();
        if (name.rfind("module.", 0) == 0) name.erase(0, 7);
        // Normalize while traversing the original Python dictionary. Building
        // a sorted map first would change which duplicate normalized key wins.
        state[name] = entry.value().toTensor();
    }
    load_normalized_dict(state);
}
}  // namespace uf::dcvc2021
