// Copyright (c) Microsoft Corporation. Licensed under the MIT License.
// Native adaptation of DCVC-TCM/src/models/video_net_dmc.py.
#include "dcvc_tcm_internal.hpp"
#include "uf/process.hpp"
#include "uf/torch_pickle.hpp"
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace uf::tcm {
namespace {
torch::nn::Sequential hyper_encoder(std::int64_t channels) {
    torch::nn::Sequential result;
    result->push_back(detail::conv(channels, 64)); result->push_back(detail::leaky());
    result->push_back(detail::conv(64, 64, 3, 2)); result->push_back(detail::leaky());
    result->push_back(detail::conv(64, 64, 3, 2)); return result;
}
torch::nn::Sequential hyper_decoder(std::int64_t channels) {
    torch::nn::Sequential result;
    result->push_back(detail::deconv(64, channels)); result->push_back(detail::leaky());
    result->push_back(detail::deconv(channels, channels * 3 / 2)); result->push_back(detail::leaky());
    result->push_back(detail::deconv(channels * 3 / 2, channels * 2, 1)); return result;
}
std::pair<torch::Tensor, torch::Tensor> halves(const torch::Tensor& tensor) {
    const auto parts = tensor.chunk(2, 1); return {parts.at(0), parts.at(1)};
}
torch::Tensor total_bits(const torch::Tensor& probability) {
    return (-1.0 * torch::log(probability + 1e-5) / std::log(2.0)).clamp(0, 50).sum();
}
}  // namespace
DMCImpl::DMCImpl(std::int64_t window_size) {
    optic_flow = register_module("optic_flow", Spynet());
    mv_encoder = register_module("mv_encoder", torch::nn::Sequential());
    for (int i = 0; i < 3; ++i) {
        mv_encoder->push_back(detail::conv(i ? channel_mv : 2, channel_mv, 3, 2));
        mv_encoder->push_back(GDN(channel_mv));
        mv_encoder->push_back(ResBlock(channel_mv, 0.01, false));
        mv_encoder->push_back(detail::leaky(0.1));
    }
    mv_encoder->push_back(detail::conv(channel_mv, channel_mv, 3, 2));
    mv_prior_encoder = register_module("mv_prior_encoder", hyper_encoder(channel_mv));
    mv_prior_decoder = register_module("mv_prior_decoder", hyper_decoder(channel_mv));
    mv_decoder = register_module("mv_decoder", torch::nn::Sequential());
    mv_decoder->push_back(detail::deconv(channel_mv, channel_mv));
    mv_decoder->push_back(detail::leaky(0.1)); mv_decoder->push_back(ResBlock(channel_mv, 0.01, false));
    mv_decoder->push_back(GDN(channel_mv, true));
    mv_decoder->push_back(detail::deconv(channel_mv, channel_mv)); mv_decoder->push_back(GDN(channel_mv, true));
    mv_decoder->push_back(detail::deconv(channel_mv, channel_mv)); mv_decoder->push_back(GDN(channel_mv, true));
    mv_decoder->push_back(detail::deconv(channel_mv, 2));
    feature_adaptor_I = register_module("feature_adaptor_I", detail::conv(3, channel_N));
    feature_adaptor_P = register_module("feature_adaptor_P", detail::conv(channel_N, channel_N, 1));
    feature_extractor = register_module("feature_extractor", FeatureExtractor());
    context_fusion_net = register_module("context_fusion_net", MultiScaleContextFusion());
    contextual_encoder = register_module("contextual_encoder", ContextualEncoder());
    contextual_hyper_prior_encoder = register_module("contextual_hyper_prior_encoder", hyper_encoder(channel_M));
    contextual_hyper_prior_decoder = register_module("contextual_hyper_prior_decoder", hyper_decoder(channel_M));
    temporal_prior_encoder = register_module("temporal_prior_encoder", TemporalPriorEncoder());
    contextual_entropy_parameter = register_module("contextual_entropy_parameter", torch::nn::Sequential());
    contextual_entropy_parameter->push_back(detail::conv(channel_M * 12 / 3, channel_M * 10 / 3));
    contextual_entropy_parameter->push_back(detail::leaky());
    contextual_entropy_parameter->push_back(detail::conv(channel_M * 10 / 3, channel_M * 8 / 3));
    contextual_entropy_parameter->push_back(detail::leaky());
    contextual_entropy_parameter->push_back(detail::conv(channel_M * 8 / 3, channel_M * 6 / 3));
    contextual_decoder = register_module("contextual_decoder", ContextualDecoder());
    recon_generation_net = register_module("recon_generation_net", ReconGeneration());
    bit_estimator_z = register_module("bit_estimator_z", BitEstimator(channel_N));
    bit_estimator_z_mv = register_module("bit_estimator_z_mv", BitEstimator(channel_N));
    ms_ssim_loss = register_module("ms_ssim_loss", MSSSIM(window_size));
}
Contexts DMCImpl::multi_scale_feature_extractor(const torch::Tensor& reference, const torch::Tensor& feature) {
    return feature_extractor->forward(feature.defined() ? feature_adaptor_P(feature) : feature_adaptor_I(reference));
}
std::pair<Contexts, torch::Tensor> DMCImpl::motion_compensation(
    const torch::Tensor& reference, const torch::Tensor& feature, const torch::Tensor& motion) {
    const auto warped = flow_warp(reference, motion);
    const auto half = bilinear_down(motion) / 2, quarter = bilinear_down(half) / 2;
    const auto features = multi_scale_feature_extractor(reference, feature);
    return {context_fusion_net->forward({flow_warp(features[0], motion), flow_warp(features[1], half),
                                        flow_warp(features[2], quarter)}), warped};
}
torch::Tensor DMCImpl::quant(const torch::Tensor& input) { return torch::round(input); }
std::pair<torch::Tensor, torch::Tensor> DMCImpl::get_y_bits_probs(
    const torch::Tensor& y, const torch::Tensor& sigma) {
    const auto scales = sigma.clamp(1e-5, 1e10);
    const auto probability = dcvc2021::laplace_cdf(y + 0.5, scales) - dcvc2021::laplace_cdf(y - 0.5, scales);
    return {total_bits(probability), probability};
}
std::pair<torch::Tensor, torch::Tensor> DMCImpl::get_z_bits_probs(const torch::Tensor& z, BitEstimator& estimator) {
    const auto probability = estimator(z + 0.5) - estimator(z - 0.5);
    return {total_bits(probability), probability};
}
ForwardResult DMCImpl::forward_one_frame(const torch::Tensor& input, const torch::Tensor& reference,
                                        const torch::Tensor& reference_feature) {
    ProcessInterrupt::check();
    ForwardResult result;
    result.mv_y = mv_encoder->forward(optic_flow(input, reference));
    result.mv_z = mv_prior_encoder->forward(result.mv_y);
    const auto mv_z_hat = quant(result.mv_z);
    const auto motion_params = halves(mv_prior_decoder->forward(mv_z_hat));
    const auto mv_q = quant(result.mv_y - motion_params.second);
    result.mv = mv_decoder->forward(mv_q + motion_params.second);
    const auto compensation = motion_compensation(reference, reference_feature, result.mv);
    const auto& contexts = compensation.first;
    result.warp_frame = compensation.second;
    result.y = contextual_encoder->forward(input, contexts);
    result.z = contextual_hyper_prior_encoder->forward(result.y);
    const auto z_hat = quant(result.z);
    const auto hierarchical = contextual_hyper_prior_decoder->forward(z_hat);
    result.temporal_params = temporal_prior_encoder->forward(contexts);
    const auto params = halves(contextual_entropy_parameter->forward(torch::cat({result.temporal_params, hierarchical}, 1)));
    const auto y_q = quant(result.y - params.second);
    const auto residual_feature = contextual_decoder->forward(y_q + params.second, contexts);
    // Python passes the 32-channel residual first, despite naming the first
    // ReconGeneration argument "ctx"; preserve that concatenation order.
    const auto reconstruction = recon_generation_net(residual_feature, contexts[0]);
    result.feature = reconstruction.first; result.recon_image = reconstruction.second;
    result.mse_loss = (result.recon_image - input).pow(2).mean();
    result.me_mse = (result.warp_frame - input).pow(2).mean();
    result.recon_ms_ssim = ms_ssim_loss(result.recon_image, input);
    result.me_ms_ssim = ms_ssim_loss(result.warp_frame, input);
    result.bit_y = get_y_bits_probs(y_q, params.first).first;
    result.bit_mv_y = get_y_bits_probs(mv_q, motion_params.first).first;
    result.bit_z = get_z_bits_probs(z_hat, bit_estimator_z).first;
    result.bit_mv_z = get_z_bits_probs(mv_z_hat, bit_estimator_z_mv).first;
    const auto pixels = input.size(0) * input.size(2) * input.size(3);
    result.bpp_y = result.bit_y / pixels; result.bpp_z = result.bit_z / pixels;
    result.bpp_mv_y = result.bit_mv_y / pixels; result.bpp_mv_z = result.bit_mv_z / pixels;
    result.bpp = result.bpp_y + result.bpp_z + result.bpp_mv_y + result.bpp_mv_z;
    result.bit = result.bit_y + result.bit_z + result.bit_mv_y + result.bit_mv_z;
    return result;
}
ForwardResult DMCImpl::forward(const torch::Tensor&, const torch::Tensor&, const torch::Tensor&) {
    throw std::logic_error("TCM DMC defines forward_one_frame; Python's generic forward is not implemented");
}
void DMCImpl::update(bool force) {
    // As in Python, repeated update(false) replaces this pointer while models
    // retaining their old coder skip reassignment. Use force=true to refresh.
    entropy_coder = std::make_shared<EntropyCoder>();
    bit_estimator_z_mv->update(force, entropy_coder);
    bit_estimator_z->update(force, entropy_coder);
    gaussian_encoder.update(force, entropy_coder);
}
CompressedLatents DMCImpl::compress(const torch::Tensor& y, const torch::Tensor& z,
    const torch::Tensor& mv_y, const torch::Tensor& mv_z, const torch::Tensor& temporal_params) {
    if (!entropy_coder) throw std::runtime_error("TCM DMC must be updated before coding");
    entropy_coder->reset_encoder();
    const auto mv_z_hat = quant(mv_z); bit_estimator_z_mv->encode(mv_z_hat);
    const auto motion_params = halves(mv_prior_decoder->forward(mv_z_hat));
    gaussian_encoder.encode(quant(mv_y - motion_params.second), motion_params.first);
    const auto z_hat = quant(z); bit_estimator_z->encode(z_hat);
    const auto hierarchical = contextual_hyper_prior_decoder->forward(z_hat);
    const auto params = halves(contextual_entropy_parameter->forward(torch::cat({temporal_params, hierarchical}, 1)));
    gaussian_encoder.encode(quant(y - params.second), params.first);
    auto stream = entropy_coder->flush_encoder();
    if (stream.size() > std::numeric_limits<std::uint64_t>::max() / 8) throw std::overflow_error("TCM stream bit count");
    return {static_cast<std::uint64_t>(stream.size()) * 8, std::move(stream)};
}
DecodedFrame DMCImpl::decompress(const torch::Tensor& reference, const torch::Tensor& reference_feature,
    const std::string& stream, std::int64_t height, std::int64_t width) {
    if (!entropy_coder) throw std::runtime_error("TCM DMC must be updated before decoding");
    entropy_coder->set_stream(stream);
    const auto device = parameters().at(0).device();
    const auto size = dcvc2021::downsampled_shape(height, width, 64);
    const auto mv_z = bit_estimator_z_mv->decode_stream(size).to(device);
    const auto motion_params = halves(mv_prior_decoder->forward(mv_z));
    const auto mv_q = gaussian_encoder.decode_stream(motion_params.first).to(device);
    const auto motion = mv_decoder->forward(mv_q + motion_params.second);
    const auto contexts = motion_compensation(reference, reference_feature, motion).first;
    const auto z = bit_estimator_z->decode_stream(size).to(device);
    const auto hierarchical = contextual_hyper_prior_decoder->forward(z);
    const auto temporal = temporal_prior_encoder->forward(contexts);
    const auto params = halves(contextual_entropy_parameter->forward(torch::cat({temporal, hierarchical}, 1)));
    const auto y_q = gaussian_encoder.decode_stream(params.first).to(device);
    const auto residual = contextual_decoder->forward(y_q + params.second, contexts);
    const auto result = recon_generation_net(residual, contexts[0]);
    return {result.second.clamp(0, 1), result.first};
}
FrameResult DMCImpl::encode_decode(const torch::Tensor& input, const torch::Tensor& reference,
    const torch::Tensor& reference_feature, std::optional<std::filesystem::path> output,
    std::int64_t width, std::int64_t height) {
    const auto encoded = forward_one_frame(input, reference, reference_feature);
    if (!output) return {encoded.recon_image, encoded.feature, encoded.bit_y.item<double>(),
        encoded.bit_z.item<double>(), encoded.bit_mv_y.item<double>(), encoded.bit_mv_z.item<double>(),
        encoded.bit.item<double>(), 0};
    const auto compressed = compress(encoded.y, encoded.z, encoded.mv_y, encoded.mv_z, encoded.temporal_params);
    write_p(*output, compressed.string);
    if (!std::filesystem::is_regular_file(*output)) throw std::runtime_error("TCM output is not a regular file");
    const auto bits = static_cast<double>(std::filesystem::file_size(*output)) * 8;
    const auto stream = read_p(*output);
    const auto start = std::chrono::system_clock::now();
    const auto decoded = decompress(reference, reference_feature, stream, height, width);
    const auto elapsed = std::chrono::duration<double>(std::chrono::system_clock::now() - start).count();
    return {decoded.x_hat, decoded.feature, 0, 0, 0, 0, bits, elapsed};
}
void DMCImpl::load_normalized_dict(const std::map<std::string, torch::Tensor>& state, bool strict) {
    std::map<std::string, torch::Tensor> targets;
    for (const auto& item : named_parameters(true)) targets.emplace(item.key(), item.value());
    for (const auto& item : named_buffers(true)) targets.emplace(item.key(), item.value());
    for (const auto& item : targets) {
        const auto found = state.find(item.first);
        if (found == state.end()) {
            if (strict) throw std::runtime_error("TCM checkpoint is missing: " + item.first);
            continue;
        }
        if (found->second.sizes() != item.second.sizes()) throw std::runtime_error("TCM checkpoint shape differs: " + item.first);
    }
    if (strict) for (const auto& item : state)
        if (!targets.count(item.first)) throw std::runtime_error("Unexpected TCM checkpoint parameter: " + item.first);
    torch::NoGradGuard no_grad;
    for (auto& item : targets) if (const auto found = state.find(item.first); found != state.end()) item.second.copy_(found->second);
    // Entropy lists are separate cached state; loading does not refresh them.
}
void DMCImpl::load_dict(const std::map<std::string, torch::Tensor>& state, bool strict) {
    std::map<std::string, torch::Tensor> normalized;
    for (const auto& item : state) {
        auto name = item.first; if (name.rfind("module.", 0) == 0) name.erase(0, 7);
        normalized[name] = item.second;
    }
    load_normalized_dict(normalized, strict);
}
void DMCImpl::load_checkpoint(const std::filesystem::path& path, bool strict) {
    auto value = load_torch_pickle_cpu(path);
    if (!value.isGenericDict()) throw std::runtime_error("TCM checkpoint must be a dictionary");
    const auto dictionary = value.toGenericDict();
    if (const auto wrapped = dictionary.find("state_dict"); wrapped != dictionary.end()) value = wrapped->value();
    if (!value.isGenericDict()) throw std::runtime_error("TCM checkpoint state_dict must be a dictionary");
    std::map<std::string, torch::Tensor> state;
    for (const auto& entry : value.toGenericDict()) {
        if (!entry.key().isString() || !entry.value().isTensor()) throw std::runtime_error("Invalid TCM checkpoint entry");
        auto name = entry.key().toStringRef(); if (name.rfind("module.", 0) == 0) name.erase(0, 7);
        // Traverse the insertion-ordered IValue dictionary before sorting:
        // colliding prefixed/unprefixed keys retain Python's last assignment.
        state[name] = entry.value().toTensor();
    }
    load_normalized_dict(state, strict);
}
}  // namespace uf::tcm
