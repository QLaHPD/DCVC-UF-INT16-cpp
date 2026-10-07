// Copyright (c) Microsoft Corporation. Licensed under the MIT License.
#include "dcvc_hem_internal.hpp"
#include <chrono>
#include <stdexcept>

namespace uf::hem {
DMCImpl::DMCImpl(std::int64_t anchors) : CompressionModelImpl("laplace", 64, 64), anchor_num(anchors) {
    optic_flow = register_module("optic_flow", Spynet());
    const auto transforms = get_enc_dec_models(2, 2, channel_mv);
    mv_encoder = register_module("mv_encoder", transforms.first); mv_decoder = register_module("mv_decoder", transforms.second);
    const auto hyper = get_hyper_enc_dec_models(channel_mv, channel_N);
    mv_hyper_prior_encoder = register_module("mv_hyper_prior_encoder", hyper.first);
    mv_hyper_prior_decoder = register_module("mv_hyper_prior_decoder", hyper.second);
    mv_y_prior_fusion = register_module("mv_y_prior_fusion", detail::prior(channel_mv * 3, channel_mv * 3, channel_mv * 3));
    mv_y_spatial_prior = register_module("mv_y_spatial_prior", detail::prior(channel_mv * 4, channel_mv * 3, channel_mv * 2));
    feature_adaptor_I = register_module("feature_adaptor_I", conv3x3(3, channel_N));
    feature_adaptor_P = register_module("feature_adaptor_P", conv1x1(channel_N, channel_N));
    feature_extractor = register_module("feature_extractor", FeatureExtractor());
    context_fusion_net = register_module("context_fusion_net", MultiScaleContextFusion());
    contextual_encoder = register_module("contextual_encoder", ContextualEncoder(channel_N, channel_M));
    contextual_hyper_prior_encoder = register_module("contextual_hyper_prior_encoder", torch::nn::Sequential());
    contextual_hyper_prior_encoder->push_back(conv3x3(channel_M, channel_N)); contextual_hyper_prior_encoder->push_back(detail::leaky());
    contextual_hyper_prior_encoder->push_back(conv3x3(channel_N, channel_N, 2)); contextual_hyper_prior_encoder->push_back(detail::leaky());
    contextual_hyper_prior_encoder->push_back(conv3x3(channel_N, channel_N, 2));
    contextual_hyper_prior_decoder = register_module("contextual_hyper_prior_decoder", torch::nn::Sequential());
    contextual_hyper_prior_decoder->push_back(conv3x3(channel_N, channel_M)); contextual_hyper_prior_decoder->push_back(detail::leaky());
    contextual_hyper_prior_decoder->push_back(subpel_conv1x1(channel_M, channel_M, 2)); contextual_hyper_prior_decoder->push_back(detail::leaky());
    contextual_hyper_prior_decoder->push_back(conv3x3(channel_M, channel_M * 3 / 2)); contextual_hyper_prior_decoder->push_back(detail::leaky());
    contextual_hyper_prior_decoder->push_back(subpel_conv1x1(channel_M * 3 / 2, channel_M * 3 / 2, 2)); contextual_hyper_prior_decoder->push_back(detail::leaky());
    contextual_hyper_prior_decoder->push_back(conv3x3(channel_M * 3 / 2, channel_M * 2));
    temporal_prior_encoder = register_module("temporal_prior_encoder", torch::nn::Sequential());
    temporal_prior_encoder->push_back(conv3x3(channel_N, channel_M * 3 / 2, 2)); temporal_prior_encoder->push_back(detail::leaky(0.1));
    temporal_prior_encoder->push_back(conv3x3(channel_M * 3 / 2, channel_M * 2, 2));
    y_prior_fusion = register_module("y_prior_fusion", torch::nn::Sequential());
    y_prior_fusion->push_back(conv3x3(channel_M * 5, channel_M * 4)); y_prior_fusion->push_back(detail::leaky(0.2));
    y_prior_fusion->push_back(conv3x3(channel_M * 4, channel_M * 3)); y_prior_fusion->push_back(detail::leaky(0.2));
    y_prior_fusion->push_back(conv3x3(channel_M * 3, channel_M * 3));
    y_spatial_prior = register_module("y_spatial_prior", detail::prior(channel_M * 4, channel_M * 3, channel_M * 2));
    contextual_decoder = register_module("contextual_decoder", ContextualDecoder(channel_N, channel_M));
    recon_generation_net = register_module("recon_generation_net", ReconGeneration());
    mv_y_q_basic = register_parameter("mv_y_q_basic", torch::ones({1, channel_mv, 1, 1}));
    mv_y_q_scale = register_parameter("mv_y_q_scale", torch::ones({anchor_num, 1, 1, 1}));
    y_q_basic = register_parameter("y_q_basic", torch::ones({1, channel_M, 1, 1}));
    y_q_scale = register_parameter("y_q_scale", torch::ones({anchor_num, 1, 1, 1}));
    initialize_weights();
}
Contexts DMCImpl::multi_scale_feature_extractor(const Dpb& reference) {
    const auto feature = reference.ref_feature.defined() ? feature_adaptor_P(reference.ref_feature) : feature_adaptor_I(reference.ref_frame);
    return feature_extractor(feature);
}
std::pair<Contexts, torch::Tensor> DMCImpl::motion_compensation(const Dpb& reference, const torch::Tensor& motion) {
    const auto warp = flow_warp(reference.ref_frame, motion);
    const auto mv2 = bilinear_down(motion) / 2, mv3 = bilinear_down(mv2) / 2;
    const auto features = multi_scale_feature_extractor(reference);
    const Contexts contexts{flow_warp(features[0], motion), flow_warp(features[1], mv2), flow_warp(features[2], mv3)};
    return {context_fusion_net(contexts), warp};
}
std::pair<torch::Tensor, torch::Tensor> DMCImpl::get_q_scales_from_ckpt(const std::filesystem::path& path) {
    const auto state = get_state_dict(path); return {state.at("y_q_scale").reshape({-1}), state.at("mv_y_q_scale").reshape({-1})};
}
torch::Tensor DMCImpl::get_curr_mv_y_q(const torch::Tensor& scale) const {
    if (!scale.defined()) throw std::invalid_argument("HEM DMC requires a motion quality scale"); return lower_bound(mv_y_q_basic, 0.5) * scale;
}
torch::Tensor DMCImpl::get_curr_y_q(const torch::Tensor& scale) const {
    if (!scale.defined()) throw std::invalid_argument("HEM DMC requires an image quality scale"); return lower_bound(y_q_basic, 0.5) * scale;
}
struct DMCImpl::Analysis {
    Dpb dpb;
    torch::Tensor y, z, z_hat, mv_y, mv_z, mv_z_hat, warp;
    DualPriorResult residual, motion;
    DualPriorStream residual_stream, motion_stream;
};
DMCImpl::Analysis DMCImpl::analyze(const torch::Tensor& input, const Dpb& reference,
    const torch::Tensor& mv_scale, const torch::Tensor& y_scale, bool stream) {
    const auto current_mv_q = get_curr_mv_y_q(mv_scale), current_q = get_curr_y_q(y_scale);
    Analysis result;
    const auto estimated = optic_flow(input, reference.ref_frame);
    result.mv_y = mv_encoder->forward(estimated) / current_mv_q;
    result.mv_z = mv_hyper_prior_encoder->forward(result.mv_y);
    result.mv_z_hat = stream ? result.mv_z.round() : quant(result.mv_z);
    const auto mv_hyper = mv_hyper_prior_decoder->forward(result.mv_z_hat);
    const auto ref_mv = reference.ref_mv_y.defined() ? reference.ref_mv_y : torch::zeros_like(result.mv_y);
    const auto mv_params = mv_y_prior_fusion->forward(torch::cat({mv_hyper, ref_mv}, 1)).chunk(3, 1);
    torch::Tensor mv_hat;
    if (stream) {
        result.motion_stream = compress_dual_prior(result.mv_y, mv_params.at(2), mv_params.at(1), mv_params.at(0), mv_y_spatial_prior);
        mv_hat = result.motion_stream.y_hat * current_mv_q;
    } else {
        result.motion = forward_dual_prior(result.mv_y, mv_params.at(2), mv_params.at(1), mv_params.at(0), mv_y_spatial_prior);
        mv_hat = result.motion.y_hat * current_mv_q;
    }
    const auto decoded_motion = mv_decoder->forward(mv_hat);
    const auto context = motion_compensation(reference, decoded_motion); result.warp = context.second;
    result.y = contextual_encoder(input, context.first) / current_q;
    result.z = contextual_hyper_prior_encoder->forward(result.y);
    result.z_hat = stream ? result.z.round() : quant(result.z);
    const auto hierarchical = contextual_hyper_prior_decoder->forward(result.z_hat);
    const auto temporal = temporal_prior_encoder->forward(context.first[2]);
    const auto ref_y = reference.ref_y.defined() ? reference.ref_y : torch::zeros_like(result.y);
    const auto params = y_prior_fusion->forward(torch::cat({temporal, hierarchical, ref_y}, 1)).chunk(3, 1);
    torch::Tensor y_hat;
    if (stream) {
        result.residual_stream = compress_dual_prior(result.y, params.at(2), params.at(1), params.at(0), y_spatial_prior);
        y_hat = result.residual_stream.y_hat * current_q;
    } else {
        result.residual = forward_dual_prior(result.y, params.at(2), params.at(1), params.at(0), y_spatial_prior);
        y_hat = result.residual.y_hat * current_q;
    }
    const auto residual = contextual_decoder(y_hat, context.first);
    // Keep the caller's 32-channel residual first, despite the source arg names.
    const auto reconstruction = recon_generation_net(residual, context.first[0]);
    result.dpb = {reconstruction.second, reconstruction.first, y_hat, mv_hat}; return result;
}
ForwardResult DMCImpl::forward_one_frame(const torch::Tensor& input, const Dpb& reference,
                                        const torch::Tensor& mv_scale, const torch::Tensor& y_scale) {
    const auto encoded = analyze(input, reference, mv_scale, y_scale, false);
    const auto pixels = input.size(2) * input.size(3);
    const auto distortion = mse(input, encoded.dpb.ref_frame), quality = ssim(input, encoded.dpb.ref_frame);
    const auto motion_distortion = mse(input, encoded.warp);
    const auto mse_result = distortion.sum({1, 2, 3}) / pixels, me_mse = motion_distortion.sum({1, 2, 3}) / pixels;
    torch::Tensor y_for_bit, mv_y_for_bit, z_for_bit, mv_z_for_bit;
    if (is_training()) {
        y_for_bit = add_noise(encoded.residual.y_res); mv_y_for_bit = add_noise(encoded.motion.y_res);
        z_for_bit = add_noise(encoded.z); mv_z_for_bit = add_noise(encoded.mv_z);
    } else { y_for_bit = encoded.residual.y_q; mv_y_for_bit = encoded.motion.y_q; z_for_bit = encoded.z_hat; mv_z_for_bit = encoded.mv_z_hat; }
    const auto bits_y = get_y_laplace_bits(y_for_bit, encoded.residual.scales_hat);
    const auto bits_mv_y = get_y_laplace_bits(mv_y_for_bit, encoded.motion.scales_hat);
    const auto bits_z = get_z_bits(z_for_bit, bit_estimator_z), bits_mv_z = get_z_bits(mv_z_for_bit, bit_estimator_z_mv);
    const auto bpp_y = bits_y.sum({1, 2, 3}) / pixels, bpp_z = bits_z.sum({1, 2, 3}) / pixels;
    const auto bpp_mv_y = bits_mv_y.sum({1, 2, 3}) / pixels, bpp_mv_z = bits_mv_z.sum({1, 2, 3}) / pixels;
    const auto bpp = bpp_y + bpp_z + bpp_mv_y + bpp_mv_z;
    return {bpp_mv_y, bpp_mv_z, bpp_y, bpp_z, bpp, me_mse, mse_result, quality, encoded.dpb,
        bpp.sum() * pixels, bpp_y.sum() * pixels, bpp_z.sum() * pixels, bpp_mv_y.sum() * pixels, bpp_mv_z.sum() * pixels};
}
ForwardResult DMCImpl::forward(const torch::Tensor& input, const Dpb& reference,
                              const torch::Tensor& mv_scale, const torch::Tensor& y_scale) {
    return forward_one_frame(input, reference, mv_scale, y_scale);
}
CompressedFrame DMCImpl::compress(const torch::Tensor& input, const Dpb& reference,
                                 const torch::Tensor& mv_scale, const torch::Tensor& y_scale) {
    const auto encoded = analyze(input, reference, mv_scale, y_scale, true);
    if (!entropy_coder) throw std::runtime_error("HEM video model must be updated before coding");
    entropy_coder->reset_encoder(); bit_estimator_z_mv->encode(encoded.mv_z_hat);
    gaussian_encoder.encode(encoded.motion_stream.y_q_w_0, encoded.motion_stream.scales_w_0);
    gaussian_encoder.encode(encoded.motion_stream.y_q_w_1, encoded.motion_stream.scales_w_1);
    bit_estimator_z->encode(encoded.z_hat);
    gaussian_encoder.encode(encoded.residual_stream.y_q_w_0, encoded.residual_stream.scales_w_0);
    gaussian_encoder.encode(encoded.residual_stream.y_q_w_1, encoded.residual_stream.scales_w_1);
    return {encoded.dpb, entropy_coder->flush_encoder()};
}
Dpb DMCImpl::decompress(const Dpb& reference, const std::string& stream, std::int64_t height, std::int64_t width,
                        const torch::Tensor& mv_scale, const torch::Tensor& y_scale) {
    const auto current_mv_q = get_curr_mv_y_q(mv_scale), current_q = get_curr_y_q(y_scale);
    if (!entropy_coder) throw std::runtime_error("HEM video model must be updated before coding");
    entropy_coder->set_stream(stream); const auto device = parameters().at(0).device();
    const auto size = dcvc2021::downsampled_shape(height, width, 64);
    const auto mv_z_hat = bit_estimator_z_mv->decode_stream(size).to(device);
    const auto mv_hyper = mv_hyper_prior_decoder->forward(mv_z_hat);
    const auto ref_mv = reference.ref_mv_y.defined() ? reference.ref_mv_y : torch::zeros(
        {1, mv_hyper.size(1) / 2, mv_hyper.size(2), mv_hyper.size(3)}, torch::TensorOptions().device(mv_hyper.device()));
    const auto mv_params = mv_y_prior_fusion->forward(torch::cat({mv_hyper, ref_mv}, 1)).chunk(3, 1);
    const auto mv_hat = decompress_dual_prior(mv_params.at(2), mv_params.at(1), mv_params.at(0), mv_y_spatial_prior) * current_mv_q;
    const auto decoded_motion = mv_decoder->forward(mv_hat);
    const auto context = motion_compensation(reference, decoded_motion);
    const auto z_hat = bit_estimator_z->decode_stream(size).to(device);
    const auto hierarchical = contextual_hyper_prior_decoder->forward(z_hat), temporal = temporal_prior_encoder->forward(context.first[2]);
    const auto ref_y = reference.ref_y.defined() ? reference.ref_y : torch::zeros(
        {1, temporal.size(1) / 2, temporal.size(2), temporal.size(3)}, torch::TensorOptions().device(temporal.device()));
    const auto params = y_prior_fusion->forward(torch::cat({temporal, hierarchical, ref_y}, 1)).chunk(3, 1);
    const auto y_hat = decompress_dual_prior(params.at(2), params.at(1), params.at(0), y_spatial_prior) * current_q;
    const auto residual = contextual_decoder(y_hat, context.first);
    const auto reconstruction = recon_generation_net(residual, context.first[0]);
    return {reconstruction.second.clamp(0, 1), reconstruction.first, y_hat, mv_hat};
}
FrameResult DMCImpl::encode_decode(const torch::Tensor& input, const Dpb& reference,
    std::optional<std::filesystem::path> output, std::optional<std::int64_t> width,
    std::optional<std::int64_t> height, const torch::Tensor& mv_scale, const torch::Tensor& y_scale) {
    if (output) {
        const auto mv_q = get_rounded_q(mv_scale), q = get_rounded_q(y_scale);
        const auto encoded = compress(input, reference, detail::scalar(mv_q.scale), detail::scalar(q.scale));
        write_p(*output, {mv_q.index, q.index, encoded.bit_stream});
        if (!std::filesystem::is_regular_file(*output)) throw std::runtime_error("HEM P-frame output is not a regular file");
        const auto bits = static_cast<double>(std::filesystem::file_size(*output)) * 8;
        const auto stream = read_p(*output);
        const auto start = std::chrono::system_clock::now();
        if (!width || !height) throw std::invalid_argument("HEM P-frame decompression requires original dimensions");
        const auto decoded = decompress(reference, stream.bit_stream, *height, *width,
            detail::scalar(stream.mv_y_q_index / 100.0), detail::scalar(stream.y_q_index / 100.0));
        const auto elapsed = std::chrono::duration<double>(std::chrono::system_clock::now() - start).count();
        return {decoded, bits, elapsed, std::nullopt, std::nullopt, std::nullopt, std::nullopt};
    }
    const auto encoded = forward_one_frame(input, reference, mv_scale, y_scale);
    return {encoded.dpb, encoded.bit.item<double>(), 0, encoded.bit_y.item<double>(), encoded.bit_z.item<double>(),
            encoded.bit_mv_y.item<double>(), encoded.bit_mv_z.item<double>()};
}
}  // namespace uf::hem
