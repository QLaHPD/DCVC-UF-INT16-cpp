// Copyright (c) Microsoft Corporation. Licensed under the MIT License.
#include "dcvc_dc_internal.hpp"
#include <torch/cuda.h>
#include <chrono>
#include <stdexcept>

namespace uf::dc {
namespace {
DepthConvBlock block(std::int64_t in, std::int64_t out, bool inplace) {
    return DepthConvBlock(in, out, 3, 1, 0.01, 0.1, inplace);
}
ResBlock residual(std::int64_t channels, bool inplace) { return ResBlock(channels, 0.01, false, false, inplace); }
torch::nn::LeakyReLU leaky(bool inplace) {
    return torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.1).inplace(inplace));
}
std::uint8_t stored_frame_index(std::int64_t value) {
    if (value < 0 || value > 255) throw std::out_of_range("DC file frame index must fit uint8");
    return static_cast<std::uint8_t>(value);
}
void synchronize(torch::Device device) {
    // Python calls torch.cuda.synchronize unconditionally in P file mode.
    if (!device.is_cuda()) throw std::invalid_argument("DC P-frame file timing requires a CUDA input device");
    torch::cuda::synchronize(device.index());
}
}  // namespace
OffsetDiversityImpl::OffsetDiversityImpl(std::int64_t in, std::int64_t auxiliary,
    std::int64_t offsets, std::int64_t groups, double maximum, bool inplace)
    : in_channel(in), offset_num(offsets), group_num(groups), max_residue_magnitude(maximum) {
    conv_offset = register_module("conv_offset", torch::nn::Sequential());
    conv_offset->push_back(conv3x3(auxiliary, 64, 2)); conv_offset->push_back(leaky(inplace));
    conv_offset->push_back(conv3x3(64, 64)); conv_offset->push_back(leaky(inplace));
    conv_offset->push_back(conv3x3(64, 3 * group_num * offset_num));
    fusion = register_module("fusion",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channel * offset_num, in_channel, 1).groups(group_num)));
}
torch::Tensor OffsetDiversityImpl::forward(const torch::Tensor& input, const torch::Tensor& auxiliary, const torch::Tensor& flow) {
    const auto batch = input.size(0), channels = input.size(1), height = input.size(2), width = input.size(3);
    const auto parts = bilinear_up(conv_offset->forward(auxiliary)).chunk(3, 1);
    const auto mask = parts.at(2).sigmoid().view({batch * group_num * offset_num, 1, height, width});
    const auto offsets = (max_residue_magnitude * torch::cat({parts.at(0), parts.at(1)}, 1).tanh()
        + flow.repeat({1, group_num * offset_num, 1, 1})).view({batch * group_num * offset_num, 2, height, width});
    const auto repeated = input.repeat({1, offset_num, 1, 1})
        .view({batch * group_num * offset_num, channels / group_num, height, width});
    const auto warped = flow_warp(repeated, offsets) * mask;
    return fusion(warped.view({batch, channels * offset_num, height, width}));
}
FeatureExtractorImpl::FeatureExtractorImpl(bool inplace) {
    conv1 = register_module("conv1", conv3x3(48, 48)); res_block1 = register_module("res_block1", residual(48, inplace));
    conv2 = register_module("conv2", conv3x3(48, 64, 2)); res_block2 = register_module("res_block2", residual(64, inplace));
    conv3 = register_module("conv3", conv3x3(64, 96, 2)); res_block3 = register_module("res_block3", residual(96, inplace));
}
Contexts FeatureExtractorImpl::forward(const torch::Tensor& input) {
    const auto first = res_block1(conv1(input)), second = res_block2(conv2(first)), third = res_block3(conv3(second));
    return {first, second, third};
}
MultiScaleContextFusionImpl::MultiScaleContextFusionImpl(bool inplace) {
    conv3_up = register_module("conv3_up", subpel_conv3x3(96, 64, 2)); res_block3_up = register_module("res_block3_up", residual(64, inplace));
    conv3_out = register_module("conv3_out", conv3x3(96, 96)); res_block3_out = register_module("res_block3_out", residual(96, inplace));
    conv2_up = register_module("conv2_up", subpel_conv3x3(128, 48, 2)); res_block2_up = register_module("res_block2_up", residual(48, inplace));
    conv2_out = register_module("conv2_out", conv3x3(128, 64)); res_block2_out = register_module("res_block2_out", residual(64, inplace));
    conv1_out = register_module("conv1_out", conv3x3(96, 48)); res_block1_out = register_module("res_block1_out", residual(48, inplace));
}
Contexts MultiScaleContextFusionImpl::forward(const Contexts& context) {
    const auto third_up = res_block3_up(conv3_up->forward(context[2]));
    const auto third_out = res_block3_out(conv3_out(context[2]));
    const auto second_up = res_block2_up(conv2_up->forward(torch::cat({third_up, context[1]}, 1)));
    const auto second_out = res_block2_out(conv2_out(torch::cat({third_up, context[1]}, 1)));
    const auto first_out = res_block1_out(conv1_out(torch::cat({second_up, context[0]}, 1)));
    return {context[0] + first_out, context[1] + second_out, context[2] + third_out};
}
MvEncImpl::MvEncImpl(std::int64_t in, std::int64_t channels, bool inplace) {
    enc_1 = register_module("enc_1", torch::nn::Sequential());
    enc_1->push_back(ResidualBlockWithStride(in, channels, 2, inplace)); enc_1->push_back(block(channels, channels, inplace));
    enc_2 = register_module("enc_2", ResidualBlockWithStride(channels, channels, 2, inplace));
    adaptor_0 = register_module("adaptor_0", block(channels, channels, inplace));
    adaptor_1 = register_module("adaptor_1", block(channels * 2, channels, inplace));
    enc_3 = register_module("enc_3", torch::nn::Sequential());
    enc_3->push_back(ResidualBlockWithStride(channels, channels, 2, inplace)); enc_3->push_back(block(channels, channels, inplace));
    enc_3->push_back(conv3x3(channels, channels, 2));
}
torch::Tensor MvEncImpl::forward(const torch::Tensor& input, const torch::Tensor& context, const torch::Tensor& step) {
    auto result = enc_2(enc_1->forward(input) * step);
    result = context.defined() ? adaptor_1(torch::cat({result, context}, 1)) : adaptor_0(result);
    return enc_3->forward(result);
}
MvDecImpl::MvDecImpl(std::int64_t out, std::int64_t channels, bool inplace) {
    dec_1 = register_module("dec_1", torch::nn::Sequential());
    dec_1->push_back(block(channels, channels, inplace)); dec_1->push_back(ResidualBlockUpsample(channels, channels, 2, inplace));
    dec_1->push_back(block(channels, channels, inplace)); dec_1->push_back(ResidualBlockUpsample(channels, channels, 2, inplace));
    dec_1->push_back(block(channels, channels, inplace));
    dec_2 = register_module("dec_2", ResidualBlockUpsample(channels, channels, 2, inplace));
    dec_3 = register_module("dec_3", torch::nn::Sequential());
    dec_3->push_back(block(channels, channels, inplace)); dec_3->push_back(subpel_conv1x1(channels, out, 2));
}
std::pair<torch::Tensor, torch::Tensor> MvDecImpl::forward(const torch::Tensor& input, const torch::Tensor& step) {
    const auto feature = dec_1->forward(input);
    return {dec_3->forward(dec_2(feature) * step), feature};
}
ContextualEncoderImpl::ContextualEncoderImpl(bool inplace) {
    conv1 = register_module("conv1", conv3x3(51, 64, 2));
    res1 = register_module("res1", ResBlock(128, 0.1, true, true, inplace));
    conv2 = register_module("conv2", conv3x3(128, 96, 2));
    res2 = register_module("res2", ResBlock(192, 0.1, true, true, inplace));
    conv3 = register_module("conv3", conv3x3(192, 96, 2)); conv4 = register_module("conv4", conv3x3(96, 128, 2));
}
torch::Tensor ContextualEncoderImpl::forward(const torch::Tensor& input, const Contexts& context, const torch::Tensor& step) {
    const auto first = conv1(torch::cat({input, context[0]}, 1));
    const auto second = conv2(res1(torch::cat({first, context[1]}, 1)) * step);
    return conv4(conv3(res2(torch::cat({second, context[2]}, 1))));
}
ContextualDecoderImpl::ContextualDecoderImpl(bool inplace) {
    up1 = register_module("up1", subpel_conv3x3(128, 96, 2)); up2 = register_module("up2", subpel_conv3x3(96, 96, 2));
    res1 = register_module("res1", ResBlock(192, 0.1, true, true, inplace));
    up3 = register_module("up3", subpel_conv3x3(192, 64, 2));
    res2 = register_module("res2", ResBlock(128, 0.1, true, true, inplace));
    up4 = register_module("up4", subpel_conv3x3(128, 32, 2));
}
torch::Tensor ContextualDecoderImpl::forward(const torch::Tensor& input, const Contexts& context, const torch::Tensor& step) {
    const auto first = up2->forward(up1->forward(input));
    const auto second = up3->forward(res1(torch::cat({first, context[2]}, 1))) * step;
    return up4->forward(res2(torch::cat({second, context[1]}, 1)));
}
ReconGenerationImpl::ReconGenerationImpl(std::int64_t context, std::int64_t residual, bool inplace) {
    first_conv = register_module("first_conv", conv3x3(context + residual, 48));
    unet_1 = register_module("unet_1", UNet(48, 48, inplace));
    unet_2 = register_module("unet_2", UNet(48, 48, inplace));
    recon_conv = register_module("recon_conv", conv3x3(48, 3));
}
std::pair<torch::Tensor, torch::Tensor> ReconGenerationImpl::forward(const torch::Tensor& first, const torch::Tensor& second) {
    const auto feature = unet_2(unet_1(first_conv(torch::cat({first, second}, 1))));
    return {feature, recon_conv(feature)};
}
DMCImpl::DMCImpl(std::int64_t anchors, bool threaded, int parts, bool inplace)
    : CompressionModelImpl("laplace", 128, 64, threaded, parts) {
    optic_flow = register_module("optic_flow", Spynet());
    align = register_module("align", OffsetDiversity(48, 53, 2, 16, 40, inplace));
    mv_encoder = register_module("mv_encoder", MvEnc(2, 64));  // Source omits inplace.
    const auto motion_hyper = get_hyper_enc_dec_models(64, 64, false, inplace);
    mv_hyper_prior_encoder = register_module("mv_hyper_prior_encoder", motion_hyper.first);
    mv_hyper_prior_decoder = register_module("mv_hyper_prior_decoder", motion_hyper.second);
    mv_y_prior_fusion_adaptor_0 = register_module("mv_y_prior_fusion_adaptor_0", block(64, 128, inplace));
    mv_y_prior_fusion_adaptor_1 = register_module("mv_y_prior_fusion_adaptor_1", block(128, 128, inplace));
    mv_y_prior_fusion = register_module("mv_y_prior_fusion", torch::nn::Sequential());
    mv_y_prior_fusion->push_back(block(128, 192, inplace)); mv_y_prior_fusion->push_back(block(192, 192, inplace));
    mv_y_spatial_prior_adaptor_1 = register_module("mv_y_spatial_prior_adaptor_1", conv1x1(256, 192));
    mv_y_spatial_prior_adaptor_2 = register_module("mv_y_spatial_prior_adaptor_2", conv1x1(256, 192));
    mv_y_spatial_prior_adaptor_3 = register_module("mv_y_spatial_prior_adaptor_3", conv1x1(256, 192));
    mv_y_spatial_prior = register_module("mv_y_spatial_prior", torch::nn::Sequential());
    mv_y_spatial_prior->push_back(block(192, 192, inplace)); mv_y_spatial_prior->push_back(block(192, 192, inplace));
    mv_y_spatial_prior->push_back(block(192, 128, inplace));
    mv_decoder = register_module("mv_decoder", MvDec(2, 64, inplace));
    feature_adaptor_I = register_module("feature_adaptor_I", conv3x3(3, 48));
    feature_adaptor = register_module("feature_adaptor", torch::nn::ModuleList());
    for (int index = 0; index < 3; ++index) feature_adaptor->push_back(conv1x1(48, 48));
    feature_extractor = register_module("feature_extractor", FeatureExtractor(inplace));
    context_fusion_net = register_module("context_fusion_net", MultiScaleContextFusion(inplace));
    contextual_encoder = register_module("contextual_encoder", ContextualEncoder(inplace));
    const auto image_hyper = get_hyper_enc_dec_models(128, 128, true, inplace);
    contextual_hyper_prior_encoder = register_module("contextual_hyper_prior_encoder", image_hyper.first);
    contextual_hyper_prior_decoder = register_module("contextual_hyper_prior_decoder", image_hyper.second);
    temporal_prior_encoder = register_module("temporal_prior_encoder", torch::nn::Sequential());
    temporal_prior_encoder->push_back(conv3x3(96, 96, 2)); temporal_prior_encoder->push_back(leaky(inplace));
    temporal_prior_encoder->push_back(conv3x3(96, 128, 2));
    y_prior_fusion_adaptor_0 = register_module("y_prior_fusion_adaptor_0", block(256, 384, inplace));
    y_prior_fusion_adaptor_1 = register_module("y_prior_fusion_adaptor_1", block(384, 384, inplace));
    y_prior_fusion = register_module("y_prior_fusion", torch::nn::Sequential());
    y_prior_fusion->push_back(block(384, 384, inplace)); y_prior_fusion->push_back(block(384, 384, inplace));
    y_spatial_prior_adaptor_1 = register_module("y_spatial_prior_adaptor_1", conv1x1(512, 384));
    y_spatial_prior_adaptor_2 = register_module("y_spatial_prior_adaptor_2", conv1x1(512, 384));
    y_spatial_prior_adaptor_3 = register_module("y_spatial_prior_adaptor_3", conv1x1(512, 384));
    y_spatial_prior = register_module("y_spatial_prior", torch::nn::Sequential());
    y_spatial_prior->push_back(block(384, 384, inplace)); y_spatial_prior->push_back(block(384, 384, inplace));
    y_spatial_prior->push_back(block(384, 256, inplace));
    contextual_decoder = register_module("contextual_decoder", ContextualDecoder(inplace));
    recon_generation_net = register_module("recon_generation_net", ReconGeneration(48, 32, inplace));
    mv_y_q_basic_enc = register_parameter("mv_y_q_basic_enc", torch::ones({1, 64, 1, 1}));
    mv_y_q_scale_enc = register_parameter("mv_y_q_scale_enc", torch::ones({anchors, 1, 1, 1}));
    mv_y_q_basic_dec = register_parameter("mv_y_q_basic_dec", torch::ones({1, 64, 1, 1}));
    mv_y_q_scale_dec = register_parameter("mv_y_q_scale_dec", torch::ones({anchors, 1, 1, 1}));
    y_q_basic_enc = register_parameter("y_q_basic_enc", torch::ones({1, 128, 1, 1}));
    y_q_scale_enc = register_parameter("y_q_scale_enc", torch::ones({anchors, 1, 1, 1}));
    y_q_basic_dec = register_parameter("y_q_basic_dec", torch::ones({1, 64, 1, 1}));
    y_q_scale_dec = register_parameter("y_q_scale_dec", torch::ones({anchors, 1, 1, 1}));
}
void DMCImpl::load_state_dict(const StateDict& state, bool strict) {
    CompressionModelImpl::load_state_dict(state, strict);
    torch::NoGradGuard no_grad;
    mv_y_q_scale_enc_fine = detail::fine_scales(mv_y_q_scale_enc);
    mv_y_q_scale_dec_fine = detail::fine_scales(mv_y_q_scale_dec);
    y_q_scale_enc_fine = detail::fine_scales(y_q_scale_enc);
    y_q_scale_dec_fine = detail::fine_scales(y_q_scale_dec);
}
Contexts DMCImpl::multi_scale_feature_extractor(const Dpb& reference, std::int64_t index) {
    torch::Tensor feature;
    if (!reference.ref_feature.defined()) feature = feature_adaptor_I(reference.ref_frame);
    else {
        constexpr std::array<int, 4> mapping{0, 1, 0, 2};
        auto position = index % 4; if (position < 0) position += 4;
        feature = feature_adaptor->at<torch::nn::Conv2dImpl>(mapping[position]).forward(reference.ref_feature);
    }
    return feature_extractor(feature);
}
std::pair<Contexts, torch::Tensor> DMCImpl::motion_compensation(const Dpb& reference, const torch::Tensor& motion, std::int64_t index) {
    const auto warped = flow_warp(reference.ref_frame, motion);
    const auto second_flow = bilinear_down(motion) / 2, third_flow = bilinear_down(second_flow) / 2;
    const auto features = multi_scale_feature_extractor(reference, index);
    const auto initial = flow_warp(features[0], motion);
    const auto first = align(features[0], torch::cat({initial, warped, motion}, 1), motion);
    return {context_fusion_net(Contexts{first, flow_warp(features[1], second_flow), flow_warp(features[2], third_flow)}), warped};
}
std::array<torch::Tensor, 4> DMCImpl::get_q_scales_from_ckpt(const std::filesystem::path& path) {
    const auto state = get_state_dict(path);
    return {state.at("y_q_scale_enc").reshape({-1}), state.at("y_q_scale_dec").reshape({-1}),
            state.at("mv_y_q_scale_enc").reshape({-1}), state.at("mv_y_q_scale_dec").reshape({-1})};
}
torch::Tensor DMCImpl::mv_prior_param_decoder(const torch::Tensor& z, const Dpb& reference, const SliceShape& slice) {
    auto params = slice_to_y(mv_hyper_prior_decoder->forward(z), slice);
    params = reference.ref_mv_y.defined() ? mv_y_prior_fusion_adaptor_1(torch::cat({params, reference.ref_mv_y}, 1))
                                        : mv_y_prior_fusion_adaptor_0(params);
    return mv_y_prior_fusion->forward(params);
}
torch::Tensor DMCImpl::res_prior_param_decoder(const torch::Tensor& z, const Dpb& reference,
                                              const torch::Tensor& context, const SliceShape& slice) {
    const auto hierarchical = slice_to_y(contextual_hyper_prior_decoder->forward(z), slice);
    const auto temporal = temporal_prior_encoder->forward(context);
    const auto params = reference.ref_y.defined() ? y_prior_fusion_adaptor_1(torch::cat({temporal, hierarchical, reference.ref_y}, 1))
                                                : y_prior_fusion_adaptor_0(torch::cat({temporal, hierarchical}, 1));
    return y_prior_fusion->forward(params);
}
std::pair<torch::Tensor, torch::Tensor> DMCImpl::get_recon_and_feature(const torch::Tensor& y,
    const Contexts& context, const torch::Tensor& step) {
    const auto decoded = contextual_decoder(y, context, step);
    const auto result = recon_generation_net(decoded, context[0]);
    return {result.second.clamp_(0, 1), result.first};
}
torch::Tensor DMCImpl::motion_estimation_and_mv_encoding(const torch::Tensor& input, const Dpb& reference, const torch::Tensor& step) {
    return mv_encoder(optic_flow(input, reference.ref_frame), reference.ref_mv_feature, step);
}
std::array<torch::Tensor, 4> DMCImpl::get_q_for_inference(bool in_checkpoint, std::optional<std::int64_t> index) {
    return {get_curr_q(in_checkpoint ? mv_y_q_scale_enc : mv_y_q_scale_enc_fine, mv_y_q_basic_enc, index),
            get_curr_q(in_checkpoint ? mv_y_q_scale_dec : mv_y_q_scale_dec_fine, mv_y_q_basic_dec, index),
            get_curr_q(in_checkpoint ? y_q_scale_enc : y_q_scale_enc_fine, y_q_basic_enc, index),
            get_curr_q(in_checkpoint ? y_q_scale_dec : y_q_scale_dec_fine, y_q_basic_dec, index)};
}
PriorAdaptors DMCImpl::motion_adaptors() const {
    return {mv_y_spatial_prior_adaptor_1, mv_y_spatial_prior_adaptor_2, mv_y_spatial_prior_adaptor_3};
}
PriorAdaptors DMCImpl::image_adaptors() const {
    return {y_spatial_prior_adaptor_1, y_spatial_prior_adaptor_2, y_spatial_prior_adaptor_3};
}
struct DMCImpl::Analysis {
    torch::Tensor mv_z_hat, z_hat, mv_feature, mv_y_hat, y_hat;
    Contexts contexts;
    PriorResult motion, image;
    PriorStream motion_stream, image_stream;
};
DMCImpl::Analysis DMCImpl::analyze(const torch::Tensor& input, const Dpb& reference,
    const std::array<torch::Tensor, 4>& q, std::int64_t frame_index, bool stream) {
    Analysis result;
    const auto mv_y = motion_estimation_and_mv_encoding(input, reference, q[0]);
    const auto motion_pad = pad_for_y(mv_y);
    result.mv_z_hat = quant(mv_hyper_prior_encoder->forward(motion_pad.first));
    const auto motion_params = mv_prior_param_decoder(result.mv_z_hat, reference, motion_pad.second);
    if (stream) {
        result.motion_stream = compress_four_part_prior(mv_y, motion_params, motion_adaptors(), mv_y_spatial_prior);
        result.mv_y_hat = result.motion_stream.y_hat;
    } else {
        result.motion = forward_four_part_prior(mv_y, motion_params, motion_adaptors(), mv_y_spatial_prior);
        result.mv_y_hat = result.motion.y_hat;
    }
    const auto decoded_motion = mv_decoder(result.mv_y_hat, q[1]);
    result.mv_feature = decoded_motion.second;
    result.contexts = motion_compensation(reference, decoded_motion.first, frame_index).first;
    const auto y = contextual_encoder(input, result.contexts, q[2]);
    const auto image_pad = pad_for_y(y);
    result.z_hat = quant(contextual_hyper_prior_encoder->forward(image_pad.first));
    const auto image_params = res_prior_param_decoder(result.z_hat, reference, result.contexts[2], image_pad.second);
    if (stream) {
        result.image_stream = compress_four_part_prior(y, image_params, image_adaptors(), y_spatial_prior);
        result.y_hat = result.image_stream.y_hat;
    } else {
        result.image = forward_four_part_prior(y, image_params, image_adaptors(), y_spatial_prior);
        result.y_hat = result.image.y_hat;
    }
    return result;
}
CompressedFrame DMCImpl::compress(const torch::Tensor& input, const Dpb& reference,
                                  bool in_checkpoint, std::int64_t index, std::int64_t frame_index) {
    const auto q = get_q_for_inference(in_checkpoint, index);
    const auto value = analyze(input, reference, q, frame_index, true);
    if (!entropy_coder) throw std::runtime_error("DC video model must be updated before coding");
    entropy_coder->reset(); bit_estimator_z_mv->encode(value.mv_z_hat); bit_estimator_z->encode(value.z_hat);
    for (int stage = 0; stage < 4; ++stage) gaussian_encoder.encode(value.motion_stream.y_q[stage], value.motion_stream.scales[stage]);
    for (int stage = 0; stage < 4; ++stage) gaussian_encoder.encode(value.image_stream.y_q[stage], value.image_stream.scales[stage]);
    entropy_coder->flush();
    const auto reconstruction = get_recon_and_feature(value.y_hat, value.contexts, q[3]);
    const Dpb next{reconstruction.first, reconstruction.second, value.mv_feature, value.y_hat, value.mv_y_hat};
    return {next, entropy_coder->get_encoded_stream()};
}
Dpb DMCImpl::decompress(const Dpb& reference, const std::string& stream, std::int64_t height, std::int64_t width,
                        bool in_checkpoint, std::int64_t index, std::int64_t frame_index) {
    const auto q = get_q_for_inference(in_checkpoint, index);
    if (!entropy_coder) throw std::runtime_error("DC video model must be updated before coding");
    entropy_coder->set_stream(stream);
    const auto parameter = parameters().at(0);
    const auto z_size = dcvc2021::downsampled_shape(height, width, 64);
    const auto y_size = dcvc2021::downsampled_shape(height, width, 16);
    const auto slice = get_to_y_slice_shape(y_size.first, y_size.second);
    const auto mv_z_hat = bit_estimator_z_mv->decode_stream(z_size, parameter.scalar_type(), parameter.device());
    const auto z_hat = bit_estimator_z->decode_stream(z_size, parameter.scalar_type(), parameter.device());
    const auto mv_params = mv_prior_param_decoder(mv_z_hat, reference, slice);
    const auto mv_y_hat = decompress_four_part_prior(mv_params, motion_adaptors(), mv_y_spatial_prior);
    const auto motion = mv_decoder(mv_y_hat, q[1]);
    const auto contexts = motion_compensation(reference, motion.first, frame_index).first;
    const auto params = res_prior_param_decoder(z_hat, reference, contexts[2], slice);
    const auto y_hat = decompress_four_part_prior(params, image_adaptors(), y_spatial_prior);
    const auto reconstruction = get_recon_and_feature(y_hat, contexts, q[3]);
    return {reconstruction.first, reconstruction.second, motion.second, y_hat, mv_y_hat};
}
FrameResult DMCImpl::encode_decode(const torch::Tensor& input, const Dpb& reference, bool in_checkpoint, std::int64_t index,
    std::optional<std::filesystem::path> output, std::optional<std::int64_t> width,
    std::optional<std::int64_t> height, std::int64_t frame_index) {
    if (!output) {
        const auto result = forward_one_frame(input, reference, in_checkpoint, index, frame_index);
        return {result.dpb, result.bit.item<double>(), 0, 0};
    }
    synchronize(input.device());
    const auto start = std::chrono::system_clock::now();
    const auto encoded = compress(input, reference, in_checkpoint, index, frame_index);
    write_p(*output, {in_checkpoint, detail::quality_index(index), stored_frame_index(frame_index), encoded.bit_stream});
    if (!std::filesystem::is_regular_file(*output)) throw std::runtime_error("DC P-frame output is not a regular file");
    const auto bits = static_cast<double>(std::filesystem::file_size(*output)) * 8;
    synchronize(input.device());
    const auto encoded_end = std::chrono::system_clock::now();
    const auto saved = read_p(*output);
    if (!height || !width) throw std::invalid_argument("DC P-frame file decoding requires original dimensions");
    const auto decoded = decompress(reference, saved.bit_stream, *height, *width, saved.q_in_ckpt, saved.q_index, saved.frame_idx);
    synchronize(input.device());
    const auto decoded_end = std::chrono::system_clock::now();
    return {decoded, bits, std::chrono::duration<double>(encoded_end - start).count(),
            std::chrono::duration<double>(decoded_end - encoded_end).count()};
}
ForwardResult DMCImpl::forward_one_frame(const torch::Tensor& input, const Dpb& reference,
    bool in_checkpoint, std::optional<std::int64_t> index, std::int64_t frame_index) {
    const auto q = get_q_for_inference(in_checkpoint, index);
    const auto value = analyze(input, reference, q, frame_index, false);
    const auto reconstruction = get_recon_and_feature(value.y_hat, value.contexts, q[3]);
    const auto pixels = input.size(2) * input.size(3);
    const auto bits_y = get_y_laplace_bits(value.image.y_q, value.image.scales_hat);
    const auto bits_mv_y = get_y_laplace_bits(value.motion.y_q, value.motion.scales_hat);
    const auto bits_z = get_z_bits(value.z_hat, bit_estimator_z);
    const auto bits_mv_z = get_z_bits(value.mv_z_hat, bit_estimator_z_mv);
    const auto bpp_y = bits_y.sum({1, 2, 3}) / pixels, bpp_z = bits_z.sum({1, 2, 3}) / pixels;
    const auto bpp_mv_y = bits_mv_y.sum({1, 2, 3}) / pixels, bpp_mv_z = bits_mv_z.sum({1, 2, 3}) / pixels;
    const auto bpp = ((bpp_y + bpp_z) + bpp_mv_y) + bpp_mv_z;
    const Dpb next{reconstruction.first, reconstruction.second, value.mv_feature, value.y_hat, value.mv_y_hat};
    return {bpp_mv_y, bpp_mv_z, bpp_y, bpp_z, bpp, next, bpp.sum() * pixels,
            bpp_y.sum() * pixels, bpp_z.sum() * pixels, bpp_mv_y.sum() * pixels, bpp_mv_z.sum() * pixels};
}
}  // namespace uf::dc
