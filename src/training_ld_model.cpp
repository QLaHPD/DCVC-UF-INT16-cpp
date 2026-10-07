// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#include "uf/training_ld_model.hpp"
#include "uf/process.hpp"

#include <array>
#include <stdexcept>
#include <utility>

namespace uf::training {
namespace {
constexpr std::int64_t source_channels = 3 * 8 * 8;
constexpr std::int64_t latent_channels = 128;
constexpr std::int64_t hyper_channels = 128;
constexpr std::int64_t decoder_channels = 256;
constexpr std::int64_t memory_channels = 256;
constexpr std::array<std::int64_t, 8> adaptor_map{0, 1, 0, 2, 0, 2, 0, 2};
constexpr std::array<double, 3> distortion_weights{0.52, 1.33, 0.83};
}  // namespace

LdDepthStackImpl::LdDepthStackImpl(std::int64_t in, std::int64_t out,
                                 std::int64_t count, std::int64_t final_channels) {
    if (count <= 0) throw std::invalid_argument("LD convolution stack must contain a depth block");
    conv = register_module("conv", torch::nn::Sequential());
    conv->push_back(DepthConvBlock(in, out, true));
    for (std::int64_t index = 1; index < count; ++index) {
        conv->push_back(DepthConvBlock(out, out, true));
    }
    if (final_channels > 0) {
        conv->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(out, final_channels, 1)));
    }
}

torch::Tensor LdDepthStackImpl::forward(const torch::Tensor& input) {
    return run_tensor([layers = conv](const CheckpointInputs& args) mutable {
        return layers->forward(args.at(0));
    }, {input});
}

torch::Tensor LdDepthStackImpl::forward(const torch::Tensor& input, const torch::Tensor& context) {
    return run_tensor([layers = conv](const CheckpointInputs& args) mutable {
        return layers->forward(torch::cat({args.at(0), args.at(1)}, 1));
    }, {input, context});
}

torch::Tensor LdDepthStackImpl::forward(const torch::Tensor& hyper, const torch::Tensor& temporal,
                                       const torch::Tensor& step) {
    return run_tensor([layers = conv](const CheckpointInputs& args) mutable {
        return layers->forward(torch::cat({args.at(0), args.at(1) * args.at(2)}, 1));
    }, {hyper, temporal, step});
}

LdEncoderImpl::LdEncoderImpl() {
    conv1 = register_module("conv1", torch::nn::Sequential());
    conv1->push_back(DepthConvBlock(source_channels + memory_channels, decoder_channels, true));
    conv1->push_back(DepthConvBlock(decoder_channels, decoder_channels, true));
    conv2 = register_module("conv2", DepthConvBlock(decoder_channels, decoder_channels, true));
    down = register_module("down",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(decoder_channels, latent_channels, 3)
                         .stride(2).padding(1)));
}

torch::Tensor LdEncoderImpl::forward(const torch::Tensor& input, const torch::Tensor& context,
                                   const torch::Tensor& step) {
    return run_tensor([first = conv1, second = conv2, reduction = down](const CheckpointInputs& args) mutable {
        auto feature = torch::pixel_unshuffle(args.at(0), 8);
        feature = first->forward(torch::cat({feature, args.at(1)}, 1));
        feature = second->forward(feature);
        return reduction->forward(feature * args.at(2));
    }, {input, context, step});
}

LdDecoderImpl::LdDecoderImpl() {
    up = register_module("up", SubpelConv2x(latent_channels, decoder_channels, 1));
    conv1 = register_module("conv1", torch::nn::Sequential());
    conv1->push_back(DepthConvBlock(decoder_channels + memory_channels, decoder_channels, true));
    conv1->push_back(DepthConvBlock(decoder_channels, decoder_channels, true));
    conv1->push_back(DepthConvBlock(decoder_channels, decoder_channels, true));
    conv2 = register_module("conv2",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(decoder_channels, decoder_channels, 1)));
}

torch::Tensor LdDecoderImpl::forward(const torch::Tensor& input, const torch::Tensor& context,
                                   const torch::Tensor& step) {
    return run_tensor([upsample = up, first = conv1, second = conv2](const CheckpointInputs& args) mutable {
        const auto feature = upsample->forward(args.at(0));
        const auto decoded = first->forward(torch::cat({feature, args.at(1)}, 1));
        return second->forward(decoded) * args.at(2);
    }, {input, context, step});
}

LdHyperEncoderImpl::LdHyperEncoderImpl() {
    conv = register_module("conv", torch::nn::Sequential());
    conv->push_back(DepthConvBlock(latent_channels, hyper_channels, true));
    conv->push_back(ResidualBlockWithStride2(hyper_channels, hyper_channels, true, false));
    conv->push_back(ResidualBlockWithStride2(hyper_channels, hyper_channels, true, false));
}

torch::Tensor LdHyperEncoderImpl::forward(const torch::Tensor& input) {
    return run_tensor([layers = conv](const CheckpointInputs& args) mutable {
        return layers->forward(args.at(0));
    }, {input});
}

LdHyperDecoderImpl::LdHyperDecoderImpl() {
    conv = register_module("conv", torch::nn::Sequential());
    conv->push_back(ResidualBlockUpsample(hyper_channels, hyper_channels, true, false));
    conv->push_back(ResidualBlockUpsample(hyper_channels, hyper_channels, true, false));
    conv->push_back(DepthConvBlock(hyper_channels, latent_channels, true));
}

torch::Tensor LdHyperDecoderImpl::forward(const torch::Tensor& input) {
    return run_tensor([layers = conv](const CheckpointInputs& args) mutable {
        return layers->forward(args.at(0));
    }, {input});
}

LdTemporalPriorImpl::LdTemporalPriorImpl() {
    conv = register_module("conv",
        ResidualBlockWithStride2(memory_channels, latent_channels * 2, true, false));
}

torch::Tensor LdTemporalPriorImpl::forward(const torch::Tensor& input) {
    return run_tensor([layer = conv](const CheckpointInputs& args) mutable {
        return layer->forward(args.at(0));
    }, {input});
}

LdReconHeadImpl::LdReconHeadImpl() {
    conv = register_module("conv", torch::nn::Sequential());
    for (int index = 0; index < 3; ++index) {
        conv->push_back(DepthConvBlock(decoder_channels, decoder_channels, true));
    }
    head = register_module("head",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(decoder_channels, source_channels, 1)));
}

torch::Tensor LdReconHeadImpl::forward(const torch::Tensor& input, bool for_reset) {
    return run_tensor([layers = conv, output = head, for_reset](const CheckpointInputs& args) mutable {
        const auto out = output->forward(layers->forward(args.at(0)));
        return for_reset ? out : torch::pixel_shuffle(out, 8);
    }, {input});
}

DmcLdTrainingImpl::DmcLdTrainingImpl() : CompressionTrainingImpl(hyper_channels) {
    feature_adaptor_i = register_module("feature_adaptor_i",
        LdDepthStack(source_channels, memory_channels, 4));
    feature_adaptor_m = register_module("feature_adaptor_m",
        LdDepthStack(memory_channels + decoder_channels, memory_channels, 4));
    feature_extractor = register_module("feature_extractor",
        LdDepthStack(memory_channels, memory_channels, 5));
    encoder = register_module("encoder", LdEncoder());
    hyper_encoder = register_module("hyper_encoder", LdHyperEncoder());
    hyper_decoder = register_module("hyper_decoder", LdHyperDecoder());
    temporal_prior_encoder = register_module("temporal_prior_encoder", LdTemporalPrior());
    y_prior_fusion = register_module("y_prior_fusion",
        LdDepthStack(latent_channels * 3, latent_channels * 3, 3, latent_channels * 3));
    y_spatial_prior = register_module("y_spatial_prior",
        LdDepthStack(latent_channels * 4, latent_channels * 2, 2, latent_channels));
    decoder = register_module("decoder", LdDecoder());
    recon_head = register_module("recon_head", LdReconHead());
    // Python's q_* torch.ones factories follow the default floating dtype.
    const auto options = torch::TensorOptions();
    q_encoder = register_parameter("q_encoder", torch::ones({qp_count, decoder_channels}, options));
    q_decoder = register_parameter("q_decoder", torch::ones({qp_count, decoder_channels}, options));
    q_feature = register_parameter("q_feature", torch::ones({qp_count, latent_channels * 2}, options));
    initialize_weights();
}

void DmcLdTrainingImpl::set_use_checkpointing(bool enabled) {
    feature_adaptor_i->set_use_checkpointing(enabled);
    feature_adaptor_m->set_use_checkpointing(enabled);
    feature_extractor->set_use_checkpointing(enabled);
    temporal_prior_encoder->set_use_checkpointing(enabled);
    encoder->set_use_checkpointing(enabled);
    decoder->set_use_checkpointing(enabled);
    recon_head->set_use_checkpointing(enabled);
    hyper_encoder->set_use_checkpointing(enabled);
    hyper_decoder->set_use_checkpointing(enabled);
    y_prior_fusion->set_use_checkpointing(enabled);
    y_spatial_prior->set_use_checkpointing(enabled);
}

void DmcLdTrainingImpl::clear_dpb() {
    ref_feature = torch::Tensor{};
    memory = torch::Tensor{};
    ctx = torch::Tensor{};
}

void DmcLdTrainingImpl::add_ref_feature_from_frame(const torch::Tensor& frame) {
    ProcessInterrupt::check();
    ref_feature = torch::pixel_unshuffle(frame, 8);
}

void DmcLdTrainingImpl::apply_feature_adaptor() {
    ProcessInterrupt::check();
    if (!ref_feature.defined()) throw std::runtime_error("LD training requires an initial reference frame");
    if (!memory.defined()) memory = feature_adaptor_i->forward(ref_feature);
    else memory = feature_adaptor_m->forward(memory, ref_feature);
    ctx = feature_extractor->forward(memory);
}

void DmcLdTrainingImpl::set_ref_feature(const torch::Tensor& feature, bool reset_feature_memory) {
    ref_feature = feature;
    if (reset_feature_memory) {
        const auto reconstructed = recon_head->forward(feature, true);
        clear_dpb();
        ref_feature = reconstructed;
    }
}

RateDistortion DmcLdTrainingImpl::forward_one_frame(const torch::Tensor& input,
    const torch::Tensor& qp, bool reset_feature_memory) {
    ProcessInterrupt::check();
    // Retain Python's QP selection and tensor broadcasting, including a shared
    // QP for the batch. Tensor operators own index/rank/shape validation.
    const auto q_enc = select_q(q_encoder, qp);
    const auto q_dec = select_q(q_decoder, qp);
    const auto q_ctx = select_q(q_feature, qp);
    apply_feature_adaptor();
    const auto y = encoder->forward(input, ctx, q_enc);
    ProcessInterrupt::check();
    const auto z = hyper_encoder->forward(y);
    const auto z_hat = quantize(z);
    const auto temporal_parameters = temporal_prior_encoder->forward(memory);
    const auto hyper_parameters = hyper_decoder->forward(z_hat);
    const auto parameters = y_prior_fusion->forward(hyper_parameters, temporal_parameters, q_ctx);
    const auto prior = prior_two(y, parameters,
        [this](const torch::Tensor& reconstructed, const torch::Tensor& common) {
            return y_spatial_prior->forward(reconstructed, common);
        });
    const auto feature = decoder->forward(prior.reconstruction, ctx, q_dec);
    RateDistortion result;
    result.reconstruction = recon_head->forward(feature);
    set_ref_feature(feature, reset_feature_memory);
    const auto y_for_bits = add_noise(prior.residual);
    const auto z_for_bits = add_noise(z);
    const auto bit_y = y_bits(y_for_bits, prior.scales);
    const auto bit_z = z_bits(z_for_bits, qp);
    result.mse = mse(input, result.reconstruction);
    result.bits_y = bit_y.sum({1, 2, 3});
    result.bits_z = bit_z.sum({1, 2, 3});
    result.bpp = (result.bits_y + result.bits_z) / (input.size(2) * input.size(3));
    return result;
}

RateDistortion DmcLdTrainingImpl::rd_info(RateDistortion result, std::int64_t adaptor_index) {
    result.mse = result.mse * distortion_weights.at(static_cast<std::size_t>(adaptor_index));
    return result;
}

TrainingForward DmcLdTrainingImpl::forward(const torch::Tensor& input, const torch::Tensor& qp,
    const torch::Tensor& lambdas, bool get_loss_info, std::int64_t current_poc) {
    const auto poc = (current_poc % 8 + 8) % 8;
    TrainingForward output;
    output.result = rd_info(forward_one_frame(input, qp), adaptor_map[static_cast<std::size_t>(poc)]);
    output.loss = rate_distortion_loss(output.result, lambdas);
    ref_feature = ref_feature.detach();
    memory = memory.detach();
    if (get_loss_info) output.info = loss_info(output.result, output.loss, input.size(2) * input.size(3));
    return output;
}

TrainingForward DmcLdTrainingImpl::forward(const std::vector<torch::Tensor>& frames,
    const torch::Tensor& qp, const torch::Tensor& lambdas, bool get_loss_info, std::int64_t) {
    TrainingForward output;
    std::vector<torch::Tensor> losses;
    losses.reserve(frames.size());
    for (std::size_t index = 0; index < frames.size(); ++index) {
        ProcessInterrupt::check();
        const auto adaptor_index = adaptor_map[(index % 8 + 1) % 8];
        output.result = rd_info(forward_one_frame(frames[index], qp), adaptor_index);
        output.loss = rate_distortion_loss(output.result, lambdas);
        losses.push_back(output.loss.mean);
    }
    if (get_loss_info) {
        output.info = loss_info(output.result, output.loss, frames.at(0).size(2) * frames.at(0).size(3));
    }
    output.loss.mean = torch::stack(losses).mean();
    return output;
}

}  // namespace uf::training
