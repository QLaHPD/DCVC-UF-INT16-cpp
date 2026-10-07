// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#include "uf/training_image_model.hpp"
#include "uf/process.hpp"

#include <array>
#include <stdexcept>
#include <utility>

namespace uf::training {
namespace {
constexpr std::int64_t source_channels = 3 * 8 * 8;
constexpr std::int64_t encoder_channels = 384;
constexpr std::int64_t latent_channels = 256;
constexpr std::int64_t hyper_channels = 128;
}  // namespace

IntraEncoderImpl::IntraEncoderImpl() {
    enc_1 = register_module("enc_1", DepthConvBlock(source_channels, encoder_channels));
    enc_2 = register_module("enc_2", torch::nn::Sequential());
    for (int index = 0; index < 6; ++index) {
        enc_2->push_back(DepthConvBlock(encoder_channels, encoder_channels));
    }
    enc_2->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(encoder_channels, latent_channels, 3)
                                     .stride(2).padding(1)));
}

torch::Tensor IntraEncoderImpl::forward(const torch::Tensor& input, const torch::Tensor& step) {
    auto encoded = enc_1->forward(torch::pixel_unshuffle(input, 8));
    return enc_2->forward(encoded * step);
}

IntraDecoderImpl::IntraDecoderImpl() {
    dec_1 = register_module("dec_1", torch::nn::Sequential());
    dec_1->push_back(ResidualBlockUpsample(latent_channels, encoder_channels));
    for (int index = 0; index < 12; ++index) {
        dec_1->push_back(DepthConvBlock(encoder_channels, encoder_channels));
    }
    dec_2 = register_module("dec_2", DepthConvBlock(encoder_channels, source_channels));
}

torch::Tensor IntraDecoderImpl::forward(const torch::Tensor& input, const torch::Tensor& step) {
    auto decoded = dec_1->forward(input) * step;
    return torch::pixel_shuffle(dec_2->forward(decoded), 8);
}

IntraHyperEncoderImpl::IntraHyperEncoderImpl() {
    conv = register_module("conv", torch::nn::Sequential());
    conv->push_back(DepthConvBlock(latent_channels, hyper_channels));
    conv->push_back(ResidualBlockWithStride2(hyper_channels, hyper_channels));
    conv->push_back(ResidualBlockWithStride2(hyper_channels, hyper_channels));
}

torch::Tensor IntraHyperEncoderImpl::forward(const torch::Tensor& input) { return conv->forward(input); }

IntraHyperDecoderImpl::IntraHyperDecoderImpl() {
    conv = register_module("conv", torch::nn::Sequential());
    conv->push_back(ResidualBlockUpsample(hyper_channels, hyper_channels));
    conv->push_back(ResidualBlockUpsample(hyper_channels, hyper_channels));
    conv->push_back(DepthConvBlock(hyper_channels, latent_channels));
}

torch::Tensor IntraHyperDecoderImpl::forward(const torch::Tensor& input) { return conv->forward(input); }

IntraSpatialPriorImpl::IntraSpatialPriorImpl() {
    conv = register_module("conv", torch::nn::Sequential());
    for (int index = 0; index < 3; ++index) {
        conv->push_back(DepthConvBlock(latent_channels * 2, latent_channels * 2));
    }
    conv->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(latent_channels * 2, latent_channels * 2, 1)));
}

torch::Tensor IntraSpatialPriorImpl::forward(const torch::Tensor& input) { return conv->forward(input); }

IntraYPriorFusionImpl::IntraYPriorFusionImpl() {
    conv = register_module("conv", torch::nn::Sequential());
    conv->push_back(DepthConvBlock(latent_channels, latent_channels * 2));
    conv->push_back(DepthConvBlock(latent_channels * 2, latent_channels * 2));
    conv->push_back(DepthConvBlock(latent_channels * 2, latent_channels * 2));
    conv->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(latent_channels * 2, latent_channels * 2, 1)));
}

torch::Tensor IntraYPriorFusionImpl::forward(const torch::Tensor& input) { return conv->forward(input); }

DmciTrainingImpl::DmciTrainingImpl() : CompressionTrainingImpl(hyper_channels) {
    enc = register_module("enc", IntraEncoder());
    hyper_enc = register_module("hyper_enc", IntraHyperEncoder());
    hyper_dec = register_module("hyper_dec", IntraHyperDecoder());
    y_prior_fusion = register_module("y_prior_fusion", IntraYPriorFusion());
    y_spatial_prior_reduction = register_module("y_spatial_prior_reduction",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(latent_channels * 2, latent_channels, 1)));
    y_spatial_prior_adaptor_1 = register_module("y_spatial_prior_adaptor_1",
        DepthConvBlock(latent_channels * 2, latent_channels * 2, false, false, true));
    y_spatial_prior_adaptor_2 = register_module("y_spatial_prior_adaptor_2",
        DepthConvBlock(latent_channels * 2, latent_channels * 2, false, false, true));
    y_spatial_prior_adaptor_3 = register_module("y_spatial_prior_adaptor_3",
        DepthConvBlock(latent_channels * 2, latent_channels * 2, false, false, true));
    y_spatial_prior = register_module("y_spatial_prior", IntraSpatialPrior());
    dec = register_module("dec", IntraDecoder());

    // Python's q_* torch.ones factories follow the default floating dtype.
    const auto options = torch::TensorOptions();
    q_scale_enc = register_parameter("q_scale_enc", torch::ones({qp_count, encoder_channels}, options));
    q_scale_dec = register_parameter("q_scale_dec", torch::ones({qp_count, encoder_channels}, options));
    q_scale_y_enc = register_parameter("q_scale_y_enc", torch::ones({qp_count, latent_channels}, options));
    q_scale_y_dec = register_parameter("q_scale_y_dec", torch::ones({qp_count, latent_channels}, options));
    initialize_weights();
}

RateDistortion DmciTrainingImpl::forward_one_frame(const torch::Tensor& input,
    const torch::Tensor& qp, bool reconstruction_only) {
    ProcessInterrupt::check();
    // Python selects QPs before applying the tensor graph. A singleton QP can
    // broadcast across a batch; index/shape validation belongs to its operators.
    const auto q_enc = select_q(q_scale_enc, qp), q_dec = select_q(q_scale_dec, qp);
    const auto q_y_enc = select_q(q_scale_y_enc, qp), q_y_dec = select_q(q_scale_y_dec, qp);
    const auto y = enc->forward(input, q_enc);
    ProcessInterrupt::check();
    const auto z = hyper_enc->forward(y);
    const auto z_hat = quantize(z);
    auto parameters = y_prior_fusion->forward(hyper_dec->forward(z_hat));
    parameters = parameters.slice(2, 0, y.size(2)).slice(3, 0, y.size(3));
    const std::array<Combine, 3> adaptors{
        [this](const torch::Tensor& reconstructed, const torch::Tensor& common) {
            return y_spatial_prior_adaptor_1->forward(torch::cat({reconstructed, common}, 1));
        },
        [this](const torch::Tensor& reconstructed, const torch::Tensor& common) {
            return y_spatial_prior_adaptor_2->forward(torch::cat({reconstructed, common}, 1));
        },
        [this](const torch::Tensor& reconstructed, const torch::Tensor& common) {
            return y_spatial_prior_adaptor_3->forward(torch::cat({reconstructed, common}, 1));
        }
    };
    const auto prior = prior_four(y, q_y_enc, q_y_dec, parameters,
        [this](const torch::Tensor& value) { return y_spatial_prior_reduction->forward(value); },
        adaptors, [this](const torch::Tensor& value) { return y_spatial_prior->forward(value); });
    RateDistortion result;
    result.reconstruction = dec->forward(prior.reconstruction, q_dec);
    if (reconstruction_only) return result;
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

TrainingForward DmciTrainingImpl::forward(const torch::Tensor& input, const torch::Tensor& qp,
                                         const torch::Tensor& lambdas, bool get_loss_info) {
    TrainingForward output;
    output.result = forward_one_frame(input, qp);
    output.loss = rate_distortion_loss(output.result, lambdas);
    if (get_loss_info) output.info = loss_info(output.result, output.loss, input.size(2) * input.size(3));
    return output;
}

torch::Tensor DmciTrainingImpl::reconstruct(const torch::Tensor& input, const torch::Tensor& qp) {
    return forward_one_frame(input, qp, true).reconstruction;
}

}  // namespace uf::training
