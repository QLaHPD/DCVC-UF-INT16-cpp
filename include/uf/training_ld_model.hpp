// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#pragma once

#include "uf/training_common.hpp"
#include "uf/training_recompute.hpp"

namespace uf::training {

// These stacks retain the Python "conv.<index>" registration paths.
struct LdDepthStackImpl : ActivationCheckpointModule {
    LdDepthStackImpl(std::int64_t in_channels, std::int64_t out_channels,
                    std::int64_t count, std::int64_t final_channels = 0);
    torch::Tensor forward(const torch::Tensor& input);
    torch::Tensor forward(const torch::Tensor& input, const torch::Tensor& context);
    torch::Tensor forward(const torch::Tensor& hyper, const torch::Tensor& temporal,
                          const torch::Tensor& quant_step);
    torch::nn::Sequential conv{nullptr};
};
TORCH_MODULE(LdDepthStack);

struct LdEncoderImpl : ActivationCheckpointModule {
    LdEncoderImpl();
    torch::Tensor forward(const torch::Tensor& input, const torch::Tensor& context,
                          const torch::Tensor& quant_step);
    torch::nn::Sequential conv1{nullptr};
    DepthConvBlock conv2{nullptr};
    torch::nn::Conv2d down{nullptr};
};
TORCH_MODULE(LdEncoder);

struct LdDecoderImpl : ActivationCheckpointModule {
    LdDecoderImpl();
    torch::Tensor forward(const torch::Tensor& input, const torch::Tensor& context,
                          const torch::Tensor& quant_step);
    SubpelConv2x up{nullptr};
    torch::nn::Sequential conv1{nullptr};
    torch::nn::Conv2d conv2{nullptr};
};
TORCH_MODULE(LdDecoder);

struct LdHyperEncoderImpl : ActivationCheckpointModule {
    LdHyperEncoderImpl();
    torch::Tensor forward(const torch::Tensor& input);
    torch::nn::Sequential conv{nullptr};
};
TORCH_MODULE(LdHyperEncoder);

struct LdHyperDecoderImpl : ActivationCheckpointModule {
    LdHyperDecoderImpl();
    torch::Tensor forward(const torch::Tensor& input);
    torch::nn::Sequential conv{nullptr};
};
TORCH_MODULE(LdHyperDecoder);

struct LdTemporalPriorImpl : ActivationCheckpointModule {
    LdTemporalPriorImpl();
    torch::Tensor forward(const torch::Tensor& input);
    ResidualBlockWithStride2 conv{nullptr};
};
TORCH_MODULE(LdTemporalPrior);

struct LdReconHeadImpl : ActivationCheckpointModule {
    LdReconHeadImpl();
    torch::Tensor forward(const torch::Tensor& input, bool for_reset = false);
    torch::nn::Sequential conv{nullptr};
    torch::nn::Conv2d head{nullptr};
};
TORCH_MODULE(LdReconHead);

// Differentiable src/models/video_model_ld.py graph. Bitstream inference uses
// the existing native codec instead. Checkpointing uses non-reentrant native
// saved-tensor hooks on the same module boundaries as the Python CkptModules.
struct DmcLdTrainingImpl : CompressionTrainingImpl {
    // Tensor operators own QP selection/batch broadcasting and shape checks.
    DmcLdTrainingImpl();
    static constexpr std::int64_t frame_delay = 1;
    void set_use_checkpointing(bool enabled = true);
    void clear_dpb();
    void add_ref_feature_from_frame(const torch::Tensor& frame);
    void apply_feature_adaptor();
    void set_ref_feature(const torch::Tensor& feature, bool reset_feature_memory = false);
    RateDistortion forward_one_frame(const torch::Tensor& input, const torch::Tensor& qp,
                                     bool reset_feature_memory = false);
    TrainingForward forward(const torch::Tensor& input, const torch::Tensor& qp,
        const torch::Tensor& lambdas, bool get_loss_info = false, std::int64_t current_poc = 0);
    // The result and per-item losses describe the final frame; mean is the
    // cascaded mean, matching the Python forward/list diagnostic behavior.
    TrainingForward forward(const std::vector<torch::Tensor>& frames, const torch::Tensor& qp,
        const torch::Tensor& lambdas, bool get_loss_info = false, std::int64_t current_poc = 0);

    LdDepthStack feature_adaptor_i{nullptr}, feature_adaptor_m{nullptr}, feature_extractor{nullptr};
    LdEncoder encoder{nullptr};
    LdHyperEncoder hyper_encoder{nullptr};
    LdHyperDecoder hyper_decoder{nullptr};
    LdTemporalPrior temporal_prior_encoder{nullptr};
    LdDepthStack y_prior_fusion{nullptr}, y_spatial_prior{nullptr};
    LdDecoder decoder{nullptr};
    LdReconHead recon_head{nullptr};
    torch::Tensor q_encoder, q_decoder, q_feature;
    torch::Tensor ref_feature, memory, ctx;

private:
    static RateDistortion rd_info(RateDistortion result, std::int64_t adaptor_index);
};
TORCH_MODULE(DmcLdTraining);

}  // namespace uf::training
