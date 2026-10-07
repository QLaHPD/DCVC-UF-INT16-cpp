// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#pragma once

#include "uf/training_common.hpp"
#include "uf/training_recompute.hpp"

namespace uf::training {

struct HtDepthStackImpl : ActivationCheckpointModule {
    HtDepthStackImpl(std::int64_t in_channels, std::int64_t out_channels,
                    std::int64_t count, bool dcb2 = false, std::int64_t final_channels = 0);
    torch::Tensor forward(const torch::Tensor& input);
    torch::Tensor forward(const torch::Tensor& input, const torch::Tensor& context);
    torch::nn::Sequential conv{nullptr};
};
TORCH_MODULE(HtDepthStack);

struct HtEncoderImpl : ActivationCheckpointModule {
    explicit HtEncoderImpl(bool is_hts);
    torch::Tensor forward(const torch::Tensor& input, const torch::Tensor& context,
                          const torch::Tensor& quant_step);
    torch::nn::Sequential conv1{nullptr};
    torch::nn::Conv2d down{nullptr};
};
TORCH_MODULE(HtEncoder);

struct HtDecoderImpl : ActivationCheckpointModule {
    explicit HtDecoderImpl(bool is_hts);
    torch::Tensor forward(const torch::Tensor& input, const torch::Tensor& context,
                          const torch::Tensor& quant_step);
    SubpelConv2x up{nullptr};
    torch::nn::Sequential conv1{nullptr};
};
TORCH_MODULE(HtDecoder);

struct HtHyperEncoderImpl : ActivationCheckpointModule {
    explicit HtHyperEncoderImpl(bool is_hts);
    torch::Tensor forward(const torch::Tensor& input);
    torch::nn::Sequential conv{nullptr};
};
TORCH_MODULE(HtHyperEncoder);

struct HtHyperDecoderImpl : ActivationCheckpointModule {
    explicit HtHyperDecoderImpl(bool is_hts);
    torch::Tensor forward(const torch::Tensor& input);
    torch::nn::Sequential conv{nullptr};
};
TORCH_MODULE(HtHyperDecoder);

struct HtTemporalPriorImpl : ActivationCheckpointModule {
    explicit HtTemporalPriorImpl(bool is_hts);
    torch::Tensor forward(const torch::Tensor& input, const torch::Tensor& quant_step);
    ResidualBlockWithStride2 conv{nullptr};
};
TORCH_MODULE(HtTemporalPrior);

struct HtReconHeadImpl : ActivationCheckpointModule {
    explicit HtReconHeadImpl(bool is_hts);
    std::vector<torch::Tensor> forward(const torch::Tensor& input);
    // Pixel-unshuffled last frame, used when resetting temporal memory.
    torch::Tensor for_reset(const torch::Tensor& input);
    torch::nn::ModuleList conv1{nullptr}, conv2{nullptr}, conv{nullptr};
    const bool is_hts;
};
TORCH_MODULE(HtReconHead);

struct HtRateDistortion {
    std::vector<torch::Tensor> reconstructions, mse;
    torch::Tensor bpp, bits_y, bits_z;
};

struct HtTrainingForward {
    HtRateDistortion result;
    // Only scalar rate/distortion fields; reconstructions are in result above.
    RateDistortion weighted;
    Loss loss;
    std::optional<LossInfo> info;
};

// Differentiable HT-S/HT-L graphs from src/models/video_model_ht.py.
// The Python CkptModule regions use native non-reentrant saved-tensor hooks.
struct DmcHtTrainingImpl : CompressionTrainingImpl {
    // QP selection supports broadcasting, including one QP shared by the batch.
    explicit DmcHtTrainingImpl(const std::string& model_structure = "hts");
    static constexpr std::int64_t frame_delay = 8;
    void set_use_checkpointing(bool enabled = true);
    void clear_dpb();
    void add_ref_feature_from_frame(const torch::Tensor& frame);
    void apply_feature_adaptor();
    void set_ref_feature(const torch::Tensor& feature, bool reset_feature_memory = false);
    HtRateDistortion forward_one_frame(const torch::Tensor& input, const torch::Tensor& qp,
                                       bool reset_feature_memory = false);
    HtTrainingForward forward(const torch::Tensor& input, const torch::Tensor& qp,
        const torch::Tensor& lambdas, bool get_loss_info = false, std::int64_t current_poc = 0);
    // Mean loss across groups; diagnostics/other fields describe the final group.
    HtTrainingForward forward(const std::vector<torch::Tensor>& groups, const torch::Tensor& qp,
        const torch::Tensor& lambdas, bool get_loss_info = false, std::int64_t current_poc = 0);
    static RateDistortion rd_info(const HtRateDistortion& result);

    const bool is_hts;
    HtDepthStack feature_adaptor_i{nullptr}, feature_adaptor_m{nullptr}, feature_extractor{nullptr};
    HtEncoder encoder{nullptr};
    HtHyperEncoder hyper_encoder{nullptr};
    HtHyperDecoder hyper_decoder{nullptr};
    HtTemporalPrior temporal_prior_encoder{nullptr};
    HtDepthStack y_prior_fusion{nullptr};
    torch::nn::Conv2d y_spatial_prior_reduction{nullptr};
    DepthConvBlock y_spatial_prior_adaptor_1{nullptr}, y_spatial_prior_adaptor_2{nullptr},
                   y_spatial_prior_adaptor_3{nullptr};
    HtDepthStack y_spatial_prior{nullptr};
    HtDecoder decoder{nullptr};
    HtReconHead recon_head{nullptr};
    torch::Tensor q_encoder, q_decoder, q_feature;
    torch::Tensor ref_feature, memory, ctx;
};
TORCH_MODULE(DmcHtTraining);

}  // namespace uf::training
