// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#pragma once

#include "uf/training_common.hpp"

namespace uf::training {

struct IntraEncoderImpl : torch::nn::Module {
    IntraEncoderImpl();
    torch::Tensor forward(const torch::Tensor& input, const torch::Tensor& quant_step);
    DepthConvBlock enc_1{nullptr};
    torch::nn::Sequential enc_2{nullptr};
};
TORCH_MODULE(IntraEncoder);

struct IntraDecoderImpl : torch::nn::Module {
    IntraDecoderImpl();
    torch::Tensor forward(const torch::Tensor& input, const torch::Tensor& quant_step);
    torch::nn::Sequential dec_1{nullptr};
    DepthConvBlock dec_2{nullptr};
};
TORCH_MODULE(IntraDecoder);

struct IntraHyperEncoderImpl : torch::nn::Module {
    IntraHyperEncoderImpl();
    torch::Tensor forward(const torch::Tensor& input);
    torch::nn::Sequential conv{nullptr};
};
TORCH_MODULE(IntraHyperEncoder);

struct IntraHyperDecoderImpl : torch::nn::Module {
    IntraHyperDecoderImpl();
    torch::Tensor forward(const torch::Tensor& input);
    torch::nn::Sequential conv{nullptr};
};
TORCH_MODULE(IntraHyperDecoder);

struct IntraSpatialPriorImpl : torch::nn::Module {
    IntraSpatialPriorImpl();
    torch::Tensor forward(const torch::Tensor& input);
    torch::nn::Sequential conv{nullptr};
};
TORCH_MODULE(IntraSpatialPrior);

struct IntraYPriorFusionImpl : torch::nn::Module {
    IntraYPriorFusionImpl();
    torch::Tensor forward(const torch::Tensor& input);
    torch::nn::Sequential conv{nullptr};
};
TORCH_MODULE(IntraYPriorFusion);

// Full differentiable image graph from src/models/image_model.py. Native
// inference proxies remain the bitstream interface; this graph owns the
// nn.Module parameters, likelihoods and loss needed for training.
struct DmciTrainingImpl : CompressionTrainingImpl {
    DmciTrainingImpl();
    // QP selection and batch broadcasting follow LibTorch, as in Python.
    // A singleton QP need not have the same length as the input batch.
    RateDistortion forward_one_frame(const torch::Tensor& input, const torch::Tensor& qp,
                                     bool reconstruction_only = false);
    TrainingForward forward(const torch::Tensor& input, const torch::Tensor& qp,
                             const torch::Tensor& lambdas, bool get_loss_info = false);
    torch::Tensor reconstruct(const torch::Tensor& input, const torch::Tensor& qp);

    IntraEncoder enc{nullptr};
    IntraHyperEncoder hyper_enc{nullptr};
    IntraHyperDecoder hyper_dec{nullptr};
    IntraYPriorFusion y_prior_fusion{nullptr};
    torch::nn::Conv2d y_spatial_prior_reduction{nullptr};
    DepthConvBlock y_spatial_prior_adaptor_1{nullptr}, y_spatial_prior_adaptor_2{nullptr},
                   y_spatial_prior_adaptor_3{nullptr};
    IntraSpatialPrior y_spatial_prior{nullptr};
    IntraDecoder dec{nullptr};
    torch::Tensor q_scale_enc, q_scale_dec, q_scale_y_enc, q_scale_y_dec;
};
TORCH_MODULE(DmciTraining);

}  // namespace uf::training
