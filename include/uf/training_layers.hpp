// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
// Native adaptation of src/layers/layers.py for differentiable LibTorch models.
#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <utility>
#include <vector>

namespace uf::training {

torch::Tensor lower_bound(const torch::Tensor& input, double bound);
torch::Tensor quantize(const torch::Tensor& input);
torch::Tensor rgb_to_ycbcr(const torch::Tensor& rgb, bool is_bgr = false);
torch::Tensor ycbcr_to_rgb(const torch::Tensor& ycbcr, bool is_bgr = false, bool clamp = true);
// Match [:, :1, :, :] and [:, 1:, :, :] before pooling all chroma channels.
std::pair<torch::Tensor, torch::Tensor> yuv444_to_420(const torch::Tensor& yuv);

// Parameter selection follows [:, :, layer:layer+1, None], with the inserted
// axis before any trailing axes and rank validation delegated to LibTorch.
torch::Tensor bit_estimator_z_probability(const torch::Tensor& input,
    const torch::Tensor& h, const torch::Tensor& b, const torch::Tensor& a);
torch::Tensor bit_estimator_z_mass(const torch::Tensor& input,
    const torch::Tensor& h, const torch::Tensor& b, const torch::Tensor& a);
torch::Tensor gaussian_probability(const torch::Tensor& values, const torch::Tensor& scales);
std::pair<torch::Tensor, torch::Tensor> mse_yuv_rgb(const torch::Tensor& input,
                                                 const torch::Tensor& reconstruction);
torch::Tensor mse_weighted_average(const torch::Tensor& mse_yuv,
                                   const torch::Tensor& mse_rgb, std::int64_t pixels);
// Required list entries are checked; trailing entries are ignored as in Python.
torch::Tensor mse_eight_frame_sum(const std::vector<torch::Tensor>& mse,
                                 const std::vector<double>& distortion_weights);

struct WSiLUImpl : torch::nn::Module {
    torch::Tensor forward(const torch::Tensor& input);
};
TORCH_MODULE(WSiLU);

struct WSiLUChunkAddImpl : torch::nn::Module {
    WSiLUChunkAddImpl();
    // Four explicit slice axes match Python's [:, offset::4, :, :].
    torch::Tensor forward(const torch::Tensor& input);
    WSiLU wsilu{nullptr};
};
TORCH_MODULE(WSiLUChunkAdd);

struct DepthConvBlockImpl : torch::nn::Module {
    DepthConvBlockImpl(std::int64_t in_channels, std::int64_t out_channels,
                      bool dcb2 = false, bool shortcut = false, bool force_adaptor = false);
    torch::Tensor forward(const torch::Tensor& input);
    torch::nn::Conv2d adaptor{nullptr};
    torch::nn::Sequential dc{nullptr}, ffn{nullptr};
    bool shortcut;
};
TORCH_MODULE(DepthConvBlock);

struct SubpelConv2xImpl : torch::nn::Module {
    SubpelConv2xImpl(std::int64_t in_channels, std::int64_t out_channels,
                    std::int64_t kernel_size, std::int64_t padding = 0, bool force_bias = false);
    torch::Tensor forward(const torch::Tensor& input);
    torch::nn::Sequential conv{nullptr};
};
TORCH_MODULE(SubpelConv2x);

struct ResidualBlockUpsampleImpl : torch::nn::Module {
    ResidualBlockUpsampleImpl(std::int64_t in_channels, std::int64_t out_channels,
                             bool dcb2 = false, bool shortcut = true, bool force_bias = false);
    torch::Tensor forward(const torch::Tensor& input);
    SubpelConv2x up{nullptr};
    DepthConvBlock conv{nullptr};
};
TORCH_MODULE(ResidualBlockUpsample);

struct ResidualBlockWithStride2Impl : torch::nn::Module {
    ResidualBlockWithStride2Impl(std::int64_t in_channels, std::int64_t out_channels,
                                bool dcb2 = false, bool shortcut = true);
    torch::Tensor forward(const torch::Tensor& input);
    torch::nn::Conv2d down{nullptr};
    DepthConvBlock conv{nullptr};
};
TORCH_MODULE(ResidualBlockWithStride2);

}  // namespace uf::training
