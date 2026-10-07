// Copyright (c) Microsoft Corporation. Licensed under the MIT License.
#pragma once
#include <torch/torch.h>
#include <utility>
#include <variant>

namespace uf::transforms {
using Yuv420 = std::pair<torch::Tensor, torch::Tensor>;
using Upsampled = std::variant<torch::Tensor, Yuv420>;
// Torch helpers retain the device and graph; chunk(3, -3) and broadcasting
// determine valid channel counts, with no separate exactly-three guard.
torch::Tensor rgb2ycbcr(const torch::Tensor& rgb, bool is_bgr = false);
torch::Tensor ycbcr2rgb(const torch::Tensor& ycbcr, bool is_bgr = false, bool clamp = true);
// Array helpers use CPU tensors to represent NumPy arrays. RGB is HWC;
// Y and UV are CHW. Chroma zoom preserves SciPy's default endpoint coordinates,
// constant boundary behavior, spline prefilter and input dtype.
torch::Tensor rgb2ycbcr_np(const torch::Tensor& rgb, bool quant = false);
torch::Tensor ycbcr420_to_444_np(const torch::Tensor& y, const torch::Tensor& uv, int order = 0);
Upsampled ycbcr420_to_444_np(const torch::Tensor& y, const torch::Tensor& uv, int order, bool separate);
// Four explicit slice axes, followed by avg_pool2d on all channels after Y.
Yuv420 yuv_444_to_420(const torch::Tensor& yuv);
}  // namespace uf::transforms
