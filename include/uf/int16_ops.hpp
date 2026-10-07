#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <vector>

namespace uf::int16 {

constexpr std::int64_t feature_scale = 512;
constexpr std::int64_t weight_scale = 8192;
constexpr const char* arithmetic_id = "uf-int16-q9-w13-acc64-rna-sat-v1";
constexpr const char* kernel_revision = "tiled-v2";

bool cuda_compiled();

torch::Tensor saturate(const torch::Tensor& value);
// Preparation only: detach, copy to CPU Float64, reject non-finite input,
// round to nearest even, convert to Int64, then saturate to Int16. Preserve
// the Int64 conversion boundary; the scaled result is not pre-clamped.
torch::Tensor quantize_parameter(const torch::Tensor& value, double scale = feature_scale);
torch::Tensor quantize_parameter(const torch::Tensor& value, const torch::Tensor& scale);
// Scalar half-divisors use Python // floor semantics; torch.div handles zero.
torch::Tensor divide_round(const torch::Tensor& value, std::int64_t denominator);
torch::Tensor divide_round(const torch::Tensor& value, const torch::Tensor& denominator);
torch::Tensor add(const std::vector<torch::Tensor>& values);
torch::Tensor multiply(const torch::Tensor& value, const torch::Tensor& scale);
torch::Tensor reciprocal(const torch::Tensor& value);
torch::Tensor symbols(const torch::Tensor& value);
torch::Tensor from_symbols(const torch::Tensor& value);
torch::Tensor from_bytes(const torch::Tensor& value);
// CPU lookup follows table[value.long()+32768] on its first dimension,
// retaining shaped/negative/empty indices and trailing table dimensions.
// CUDA dispatch retains the native extension's one-dimensional table checks.
torch::Tensor lookup(const torch::Tensor& value, const torch::Tensor& table);
// CPU WSiLU4 uses four out[:, offset::4] slices and add's broadcasting;
// rank/channel errors come from indexing/add. CUDA uses the native kernel.
torch::Tensor wsilu4(const torch::Tensor& value, const torch::Tensor& table);

// Exact signed INT16 convolution: INT64 accumulation, nearest ties away from
// zero after division by 8192, feature-scale bias, and saturation.
// Independent CPU reference using Python's pad/unfold/reshape/matmul sequence.
// Input and weights must be CPU Int16. Bias uses bias.long()[None,:,None,None]
// indexing, retaining trailing axes and LibTorch broadcast/rank errors. Signed
// padding is handled by LibTorch, including cropping for negative padding.
torch::Tensor conv2d_reference(const torch::Tensor& input, const torch::Tensor& weight,
                               const torch::Tensor& bias = {}, std::int64_t stride_h = 1,
                               std::int64_t stride_w = 1, std::int64_t pad_h = 0,
                               std::int64_t pad_w = 0, std::int64_t groups = 1);
// CUDA dispatch checks these Int64 geometry values before converting to the
// extension's native int arguments. CPU reference geometry retains Int64.
torch::Tensor conv2d(const torch::Tensor& input, const torch::Tensor& weight,
                     const torch::Tensor& bias = {}, std::int64_t stride_h = 1,
                     std::int64_t stride_w = 1, std::int64_t pad_h = 0,
                     std::int64_t pad_w = 0, std::int64_t groups = 1);

}  // namespace uf::int16
