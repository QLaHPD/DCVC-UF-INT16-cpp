// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#pragma once

#include "uf/evaluation_json.hpp"
#include <torch/torch.h>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace uf::evaluation {

double calc_psnr(const torch::Tensor& image, const torch::Tensor& reconstruction, double data_range = 255);
torch::Tensor fspecial_gauss(std::int64_t size, double sigma);
std::pair<torch::Tensor, torch::Tensor> calc_ssim(
    const torch::Tensor& image, const torch::Tensor& reconstruction, double data_range = 255);
double calc_msssim(const torch::Tensor& image, const torch::Tensor& reconstruction, double data_range = 255);
double calc_msssim_rgb(const torch::Tensor& image, const torch::Tensor& reconstruction, double data_range = 255);

struct FrameQuality {
    // RGB: one value. YUV: combined, Y, U, V.
    std::vector<double> psnr, msssim;
};

// Counts/pixels and individual bit values use UInt64. Accumulated bit totals
// and count-times-pixels use UInt128 internally before binary64 true division.
// The explicit count selects the prefix used for aggregates. Verbose arrays
// retain their full input lengths, as maintained Python generate_log_json does.
Json generate_log_json(std::uint64_t frame_count, std::uint64_t frame_pixels, double test_time,
    const std::vector<int>& frame_types, const std::vector<std::uint64_t>& bits,
    const std::vector<FrameQuality>& quality, bool verbose = false,
    std::optional<double> average_encoding_time = std::nullopt,
    std::optional<double> average_decoding_time = std::nullopt);
// Convenience form for equal-length arrays, counting every quality entry.
Json generate_log_json(std::uint64_t frame_pixels, double test_time,
    const std::vector<int>& frame_types, const std::vector<std::uint64_t>& bits,
    const std::vector<FrameQuality>& quality, bool verbose = false,
    std::optional<double> average_encoding_time = std::nullopt,
    std::optional<double> average_decoding_time = std::nullopt);

}  // namespace uf::evaluation
