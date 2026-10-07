// PSNR follows the original Microsoft DCVC test_video.py (MIT).
// MS-SSIM adapts pytorch-msssim 0.2.1, Copyright 2020 Gongfan Fang,
// Zhejiang University. MIT license retained in licenses/PyTorch-MSSSIM.txt.
#include "dcvc2021_evaluation_internal.hpp"
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace uf::dcvc2021 {
namespace {
torch::Tensor gaussian_filter(torch::Tensor input, const torch::Tensor& window) {
    // The separable valid convolutions run along height, then width.
    const auto channels = input.size(1);
    const std::vector<std::int64_t> zero_padding{0, 0};
    for (std::int64_t axis = 0; axis < 2; ++axis) {
        if (input.size(axis + 2) >= window.size(-1)) {
            input = torch::conv2d(input, window.transpose(axis + 2, 3), torch::Tensor{},
                {1, 1}, zero_padding, {1, 1}, channels);
        } else {
            std::cerr << "Skipping Gaussian Smoothing at dimension 2+" << axis << '\n';
        }
    }
    return input;
}

std::pair<torch::Tensor, torch::Tensor> ssim_channels(
    const torch::Tensor& x, const torch::Tensor& y, const torch::Tensor& cpu_window) {
    constexpr double c1 = 0.01 * 0.01, c2 = 0.03 * 0.03;
    const auto window = cpu_window.to(x.device(), x.scalar_type());
    const auto mean_x = gaussian_filter(x, window), mean_y = gaussian_filter(y, window);
    const auto mean_x_sq = mean_x.pow(2), mean_y_sq = mean_y.pow(2), mean_xy = mean_x * mean_y;
    const auto variance_x = 1.0 * (gaussian_filter(x * x, window) - mean_x_sq);
    const auto variance_y = 1.0 * (gaussian_filter(y * y, window) - mean_y_sq);
    const auto covariance = 1.0 * (gaussian_filter(x * y, window) - mean_xy);
    const auto cs = (2 * covariance + c2) / (variance_x + variance_y + c2);
    const auto ssim = ((2 * mean_xy + c1) / (mean_x_sq + mean_y_sq + c1)) * cs;
    return {ssim.flatten(2).mean({-1}), cs.flatten(2).mean({-1})};
}
}  // namespace

double evaluation_psnr(const torch::Tensor& reconstruction, const torch::Tensor& original) {
    // Keep the tensor's dtype/device. The root UF metric instead uses doubles
    // and a different perfect-image policy; it is not used for this family.
    const auto mse = (reconstruction - original).pow(2).mean();
    return (20 * torch::log10(1.0 / torch::sqrt(mse))).item<double>();
}

double evaluation_msssim(const torch::Tensor& reconstruction, const torch::Tensor& original) {
    if (reconstruction.sizes() != original.sizes()) throw std::invalid_argument("MS-SSIM input dimensions differ");
    auto x = reconstruction, y = original;
    for (auto dimension = x.dim() - 1; dimension > 1; --dimension) {
        x = x.squeeze(dimension); y = y.squeeze(dimension);
    }
    if (x.scalar_type() != y.scalar_type() || x.device().type() != y.device().type()) {
        throw std::invalid_argument("MS-SSIM input tensor types differ");
    }
    // This workflow reads NCHW RGB. The package's independent 3D volume API
    // is not an input format of the original DCVC evaluation script.
    if (x.dim() != 4) throw std::invalid_argument("DCVC evaluation MS-SSIM requires NCHW input");
    if (std::min(x.size(-2), x.size(-1)) <= 160) {
        throw std::invalid_argument("Image size should be larger than 160 due to the 4 downsamplings in ms-ssim");
    }
    const auto weights = torch::tensor({0.0448, 0.2856, 0.3001, 0.2363, 0.1333},
        torch::TensorOptions().dtype(torch::kFloat32)).to(x.device(), x.scalar_type());
    auto coordinates = torch::arange(11, torch::TensorOptions().dtype(torch::kInt64)).to(torch::kFloat32);
    coordinates.sub_(5);
    auto gaussian = torch::exp(-coordinates.pow(2) / (2 * 1.5 * 1.5));
    gaussian.div_(gaussian.sum());
    const auto window = gaussian.reshape({1, 1, 11}).repeat({x.size(1), 1, 1, 1});
    std::vector<torch::Tensor> responses;
    responses.reserve(5);
    for (std::int64_t level = 0; level < 5; ++level) {
        auto result = ssim_channels(x, y, window);
        if (level < 4) {
            responses.push_back(torch::relu(result.second));
            const std::vector<std::int64_t> padding{x.size(2) % 2, x.size(3) % 2};
            // PyTorch's default count_include_pad=true is part of the metric.
            x = torch::avg_pool2d(x, {2, 2}, {2, 2}, padding, false, true);
            y = torch::avg_pool2d(y, {2, 2}, {2, 2}, padding, false, true);
        } else responses.push_back(torch::relu(result.first));
    }
    return torch::stack(responses, 0).pow(weights.view({-1, 1, 1})).prod(0).mean().item<double>();
}
}  // namespace uf::dcvc2021
