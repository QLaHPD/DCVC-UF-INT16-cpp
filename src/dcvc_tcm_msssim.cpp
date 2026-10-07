// Copyright 2020 Gongfan Fang, Zhejiang University.
// Adapted from pytorch-msssim 0.2.0. MIT notice: licenses/PyTorch-MSSSIM.txt.
#include "uf/dcvc_tcm.hpp"
#include "uf/process.hpp"
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace uf::tcm {
namespace {
torch::Tensor filter(const torch::Tensor& input, const torch::Tensor& window) {
    // Version 0.2.0 runs width first, then height, with no skip branch.
    const std::vector<std::int64_t> zero_padding{0, 0};
    const auto horizontal = torch::conv2d(input, window, torch::Tensor{},
        {1, 1}, zero_padding, {1, 1}, input.size(1));
    return torch::conv2d(horizontal, window.transpose(2, 3), torch::Tensor{},
        {1, 1}, zero_padding, {1, 1}, input.size(1));
}
std::pair<torch::Tensor, torch::Tensor> ssim(const torch::Tensor& x, const torch::Tensor& y,
                                           const torch::Tensor& cpu_window) {
    constexpr double c1 = 0.01 * 0.01, c2 = 0.03 * 0.03;
    const auto window = cpu_window.to(x.device(), x.scalar_type());
    const auto mean_x = filter(x, window), mean_y = filter(y, window);
    const auto mx2 = mean_x.pow(2), my2 = mean_y.pow(2), mxy = mean_x * mean_y;
    const auto vx = 1.0 * (filter(x * x, window) - mx2);
    const auto vy = 1.0 * (filter(y * y, window) - my2);
    const auto covariance = 1.0 * (filter(x * y, window) - mxy);
    const auto cs = (2 * covariance + c2) / (vx + vy + c2);
    const auto response = ((2 * mxy + c1) / (mx2 + my2 + c1)) * cs;
    return {response.flatten(2).mean({-1}), cs.flatten(2).mean({-1})};
}
}  // namespace
MSSSIMImpl::MSSSIMImpl(std::int64_t width, bool size_average) : size_average_(size_average) {
    auto coordinates = torch::arange(width, torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt64)).to(torch::kFloat32);
    coordinates.sub_(width / 2);
    auto gaussian = torch::exp(-coordinates.pow(2) / (2 * 1.5 * 1.5));
    gaussian.div_(gaussian.sum());
    window_ = gaussian.view({1, 1, width}).repeat({3, 1, 1, 1});
}
torch::Tensor MSSSIMImpl::forward(const torch::Tensor& input, const torch::Tensor& reference) {
    if (input.dim() != 4) throw std::invalid_argument("TCM MS-SSIM input must be four dimensional");
    if (input.scalar_type() != reference.scalar_type() || input.device().type() != reference.device().type())
        throw std::invalid_argument("TCM MS-SSIM input tensor types differ");
    if (input.sizes() != reference.sizes()) throw std::invalid_argument("TCM MS-SSIM input dimensions differ");
    const auto width = window_.size(-1);
    if (width % 2 != 1) throw std::invalid_argument("TCM MS-SSIM window size must be odd");
    if (width - 1 > std::numeric_limits<std::int64_t>::max() / 16 ||
        std::min(input.size(2), input.size(3)) <= (width - 1) * 16)
        throw std::invalid_argument("TCM MS-SSIM image is too small for four downsamplings");
    const auto weights = torch::tensor({0.0448, 0.2856, 0.3001, 0.2363, 0.1333},
        torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32)).to(input.device(), input.scalar_type());
    auto x = input, y = reference;
    std::vector<torch::Tensor> responses;
    for (int level = 0; level < 5; ++level) {
        ProcessInterrupt::check();
        const auto result = ssim(x, y, window_);
        if (level < 4) {
            responses.push_back(torch::relu(result.second));
            const std::vector<std::int64_t> padding{x.size(2) % 2, x.size(3) % 2};
            x = torch::avg_pool2d(x, {2, 2}, {2, 2}, padding, false, true);
            y = torch::avg_pool2d(y, {2, 2}, {2, 2}, padding, false, true);
        } else responses.push_back(torch::relu(result.first));
    }
    const auto result = torch::stack(responses, 0).pow(weights.view({-1, 1, 1})).prod(0);
    return size_average_ ? result.mean() : result.mean(1);
}
}  // namespace uf::tcm
