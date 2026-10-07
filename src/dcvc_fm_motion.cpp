// Copyright (c) Microsoft Corporation. Licensed under the MIT License.
#include "uf/dcvc_fm.hpp"
#include <cstdlib>
#include <iostream>
#include <mutex>

namespace uf::fm {
#ifdef UF_HAS_FM_MOTION_CUDA
namespace detail {
void block_mc_forward_cuda(torch::Tensor& out, const torch::Tensor& im,
                           const torch::Tensor& flow, int b, int c, int h, int w);
}
#endif
namespace {
std::array<std::map<std::vector<std::int64_t>, torch::Tensor>, 9> grids;
std::mutex grid_mutex;
bool force_grid = false;
int cache_slot(torch::Device device) {
    const auto index = device.is_cpu() ? 8 : static_cast<int>(device.index());
    if (index < 0 || index > 8) throw std::out_of_range("FM warp cache device index");
    return index;
}
}
void set_force_recalculate_grid(bool value) {
    std::lock_guard<std::mutex> lock(grid_mutex);
    force_grid = value;
}
torch::Tensor torch_warp(const torch::Tensor& feature, const torch::Tensor& flow) {
    if (feature.dim() != 4 || flow.dim() != 4 || flow.size(1) != 2)
        throw std::invalid_argument("FM warp requires NCHW features and two-channel flow");
    torch::Tensor grid;
    {
        std::lock_guard<std::mutex> lock(grid_mutex);
        auto& cache = grids.at(cache_slot(flow.device()));
        const auto key = flow.sizes().vec();
        if (!cache.count(key) || force_grid) {
            const auto options = torch::TensorOptions().device(flow.device()).dtype(torch::kFloat32);
            const auto horizontal = torch::linspace(-1, 1, flow.size(3), options)
                .view({1, 1, 1, flow.size(3)}).expand({flow.size(0), -1, flow.size(2), -1});
            const auto vertical = torch::linspace(-1, 1, flow.size(2), options)
                .view({1, 1, flow.size(2), 1}).expand({flow.size(0), -1, -1, flow.size(3)});
            cache[key] = torch::cat({horizontal, vertical}, 1);
        }
        // Source uses the feature's device to look up a grid created for flow.
        grid = grids.at(cache_slot(feature.device())).at(key);
    }
    const auto normalized = torch::cat({flow.slice(1, 0, 1) / ((feature.size(3) - 1.0) / 2.0),
                                       flow.slice(1, 1, 2) / ((feature.size(2) - 1.0) / 2.0)}, 1);
    return torch::nn::functional::grid_sample(feature, (grid + normalized).permute({0, 2, 3, 1}),
        torch::nn::functional::GridSampleFuncOptions().mode(torch::kBilinear)
            .padding_mode(torch::kBorder).align_corners(true));
}
torch::Tensor flow_warp(const torch::Tensor& feature, const torch::Tensor& flow) {
    if (feature.scalar_type() == torch::kFloat16)
        return torch_warp(feature.to(torch::kFloat32), flow.to(torch::kFloat32)).to(torch::kFloat16);
    return torch_warp(feature, flow);
}
torch::Tensor block_mc_func(const torch::Tensor& feature, const torch::Tensor& flow) {
#ifdef UF_HAS_FM_MOTION_CUDA
    torch::NoGradGuard no_grad;
    auto out = torch::empty_like(feature);
    if (feature.dim() != 4) throw std::invalid_argument("FM motion features require NCHW dimensions");
    detail::block_mc_forward_cuda(out, feature, flow, feature.size(0), feature.size(1), feature.size(2), feature.size(3));
    return out;
#else
    static std::once_flag warning;
    std::call_once(warning, [] {
        if (!std::getenv("SUPPRESS_CUSTOM_KERNEL_WARNING"))
            std::cout << "cannot import motion compensation in cuda, fallback to pytorch grid_sample.\n";
    });
    return flow_warp(feature, flow);
#endif
}
}  // namespace uf::fm
