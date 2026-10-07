// Copyright (c) Microsoft Corporation. Licensed under the MIT License.
// Zoom traversal adapts SciPy 1.15.3 NI_ZoomShift (Copyright 2003-2005
// Peter J. Verveer), under the BSD notice retained in licenses/SciPy.txt.
#include "uf/dcvc_dc_media.hpp"
#include "dcvc_dc_splines.hpp"
#include "dcvc_dc_numpy.hpp"
#include <torch/nn/functional.h>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace uf::dc {
namespace {
constexpr double kr = 0.2126, kg = 0.7152, kb = 0.0722;
void check_tensor(const torch::Tensor& input) {
    if (!input.defined() || !input.is_floating_point() ||
        (input.dim() != 3 && input.dim() != 4) || input.size(-3) != 3)
        throw std::invalid_argument("Expected a 3D or 4D floating tensor with three channels");
}
void check_array(const torch::Tensor& input, std::int64_t channels) {
    if (!input.defined() || !input.is_floating_point() || !input.device().is_cpu() ||
        input.dim() != 3 || input.size(0) != channels)
        throw std::invalid_argument("DC array transform requires a floating CPU CHW tensor");
}
torch::Tensor box_down(const torch::Tensor& plane) {
    const auto h = plane.size(1), w = plane.size(2);
    const auto dtype = plane.scalar_type();
    const auto input = plane.to(dtype == torch::kFloat64 ? torch::kFloat64 : torch::kFloat32).contiguous();
    auto result = torch::empty({1, h / 2, w / 2}, input.options());
    const auto reduce = [&](auto* source, auto* destination) {
        using Number = std::remove_pointer_t<decltype(source)>;
        for (std::int64_t row = 0; row < h / 2; ++row)
            for (std::int64_t column = 0; column < w / 2; ++column) {
                const auto* first = source + 2 * row * w + 2 * column;
                Number total = 0;
                if (w == 2) total += detail::numpy_pairwise_sum(first, 4);
                else {
                    total += detail::numpy_pairwise_sum(first, 2);
                    total += detail::numpy_pairwise_sum(first + w, 2);
                }
                destination[row * (w / 2) + column] = total / 4;
            }
    };
    if (dtype == torch::kFloat64) reduce(input.data_ptr<double>(), result.data_ptr<double>());
    else reduce(input.data_ptr<float>(), result.data_ptr<float>());
    return result.to(dtype);
}
void check_even(const torch::Tensor& input) {
    check_array(input, 3);
    if (input.size(1) % 2 || input.size(2) % 2)
        throw std::invalid_argument("DC 4:2:0 conversion requires even height and width");
}
std::int64_t mirror(std::int64_t index, std::int64_t size) {
    if (size <= 1) return 0;
    const auto period = 2 * size - 2;
    index %= period;
    if (index < 0) index += period;
    return index < size ? index : period - index;
}
struct Knot {
    std::array<std::int64_t, 6> indexes{};
    std::array<double, 6> weights{};
    bool outside = false;
};
std::vector<Knot> knots(std::int64_t input, std::int64_t output, int order) {
    std::vector<Knot> result(static_cast<std::size_t>(output));
    const auto scale = output > 1 ? static_cast<double>(input - 1) / (output - 1) : 1.0;
    for (std::int64_t index = 0; index < output; ++index) {
        auto& knot = result[static_cast<std::size_t>(index)];
        const auto coordinate = static_cast<double>(index) * scale;
        // Constant mode checks the coordinate before mirroring its footprint.
        if (coordinate < 0 || coordinate > input - 1) { knot.outside = true; continue; }
        const auto start = static_cast<std::int64_t>(std::floor(order & 1 ? coordinate : coordinate + 0.5)) - order / 2;
        for (int offset = 0; offset <= order; ++offset)
            knot.indexes[offset] = mirror(start + offset, input);
        if (order) detail::get_spline_interpolation_weights(coordinate, order, knot.weights.data());
        else knot.weights[0] = 1;
    }
    return result;
}
void prefilter(torch::Tensor& coefficients, int order) {
    auto* data = coefficients.data_ptr<double>();
    const std::array<std::int64_t, 3> sizes{{coefficients.size(0), coefficients.size(1), coefficients.size(2)}};
    const std::array<std::int64_t, 3> strides{{sizes[1] * sizes[2], sizes[2], 1}};
    double poles[detail::MAX_SPLINE_FILTER_POLES]; int count = 0;
    detail::get_filter_poles(order, &count, poles);
    // scipy.ndimage.spline_filter traverses axes 0, 1, 2 in Float64, including
    // the channel axis whose zoom factor is one.
    for (int axis = 0; axis < 3; ++axis) {
        const auto length = sizes[axis], stride = strides[axis];
        if (length <= 1) continue;
        std::vector<double> line(static_cast<std::size_t>(length));
        const auto blocks = coefficients.numel() / (length * stride);
        for (std::int64_t block = 0; block < blocks; ++block) {
            for (std::int64_t offset = 0; offset < stride; ++offset) {
                const auto base = block * length * stride + offset;
                for (std::int64_t index = 0; index < length; ++index) line[index] = data[base + index * stride];
                detail::apply_filter(line.data(), length, poles, count, detail::NI_EXTEND_CONSTANT);
                for (std::int64_t index = 0; index < length; ++index) data[base + index * stride] = line[index];
            }
        }
    }
}
torch::Tensor zoom_chroma(const torch::Tensor& uv, int order) {
    check_array(uv, 2);
    if (order < 0 || order > 5) throw std::invalid_argument("Spline order must be between zero and five");
    if (uv.scalar_type() != torch::kFloat32 && uv.scalar_type() != torch::kFloat64)
        throw std::invalid_argument("DC SciPy zoom supports Float32 and Float64 arrays");
    const auto h = uv.size(1), w = uv.size(2);
    if (h > std::numeric_limits<std::int64_t>::max() / 2 || w > std::numeric_limits<std::int64_t>::max() / 2)
        throw std::overflow_error("DC chroma zoom dimensions exceed native integers");
    auto result = torch::zeros({2, 2 * h, 2 * w}, uv.options().dtype(torch::kFloat64));
    if (!h || !w) return result.to(uv.scalar_type());
    // Clone before filtering so a Float64 input is never modified by prefilter.
    auto coefficients = uv.detach().to(torch::kFloat64).contiguous().clone();
    if (order > 1) prefilter(coefficients, order);
    const auto channels = knots(2, 2, order), rows = knots(h, 2 * h, order), columns = knots(w, 2 * w, order);
    const auto* data = coefficients.data_ptr<double>();
    auto* output = result.data_ptr<double>();
    for (std::int64_t channel = 0; channel < 2; ++channel) {
        const auto& ck = channels[channel];
        for (std::int64_t row = 0; row < 2 * h; ++row) {
            const auto& rk = rows[row];
            for (std::int64_t column = 0; column < 2 * w; ++column) {
                const auto& wk = columns[column]; double value = 0;
                if (!ck.outside && !rk.outside && !wk.outside) {
                    for (int c = 0; c <= order; ++c)
                        for (int y = 0; y <= order; ++y)
                            for (int x = 0; x <= order; ++x) {
                                auto term = data[(ck.indexes[c] * h + rk.indexes[y]) * w + wk.indexes[x]];
                                if (order) { term *= ck.weights[c]; term *= rk.weights[y]; term *= wk.weights[x]; }
                                value += term;
                            }
                }
                output[(channel * 2 * h + row) * 2 * w + column] = value;
            }
        }
    }
    return result.to(uv.scalar_type());
}
}  // namespace

torch::Tensor rgb2ycbcr(const torch::Tensor& rgb) {
    check_tensor(rgb);
    const auto parts = rgb.chunk(3, -3);
    const auto y = kr * parts[0] + kg * parts[1] + kb * parts[2];
    const auto cb = 0.5 * (parts[2] - y) / (1 - kb) + 0.5;
    const auto cr = 0.5 * (parts[0] - y) / (1 - kr) + 0.5;
    return torch::cat({y, cb, cr}, -3);
}
torch::Tensor ycbcr2rgb(const torch::Tensor& ycbcr) {
    check_tensor(ycbcr);
    const auto parts = ycbcr.chunk(3, -3);
    const auto r = parts[0] + (2 - 2 * kr) * (parts[2] - 0.5);
    const auto b = parts[0] + (2 - 2 * kb) * (parts[1] - 0.5);
    const auto g = (parts[0] - kr * r - kb * b) / kg;
    return torch::cat({r, g, b}, -3);
}
torch::Tensor rgb_to_ycbcr(const torch::Tensor& rgb) {
    check_array(rgb, 3); return rgb2ycbcr(rgb).clamp(0, 1);
}
torch::Tensor ycbcr_to_rgb(const torch::Tensor& yuv) {
    check_array(yuv, 3); return ycbcr2rgb(yuv).clamp(0, 1);
}
Yuv420 rgb_to_ycbcr420(const torch::Tensor& rgb) {
    check_even(rgb);
    const auto parts = rgb2ycbcr(rgb).chunk(3, 0);
    return {parts[0].clamp(0, 1), torch::cat({box_down(parts[1]), box_down(parts[2])}, 0).clamp(0, 1)};
}
Yuv420 ycbcr444_to_420(const torch::Tensor& yuv) {
    check_even(yuv);
    const auto parts = yuv.chunk(3, 0);
    return {parts[0].clamp(0, 1), torch::cat({box_down(parts[1]), box_down(parts[2])}, 0).clamp(0, 1)};
}
torch::Tensor ycbcr420_to_rgb(const torch::Tensor& y, const torch::Tensor& uv, int order) {
    const auto chroma = zoom_chroma(uv, order);
    const auto cb = chroma.slice(0, 0, 1), cr = chroma.slice(0, 1, 2);
    // The evaluation writer passes a squeezed 2D Y array. Broadcasting that
    // array with a 1xHxW chroma plane is part of the reference behavior.
    const auto r = y + (2 - 2 * kr) * (cr - 0.5);
    const auto b = y + (2 - 2 * kb) * (cb - 0.5);
    const auto g = (y - kr * r - kb * b) / kg;
    return torch::cat({r, g, b}, 0).clamp(0, 1);
}
torch::Tensor ycbcr420_to_444(const torch::Tensor& y, const torch::Tensor& uv, int order) {
    return torch::cat({y, zoom_chroma(uv, order)}, 0);
}
YuvPlanes yuv_444_to_420(const YuvPlanes& yuv, const std::string& mode) {
    if (mode != "avg_pool") throw std::invalid_argument("Invalid downsampling mode: " + mode);
    const auto options = torch::nn::functional::AvgPool2dFuncOptions(2).stride(2);
    return {yuv[0], torch::nn::functional::avg_pool2d(yuv[1], options), torch::nn::functional::avg_pool2d(yuv[2], options)};
}
YuvPlanes yuv_444_to_420(const torch::Tensor& yuv, const std::string& mode) {
    const auto parts = yuv.chunk(3, 1);
    if (parts.size() != 3) throw std::invalid_argument("Expected three YUV planes");
    return yuv_444_to_420(YuvPlanes{{parts[0], parts[1], parts[2]}}, mode);
}
YuvPlanes yuv_420_to_444_tuple(const YuvPlanes& yuv, const std::string& mode) {
    if (mode != "bilinear" && mode != "nearest") throw std::invalid_argument("Invalid upsampling mode: " + mode);
    for (const auto& plane : yuv) if (!plane.defined()) throw std::invalid_argument("Expected three Torch tensors");
    // The Python source passes align_corners=False even for nearest, which
    // raises; preserve that behavior instead of silently changing interpolation.
    if (mode == "nearest") throw std::invalid_argument("align_corners is only available for linear interpolation modes");
    const auto options = torch::nn::functional::InterpolateFuncOptions()
        .scale_factor(std::vector<double>{2, 2}).mode(torch::kBilinear).align_corners(false);
    return {yuv[0], torch::nn::functional::interpolate(yuv[1], options), torch::nn::functional::interpolate(yuv[2], options)};
}
torch::Tensor yuv_420_to_444(const YuvPlanes& yuv, const std::string& mode) {
    const auto planes = yuv_420_to_444_tuple(yuv, mode);
    return torch::cat({planes[0], planes[1], planes[2]}, 1);
}
}  // namespace uf::dc
