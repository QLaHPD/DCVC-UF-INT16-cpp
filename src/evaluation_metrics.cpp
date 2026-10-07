// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#include "uf/evaluation_metrics.hpp"
#include "uf/process.hpp"
#include "numpy_reductions.hpp"
#include "python_integer_ratio.hpp"

#include <nlohmann/json.hpp>
#include <ATen/ops/fft_irfftn.h>
#include <ATen/ops/fft_rfftn.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace uf::evaluation {
namespace {

torch::Tensor float64_cpu(const torch::Tensor& input) {
    if (!input.defined()) throw std::invalid_argument("Evaluation metric requires a defined tensor");
    return input.detach().to(torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
}

double array_sum(const torch::Tensor& values) {
    // NumPy's default reduction visits physical keep-order axes. A dense CHW
    // view of an HWC PNG must retain HWC order rather than be flattened as CHW.
    auto data = values;
    if (!data.is_non_overlapping_and_dense()) {
        std::vector<std::int64_t> axes(static_cast<std::size_t>(data.dim()));
        std::iota(axes.begin(), axes.end(), 0);
        std::stable_sort(axes.begin(), axes.end(), [&](std::int64_t first, std::int64_t second) {
            return data.stride(first) > data.stride(second);
        });
        data = data.permute(axes).contiguous();
    }
    return 0.0 + numpy_detail::pairwise_sum(data.data_ptr<double>(), data.numel());
}
double array_mean(const torch::Tensor& values) {
    return values.numel() ? array_sum(values) / values.numel() : std::numeric_limits<double>::quiet_NaN();
}

// SciPy 1.15.3's real FFT fast lengths have only factors 2, 3 and 5.
std::int64_t fast_length(std::int64_t target) {
    if (target <= 0) throw std::invalid_argument("Invalid evaluation FFT length");
    std::int64_t best = std::numeric_limits<std::int64_t>::max();
    for (std::int64_t a = 1;;) {
        for (std::int64_t b = a;;) {
            for (std::int64_t c = b;;) {
                if (c >= target) { best = std::min(best, c); break; }
                if (c >= best || c > std::numeric_limits<std::int64_t>::max() / 5) break;
                c *= 5;
            }
            if (b >= best || b > std::numeric_limits<std::int64_t>::max() / 3) break;
            b *= 3;
        }
        if (a >= best || a > std::numeric_limits<std::int64_t>::max() / 2) break;
        a *= 2;
    }
    if (best == std::numeric_limits<std::int64_t>::max()) {
        throw std::overflow_error("Evaluation FFT padding length overflow");
    }
    return best;
}

struct WindowConvolution {
    std::array<std::int64_t, 2> shape;
    torch::Tensor window_fft;
    std::int64_t height, width, start_y, start_x, valid_height, valid_width;
    WindowConvolution(const torch::Tensor& window, const torch::Tensor& image)
        : height(image.size(0)), width(image.size(1)) {
        const auto wh = window.size(0), ww = window.size(1);
        if (!((height >= wh && width >= ww) || (height <= wh && width <= ww))) {
            throw std::invalid_argument("Valid convolution requires one input to cover the other in every dimension");
        }
        if (height > std::numeric_limits<std::int64_t>::max() - wh + 1 ||
            width > std::numeric_limits<std::int64_t>::max() - ww + 1) {
            throw std::overflow_error("Evaluation convolution dimensions overflow");
        }
        shape = {fast_length(height + wh - 1), fast_length(width + ww - 1)};
        start_y = std::min(height, wh) - 1;
        start_x = std::min(width, ww) - 1;
        valid_height = std::max(height, wh) - std::min(height, wh) + 1;
        valid_width = std::max(width, ww) - std::min(width, ww) + 1;
        window_fft = torch::fft_rfftn(window, shape, std::array<std::int64_t, 2>{0, 1}, "backward");
    }
    torch::Tensor operator()(const torch::Tensor& image) const {
        ProcessInterrupt::check();
        auto spectrum = torch::fft_rfftn(image, shape, std::array<std::int64_t, 2>{0, 1}, "backward");
        auto output = torch::fft_irfftn(window_fft * spectrum, shape,
                                       std::array<std::int64_t, 2>{0, 1}, "backward");
        return output.narrow(0, start_y, valid_height).narrow(1, start_x, valid_width).clone();
    }
};

torch::Tensor reflect_downsample(const torch::Tensor& image) {
    // ndimage.convolve with an even 2x2 filter/origin=0 takes current and
    // following pixels. Reflect repeats the edge; then [::2, ::2] is selected.
    const auto source = image.contiguous();
    const auto height = source.size(0), width = source.size(1);
    const auto rows = height / 2 + height % 2, columns = width / 2 + width % 2;
    auto result = torch::empty({rows, columns}, source.options());
    const auto* input = source.data_ptr<double>();
    auto* output = result.data_ptr<double>();
    for (std::int64_t y = 0; y < rows; ++y) {
        ProcessInterrupt::check();
        const auto y0 = y * 2, y1 = std::min(y0 + 1, height - 1);
        for (std::int64_t x = 0; x < columns; ++x) {
            const auto x0 = x * 2, x1 = std::min(x0 + 1, width - 1);
            double value = 0;
            value += input[y0 * width + x0] * 0.25;
            value += input[y0 * width + x1] * 0.25;
            value += input[y1 * width + x0] * 0.25;
            value += input[y1 * width + x1] * 0.25;
            output[y * columns + x] = value;
        }
    }
    return result;
}

void image_pair(const torch::Tensor& first, const torch::Tensor& second) {
    if (first.dim() != 2 || second.dim() != 2 || first.sizes() != second.sizes()) {
        throw std::invalid_argument("SSIM requires two 2D images with identical shapes");
    }
}

}  // namespace

double calc_psnr(const torch::Tensor& image, const torch::Tensor& reconstruction, double data_range) {
    const torch::NoGradGuard no_grad;
    ProcessInterrupt::check();
    const auto error = float64_cpu(image) - float64_cpu(reconstruction);
    const auto mse = array_mean(error.square());
    if (!std::isfinite(mse)) return -999.9;
    const auto value = mse > 1e-10 ? 10 * std::log10(data_range * data_range / mse) : 999.9;
    return std::min(value, 99.9);
}

torch::Tensor fspecial_gauss(std::int64_t size, double sigma) {
    const torch::NoGradGuard no_grad;
    if (size <= 0) return torch::empty({0, 0}, torch::TensorOptions().dtype(torch::kFloat64));
    if (size > std::numeric_limits<std::int64_t>::max() / size) {
        throw std::overflow_error("Gaussian window dimensions overflow");
    }
    auto window = torch::empty({size, size}, torch::TensorOptions().dtype(torch::kFloat64));
    auto* values = window.data_ptr<double>();
    const auto begin = 1 - size / 2 - size % 2;
    const auto square_sigma = sigma * sigma;
    const auto denominator = 2.0 * square_sigma;
    for (std::int64_t row = 0; row < size; ++row) {
        ProcessInterrupt::check();
        for (std::int64_t column = 0; column < size; ++column) {
            const auto y = begin + row, x = begin + column;
            values[row * size + column] = std::exp(-static_cast<double>(x * x + y * y) / denominator);
        }
    }
    const auto total = array_sum(window);
    for (std::int64_t index = 0; index < window.numel(); ++index) {
        volatile double normalized = values[index] / total;
        values[index] = normalized;
    }
    return window;
}

std::pair<torch::Tensor, torch::Tensor> calc_ssim(
    const torch::Tensor& image, const torch::Tensor& reconstruction, double data_range) {
    const torch::NoGradGuard no_grad;
    const auto first = float64_cpu(image), second = float64_cpu(reconstruction);
    image_pair(first, second);
    if (!first.numel()) {
        auto empty = torch::empty({0}, first.options());
        return {empty, empty.clone()};
    }
    const WindowConvolution convolve(fspecial_gauss(11, 1.5), first);
    const auto mu1 = convolve(first), mu2 = convolve(second);
    const auto mu1_sq = mu1 * mu1, mu2_sq = mu2 * mu2, mu12 = mu1 * mu2;
    const auto variance1 = convolve(first * first) - mu1_sq;
    const auto variance2 = convolve(second * second) - mu2_sq;
    const auto covariance = convolve(first * second) - mu12;
    const auto c1 = (0.01 * data_range) * (0.01 * data_range);
    const auto c2 = (0.03 * data_range) * (0.03 * data_range);
    const auto cs = (2.0 * covariance + c2) / (variance1 + variance2 + c2);
    const auto ssim = ((2.0 * mu12 + c1) * (2.0 * covariance + c2)) /
                      ((mu1_sq + mu2_sq + c1) * (variance1 + variance2 + c2));
    return {ssim, cs};
}

double calc_msssim(const torch::Tensor& image, const torch::Tensor& reconstruction, double data_range) {
    const torch::NoGradGuard no_grad;
    auto first = float64_cpu(image), second = float64_cpu(reconstruction);
    image_pair(first, second);
    if (first.size(0) < 88 || first.size(1) < 88) {
        throw std::invalid_argument("MS-SSIM requires height and width of at least 88");
    }
    const bool four_levels = first.size(0) < 176 || first.size(1) < 176;
    const std::vector<double> weights = four_levels
        ? std::vector<double>{0.0517, 0.3295, 0.3462, 0.2726}
        : std::vector<double>{0.0448, 0.2856, 0.3001, 0.2363, 0.1333};
    double cs_product = 1, last_ssim = 0;
    for (std::size_t level = 0; level < weights.size(); ++level) {
        ProcessInterrupt::check();
        const auto maps = calc_ssim(first, second, data_range);
        last_ssim = array_mean(maps.first);
        const auto cs = array_mean(maps.second);
        if (level + 1 != weights.size()) cs_product *= std::pow(cs, weights[level]);
        first = reflect_downsample(first);
        second = reflect_downsample(second);
    }
    // Do not clamp negative contrast or normalize weights: the reference does
    // neither, and fractional powers may therefore produce NaN.
    return cs_product * std::pow(last_ssim, weights.back());
}

double calc_msssim_rgb(const torch::Tensor& image, const torch::Tensor& reconstruction, double data_range) {
    if (image.dim() != 3 || reconstruction.dim() != 3 || image.size(0) < 3 || reconstruction.size(0) < 3) {
        throw std::invalid_argument("RGB MS-SSIM requires 3xHxW images");
    }
    double value = 0;
    for (std::int64_t channel = 0; channel < 3; ++channel) {
        value += calc_msssim(image.select(0, channel), reconstruction.select(0, channel), data_range);
    }
    return value / 3;
}

Json generate_log_json(std::uint64_t frame_count, std::uint64_t frame_pixels, double test_time,
    const std::vector<int>& frame_types, const std::vector<std::uint64_t>& bits,
    const std::vector<FrameQuality>& quality, bool verbose,
    std::optional<double> average_encoding_time, std::optional<double> average_decoding_time) {
    if (!frame_pixels || !frame_count || frame_count > quality.size() ||
        frame_count > frame_types.size() || frame_count > bits.size()) {
        throw std::invalid_argument("Invalid evaluation frame statistics");
    }
    const auto fields = quality.front().psnr.size();
    if (fields != 1 && fields != 4) throw std::invalid_argument("Evaluation quality requires RGB or YUV fields");
    static_assert(std::numeric_limits<std::size_t>::digits <= 64,
                  "UF report frame counts must fit UInt64");
    struct Sums {
        std::uint64_t count = 0;
        // At most UInt64 frames, each with UInt64 bits: the full sum fits
        // UInt128, including totals that Python accepts above UINT64_MAX.
        python_detail::RatioInteger bits = 0;
        std::array<double, 4> psnr{}, ssim{};
    };
    std::array<Sums, 2> sums;
    for (std::size_t index = 0; index < frame_count; ++index) {
        ProcessInterrupt::check();
        if (quality[index].psnr.size() != fields || quality[index].msssim.size() != fields) {
            throw std::invalid_argument("Evaluation quality fields differ between frames");
        }
        auto& target = sums[frame_types[index] == 0 ? 0 : 1];
        ++target.count;
        target.bits += bits[index];
        for (std::size_t field = 0; field < fields; ++field) {
            target.psnr[field] += quality[index].psnr[field];
            target.ssim[field] += quality[index].msssim[field];
        }
    }
    const std::array<std::string, 4> suffix{"", "_y", "_u", "_v"};
    Json log{{"frame_pixel_num", frame_pixels}, {"i_frame_num", sums[0].count}, {"p_frame_num", sums[1].count}};
    for (std::size_t type = 0; type < 2; ++type) {
        const std::string prefix = type ? "ave_p_frame_" : "ave_i_frame_";
        const auto& value = sums[type];
        // Python first divides two integers, then divides that Double result
        // by pixels. Empty groups instead retain integer zero in JSON.
        log[prefix + "bpp"] = value.count
            ? Json(python_detail::integer_ratio(value.bits, value.count) / frame_pixels) : Json(0);
        for (std::size_t field = 0; field < fields; ++field) {
            log[prefix + "psnr" + suffix[field]] = value.count ? Json(value.psnr[field] / value.count) : Json(0);
            log[prefix + "msssim" + suffix[field]] = value.count ? Json(value.ssim[field] / value.count) : Json(0);
        }
    }
    if (verbose) {
        log["frame_bpp"] = Json::array();
        for (const auto value : bits) {
            ProcessInterrupt::check();
            // This field uses NumPy array division, with floating operands.
            log["frame_bpp"].push_back(static_cast<double>(value) / frame_pixels);
        }
        for (std::size_t field = 0; field < fields; ++field) {
            log["frame_psnr" + suffix[field]] = Json::array();
            log["frame_msssim" + suffix[field]] = Json::array();
            if (field == 0) log["frame_type"] = frame_types;
        }
        for (std::size_t index = 0; index < quality.size(); ++index) {
            ProcessInterrupt::check();
            if (quality[index].psnr.size() != fields || quality[index].msssim.size() != fields) {
                throw std::invalid_argument("Evaluation quality fields differ between frames");
            }
            for (std::size_t field = 0; field < fields; ++field) {
                log["frame_psnr" + suffix[field]].push_back(quality[index].psnr[field]);
                log["frame_msssim" + suffix[field]].push_back(quality[index].msssim[field]);
            }
        }
    }
    log["test_time"] = test_time;
    // Python multiplies its integer count and pixels before true division.
    log["ave_all_frame_bpp"] = python_detail::integer_ratio(sums[0].bits + sums[1].bits,
        static_cast<python_detail::RatioInteger>(frame_count) * frame_pixels);
    log["ave_all_frame_psnr"] = (sums[0].psnr[0] + sums[1].psnr[0]) / frame_count;
    log["ave_all_frame_msssim"] = (sums[0].ssim[0] + sums[1].ssim[0]) / frame_count;
    if (average_encoding_time && average_decoding_time) {
        log["avg_frame_encoding_time"] = *average_encoding_time;
        log["avg_frame_decoding_time"] = *average_decoding_time;
    }
    for (std::size_t field = 1; field < fields; ++field) {
        log["ave_all_frame_psnr" + suffix[field]] = (sums[0].psnr[field] + sums[1].psnr[field]) / frame_count;
        log["ave_all_frame_msssim" + suffix[field]] = (sums[0].ssim[field] + sums[1].ssim[field]) / frame_count;
    }
    return log;
}

Json generate_log_json(std::uint64_t frame_pixels, double test_time,
    const std::vector<int>& frame_types, const std::vector<std::uint64_t>& bits,
    const std::vector<FrameQuality>& quality, bool verbose,
    std::optional<double> average_encoding_time, std::optional<double> average_decoding_time) {
    if (frame_types.size() != quality.size() || bits.size() != quality.size()) {
        throw std::invalid_argument("Invalid evaluation frame statistics");
    }
    return generate_log_json(static_cast<std::uint64_t>(quality.size()), frame_pixels, test_time,
        frame_types, bits, quality, verbose, average_encoding_time, average_decoding_time);
}

}  // namespace uf::evaluation
