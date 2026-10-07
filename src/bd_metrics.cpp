// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#include "uf/bd_metrics.hpp"
#include "uf/process.hpp"
#include "numpy_reductions.hpp"

#include <torch/torch.h>
#include <ATen/ops/linalg_lstsq.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>

namespace uf::evaluation {
namespace {

double multiply(double first, double second) {
    volatile double product = first * second;
    return product;
}

std::vector<double> log_rates(const std::vector<double>& rates) {
    std::vector<double> result;
    result.reserve(rates.size());
    for (const auto value : rates) result.push_back(std::log(value));
    return result;
}

using Polynomial = std::array<double, 4>;

Polynomial fit_cubic(const std::vector<double>& x, const std::vector<double>& y, std::size_t warning_site) {
    ProcessInterrupt::check();
    if (x.empty() || x.size() != y.size() ||
        x.size() > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
        throw std::invalid_argument("BD metric requires equal nonempty rate and quality lists");
    }
    const auto count = static_cast<std::int64_t>(x.size());
    auto matrix = torch::empty({count, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
    auto target = torch::empty({count}, matrix.options());
    auto* values = matrix.data_ptr<double>();
    auto* expected = target.data_ptr<double>();
    std::array<double, 4> scales{};
    for (std::size_t row = 0; row < x.size(); ++row) {
        if (!std::isfinite(x[row]) || !std::isfinite(y[row])) {
            throw std::invalid_argument("BD polynomial input contains nonfinite values");
        }
        // np.polyfit first applies asarray(...)+0.0 to both coordinates,
        // including its conversion of negative zero to positive zero.
        const auto coordinate = x[row] + 0.0;
        const auto square = multiply(coordinate, coordinate);
        values[row * 4] = multiply(square, coordinate);
        values[row * 4 + 1] = square;
        values[row * 4 + 2] = coordinate;
        values[row * 4 + 3] = 1.0;
        expected[row] = y[row] + 0.0;
        for (std::size_t column = 0; column < scales.size(); ++column) {
            scales[column] += multiply(values[row * 4 + column], values[row * 4 + column]);
        }
    }
    // NumPy polyfit normalizes each Vandermonde column before GELSD and uses
    // len(x)*eps as its singular-value cutoff.
    for (auto& scale : scales) {
        scale = std::sqrt(scale);
        if (!std::isfinite(scale) || scale == 0) throw std::invalid_argument("Invalid BD polynomial basis norm");
    }
    for (std::size_t row = 0; row < x.size(); ++row) {
        for (std::size_t column = 0; column < scales.size(); ++column) {
            values[row * 4 + column] /= scales[column];
        }
    }
    const auto fitted = torch::linalg_lstsq(matrix, target,
        static_cast<double>(count) * std::numeric_limits<double>::epsilon(), c10::string_view("gelsd"));
    static std::array<std::atomic<bool>, 4> warned{};
    if (std::get<2>(fitted).item<std::int64_t>() != 4 && !warned.at(warning_site).exchange(true)) {
        std::cerr << "warning: BD metric polynomial fit may be poorly conditioned\n";
    }
    const auto solution = std::get<0>(fitted).contiguous();
    const auto* coefficients = solution.data_ptr<double>();
    Polynomial result;
    for (std::size_t column = 0; column < result.size(); ++column) result[column] = coefficients[column] / scales[column];
    return result;
}

double polynomial_integral(const Polynomial& coefficients, double value) {
    double result = coefficients[0] / 4.0;
    result = multiply(result, value) + coefficients[1] / 3.0;
    result = multiply(result, value) + coefficients[2] / 2.0;
    result = multiply(result, value) + coefficients[3];
    return multiply(result, value);
}

int sign(double value) { return value > 0 ? 1 : value < 0 ? -1 : 0; }

double endpoint_derivative(double h0, double h1, double slope0, double slope1) {
    auto value = (multiply(2.0 * h0 + h1, slope0) - multiply(h0, slope1)) / (h0 + h1);
    if (sign(value) != sign(slope0)) value = 0;
    else if (sign(slope0) != sign(slope1) && std::abs(value) > 3.0 * std::abs(slope0)) value = 3.0 * slope0;
    return value;
}

class Pchip {
public:
    Pchip(std::vector<double> x, std::vector<double> y) : x_(std::move(x)) {
        if (x_.size() < 2 || x_.size() != y.size()) throw std::invalid_argument("PCHIP requires at least two coordinates");
        // Preserve bd-metric's independent np.sort calls, including the
        // reference behavior for nonmonotonic input quality/rate pairing.
        std::sort(x_.begin(), x_.end());
        std::sort(y.begin(), y.end());
        const auto count = x_.size();
        std::vector<double> h(count - 1), slopes(count - 1), derivatives(count, 0);
        for (std::size_t index = 0; index < count; ++index) {
            if (!std::isfinite(x_[index]) || !std::isfinite(y[index])) {
                throw std::invalid_argument("PCHIP coordinates must be finite");
            }
            if (index + 1 < count) {
                h[index] = x_[index + 1] - x_[index];
                if (h[index] <= 0) throw std::invalid_argument("PCHIP x coordinates must be strictly increasing");
                slopes[index] = (y[index + 1] - y[index]) / h[index];
            }
        }
        if (count == 2) derivatives[0] = derivatives[1] = slopes[0];
        else {
            for (std::size_t index = 1; index + 1 < count; ++index) {
                if (slopes[index - 1] == 0 || slopes[index] == 0 || sign(slopes[index - 1]) != sign(slopes[index])) continue;
                const auto w1 = 2.0 * h[index] + h[index - 1];
                const auto w2 = h[index] + 2.0 * h[index - 1];
                const auto inverse = (w1 / slopes[index - 1] + w2 / slopes[index]) / (w1 + w2);
                derivatives[index] = 1.0 / inverse;
            }
            derivatives.front() = endpoint_derivative(h[0], h[1], slopes[0], slopes[1]);
            derivatives.back() = endpoint_derivative(h[count - 2], h[count - 3], slopes[count - 2], slopes[count - 3]);
        }
        coefficients_.reserve(count - 1);
        for (std::size_t index = 0; index + 1 < count; ++index) {
            const auto t = (derivatives[index] + derivatives[index + 1] - 2.0 * slopes[index]) / h[index];
            coefficients_.push_back({t / h[index], (slopes[index] - derivatives[index]) / h[index] - t,
                                     derivatives[index], y[index]});
        }
    }
    double operator()(double value) const {
        const auto found = std::upper_bound(x_.begin(), x_.end(), value);
        const auto index = found == x_.begin() ? 0u :
            std::min(static_cast<std::size_t>(found - x_.begin() - 1), coefficients_.size() - 1);
        const auto offset = value - x_[index];
        const auto& c = coefficients_[index];
        auto result = multiply(c[0], offset) + c[1];
        result = multiply(result, offset) + c[2];
        return multiply(result, offset) + c[3];
    }
private:
    std::vector<double> x_;
    std::vector<Polynomial> coefficients_;
};

double integrate_samples(const Pchip& curve, double begin, double end) {
    constexpr std::size_t points = 100;
    const auto delta = end - begin;
    const auto step = delta / (points - 1);
    std::array<double, points> values;
    for (std::size_t index = 0; index < points; ++index) {
        ProcessInterrupt::check();
        const auto displacement = step == 0 ? multiply(static_cast<double>(index) / (points - 1), delta)
                                             : multiply(static_cast<double>(index), step);
        const auto sample = index + 1 == points ? end : begin + displacement;
        values[index] = curve(sample);
    }
    std::array<double, points - 1> terms;
    for (std::size_t index = 0; index < terms.size(); ++index) {
        terms[index] = multiply(step, values[index + 1] + values[index]) / 2.0;
    }
    // NumPy trapz reduces the contiguous, materialized Float64 trapezoids
    // with its eight-accumulator pairwise sum, adding to the +0 identity.
    return 0.0 + numpy_detail::pairwise_sum(terms.data(), static_cast<std::int64_t>(terms.size()));
}

double difference(const std::vector<double>& x1, const std::vector<double>& y1,
                  const std::vector<double>& x2, const std::vector<double>& y2,
                  bool piecewise, std::size_t warning_site) {
    const torch::NoGradGuard no_grad;
    // The installed reference computes both polynomial fits even when the
    // piecewise branch is selected; retain its validation/rank-warning behavior.
    const auto first = fit_cubic(x1, y1, warning_site), second = fit_cubic(x2, y2, warning_site + 1);
    const auto begin = std::max(*std::min_element(x1.begin(), x1.end()), *std::min_element(x2.begin(), x2.end()));
    const auto end = std::min(*std::max_element(x1.begin(), x1.end()), *std::max_element(x2.begin(), x2.end()));
    double integral1, integral2;
    if (piecewise) {
        integral1 = integrate_samples(Pchip(x1, y1), begin, end);
        integral2 = integrate_samples(Pchip(x2, y2), begin, end);
    } else {
        integral1 = polynomial_integral(first, end) - polynomial_integral(first, begin);
        integral2 = polynomial_integral(second, end) - polynomial_integral(second, begin);
    }
    // The reference neither rejects reversed overlap nor special-cases a
    // zero-length interval. Extrapolation and IEEE NaN/Inf remain visible.
    return (integral2 - integral1) / (end - begin);
}

}  // namespace

double bd_rate(const std::vector<double>& base_rate, const std::vector<double>& base_quality,
               const std::vector<double>& rate, const std::vector<double>& quality, bool piecewise) {
    return (std::exp(difference(base_quality, log_rates(base_rate), quality, log_rates(rate), piecewise, 0)) - 1) * 100;
}

double bd_psnr(const std::vector<double>& base_rate, const std::vector<double>& base_quality,
               const std::vector<double>& rate, const std::vector<double>& quality, bool piecewise) {
    return difference(log_rates(base_rate), base_quality, log_rates(rate), quality, piecewise, 2);
}

}  // namespace uf::evaluation
