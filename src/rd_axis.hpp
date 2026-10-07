// Linear limits, AutoLocator and ScalarFormatter adapted from Matplotlib 3.10.9.
// Copyright (c) 2012- Matplotlib Development Team; All Rights Reserved.
// License and modification summary: licenses/Matplotlib.txt, licenses/README.md.
// Float divmod follows CPython 3.10; retained license: licenses/CPython-3.10.20.txt.
// Decimal refinement adapts NumPy 1.26.1 calculation.c; license: licenses/NumPy.txt.
#pragma once

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace uf::evaluation::plot_detail {

using Limits = std::pair<double, double>;

inline Limits nonsingular(double first, double second, double expander, double tiny = 1e-15) {
    if (!std::isfinite(first) || !std::isfinite(second)) return {-expander, expander};
    if (second < first) std::swap(first, second);
    const auto maximum = std::max(std::abs(first), std::abs(second));
    if (maximum < (1e6 / tiny) * std::numeric_limits<double>::min()) {
        first = -expander; second = expander;
    } else if (second - first <= maximum * tiny) {
        first -= expander * std::abs(first);
        second += expander * std::abs(second);
    }
    return {first, second};
}

inline Limits linear_limits(double first, double second) {
    auto limits = nonsingular(first, second, 0.05);
    auto margin = (limits.second - limits.first) * 0.05;
    if (!std::isfinite(margin)) margin = 0;
    limits.first -= margin; limits.second += margin;
    // AutoLocator.view_limits in the default axes.autolimit_mode='data'.
    return nonsingular(limits.first, limits.second, 1e-12, 1e-13);
}

inline Limits zero_left(double right) {
    const bool reverse = 0.0 > right;
    auto limits = nonsingular(0, right, 0.05);
    if (reverse) std::swap(limits.first, limits.second);
    return limits;
}

inline bool tick_visible(double value, Limits limits) {
    if (limits.first > limits.second) std::swap(limits.first, limits.second);
    const auto slop = (limits.second - limits.first) * 1e-10;
    return limits.first - slop <= value && value <= limits.second + slop;
}

// Positive divisor; retain float divmod's snapped quotient and signed remainder
// rather than floor(x / step), which differs close to a multiple of the step.
inline Limits positive_divmod(double value, double step) {
    auto remainder = std::fmod(value, step);
    auto quotient = (value - remainder) / step;
    if (remainder) {
        if (remainder < 0) { remainder += step; quotient -= 1; }
    } else remainder = 0.0;
    double floor_quotient;
    if (quotient) {
        floor_quotient = std::floor(quotient);
        if (quotient - floor_quotient > 0.5) floor_quotient += 1;
    } else floor_quotient = std::copysign(0.0, value / step);
    return {floor_quotient, remainder};
}

inline bool close_edge(double fraction, double edge, double step, double offset) {
    double tolerance = 1e-10;
    if (offset > 0) {
        const auto digits = std::log10(offset / step);
        tolerance = std::min(0.4999, std::max(1e-10, std::pow(10.0, digits - 12)));
    }
    return std::abs(fraction - edge) < tolerance;
}

inline std::vector<double> auto_ticks(Limits view, double axis_pixels, bool horizontal) {
    const auto limits = nonsingular(view.first, view.second, 1e-13, 1e-14);
    // X/Y Axis.get_tick_space: 3/2 times the default 10-point tick font size.
    const auto bins = std::clamp(std::floor(axis_pixels * 72.0 / 100.0 /
        (10 * (horizontal ? 3 : 2))), 1.0, 9.0);
    const auto span = std::abs(limits.second - limits.first);
    const auto mean = (limits.second + limits.first) / 2;
    const auto offset = std::abs(mean) / span < 100 ? 0.0 :
        std::copysign(std::pow(10.0, std::floor(std::log10(std::abs(mean)))), mean);
    const auto scale = std::pow(10.0, std::floor(std::log10(span / bins)));
    if (!std::isfinite(scale) || scale <= 0 || !std::isfinite(offset))
        throw std::runtime_error("Cannot locate ticks for nonfinite linear scale");
    const auto minimum = limits.first - offset, maximum = limits.second - offset;
    const auto raw_step = (maximum - minimum) / bins;
    // AutoLocator [1,2,2.5,5,10], extended by MaxNLocator._staircase.
    constexpr std::array<double, 10> staircase{0.1, 0.2, 0.25, 0.5, 1, 2, 2.5, 5, 10, 20};
    std::size_t selected = staircase.size() - 1;
    for (std::size_t index = 0; index < staircase.size(); ++index) {
        if (staircase[index] * scale >= raw_step) { selected = index; break; }
    }
    std::vector<double> ticks;
    for (auto index = selected + 1; index > 0;) {
        const auto step = staircase[--index] * scale;
        if (!std::isfinite(step) || step <= 0) throw std::runtime_error("Invalid linear tick step");
        const auto base = positive_divmod(minimum, step).first * step;
        const auto lower = positive_divmod(minimum - base, step);
        const auto upper = positive_divmod(maximum - base, step);
        const auto low = lower.first + (close_edge(lower.second / step, 1, step, std::abs(offset)) ? 1 : 0);
        const auto high = upper.first + (close_edge(upper.second / step, 0, step, std::abs(offset)) ? 0 : 1);
        const auto count = high - low + 1;
        if (!std::isfinite(count) || count < 0 || count >= static_cast<double>(ticks.max_size()))
            throw std::length_error("Linear tick count is too large");
        ticks.resize(static_cast<std::size_t>(count));
        std::size_t visible = 0;
        for (std::size_t tick = 0; tick < ticks.size(); ++tick) {
            // Separate multiplication/addition matches NumPy's array operations.
            const auto value = (low + static_cast<double>(tick)) * step + base;
            if (value >= minimum && value <= maximum) ++visible;
            ticks[tick] = value + offset;
        }
        if (visible >= 2) break;
    }
    return ticks;
}

inline std::string unicode_minus(std::string value) {
    for (std::size_t index = 0; (index = value.find('-', index)) != std::string::npos; index += 3)
        value.replace(index, 1, "\xE2\x88\x92");
    return value;
}

inline std::string fixed_label(double value, int decimals) {
    static_assert(std::numeric_limits<double>::is_iec559 &&
                  std::numeric_limits<double>::digits == 53 &&
                  std::numeric_limits<double>::max_exponent == 1024,
                  "Linear tick formatting requires IEEE binary64");
    // Binary64's 309 integer digits plus at most 327 fractional digits fit.
    std::array<char, 1024> buffer{};
    const auto result = std::to_chars(buffer.data(), buffer.data() + buffer.size(),
                                      value, std::chars_format::fixed, decimals);
    if (result.ec != std::errc{}) throw std::runtime_error("Cannot format linear tick label");
    return std::string(buffer.data(), static_cast<std::size_t>(result.ptr - buffer.data()));
}

inline double decimal_double(std::string_view text) {
    double value;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size())
        throw std::runtime_error("Cannot parse rounded linear tick value");
    return value;
}

inline double numpy_round_scale(int decimals) {
    // NumPy power_of_ten uses a small table, then repeated Double products.
    constexpr std::array<double, 9> small{1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8};
    if (decimals < 9) return small[static_cast<std::size_t>(decimals)];
    double scale = 1e9;
    for (int exponent = 9; exponent < decimals; ++exponent) scale *= 10;
    return scale;
}

inline std::string format_offset(double value) {
    const auto exponent = static_cast<int>(std::floor(std::log10(std::abs(value))));
    // format_data uses 10**integer_exponent. Positive powers are exact Python
    // integers, rounded on conversion for the floating division; negative
    // powers take float pow. Float-base powers elsewhere use std::pow.
    const auto power = exponent < 0 ? std::pow(10.0, exponent) :
        decimal_double("1e" + std::to_string(exponent));
    const auto normalized = value / power;
    // _compute_offset produces a NumPy Double scalar. Under the installed
    // default legacy promotion, powers beyond UInt64 (10**20 and above)
    // fall back to object/Python float arithmetic. The 10**19 power promotes
    // through UInt64 to Double; weak modes retain Double for larger integers.
    // Other powers retain NumPy's scalar round -> array round. Promotion
    // warnings/error policy remains under review. No interpreter is invoked.
    const char* environment = std::getenv("NPY_PROMOTION_STATE");
    const bool weak = environment && (std::string_view(environment) == "weak" ||
                                      std::string_view(environment) == "weak_and_warn");
    const auto factor = numpy_round_scale(10);
    const auto significand = exponent >= 20 && !weak ? decimal_double(fixed_label(normalized, 10)) :
        std::nearbyint(normalized * factor) / factor;
    std::string text;
    if (std::fmod(significand, 1.0) == 0) text = fixed_label(significand == 0 ? 0.0 : significand, 0);
    else {
        std::array<char, 64> buffer{};
        const auto result = std::to_chars(buffer.data(), buffer.data() + buffer.size(),
                                          significand, std::chars_format::general, 10);
        if (result.ec != std::errc{}) throw std::runtime_error("Cannot format linear tick offset");
        text.assign(buffer.data(), static_cast<std::size_t>(result.ptr - buffer.data()));
    }
    if (exponent) text += "e" + std::to_string(exponent);
    return text;
}

struct ScalarLabels {
    std::vector<std::string> ticks;
    std::string annotation;
};

inline ScalarLabels scalar_labels(const std::vector<double>& ticks, Limits view) {
    ScalarLabels result;
    if (ticks.empty()) return result;
    if (view.first > view.second) std::swap(view.first, view.second);
    std::vector<double> visible;
    for (const auto tick : ticks) if (view.first <= tick && tick <= view.second) visible.push_back(tick);
    double offset = 0;
    int order = 0;
    if (!visible.empty()) {
        const auto endpoints = std::minmax_element(visible.begin(), visible.end());
        const auto minimum = *endpoints.first, maximum = *endpoints.second;
        if (minimum != maximum && !(minimum <= 0 && maximum >= 0)) {
            const auto small = std::min(std::abs(minimum), std::abs(maximum));
            const auto large = std::max(std::abs(minimum), std::abs(maximum));
            const auto start = static_cast<int>(std::ceil(std::log10(large)));
            const auto divided = [](double number, int exponent) {
                const auto step = std::pow(10.0, exponent);
                if (step <= 0) throw std::runtime_error("Linear tick offset scale underflow");
                return positive_divmod(number, step).first;
            };
            auto differing = start;
            while (divided(small, differing) == divided(large, differing)) --differing;
            auto exponent = differing + 1;
            if ((large - small) / std::pow(10.0, exponent) <= 1e-2) {
                differing = start;
                while (divided(large, differing) - divided(small, differing) <= 1) --differing;
                exponent = differing + 1;
            }
            const auto leading = divided(large, exponent);
            // Default offset_threshold=4: save at least four printed digits.
            if (leading >= 1e3) offset = std::copysign(1.0, minimum) * leading * std::pow(10.0, exponent);
        }
        const auto magnitude = std::max(std::abs(minimum), std::abs(maximum));
        const auto exponent = offset ? std::floor(std::log10(view.second - view.first)) :
            magnitude == 0 ? 0.0 : std::floor(std::log10(magnitude));
        // Default axes.formatter.limits=(-5, 6).
        if (exponent <= -5 || exponent >= 6) order = static_cast<int>(exponent);
    }
    const auto scale = std::pow(10.0, order);
    std::vector<double> adjusted;
    adjusted.reserve(ticks.size() + (ticks.size() < 2 ? 2 : 0));
    for (const auto tick : ticks) adjusted.push_back((tick - offset) / scale);
    if (ticks.size() < 2) {
        adjusted.push_back((view.first - offset) / scale);
        adjusted.push_back((view.second - offset) / scale);
    }
    const auto endpoints = std::minmax_element(adjusted.begin(), adjusted.end());
    auto range = *endpoints.second - *endpoints.first;
    if (range == 0) range = std::max(std::abs(*endpoints.first), std::abs(*endpoints.second));
    if (range == 0) range = 1;
    if (!std::isfinite(range) || range <= 0) throw std::runtime_error("Invalid linear tick formatting range");
    adjusted.resize(ticks.size());
    const auto range_order = static_cast<int>(std::floor(std::log10(range)));
    auto decimals = std::max(0, 3 - range_order);
    const auto threshold = 1e-3 * std::pow(10.0, range_order);
    while (decimals >= 0) {
        const auto factor = numpy_round_scale(decimals);
        double deviation = 0;
        for (const auto value : adjusted) {
            const auto rounded = decimals == 0 ? std::nearbyint(value) :
                std::nearbyint(value * factor) / factor;
            const auto error = std::abs(value - rounded);
            if (std::isnan(error)) { deviation = error; break; }
            deviation = std::max(deviation, error);
        }
        if (deviation < threshold) --decimals;
        else break;
    }
    ++decimals;
    result.ticks.reserve(adjusted.size());
    for (const auto value : adjusted)
        result.ticks.push_back(unicode_minus(fixed_label(std::abs(value) < 1e-8 ? 0.0 : value, decimals)));
    if (order) result.annotation = "1e" + std::to_string(order);
    if (offset) result.annotation += (offset > 0 ? "+" : "") + format_offset(offset);
    result.annotation = unicode_minus(std::move(result.annotation));
    return result;
}

}  // namespace uf::evaluation::plot_detail
