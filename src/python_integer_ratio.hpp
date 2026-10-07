#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace uf::python_detail {

// The Linux GCC/Clang UF build supports a two-word unsigned integer. Keep
// bit totals and count * pixels exact, as Python does with its integers.
using RatioInteger = unsigned __int128;

inline int ratio_bit_length(RatioInteger value) {
    int result = 0;
    for (; value; value >>= 1) ++result;
    return result;
}

// Correctly rounded nonnegative integer true division within the report's
// UInt128 numerator / denominator domain. Like CPython long_true_divide,
// round the exact ratio to an even binary64 significand, then scale exactly.
// Nonzero ratios here are normal binary64 values (exponents -128 through 127).
inline double integer_ratio(RatioInteger numerator, RatioInteger denominator) {
    static_assert(std::numeric_limits<double>::is_iec559 &&
                  std::numeric_limits<double>::radix == 2 &&
                  std::numeric_limits<double>::digits == 53,
                  "UF report ratios require IEEE binary64");
    if (!denominator) throw std::invalid_argument("Integer ratio division by zero");
    if (!numerator) return 0.0;

    int exponent = ratio_bit_length(numerator) - ratio_bit_length(denominator);
    if (exponent >= 0) {
        if (numerator < (denominator << exponent)) --exponent;
    } else {
        if ((numerator << -exponent) < denominator) --exponent;
    }

    const auto normalized_denominator = exponent >= 0 ? denominator << exponent : denominator;
    // When the normalized numerator needs 129 bits, unsigned wrap followed by
    // subtraction still gives the exact remainder, which is < denominator.
    // A shift of 128 has zero low 128 bits; do not perform that undefined shift.
    const auto normalized_numerator = exponent >= 0 ? numerator :
        (exponent == -128 ? RatioInteger{0} : numerator << -exponent);
    auto remainder = normalized_numerator - normalized_denominator;
    std::uint64_t significand = 1;
    for (int bit = 0; bit < 52; ++bit) {
        const bool carry = (remainder >> 127) != 0;
        remainder <<= 1;
        const bool next = carry || remainder >= normalized_denominator;
        if (next) remainder -= normalized_denominator;
        significand = (significand << 1) | static_cast<std::uint64_t>(next);
    }

    const bool carry = (remainder >> 127) != 0;
    const auto twice_remainder = remainder << 1;
    if (carry || twice_remainder > normalized_denominator ||
        (twice_remainder == normalized_denominator && (significand & 1))) {
        ++significand;
    }
    // The rounded significand is <= 2^53, hence this conversion and scaling
    // are exact; no second rounding occurs, including a significand carry.
    return std::ldexp(static_cast<double>(significand), exponent - 52);
}

}  // namespace uf::python_detail
