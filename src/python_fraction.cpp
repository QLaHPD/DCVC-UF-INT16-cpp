// Adapted from CPython 3.10.20 fractions.py, numbers.Rational.__float__,
// floatobject.c integer ratios and longobject.c integer true division.
// See licenses/CPython-3.10.20.txt.
#include "uf/python_numeric.hpp"

#include <openssl/bn.h>
#include <openssl/crypto.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace uf {
namespace {

static_assert(std::numeric_limits<double>::is_iec559 &&
              std::numeric_limits<double>::radix == 2 &&
              std::numeric_limits<double>::digits == 53 &&
              std::numeric_limits<double>::max_exponent == 1024 &&
              std::numeric_limits<double>::min_exponent == -1021,
              "UF fraction conversion requires IEEE binary64");

using Big = std::unique_ptr<BIGNUM, decltype(&BN_free)>;
using Context = std::unique_ptr<BN_CTX, decltype(&BN_CTX_free)>;

void checked(int success) {
    if (!success) throw std::runtime_error("Cannot compute UF rational number");
}

Big number() {
    Big result(BN_new(), &BN_free);
    if (!result) throw std::bad_alloc();
    return result;
}

Big decimal(std::string_view digits) {
    check_python_integer_digits(digits.size());
    if (digits.empty() || digits.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::length_error("Fraction integer exceeds native decimal conversion range");
    }
    const std::string text(digits);
    BIGNUM* raw = nullptr;
    const auto used = BN_dec2bn(&raw, text.c_str());
    Big result(raw, &BN_free);
    if (!result || used != static_cast<int>(digits.size())) {
        throw std::runtime_error("Cannot parse UF rational integer");
    }
    return result;
}

Big power_ten(const BIGNUM* exponent, BN_CTX* context) {
    auto base = number(), result = number();
    checked(BN_set_word(base.get(), 10));
    checked(BN_exp(result.get(), base.get(), exponent, context));
    return result;
}

bool digit(char value) { return value >= '0' && value <= '9'; }

struct Rational {
    Big numerator{number()}, denominator{number()};
};

Rational parse(std::string_view value) {
    const auto normalized = python_numeric_ascii(python_strip(value));
    value = normalized;
    const auto invalid = []() -> void { throw std::invalid_argument("Invalid literal for Fraction"); };
    bool negative = false;
    if (!value.empty() && (value.front() == '+' || value.front() == '-')) {
        negative = value.front() == '-';
        value.remove_prefix(1);
    }
    if (value.empty() || (!digit(value.front()) &&
        !(value.front() == '.' && value.size() > 1 && digit(value[1])))) invalid();
    std::size_t position = 0;
    while (position < value.size() && digit(value[position])) ++position;
    const auto whole = value.substr(0, position);
    std::string_view denominator, fractional, exponent;
    bool exponent_negative = false;
    if (position < value.size() && value[position] == '/') {
        const auto start = ++position;
        while (position < value.size() && digit(value[position])) ++position;
        if (start == position || position != value.size()) invalid();
        denominator = value.substr(start);
    } else {
        if (position < value.size() && value[position] == '.') {
            const auto start = ++position;
            while (position < value.size() && digit(value[position])) ++position;
            fractional = value.substr(start, position - start);
        }
        if (position < value.size() && (value[position] == 'e' || value[position] == 'E')) {
            ++position;
            if (position < value.size() && (value[position] == '+' || value[position] == '-')) {
                exponent_negative = value[position++] == '-';
            }
            const auto start = position;
            while (position < value.size() && digit(value[position])) ++position;
            if (start == position) invalid();
            exponent = value.substr(start, position - start);
        }
        if (position != value.size()) invalid();
    }
    // Fraction's regular expression validates the full literal before any
    // integer conversion or digit-limit check.
    Rational result;
    result.numerator = decimal(whole.empty() ? std::string_view("0") : whole);
    Context context(BN_CTX_new(), &BN_CTX_free);
    if (!context) throw std::bad_alloc();
    if (!denominator.empty()) result.denominator = decimal(denominator);
    else {
        checked(BN_set_word(result.denominator.get(), 1));
        if (!fractional.empty()) {
            auto places = number();
            if (fractional.size() > std::numeric_limits<BN_ULONG>::max()) {
                throw std::length_error("Fraction decimal scale exceeds native range");
            }
            checked(BN_set_word(places.get(), static_cast<BN_ULONG>(fractional.size())));
            result.denominator = power_ten(places.get(), context.get());
            auto tail = decimal(fractional);
            checked(BN_mul(result.numerator.get(), result.numerator.get(), result.denominator.get(), context.get()));
            checked(BN_add(result.numerator.get(), result.numerator.get(), tail.get()));
        }
        if (!exponent.empty()) {
            auto places = decimal(exponent);
            auto scale = power_ten(places.get(), context.get());
            auto* target = exponent_negative ? result.denominator.get() : result.numerator.get();
            checked(BN_mul(target, target, scale.get(), context.get()));
        }
    }
    if (BN_is_zero(result.denominator.get())) throw std::domain_error("Fraction denominator is zero");
    auto divisor = number();
    checked(BN_gcd(divisor.get(), result.numerator.get(), result.denominator.get(), context.get()));
    auto numerator = number(), reduced_denominator = number();
    checked(BN_div(numerator.get(), nullptr, result.numerator.get(), divisor.get(), context.get()));
    checked(BN_div(reduced_denominator.get(), nullptr, result.denominator.get(), divisor.get(), context.get()));
    result.numerator = std::move(numerator);
    result.denominator = std::move(reduced_denominator);
    BN_set_negative(result.numerator.get(), negative);
    return result;
}

std::uint64_t magnitude(const BIGNUM* value) {
    std::array<unsigned char, 8> bytes{};
    if (BN_bn2binpad(value, bytes.data(), static_cast<int>(bytes.size())) != static_cast<int>(bytes.size())) {
        throw std::overflow_error("Fraction significand exceeds native range");
    }
    std::uint64_t result = 0;
    for (const auto byte : bytes) result = (result << 8) | byte;
    return result;
}

struct FreeText { void operator()(char* text) const { OPENSSL_free(text); } };

std::string integer_text(const BIGNUM* value) {
    std::unique_ptr<char, FreeText> text(BN_bn2dec(value));
    if (!text) throw std::bad_alloc();
    std::string result(text.get());
    check_python_integer_digits(result.size() - (result.front() == '-'));
    return result;
}

std::string fraction_text(const Rational& rational) {
    auto result = integer_text(rational.numerator.get());
    if (!BN_is_one(rational.denominator.get())) {
        result += '/';
        result += integer_text(rational.denominator.get());
    }
    return result;
}

}  // namespace

std::string python_fraction_text(std::string_view value) {
    return fraction_text(parse(value));
}

std::string python_fraction_text(double value) {
    if (std::isinf(value)) throw std::overflow_error("cannot convert Infinity to integer ratio");
    if (std::isnan(value)) throw std::invalid_argument("cannot convert NaN to integer ratio");
    if (value == 0.0) return "0";
    int exponent;
    const auto part = std::frexp(std::abs(value), &exponent);
    // Scaling the binary significand is exact, including subnormal inputs.
    auto significand = static_cast<std::uint64_t>(std::ldexp(part, 53));
    exponent -= 53;
    while (!(significand & 1u)) { significand >>= 1; ++exponent; }
    Rational rational;
    rational.numerator = decimal(std::to_string(significand));
    checked(BN_set_word(rational.denominator.get(), 1));
    auto shifted = number();
    if (exponent >= 0) {
        checked(BN_lshift(shifted.get(), rational.numerator.get(), exponent));
        rational.numerator = std::move(shifted);
    } else {
        checked(BN_lshift(shifted.get(), rational.denominator.get(), -exponent));
        rational.denominator = std::move(shifted);
    }
    BN_set_negative(rational.numerator.get(), std::signbit(value));
    return fraction_text(rational);
}

double python_fraction_double(std::string_view value) {
    auto rational = parse(value);
    const bool negative = BN_is_negative(rational.numerator.get());
    BN_set_negative(rational.numerator.get(), 0);
    const auto signed_result = [negative](double result) { return negative ? -result : result; };
    if (BN_is_zero(rational.numerator.get())) return 0.0;
    const int numerator_bits = BN_num_bits(rational.numerator.get());
    const int denominator_bits = BN_num_bits(rational.denominator.get());
    // CPython's small-integer fast path uses exact operands and one division.
    if (numerator_bits <= 53 && denominator_bits <= 53) {
        return signed_result(static_cast<double>(magnitude(rational.numerator.get())) /
                             static_cast<double>(magnitude(rational.denominator.get())));
    }
    const int difference = numerator_bits - denominator_bits;
    const auto overflow = []() -> void {
        throw std::overflow_error("integer division result too large for a float");
    };
    if (difference > 1024) overflow();
    if (difference < -1075) return signed_result(0.0);
    const int shift = std::max(difference, -1021) - 55;
    Context context(BN_CTX_new(), &BN_CTX_free);
    if (!context) throw std::bad_alloc();
    auto shifted = number(), quotient = number(), remainder = number();
    if (shift <= 0) {
        checked(BN_lshift(shifted.get(), rational.numerator.get(), -shift));
        checked(BN_div(quotient.get(), remainder.get(), shifted.get(), rational.denominator.get(), context.get()));
    } else {
        // Shifting the denominator keeps all discarded numerator bits in the
        // exact remainder, equivalent to CPython's shift/division sticky flag.
        checked(BN_lshift(shifted.get(), rational.denominator.get(), shift));
        checked(BN_div(quotient.get(), remainder.get(), rational.numerator.get(), shifted.get(), context.get()));
    }
    const int bits = BN_num_bits(quotient.get());
    const int extra = std::max(bits, -1021 - shift) - 53;
    if (extra != 2 && extra != 3) throw std::runtime_error("Invalid fraction rounding geometry");
    auto rounded = magnitude(quotient.get());
    const auto mask = std::uint64_t{1} << (extra - 1);
    const auto low = rounded | (BN_is_zero(remainder.get()) ? 0u : 1u);
    if ((low & mask) && (low & (3 * mask - 1))) rounded += mask;
    rounded &= ~(2 * mask - 1);
    if (shift + bits >= 1024 && (shift + bits > 1024 || rounded == (std::uint64_t{1} << bits))) overflow();
    return signed_result(std::ldexp(static_cast<double>(rounded), shift));
}

}  // namespace uf
