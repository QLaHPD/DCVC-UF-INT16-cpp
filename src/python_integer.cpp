#include "uf/python_integer.hpp"
#include "uf/python_numeric.hpp"

#include <cstddef>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace uf {
namespace {

bool ascii_space(char value) {
    return value == ' ' || value == '\t' || value == '\n' ||
           value == '\r' || value == '\v' || value == '\f';
}

bool ascii_digit(char value) {
    return value >= '0' && value <= '9';
}

std::string add_magnitudes(std::string_view first, std::string_view second) {
    const auto count = first.size() > second.size() ? first.size() : second.size();
    std::string result;
    if (count >= result.max_size()) throw std::length_error("Integer sum is too large");
    result.resize(count + 1, '0');
    unsigned carry = 0;
    for (std::size_t offset = 0; offset < count; ++offset) {
        const auto a = offset < first.size() ? first[first.size() - offset - 1] - '0' : 0;
        const auto b = offset < second.size() ? second[second.size() - offset - 1] - '0' : 0;
        const auto total = a + b + carry;
        result[count - offset] = static_cast<char>('0' + total % 10);
        carry = total / 10;
    }
    if (carry) result[0] = static_cast<char>('0' + carry);
    else result.erase(0, 1);
    return result;
}

// first >= second, with canonical unsigned decimal operands.
std::string subtract_magnitudes(std::string_view first, std::string_view second) {
    std::string result(first);
    int borrow = 0;
    for (std::size_t offset = 0; offset < first.size(); ++offset) {
        const auto b = offset < second.size() ? second[second.size() - offset - 1] - '0' : 0;
        auto digit = first[first.size() - offset - 1] - '0' - b - borrow;
        borrow = digit < 0;
        if (borrow) digit += 10;
        result[first.size() - offset - 1] = static_cast<char>('0' + digit);
    }
    const auto nonzero = result.find_first_not_of('0');
    if (nonzero == std::string::npos) return "0";
    result.erase(0, nonzero);
    return result;
}

std::vector<std::uint32_t> words(std::uint64_t value) {
    std::vector<std::uint32_t> result{static_cast<std::uint32_t>(value)};
    if (value >> 32) result.push_back(static_cast<std::uint32_t>(value >> 32));
    return result;
}

}  // namespace

PythonInteger::PythonInteger(std::int64_t value) : decimal_(std::to_string(value)) {}

PythonInteger PythonInteger::from_unsigned(std::uint64_t value) {
    PythonInteger result;
    result.decimal_ = std::to_string(value);
    return result;
}

PythonInteger& PythonInteger::operator+=(const PythonInteger& other) {
    const bool negative = decimal_.front() == '-', other_negative = other.decimal_.front() == '-';
    const std::string_view first(decimal_.data() + negative, decimal_.size() - negative);
    const std::string_view second(other.decimal_.data() + other_negative, other.decimal_.size() - other_negative);
    std::string magnitude;
    bool result_negative = negative;
    if (negative == other_negative) magnitude = add_magnitudes(first, second);
    else if (first.size() > second.size() || (first.size() == second.size() && first >= second)) {
        magnitude = subtract_magnitudes(first, second);
    } else {
        magnitude = subtract_magnitudes(second, first);
        result_negative = other_negative;
    }
    decimal_ = result_negative && magnitude != "0" ? "-" + magnitude : magnitude;
    // Arithmetic results do not pass through Python's decimal-input limit.
    return *this;
}

PythonInteger PythonInteger::operator*(const PythonInteger& other) const {
    PythonInteger result;
    if (decimal_ == "0" || other.decimal_ == "0") return result;
    const bool negative = decimal_.front() == '-', other_negative = other.decimal_.front() == '-';
    const std::string_view first(decimal_.data() + negative, decimal_.size() - negative);
    const std::string_view second(other.decimal_.data() + other_negative, other.decimal_.size() - other_negative);
    std::vector<unsigned char> digits;
    if (second.size() > digits.max_size() || second.size() > result.decimal_.max_size() ||
        first.size() > digits.max_size() - second.size() ||
        first.size() > result.decimal_.max_size() - second.size())
        throw std::length_error("Integer product is too large");
    digits.resize(first.size() + second.size());
    for (std::size_t i = 0; i < first.size(); ++i) {
        unsigned carry = 0;
        const auto a = static_cast<unsigned>(first[first.size() - i - 1] - '0');
        for (std::size_t j = 0; j < second.size(); ++j) {
            const auto b = static_cast<unsigned>(second[second.size() - j - 1] - '0');
            const auto total = digits[i + j] + a * b + carry;
            digits[i + j] = static_cast<unsigned char>(total % 10);
            carry = total / 10;
        }
        digits[i + second.size()] = static_cast<unsigned char>(carry);
    }
    while (digits.size() > 1 && digits.back() == 0) digits.pop_back();
    result.decimal_.clear();
    if (negative != other_negative) result.decimal_.push_back('-');
    for (auto digit = digits.rbegin(); digit != digits.rend(); ++digit)
        result.decimal_.push_back(static_cast<char>('0' + *digit));
    return result;
}

double PythonInteger::as_double() const {
    static_assert(std::numeric_limits<double>::is_iec559 &&
                  std::numeric_limits<double>::radix == 2 &&
                  std::numeric_limits<double>::digits == 53 &&
                  std::numeric_limits<double>::max_exponent == 1024,
                  "Python integer conversion requires IEEE binary64");
    if (decimal_ == "0") return 0.0;
    const auto magnitude = seed_words();
    if (magnitude.size() > 32) throw std::overflow_error("int too large to convert to float");
    auto highest = magnitude.back();
    std::size_t high_bits = 0;
    for (; highest; highest >>= 1) ++high_bits;
    const auto bits = (magnitude.size() - 1) * 32 + high_bits;
    const auto bit = [&](std::size_t index) { return (magnitude[index / 32] >> (index % 32)) & 1u; };
    const auto shift = bits > 53 ? bits - 53 : 0;
    std::uint64_t significand = 0;
    for (auto index = bits; index > shift;)
        significand = (significand << 1) | bit(--index);
    if (shift && bit(shift - 1)) {
        bool sticky = false;
        for (std::size_t index = 0; index + 1 < shift; ++index) {
            if (bit(index)) { sticky = true; break; }
        }
        if (sticky || (significand & 1u)) ++significand;
    }
    // A rounding carry at exponent 1024 overflows; avoid asking ldexp to
    // overflow, just as PyLong_AsDouble checks the rounded exponent first.
    if (bits == 1024 && significand == (std::uint64_t{1} << 53))
        throw std::overflow_error("int too large to convert to float");
    const auto converted = std::ldexp(static_cast<double>(significand), static_cast<int>(shift));
    return decimal_.front() == '-' ? -converted : converted;
}

PythonInteger::PythonInteger(std::string_view value) {
    const auto normalized = python_numeric_ascii(value);
    value = normalized;
    while (!value.empty() && ascii_space(value.front())) value.remove_prefix(1);
    while (!value.empty() && ascii_space(value.back())) value.remove_suffix(1);
    bool negative = false;
    if (!value.empty() && (value.front() == '-' || value.front() == '+')) {
        negative = value.front() == '-';
        value.remove_prefix(1);
    }
    if (value.empty()) throw std::invalid_argument("Integer must contain decimal digits");
    std::string digits;
    digits.reserve(value.size());
    for (std::size_t index = 0; index < value.size(); ++index) {
        const auto digit = value[index];
        if (ascii_digit(digit)) {
            digits.push_back(digit);
        } else if (digit != '_' || index == 0 || index + 1 == value.size() ||
                   !ascii_digit(value[index - 1]) || !ascii_digit(value[index + 1])) {
            throw std::invalid_argument("Integer must contain decimal digits with optional separators");
        }
    }
    // Count input digits, including leading zeroes, before canonicalization;
    // underscores and the sign do not count toward Python's decimal limit.
    check_python_integer_digits(digits.size());
    const auto first = digits.find_first_not_of('0');
    if (first == std::string::npos) {
        decimal_ = "0";
    } else {
        decimal_ = negative ? "-" : "";
        decimal_.append(digits, first, std::string::npos);
    }
}

std::vector<std::uint32_t> PythonInteger::seed_words(std::uint64_t increment) const {
    const bool negative = decimal_.front() == '-';
    std::vector<std::uint32_t> result{0};
    for (std::size_t index = negative ? 1 : 0; index < decimal_.size(); ++index) {
        std::uint64_t carry = static_cast<unsigned>(decimal_[index] - '0');
        for (auto& word : result) {
            const auto next = static_cast<std::uint64_t>(word) * 10 + carry;
            word = static_cast<std::uint32_t>(next);
            carry = next >> 32;
        }
        if (carry) result.push_back(static_cast<std::uint32_t>(carry));
    }
    if (!negative) {
        auto carry = increment;
        for (std::size_t index = 0; carry; ++index) {
            if (index == result.size()) result.push_back(0);
            const auto next = static_cast<std::uint64_t>(result[index]) +
                              static_cast<std::uint32_t>(carry);
            result[index] = static_cast<std::uint32_t>(next);
            carry = (carry >> 32) + (next >> 32);
        }
        return result;
    }
    if (result.size() <= 2) {
        auto magnitude = static_cast<std::uint64_t>(result[0]);
        if (result.size() == 2) magnitude |= static_cast<std::uint64_t>(result[1]) << 32;
        if (increment >= magnitude) return words(increment - magnitude);
    }
    // Here abs(seed) > increment. Subtract the offset from the magnitude;
    // propagating a borrow can consume words above the offset's two words.
    std::uint64_t borrow = 0;
    for (std::size_t index = 0; increment || borrow; ++index) {
        const auto subtrahend = static_cast<std::uint32_t>(increment) + borrow;
        const auto current = static_cast<std::uint64_t>(result[index]);
        result[index] = static_cast<std::uint32_t>(current - subtrahend);
        borrow = current < subtrahend ? 1 : 0;
        increment >>= 32;
    }
    while (result.size() > 1 && result.back() == 0) result.pop_back();
    return result;
}

}  // namespace uf
