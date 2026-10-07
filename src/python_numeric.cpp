#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "uf/python_numeric.hpp"

#include "python_unicode_data.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <locale.h>
#include <stdexcept>

namespace uf {
namespace {

constexpr std::uint32_t invalid_codepoint = 0x110000u;

bool ascii_space(char value) {
    return value == ' ' || value == '\t' || value == '\n' ||
           value == '\r' || value == '\v' || value == '\f';
}

bool ascii_digit(char value) { return value >= '0' && value <= '9'; }

struct NumericLocale {
    locale_t value{newlocale(LC_NUMERIC_MASK, "C", nullptr)};
    NumericLocale() {
        if (!value) throw std::runtime_error("Cannot create numeric locale");
    }
    ~NumericLocale() { freelocale(value); }
};

std::uint32_t next_codepoint(std::string_view value, std::size_t& position) {
    const auto start = position;
    const auto first = static_cast<unsigned char>(value[position]);
    if (first < 0x80u) { ++position; return first; }
    std::size_t length = 0;
    std::uint32_t codepoint = 0, minimum = 0;
    if (first >= 0xc2u && first <= 0xdfu) {
        length = 2; codepoint = first & 0x1fu; minimum = 0x80u;
    } else if (first >= 0xe0u && first <= 0xefu) {
        length = 3; codepoint = first & 0x0fu; minimum = 0x800u;
    } else if (first >= 0xf0u && first <= 0xf4u) {
        length = 4; codepoint = first & 0x07u; minimum = 0x10000u;
    }
    if (!length || value.size() - start < length) {
        ++position;
        return invalid_codepoint;
    }
    for (std::size_t index = 1; index < length; ++index) {
        const auto byte = static_cast<unsigned char>(value[start + index]);
        if ((byte & 0xc0u) != 0x80u) {
            ++position;
            return invalid_codepoint;
        }
        codepoint = (codepoint << 6u) | (byte & 0x3fu);
    }
    if (codepoint < minimum || codepoint > 0x10ffffu ||
        (codepoint >= 0xd800u && codepoint <= 0xdfffu)) {
        ++position;
        return invalid_codepoint;
    }
    position += length;
    return codepoint;
}

int decimal_digit(std::uint32_t codepoint) {
    const auto& zero = python_unicode_data::decimal_zero;
    auto found = std::upper_bound(zero.begin(), zero.end(), codepoint);
    if (found == zero.begin()) return -1;
    const auto offset = codepoint - *--found;
    return offset < 10u ? static_cast<int>(offset) : -1;
}

bool unicode_space(std::uint32_t codepoint) {
    const auto& spaces = python_unicode_data::non_ascii_space;
    return std::binary_search(spaces.begin(), spaces.end(), codepoint);
}

bool text_space(std::uint32_t codepoint) {
    return (codepoint < 128u && (ascii_space(static_cast<char>(codepoint)) ||
            (codepoint >= 0x1cu && codepoint <= 0x1fu))) || unicode_space(codepoint);
}

bool line_boundary(std::uint32_t codepoint) {
    return (codepoint >= 0x0au && codepoint <= 0x0du) ||
           (codepoint >= 0x1cu && codepoint <= 0x1eu) ||
           codepoint == 0x85u || codepoint == 0x2028u || codepoint == 0x2029u;
}

std::uint32_t text_codepoint(std::string_view value, std::size_t& position) {
    const auto codepoint = next_codepoint(value, position);
    if (codepoint == invalid_codepoint) throw std::invalid_argument("Invalid UTF-8 text file");
    return codepoint;
}

std::size_t environment_digit_limit() {
    const char* setting = std::getenv("PYTHONINTMAXSTRDIGITS");
    if (!setting || !*setting) return 4300;
    errno = 0;
    char* end = nullptr;
    const long limit = std::strtol(setting, &end, 10);
    if (end == setting || *end || errno == ERANGE ||
        limit < 0 || limit > std::numeric_limits<int>::max() ||
        (limit != 0 && limit < 640)) {
        throw std::runtime_error("PYTHONINTMAXSTRDIGITS: invalid limit; must be >= 640 or 0 for unlimited");
    }
    return static_cast<std::size_t>(limit);
}

}  // namespace

bool python_decimal_digits(std::string_view value) {
    if (value.empty()) return false;
    for (std::size_t position = 0; position < value.size();) {
        if (decimal_digit(next_codepoint(value, position)) < 0) return false;
    }
    return true;
}

std::string python_strip(std::string_view value) {
    std::size_t position = 0, first = value.size(), last = 0;
    while (position < value.size()) {
        const auto start = position;
        if (!text_space(text_codepoint(value, position))) {
            if (first == value.size()) first = start;
            last = position;
        }
    }
    return first == value.size() ? std::string{} : std::string(value.substr(first, last - first));
}

std::vector<std::string> python_split_whitespace(std::string_view value) {
    std::vector<std::string> words;
    std::size_t position = 0, start = 0;
    bool in_word = false;
    while (position < value.size()) {
        const auto end = position;
        if (text_space(text_codepoint(value, position))) {
            if (in_word) words.emplace_back(value.substr(start, end - start));
            in_word = false;
        } else if (!in_word) {
            start = end;
            in_word = true;
        }
    }
    if (in_word) words.emplace_back(value.substr(start));
    return words;
}

std::vector<std::string> python_splitlines(std::string_view value) {
    std::vector<std::string> lines;
    std::size_t position = 0, start = 0;
    while (position < value.size()) {
        const auto end = position;
        const auto codepoint = text_codepoint(value, position);
        if (!line_boundary(codepoint)) continue;
        lines.emplace_back(value.substr(start, end - start));
        if (codepoint == '\r' && position < value.size() && value[position] == '\n') ++position;
        start = position;
    }
    if (start < value.size()) lines.emplace_back(value.substr(start));
    return lines;
}

std::string python_signed_digit_prefix(std::string_view value) {
    // Environment strings use surrogate escapes for invalid Unix bytes.
    // Unlike strict text-file helpers, preserve those nondigit bytes while
    // finding the strip boundary, then stop the numeric prefix at them.
    std::size_t position = 0, first = value.size(), last = 0;
    while (position < value.size()) {
        const auto start = position;
        if (!text_space(next_codepoint(value, position))) {
            if (first == value.size()) first = start;
            last = position;
        }
    }
    if (first == value.size()) return {};
    value = value.substr(first, last - first);
    position = 0;
    std::size_t end = 0;
    const auto& extra_digits = python_unicode_data::non_decimal_digit;
    while (position < value.size()) {
        const auto start = position;
        const auto codepoint = next_codepoint(value, position);
        if (!(start == 0 && (codepoint == '+' || codepoint == '-')) &&
            decimal_digit(codepoint) < 0 &&
            !std::binary_search(extra_digits.begin(), extra_digits.end(), codepoint)) break;
        end = position;
    }
    return std::string(value.substr(0, end));
}

std::string python_numeric_ascii(std::string_view value) {
    if (std::none_of(value.begin(), value.end(), [](unsigned char byte) { return byte >= 0x80u; })) {
        return std::string(value);
    }
    std::string result;
    result.reserve(value.size());
    std::size_t position = 0;
    while (position < value.size()) {
        const auto codepoint = next_codepoint(value, position);
        if (codepoint < 127u) result.push_back(static_cast<char>(codepoint));
        else if (unicode_space(codepoint)) result.push_back(' ');
        else {
            const int digit = decimal_digit(codepoint);
            if (digit < 0) {
                // CPython truncates at a '?' for an unsupported character.
                // Invalid Unix argv bytes become nonnumeric surrogate escapes
                // in Python and are rejected by this same boundary.
                result.push_back('?');
                break;
            }
            result.push_back(static_cast<char>('0' + digit));
        }
    }
    return result;
}

std::string python_float_text(std::string_view value) {
    const auto normalized = python_numeric_ascii(value);
    value = normalized;
    while (!value.empty() && ascii_space(value.front())) value.remove_prefix(1);
    while (!value.empty() && ascii_space(value.back())) value.remove_suffix(1);
    const auto invalid = []() -> void { throw std::invalid_argument("Expected a decimal number"); };
    std::string cleaned;
    cleaned.reserve(value.size());
    for (std::size_t index = 0; index < value.size(); ++index) {
        if (value[index] != '_') cleaned.push_back(value[index]);
        else if (index == 0 || index + 1 == value.size() ||
                 !ascii_digit(value[index - 1]) || !ascii_digit(value[index + 1])) invalid();
    }
    if (cleaned.empty()) invalid();
    std::size_t index = cleaned.front() == '+' || cleaned.front() == '-' ? 1 : 0;
    auto special = cleaned.substr(index);
    for (auto& letter : special) if (letter >= 'A' && letter <= 'Z') letter += 'a' - 'A';
    if (special == "inf" || special == "infinity" || special == "nan") return cleaned;
    bool digits = false;
    while (index < cleaned.size() && ascii_digit(cleaned[index])) { digits = true; ++index; }
    if (index < cleaned.size() && cleaned[index] == '.') {
        ++index;
        while (index < cleaned.size() && ascii_digit(cleaned[index])) { digits = true; ++index; }
    }
    if (!digits) invalid();
    if (index < cleaned.size() && (cleaned[index] == 'e' || cleaned[index] == 'E')) {
        ++index;
        if (index < cleaned.size() && (cleaned[index] == '+' || cleaned[index] == '-')) ++index;
        const auto exponent = index;
        while (index < cleaned.size() && ascii_digit(cleaned[index])) ++index;
        if (index == exponent) invalid();
    }
    if (index != cleaned.size()) invalid();
    return cleaned;
}

double python_float(std::string_view value) {
    const auto decimal = python_float_text(value);
    static const NumericLocale locale;
    char* end = nullptr;
    const double result = strtod_l(decimal.c_str(), &end, locale.value);
    if (end != decimal.c_str() + decimal.size()) throw std::invalid_argument("Expected a decimal number");
    return result;
}

bool python_negative_number(std::string_view value) {
    if (value.empty() || value.front() != '-') return false;
    if (value.back() == '\n') value.remove_suffix(1);
    std::size_t position = 1, integral = 0, fractional = 0;
    bool point = false;
    while (position < value.size()) {
        const auto codepoint = next_codepoint(value, position);
        if (codepoint == '.' && !point) { point = true; continue; }
        if (decimal_digit(codepoint) < 0) return false;
        if (point) ++fractional; else ++integral;
    }
    return point ? fractional != 0 : integral != 0;
}

std::size_t python_integer_digit_limit() {
    static const auto limit = environment_digit_limit();
    return limit;
}

void check_python_integer_digits(std::size_t count) {
    const auto limit = python_integer_digit_limit();
    if (limit && count > limit) {
        throw std::invalid_argument("Integer string has " + std::to_string(count) +
            " digits, exceeding the limit of " + std::to_string(limit) +
            "; set PYTHONINTMAXSTRDIGITS to increase or disable the limit");
    }
}

}  // namespace uf
