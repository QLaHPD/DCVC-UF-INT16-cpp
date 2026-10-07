#pragma once

#include <nlohmann/json.hpp>
#include "uf/python_json.hpp"

#include <array>
#include <algorithm>
#include <charconv>
#include <cstdint>
#include <cstddef>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace uf {

class PythonJsonAttributeError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class PythonJsonTypeError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

namespace json_compat_detail {

inline const char* python_type_name(const nlohmann::json& value) {
    if (value.is_null()) return "NoneType";
    if (value.is_boolean()) return "bool";
    if (value.is_number_float()) return "float";
    if (value.is_number_integer() || value.is_number_unsigned() || python_json_extended_integer(value)) return "int";
    if (value.is_string()) return "str";
    if (value.is_array()) return "list";
    if (value.is_object()) return "dict";
    return value.type_name();  // Native binary/discarded values are not JSON loads.
}

[[noreturn]] inline void missing_attribute(const nlohmann::json& value, const std::string& name) {
    throw PythonJsonAttributeError("'" + std::string(python_type_name(value)) +
        "' object has no attribute '" + name + "'");
}

}  // namespace json_compat_detail

// dict.get distinguishes an absent key from a receiver without a get method.
// Returned references alias the dictionary or fallback: both must outlive use.
inline const nlohmann::json& python_json_dict_get(const nlohmann::json& value,
    const std::string& key, const nlohmann::json& fallback) {
    if (!value.is_object()) json_compat_detail::missing_attribute(value, "get");
    const auto found = value.find(key);
    return found == value.end() ? fallback : *found;
}

inline const nlohmann::json& python_json_dict_get(const nlohmann::json& value,
    const std::string& key) {
    static const nlohmann::json missing(nullptr);
    return python_json_dict_get(value, key, missing);
}

// The maintained artifact checks use a string needle with Python's `in` on a
// dict, list or string. Scalar receivers raise TypeError rather than false.
inline bool python_json_string_in(const std::string& needle, const nlohmann::json& value) {
    if (value.is_object()) return value.contains(needle);
    if (value.is_array()) {
        for (const auto& item : value)
            if (item.is_string() && item.get_ref<const std::string&>() == needle) return true;
        return false;
    }
    if (value.is_string()) return value.get_ref<const std::string&>().find(needle) != std::string::npos;
    throw std::invalid_argument("argument of type '" +
        std::string(json_compat_detail::python_type_name(value)) + "' is not iterable");
}

inline bool python_json_endswith(const nlohmann::json& value, const std::string& suffix) {
    if (!value.is_string()) json_compat_detail::missing_attribute(value, "endswith");
    const auto& text = value.get_ref<const std::string&>();
    return text.size() >= suffix.size() &&
        text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// Python truth values for the scalar/container types represented by JSON.
// json::empty() treats even an empty string as a nonempty scalar; strings,
// false and numeric zero therefore require explicit handling.
inline bool json_truthy(const nlohmann::json& value) {
    if (python_json_extended_integer(value)) return python_json_extended_integer_decimal(value) != "0";
    if (value.is_null() || value.is_discarded()) return false;
    if (value.is_boolean()) return value.get<bool>();
    if (value.is_number_unsigned()) return value.get<std::uint64_t>() != 0;
    if (value.is_number_integer()) return value.get<std::int64_t>() != 0;
    if (value.is_number_float()) return value.get<double>() != 0;
    if (value.is_string()) return !value.get_ref<const std::string&>().empty();
    return !value.empty();
}

// Python compares st_size (an integer) with the JSON value without truncating
// fractions or wrapping negatives. Booleans and exactly integral floats can
// compare equal; converting a large integer through Double can lose precision.
inline bool json_file_size_equal(std::uintmax_t size, const nlohmann::json& expected) {
    if (expected.is_boolean()) return size == static_cast<std::uintmax_t>(expected.get<bool>());
    if (expected.is_number_unsigned()) return size == expected.get<std::uint64_t>();
    if (expected.is_number_integer()) {
        const auto value = expected.get<std::int64_t>();
        return value >= 0 && size == static_cast<std::uintmax_t>(value);
    }
    if (expected.is_number_float()) {
        const auto value = expected.get<double>();
        return std::isfinite(value) && value >= 0 && std::trunc(value) == value &&
            value < std::ldexp(1.0, std::numeric_limits<std::uintmax_t>::digits) &&
            size == static_cast<std::uintmax_t>(value);
    }
    return false;
}

namespace json_compat_detail {

inline std::uint64_t unsigned_integer(const nlohmann::json& value) {
    return value.is_boolean() ? static_cast<std::uint64_t>(value.get<bool>())
                              : value.get<std::uint64_t>();
}

inline bool float_integer_equal(double value, const nlohmann::json& integer) {
    if (!std::isfinite(value) || std::trunc(value) != value) return false;
    if (integer.is_boolean() || integer.is_number_unsigned()) {
        const auto upper = std::ldexp(1.0, std::numeric_limits<std::uint64_t>::digits);
        return value >= 0.0 && value < upper &&
               static_cast<std::uint64_t>(value) == unsigned_integer(integer);
    }
    const auto upper = std::ldexp(1.0, std::numeric_limits<std::int64_t>::digits);
    return value >= -upper && value < upper &&
           static_cast<std::int64_t>(value) == integer.get<std::int64_t>();
}

inline bool integer_equal(const nlohmann::json& first, const nlohmann::json& second) {
    const bool first_unsigned = first.is_boolean() || first.is_number_unsigned();
    const bool second_unsigned = second.is_boolean() || second.is_number_unsigned();
    if (first_unsigned && second_unsigned) return unsigned_integer(first) == unsigned_integer(second);
    if (!first_unsigned && !second_unsigned)
        return first.get<std::int64_t>() == second.get<std::int64_t>();
    const auto& positive = first_unsigned ? first : second;
    const auto& signed_value = first_unsigned ? second : first;
    const auto integer = signed_value.get<std::int64_t>();
    return integer >= 0 && static_cast<std::uint64_t>(integer) == unsigned_integer(positive);
}

}  // namespace json_compat_detail

// Python equality for native JSON and the private exact integer payload.
// Never round an integer to
// double to compare it. Container elements use identity before value equality,
// as CPython's list/dict comparisons do; a scalar NaN remains unequal to itself.
// Adapted from CPython 3.10 equality policies; see licenses/CPython-3.10.20.txt.
// Parser-owned NaN identities still need separate representation work.
inline bool json_python_equal(const nlohmann::json& first, const nlohmann::json& second) {
    if (python_json_extended_integer(first) || python_json_extended_integer(second)) {
        const auto& integer = python_json_extended_integer(first) ? first : second;
        const auto& other = &integer == &first ? second : first;
        const auto decimal = python_json_extended_integer_decimal(integer);
        if (python_json_extended_integer(other))
            return decimal == python_json_extended_integer_decimal(other);
        if (other.is_number_integer()) return decimal == other.dump();
        if (other.is_boolean()) return decimal == (other.get<bool>() ? "1" : "0");
        if (!other.is_number_float()) return false;
        const auto number = other.get<double>();
        if (!std::isfinite(number) || std::trunc(number) != number) return false;
        if (number == 0) return decimal == "0";
        // A finite integral binary64 has at most 309 decimal digits. Fixed
        // precision zero renders its exact integer, without rounding the int
        // through double; signs and negative zero follow Python equality.
        std::array<char, 384> text{};
        const auto converted = std::to_chars(text.data(), text.data() + text.size(),
                                            number, std::chars_format::fixed, 0);
        if (converted.ec != std::errc{}) throw std::runtime_error("Cannot format JSON integer comparison");
        return decimal == std::string_view(text.data(), static_cast<std::size_t>(converted.ptr - text.data()));
    }
    const bool first_numeric = first.is_number() || first.is_boolean();
    const bool second_numeric = second.is_number() || second.is_boolean();
    if (first_numeric || second_numeric) {
        if (!first_numeric || !second_numeric) return false;
        if (first.is_number_float() && second.is_number_float())
            return first.get<double>() == second.get<double>();
        if (first.is_number_float()) return json_compat_detail::float_integer_equal(first.get<double>(), second);
        if (second.is_number_float()) return json_compat_detail::float_integer_equal(second.get<double>(), first);
        return json_compat_detail::integer_equal(first, second);
    }
    if (first.type() != second.type()) return false;
    if (first.is_null()) return true;
    if (first.is_string()) return first.get_ref<const std::string&>() == second.get_ref<const std::string&>();
    if (first.is_array()) {
        if (first.size() != second.size()) return false;
        for (std::size_t i = 0; i < first.size(); ++i) {
            const auto& left = first[i];
            const auto& right = second[i];
            if (&left != &right && !json_python_equal(left, right)) return false;
        }
        return true;
    }
    if (first.is_object()) {
        if (first.size() != second.size()) return false;
        for (auto item = first.begin(); item != first.end(); ++item) {
            const auto other = second.find(item.key());
            if (other == second.end() ||
                (&item.value() != &other.value() && !json_python_equal(item.value(), other.value()))) return false;
        }
        return true;
    }
    // Binary/discarded values are not Python json.loads values.
    return false;
}

namespace json_compat_detail {

inline std::string integer_decimal(const nlohmann::json& value) {
    if (python_json_extended_integer(value))
        return std::string(python_json_extended_integer_decimal(value));
    return value.is_boolean() ? (value.get<bool>() ? "1" : "0") : value.dump();
}

inline int compare_integer_decimal(std::string_view first, std::string_view second) {
    const bool first_negative = first.front() == '-', second_negative = second.front() == '-';
    if (first_negative != second_negative) return first_negative ? -1 : 1;
    if (first_negative) first.remove_prefix(1);
    if (second_negative) second.remove_prefix(1);
    const int magnitude = first.size() < second.size() ? -1 : first.size() > second.size() ? 1
        : first < second ? -1 : first > second ? 1 : 0;
    return first_negative ? -magnitude : magnitude;
}

inline std::string integral_float_decimal(double value) {
    if (value == 0) return "0";
    std::array<char, 384> text{};
    const auto result = std::to_chars(text.data(), text.data() + text.size(),
                                      value, std::chars_format::fixed, 0);
    if (result.ec != std::errc{}) throw std::runtime_error("Cannot format JSON integer comparison");
    return std::string(text.data(), static_cast<std::size_t>(result.ptr - text.data()));
}

}  // namespace json_compat_detail

// Python < for represented numbers, strings and lists. Mixed int/float
// ordering compares the exact integer to floor(float), retaining its fraction,
// rather than rounding a large integer through double. NaN comparisons are
// false; unsupported type pairs raise TypeError. Lists compare lexically.
inline bool json_python_less(const nlohmann::json& first, const nlohmann::json& second) {
    const auto numeric = [](const auto& value) {
        return value.is_number() || value.is_boolean() || python_json_extended_integer(value);
    };
    if (numeric(first) && numeric(second)) {
        if (first.is_number_float() && second.is_number_float())
            return first.get<double>() < second.get<double>();
        if (first.is_number_float() || second.is_number_float()) {
            const bool float_first = first.is_number_float();
            const auto number = (float_first ? first : second).get<double>();
            if (std::isnan(number)) return false;
            if (std::isinf(number)) return float_first ? number < 0 : number > 0;
            const auto floor = std::floor(number);
            const auto& integer = float_first ? second : first;
            const auto compared = json_compat_detail::compare_integer_decimal(
                json_compat_detail::integer_decimal(integer),
                json_compat_detail::integral_float_decimal(floor));
            return float_first ? compared > 0 : compared < 0 || (compared == 0 && number != floor);
        }
        return json_compat_detail::compare_integer_decimal(
            json_compat_detail::integer_decimal(first), json_compat_detail::integer_decimal(second)) < 0;
    }
    if (first.is_string() && second.is_string())
        return first.get_ref<const std::string&>() < second.get_ref<const std::string&>();
    if (first.is_array() && second.is_array()) {
        const auto size = std::min(first.size(), second.size());
        for (std::size_t index = 0; index < size; ++index) {
            if (&first[index] == &second[index] || json_python_equal(first[index], second[index])) continue;
            return json_python_less(first[index], second[index]);
        }
        return first.size() < second.size();
    }
    throw PythonJsonTypeError("'<' not supported between instances of '" +
        std::string(json_compat_detail::python_type_name(first)) + "' and '" +
        std::string(json_compat_detail::python_type_name(second)) + "'");
}

inline bool archive_completion_record(const nlohmann::json& metadata) {
    static const nlohmann::json empty_validation = nlohmann::json::object();
    const auto& validation = python_json_dict_get(metadata, "validation", empty_validation);
    if (json_truthy(python_json_dict_get(validation, "decoded_yuv_sha256"))) return true;
    if (python_json_dict_get(validation, "mode") != "artifact-hashes") return false;
    const auto& performed = python_json_dict_get(validation, "decode_performed");
    return performed.is_boolean() && !performed.get<bool>();
}

}  // namespace uf
