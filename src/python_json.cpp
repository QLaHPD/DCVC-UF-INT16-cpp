#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "uf/python_json.hpp"
#include "uf/python_path.hpp"
#include "uf/python_numeric.hpp"
#include "uf/json_compat.hpp"
#include "uf/process.hpp"
#include "uf/utf8.hpp"
#include "python_unicode_data.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>
#include <fcntl.h>
#include <unistd.h>

namespace uf {
namespace {

template<class Json>
Json parse_json_values(std::string_view input);

template<class Result, class Finish>
Result read_text_file(const std::filesystem::path& path,
                      const std::string& encoding, bool check_interrupt,
                      Finish finish) {
    // CPython pathlib/json text/context policies; see licenses/CPython-3.10.20.txt.
    const auto filename = path.string();
    if (filename.find('\0') != std::string::npos)
        throw std::invalid_argument("embedded null byte");
    int descriptor;
    for (;;) {
        if (check_interrupt) ProcessInterrupt::check();
        descriptor = open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (descriptor >= 0) break;
        const int error = errno;
        if (error == EINTR) continue;
        throw std::system_error(error, std::generic_category(),
                                "Cannot open text file: " + filename);
    }
    FileDescriptor input(descriptor);
    const auto close_input = [&] {
        const int owned = input.release();
        if (close(owned) != 0) {
            const int error = errno;
            throw std::system_error(error, std::generic_category(),
                                    "Cannot close text file: " + filename);
        }
    };
    try {
        std::array<char, 65536> block{};
        std::string bytes;
        for (;;) {
            if (check_interrupt) ProcessInterrupt::check();
            const auto count = read(input.get(), block.data(), block.size());
            if (count < 0) {
                const int error = errno;
                if (error == EINTR) continue;
                throw std::system_error(error, std::generic_category(),
                                        "Cannot read text file: " + filename);
            }
            if (count == 0) break;
            bytes.append(block.data(), static_cast<std::size_t>(count));
        }
        auto text = decode_process_text(std::move(bytes), encoding);
        return finish(std::move(text), close_input);
    } catch (...) {
        // A with/finally close may replace the active read/decode/parse error.
        // Release before close: neither this handler nor RAII retries it.
        if (input.get() >= 0) close_input();
        throw;
    }
}

template<class JsonType>
JsonType read_json_file(const std::filesystem::path& path,
                       const std::string& encoding, bool close_before_parse,
                       bool check_interrupt) {
    return read_text_file<JsonType>(path, encoding, check_interrupt,
        [&](std::string text, const auto& close_input) {
            if (close_before_parse) close_input();
            if (text.compare(0, 3, "\xef\xbb\xbf") == 0)
                throw JsonType::parse_error::create(101, std::size_t{1},
                    "Unexpected UTF-8 BOM (decode using utf-8-sig)",
                    static_cast<const JsonType*>(nullptr));
            auto parsed = parse_json_values<JsonType>(text);
            if (!close_before_parse) close_input();
            return parsed;
        });
}

std::uint32_t next_codepoint(std::string_view text, std::size_t& offset,
                             bool surrogatepass = false) {
    const auto first = static_cast<unsigned char>(text[offset++]);
    if (first < 0x80) return first;
    int continuation = 0;
    std::uint32_t codepoint = 0;
    std::uint32_t minimum = 0;
    if (first >= 0xc2 && first <= 0xdf) {
        continuation = 1; codepoint = first & 0x1f; minimum = 0x80;
    } else if (first >= 0xe0 && first <= 0xef) {
        continuation = 2; codepoint = first & 0x0f; minimum = 0x800;
    } else if (first >= 0xf0 && first <= 0xf4) {
        continuation = 3; codepoint = first & 0x07; minimum = 0x10000;
    } else {
        throw std::invalid_argument("Invalid UTF-8 in Python JSON repr");
    }
    for (int i = 0; i < continuation; ++i) {
        if (offset == text.size())
            throw std::invalid_argument("Truncated UTF-8 in Python JSON repr");
        const auto byte = static_cast<unsigned char>(text[offset++]);
        if ((byte & 0xc0) != 0x80)
            throw std::invalid_argument("Invalid UTF-8 continuation in Python JSON repr");
        codepoint = (codepoint << 6) | (byte & 0x3f);
    }
    if (codepoint < minimum || codepoint > 0x10ffff ||
        (!surrogatepass && codepoint >= 0xd800 && codepoint <= 0xdfff))
        throw std::invalid_argument("Invalid UTF-8 codepoint in Python JSON repr");
    return codepoint;
}

bool printable(std::uint32_t codepoint) {
    const auto& ranges = python_unicode_data::non_printable_ranges;
    const auto found = std::lower_bound(ranges.begin(), ranges.end(), codepoint,
        [](const python_unicode_data::CodepointRange& range, std::uint32_t value) {
            return range.last < value;
        });
    return found == ranges.end() || found->first > codepoint;
}

// Adapted from CPython unicode_repr; see licenses/CPython-3.10.20.txt.
void append_string_repr(std::string& output, std::string_view text) {
    const char quote = text.find('\'') != std::string_view::npos &&
                       text.find('"') == std::string_view::npos ? '"' : '\'';
    output.push_back(quote);
    std::size_t offset = 0;
    constexpr char hex[] = "0123456789abcdef";
    while (offset < text.size()) {
        const auto start = offset;
        const auto codepoint = next_codepoint(text, offset, true);
        if (codepoint == static_cast<unsigned char>(quote) || codepoint == '\\') {
            output.push_back('\\');
            output.push_back(static_cast<char>(codepoint));
        } else if (codepoint == '\t' || codepoint == '\n' || codepoint == '\r') {
            output.push_back('\\');
            output.push_back(codepoint == '\t' ? 't' : codepoint == '\n' ? 'n' : 'r');
        } else if ((codepoint >= 0xd800 && codepoint <= 0xdfff) || !printable(codepoint)) {
            const int digits = codepoint <= 0xff ? 2 : codepoint <= 0xffff ? 4 : 8;
            output.push_back('\\');
            output.push_back(digits == 2 ? 'x' : digits == 4 ? 'u' : 'U');
            for (int digit = digits - 1; digit >= 0; --digit)
                output.push_back(hex[(codepoint >> (digit * 4)) & 0xf]);
        } else {
            output.append(text.data() + start, offset - start);
        }
    }
    output.push_back(quote);
}

void append_hex_escape(std::string& output, std::uint32_t value) {
    constexpr char hex[] = "0123456789abcdef";
    output += "\\u";
    for (int digit = 3; digit >= 0; --digit) output.push_back(hex[(value >> (digit * 4)) & 0xf]);
}

// CPython _json string policies; see licenses/CPython-3.10.20.txt.
void append_json_string(std::string& output, std::string_view text, bool ensure_ascii,
                        nlohmann::json::error_handler_t errors = nlohmann::json::error_handler_t::strict) {
    const auto original_size = output.size();
    try {
        output.push_back('"');
        for (std::size_t offset = 0; offset < text.size();) {
            const auto start = offset;
            const auto codepoint = next_codepoint(text, offset, true);
            if (codepoint == '"' || codepoint == '\\') {
                output.push_back('\\'); output.push_back(static_cast<char>(codepoint));
            } else if (codepoint == '\b' || codepoint == '\f' || codepoint == '\n' ||
                       codepoint == '\r' || codepoint == '\t') {
                output.push_back('\\');
                output.push_back(codepoint == '\b' ? 'b' : codepoint == '\f' ? 'f' :
                                 codepoint == '\n' ? 'n' : codepoint == '\r' ? 'r' : 't');
            } else if (codepoint <= 0x1f || (ensure_ascii && codepoint >= 0x7f)) {
                if (codepoint <= 0xffff) append_hex_escape(output, codepoint);
                else {
                    const auto supplementary = codepoint - 0x10000;
                    append_hex_escape(output, 0xd800 + (supplementary >> 10));
                    append_hex_escape(output, 0xdc00 + (supplementary & 0x3ff));
                }
            } else {
                if (codepoint >= 0xd800 && codepoint <= 0xdfff)
                    throw std::invalid_argument("Cannot encode surrogate codepoint as strict UTF-8 JSON");
                output.append(text.data() + start, offset - start);
            }
        }
        output.push_back('"');
    } catch (const std::invalid_argument&) {
        output.resize(original_size);
        if (errors != nlohmann::json::error_handler_t::replace) throw;
        // Preserve the prior native invalid-byte replacement policy. Valid
        // internal surrogate codepoints take the ASCII escape path above.
        output += nlohmann::json(std::string(text)).dump(-1, ' ', ensure_ascii, errors);
    }
}

template<class Json>
void append_repr(std::string& output, const Json& value) {
    if (python_json_extended_integer(value)) {
        const auto decimal = python_json_extended_integer_decimal(value);
        check_python_integer_digits(decimal.size() - (decimal.front() == '-'));
        output.append(decimal.data(), decimal.size());
    } else if (value.is_null()) {
        output += "None";
    } else if (value.is_boolean()) {
        output += value.template get<bool>() ? "True" : "False";
    } else if (value.is_number_float()) {
        output += python_float_repr(value.template get<double>());
    } else if (value.is_number_integer()) {
        output += value.dump();
    } else if (value.is_string()) {
        append_string_repr(output, value.template get_ref<const std::string&>());
    } else if (value.is_array()) {
        output.push_back('[');
        bool first = true;
        for (const auto& item : value) {
            if (!first) output += ", ";
            first = false;
            append_repr(output, item);
        }
        output.push_back(']');
    } else if (value.is_object()) {
        output.push_back('{');
        bool first = true;
        for (auto item = value.begin(); item != value.end(); ++item) {
            if (!first) output += ", ";
            first = false;
            append_string_repr(output, item.key());
            output += ": ";
            append_repr(output, item.value());
        }
        output.push_back('}');
    } else {
        throw std::invalid_argument("Value cannot be represented as Python JSON repr");
    }
}

void append_json(std::string& output, const nlohmann::json& value,
                 bool spaces = false, bool ensure_ascii = false,
                 nlohmann::json::error_handler_t errors = nlohmann::json::error_handler_t::strict) {
    if (python_json_extended_integer(value)) {
        const auto decimal = python_json_extended_integer_decimal(value);
        check_python_integer_digits(decimal.size() - (decimal.front() == '-'));
        output.append(decimal.data(), decimal.size());
    } else if (value.is_object()) {
        output.push_back('{');
        bool first = true;
        // nlohmann::json stores keys in std::map order. Valid UTF-8 byte
        // ordering agrees with Python's Unicode codepoint ordering.
        for (auto item = value.begin(); item != value.end(); ++item) {
            if (!first) output += spaces ? ", " : ",";
            first = false;
            append_json_string(output, item.key(), ensure_ascii, errors);
            output += spaces ? ": " : ":";
            append_json(output, item.value(), spaces, ensure_ascii, errors);
        }
        output.push_back('}');
    } else if (value.is_array()) {
        output.push_back('[');
        bool first = true;
        for (const auto& item : value) {
            if (!first) output += spaces ? ", " : ",";
            first = false;
            append_json(output, item, spaces, ensure_ascii, errors);
        }
        output.push_back(']');
    } else if (value.is_number_float()) {
        output += python_json_float(value.get<double>());
    } else if (value.is_string()) {
        append_json_string(output, value.get_ref<const std::string&>(), ensure_ascii, errors);
    } else {
        if (value.is_binary() || value.is_discarded()) {
            throw std::invalid_argument("Value cannot be serialized as Python JSON");
        }
        output += value.dump(-1, ' ', ensure_ascii, errors);
    }
}

template<class Json>
void append_pretty_json(std::string& output, const Json& value,
                        std::size_t depth) {
    if (python_json_extended_integer(value)) {
        const auto decimal = python_json_extended_integer_decimal(value);
        check_python_integer_digits(decimal.size() - (decimal.front() == '-'));
        output.append(decimal.data(), decimal.size());
    } else if ((value.is_object() || value.is_array()) && !value.empty()) {
        const bool object = value.is_object();
        output.push_back(object ? '{' : '[');
        bool first = true;
        for (auto item = value.begin(); item != value.end(); ++item) {
            if (!first) output.push_back(',');
            first = false;
            output.push_back('\n');
            output.append((depth + 1) * 2, ' ');
            if (object) {
                append_json_string(output, item.key(), true);
                output += ": ";
            }
            append_pretty_json(output, item.value(), depth + 1);
        }
        output.push_back('\n');
        output.append(depth * 2, ' ');
        output.push_back(object ? '}' : ']');
    } else if (value.is_number_float()) {
        output += python_json_float(value.template get<double>());
    } else if (value.is_string()) {
        append_json_string(output, value.template get_ref<const std::string&>(), true);
    } else {
        if (value.is_binary() || value.is_discarded())
            throw std::invalid_argument("Value cannot be serialized as Python JSON");
        output += value.dump(-1, ' ', true, nlohmann::json::error_handler_t::strict);
    }
}

bool json_delimiter(char value) {
    return value == ' ' || value == '\t' || value == '\r' || value == '\n' ||
           value == '{' || value == '}' || value == '[' || value == ']' ||
           value == ',' || value == ':';
}

// Return zero for nonnumbers, one for an integer and two for a float. Python
// float(str) accepts more syntax than JSON, so validate the JSON grammar first.
int json_number_kind(std::string_view text) {
    std::size_t position = !text.empty() && text.front() == '-' ? 1 : 0;
    if (position == text.size()) return 0;
    const auto digit = [](char value) { return value >= '0' && value <= '9'; };
    if (text[position] == '0') ++position;
    else {
        if (text[position] < '1' || text[position] > '9') return 0;
        while (position < text.size() && digit(text[position])) ++position;
    }
    int kind = 1;
    if (position < text.size() && text[position] == '.') {
        kind = 2;
        const auto start = ++position;
        while (position < text.size() && digit(text[position])) ++position;
        if (position == start) return 0;
    }
    if (position < text.size() && (text[position] == 'e' || text[position] == 'E')) {
        kind = 2;
        ++position;
        if (position < text.size() && (text[position] == '+' || text[position] == '-')) ++position;
        const auto start = position;
        while (position < text.size() && digit(text[position])) ++position;
        if (position == start) return 0;
    }
    return position == text.size() ? kind : 0;
}

template<class Json>
std::size_t json_quote_end(std::string_view text, std::size_t start) {
    for (auto position = start + 1; position < text.size(); ++position) {
        if (text[position] == '\\') {
            if (++position == text.size()) break;
        } else if (text[position] == '"') return position + 1;
    }
    using Error = typename Json::parse_error;
    throw Error::create(101, text.size() + 1, "Unterminated JSON string",
                        static_cast<const Json*>(nullptr));
}

void append_wtf8(std::string& output, std::uint32_t codepoint) {
    if (codepoint < 0x80) output.push_back(static_cast<char>(codepoint));
    else if (codepoint < 0x800) {
        output.push_back(static_cast<char>(0xc0 | (codepoint >> 6)));
        output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    } else if (codepoint < 0x10000) {
        output.push_back(static_cast<char>(0xe0 | (codepoint >> 12)));
        output.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
        output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    } else {
        output.push_back(static_cast<char>(0xf0 | (codepoint >> 18)));
        output.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3f)));
        output.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
        output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    }
}

template<class Json>
std::string decode_json_string(std::string_view token, std::size_t input_offset) {
    using Error = typename Json::parse_error;
    const auto invalid = [&](const char* message, std::size_t offset) {
        throw Error::create(101, input_offset + offset + 1, message,
                            static_cast<const Json*>(nullptr));
    };
    const auto limit = token.size() - 1;  // The closing quote is not content.
    const auto hex4 = [&](std::size_t position) {
        if (position > limit || limit - position < 4) invalid("Invalid \\uXXXX escape", position);
        std::uint32_t result = 0;
        for (std::size_t count = 0; count < 4; ++count) {
            const auto digit = token[position + count];
            unsigned value;
            if (digit >= '0' && digit <= '9') value = digit - '0';
            else if (digit >= 'a' && digit <= 'f') value = digit - 'a' + 10;
            else if (digit >= 'A' && digit <= 'F') value = digit - 'A' + 10;
            else { invalid("Invalid \\uXXXX escape", position); value = 0; }
            result = (result << 4) | value;
        }
        return result;
    };
    std::string output;
    for (std::size_t position = 1; position < limit;) {
        const auto start = position;
        const auto byte = static_cast<unsigned char>(token[position]);
        if (byte < 0x20) invalid("Invalid control character in JSON string", position);
        if (byte != '\\') {
            try { (void)next_codepoint(token.substr(0, limit), position, true); }
            catch (const std::invalid_argument&) { invalid("Invalid UTF-8 in JSON string", start); }
            output.append(token.data() + start, position - start);
            continue;
        }
        if (++position == limit) invalid("Unterminated JSON string", start);
        const auto escape = token[position++];
        if (escape == 'u') {
            auto codepoint = hex4(position);
            position += 4;
            if (codepoint >= 0xd800 && codepoint <= 0xdbff && limit - position >= 6 &&
                token[position] == '\\' && token[position + 1] == 'u') {
                const auto low = hex4(position + 2);
                if (low >= 0xdc00 && low <= 0xdfff) {
                    codepoint = 0x10000 + ((codepoint - 0xd800) << 10) + (low - 0xdc00);
                    position += 6;
                }
            }
            append_wtf8(output, codepoint);
        } else {
            switch (escape) {
            case '"': case '\\': case '/': output.push_back(escape); break;
            case 'b': output.push_back('\b'); break;
            case 'f': output.push_back('\f'); break;
            case 'n': output.push_back('\n'); break;
            case 'r': output.push_back('\r'); break;
            case 't': output.push_back('\t'); break;
            default: invalid("Invalid JSON escape", start);
            }
        }
    }
    return output;
}

template<class Json>
void restore_json_values(Json& value, const std::string& marker,
                         const std::vector<Json>& numbers,
                         const std::vector<std::string>& strings,
                         const std::map<std::string, std::size_t>& protected_strings) {
    if (value.is_object() && value.size() == 1 && value.contains(marker)) {
        const auto index = value.at(marker).template get<std::size_t>();
        value = numbers.at(index);
    } else if (value.is_string()) {
        const auto found = protected_strings.find(value.template get_ref<const std::string&>());
        if (found != protected_strings.end()) value = strings.at(found->second);
    } else if (value.is_array()) {
        for (auto& item : value) restore_json_values(item, marker, numbers, strings, protected_strings);
    } else if (value.is_object()) {
        auto restored = Json::object();
        for (auto item = value.begin(); item != value.end(); ++item) {
            const auto found = protected_strings.find(item.key());
            const auto& name = found == protected_strings.end() ? item.key() : strings.at(found->second);
            auto child = std::move(item.value());
            restore_json_values(child, marker, numbers, strings, protected_strings);
            restored[name] = std::move(child);
        }
        value = std::move(restored);
    }
}

template<class Json>
Json parse_json_values(std::string_view input) {
    using Error = typename Json::parse_error;
    if (input.substr(0, 3) == "\xef\xbb\xbf")
        throw Error::create(101, std::size_t{1}, "Unexpected UTF-8 BOM (decode using utf-8-sig)",
                            static_cast<const Json*>(nullptr));
    // Decode with Python string policies before native syntax parsing. Reuse
    // a placeholder for equal decoded keys, preserving duplicate-key last-value
    // and ordered-object first-position behavior even for surrogate spellings.
    std::map<std::string, std::size_t> string_indices;
    std::vector<std::string> strings;
    for (std::size_t position = 0; position < input.size();) {
        if (input[position] == '"') {
            const auto end = json_quote_end<Json>(input, position);
            const auto decoded = decode_json_string<Json>(input.substr(position, end - position), position);
            const auto inserted = string_indices.emplace(decoded, strings.size());
            if (inserted.second) strings.push_back(decoded);
            position = end;
        } else ++position;
    }
    // These namespaces are disjoint. Every original string (keys as well as
    // values) is protected; restoration checks numeric wrappers before user
    // keys are decoded and never rechecks an already restored value. Input
    // strings can therefore equal either namespace without impersonating it.
    const std::string marker = "__uf_python_json_number__";
    const auto marker_key = Json(marker).dump();
    const std::string string_prefix = "__uf_python_json_string__";
    std::map<std::string, std::size_t> protected_strings;
    for (std::size_t index = 0; index < strings.size(); ++index)
        protected_strings.emplace(string_prefix + std::to_string(index), index);
    std::string normalized;
    normalized.reserve(input.size());
    std::vector<Json> numbers;
    for (std::size_t position = 0; position < input.size();) {
        if (input[position] == '"') {
            const auto end = json_quote_end<Json>(input, position);
            const auto decoded = decode_json_string<Json>(input.substr(position, end - position), position);
            normalized.push_back('"');
            normalized += string_prefix;
            normalized += std::to_string(string_indices.at(decoded));
            normalized.push_back('"');
            position = end;
        } else if (json_delimiter(input[position])) {
            normalized.push_back(input[position++]);
        } else {
            const auto start = position;
            while (position < input.size() && !json_delimiter(input[position])) ++position;
            const auto token = input.substr(start, position - start);
            const auto kind = json_number_kind(token);
            Json number;
            bool numeric = true;
            if (kind == 1) number = Json(python_json_integer(PythonInteger(token)));
            else if (kind == 2) number = python_float(token);
            else if (token == "NaN") number = std::numeric_limits<double>::quiet_NaN();
            else if (token == "Infinity") number = std::numeric_limits<double>::infinity();
            else if (token == "-Infinity") number = -std::numeric_limits<double>::infinity();
            else numeric = false;
            if (numeric) {
                normalized += "{" + marker_key + ":" + std::to_string(numbers.size()) + "}";
                numbers.push_back(std::move(number));
            } else normalized.append(token.data(), token.size());
        }
    }
    // The native parser validates the protected document/container syntax.
    // String validation precedes it; restore keys only after numeric markers,
    // so a user key cannot impersonate the private number wrapper.
    auto result = Json::parse(normalized);
    restore_json_values(result, marker, numbers, strings, protected_strings);
    return result;
}

}  // namespace

nlohmann::json python_json_integer(const PythonInteger& value) {
    const auto& decimal = value.decimal();
    std::int64_t signed_value{};
    const auto signed_result = std::from_chars(decimal.data(), decimal.data() + decimal.size(), signed_value);
    if (signed_result.ec == std::errc{} && signed_result.ptr == decimal.data() + decimal.size())
        return signed_value;
    if (decimal.front() != '-') {
        std::uint64_t unsigned_value{};
        const auto result = std::from_chars(decimal.data(), decimal.data() + decimal.size(), unsigned_value);
        if (result.ec == std::errc{} && result.ptr == decimal.data() + decimal.size()) return unsigned_value;
    }
    return nlohmann::json::binary(std::vector<std::uint8_t>(decimal.begin(), decimal.end()),
                                  python_json_integer_subtype);
}

nlohmann::json parse_python_json(std::string_view text) {
    return parse_json_values<nlohmann::json>(text);
}

PythonInteger python_json_int(const nlohmann::json& value) {
    if (python_json_extended_integer(value))
        return PythonInteger(python_json_extended_integer_decimal(value));
    if (value.is_boolean()) return PythonInteger(value.get<bool>() ? 1 : 0);
    if (value.is_number_unsigned()) return PythonInteger::from_unsigned(value.get<std::uint64_t>());
    if (value.is_number_integer()) return PythonInteger(value.get<std::int64_t>());
    if (value.is_string()) return PythonInteger(value.get_ref<const std::string&>());
    if (value.is_number_float()) {
        const auto number = value.get<double>();
        if (std::isnan(number)) throw std::invalid_argument("cannot convert float NaN to integer");
        if (std::isinf(number)) throw std::overflow_error("cannot convert float infinity to integer");
        std::array<char, 384> buffer{};
        const auto converted = std::to_chars(buffer.data(), buffer.data() + buffer.size(),
                                            std::trunc(number), std::chars_format::fixed, 0);
        if (converted.ec != std::errc{}) throw std::runtime_error("Cannot format Python integer conversion");
        return PythonInteger(std::string_view(buffer.data(),
            static_cast<std::size_t>(converted.ptr - buffer.data())));
    }
    throw PythonJsonTypeError("int() argument must be a string, a bytes-like object or a real number, not '" +
        std::string(json_compat_detail::python_type_name(value)) + "'");
}

double python_json_as_double(const nlohmann::json& value) {
    if (python_json_extended_integer(value))
        return PythonInteger(python_json_extended_integer_decimal(value)).as_double();
    if (value.is_boolean()) return value.get<bool>() ? 1.0 : 0.0;
    if (value.is_number_unsigned()) return PythonInteger::from_unsigned(value.get<std::uint64_t>()).as_double();
    if (value.is_number_integer()) return PythonInteger(value.get<std::int64_t>()).as_double();
    if (value.is_number_float()) return value.get<double>();
    if (value.is_string()) return python_float(value.get_ref<const std::string&>());
    throw PythonJsonTypeError("float() argument must be a string or a real number, not '" +
        std::string(json_compat_detail::python_type_name(value)) + "'");
}

std::int64_t python_json_index_int64(const nlohmann::json& value) {
    if (!value.is_number_integer() && !value.is_boolean() && !python_json_extended_integer(value))
        throw PythonJsonTypeError("'" + std::string(json_compat_detail::python_type_name(value)) +
                                  "' object cannot be interpreted as an integer");
    const auto integer = python_json_int(value);
    const auto& decimal = integer.decimal();
    std::int64_t result{};
    const auto converted = std::from_chars(decimal.data(), decimal.data() + decimal.size(), result);
    if (converted.ec != std::errc{} || converted.ptr != decimal.data() + decimal.size())
        throw std::overflow_error("Python integer exceeds native Int64 range");
    return result;
}

std::uint64_t python_json_exact_uint64(const nlohmann::json& value) {
    if (!value.is_number() && !value.is_boolean() && !python_json_extended_integer(value))
        throw PythonJsonTypeError("Archive count must be numeric, not '" +
            std::string(json_compat_detail::python_type_name(value)) + "'");
    const auto integer = python_json_int(value);
    if (!json_python_equal(value, python_json_integer(integer)))
        throw std::invalid_argument("Archive count must be an exact integer");
    const auto& decimal = integer.decimal();
    std::uint64_t result{};
    const auto converted = std::from_chars(decimal.data(), decimal.data() + decimal.size(), result);
    if (converted.ec != std::errc{} || converted.ptr != decimal.data() + decimal.size())
        throw std::overflow_error("Archive count exceeds native UInt64 range");
    return result;
}

nlohmann::ordered_json parse_python_ordered_json(std::string_view text) {
    return parse_json_values<nlohmann::ordered_json>(text);
}

std::string python_float_repr(double value) {
    if (std::isnan(value)) return "nan";
    if (std::isinf(value)) return std::signbit(value) ? "-inf" : "inf";
    if (value == 0) return std::signbit(value) ? "-0.0" : "0.0";

    const bool negative = std::signbit(value);
    std::array<char, 64> buffer{};
    const auto converted = std::to_chars(buffer.data(), buffer.data() + buffer.size(),
                                        std::fabs(value), std::chars_format::scientific);
    if (converted.ec != std::errc{}) throw std::runtime_error("Cannot format Python float");
    const std::string_view scientific(buffer.data(), static_cast<std::size_t>(converted.ptr - buffer.data()));
    const auto marker = scientific.find('e');
    if (marker == std::string_view::npos) throw std::runtime_error("Invalid scientific float format");
    auto exponent_text = scientific.substr(marker + 1);
    if (!exponent_text.empty() && exponent_text.front() == '+') exponent_text.remove_prefix(1);
    int exponent = 0;
    const auto parsed = std::from_chars(exponent_text.data(), exponent_text.data() + exponent_text.size(), exponent);
    if (parsed.ec != std::errc{} || parsed.ptr != exponent_text.data() + exponent_text.size()) {
        throw std::runtime_error("Invalid scientific float exponent");
    }

    std::string digits;
    for (const auto character : scientific.substr(0, marker)) {
        if (character != '.') digits.push_back(character);
    }
    std::string output = negative ? "-" : "";
    // CPython format_float_short('r') uses decpt <= -4 || decpt > 16,
    // where decpt is this scientific exponent plus one.
    if (exponent < -4 || exponent >= 16) {
        output.push_back(digits.front());
        if (digits.size() > 1) {
            output.push_back('.');
            output.append(digits, 1, std::string::npos);
        }
        output += exponent < 0 ? "e-" : "e+";
        const auto magnitude = std::to_string(exponent < 0 ? -exponent : exponent);
        if (magnitude.size() < 2) output.push_back('0');
        output += magnitude;
    } else {
        const int point = exponent + 1;
        if (point <= 0) {
            output += "0.";
            output.append(static_cast<std::size_t>(-point), '0');
            output += digits;
        } else if (static_cast<std::size_t>(point) >= digits.size()) {
            output += digits;
            output.append(static_cast<std::size_t>(point) - digits.size(), '0');
            output += ".0";
        } else {
            output.append(digits, 0, static_cast<std::size_t>(point));
            output.push_back('.');
            output.append(digits, static_cast<std::size_t>(point), std::string::npos);
        }
    }
    return output;
}

std::string python_json_float(double value) {
    if (std::isnan(value)) return "NaN";
    if (std::isinf(value)) return std::signbit(value) ? "-Infinity" : "Infinity";
    return python_float_repr(value);
}

std::string python_json_string(std::string_view value, bool ensure_ascii) {
    std::string output;
    append_json_string(output, value, ensure_ascii);
    return output;
}

std::string python_canonical_json(const nlohmann::json& value) {
    std::string output;
    append_json(output, value);
    return output;
}

std::string python_ascii_canonical_json(const nlohmann::json& value) {
    std::string output;
    append_json(output, value, false, true);
    return output;
}

std::string python_compact_json(const nlohmann::json& value, bool replace_invalid) {
    std::string output;
    append_json(output, value, false, true, replace_invalid
        ? nlohmann::json::error_handler_t::replace : nlohmann::json::error_handler_t::strict);
    return output;
}

std::string python_sorted_json(const nlohmann::json& value) {
    std::string output;
    append_json(output, value, true, true);
    return output;
}

std::string python_pretty_json(const nlohmann::json& value) {
    std::string output;
    append_pretty_json(output, value, 0);
    return output;
}

std::string python_pretty_json(const nlohmann::ordered_json& value) {
    std::string output;
    append_pretty_json(output, value, 0);
    return output;
}

void write_python_json_file_atomic(const std::filesystem::path& input,
                                   const nlohmann::json& data) {
    const auto path = python_path_normalize(input);
    if (path.native().find('\0') != std::string::npos)
        throw std::invalid_argument("embedded null byte");
    const auto parent = python_path_normalize(path.parent_path());
    std::filesystem::create_directories(parent);
    auto temporary = python_temporary_file("." + path.filename().native(), {}, parent);
    auto output = std::move(temporary.descriptor);
    const auto close_output = [&] {
        const int owned = output.release();
        if (close(owned) != 0) {
            const int error = errno;
            throw std::system_error(error, std::generic_category(),
                                    "Cannot close UF atomic JSON file: " + path.string());
        }
    };
    const auto unlink_temporary = [&] {
        if (unlink(temporary.path.c_str()) != 0) {
            const int error = errno;
            if (error != ENOENT)
                throw std::system_error(error, std::generic_category(), "Cannot remove UF temporary JSON file");
        }
    };
    try {
        try {
            const auto bytes = python_pretty_json(data) + "\n";
            std::size_t offset = 0;
            while (offset < bytes.size()) {
                ProcessInterrupt::check();
                const auto count = write(output.get(), bytes.data() + offset, bytes.size() - offset);
                if (count < 0) {
                    const int error = errno;
                    if (error == EINTR) continue;
                    throw std::system_error(error, std::generic_category(), "Cannot write UF atomic JSON file");
                }
                if (count == 0)
                    throw std::system_error(EIO, std::generic_category(), "Cannot write UF atomic JSON file");
                offset += static_cast<std::size_t>(count);
            }
            for (;;) {
                ProcessInterrupt::check();
                if (fsync(output.get()) == 0) break;
                const int error = errno;
                if (error == EINTR) continue;
                throw std::system_error(error, std::generic_category(), "Cannot sync UF atomic JSON file");
            }
            close_output();
        } catch (...) {
            if (output.get() >= 0) close_output();
            throw;
        }
        if (rename(temporary.path.c_str(), path.c_str()) != 0) {
            const int error = errno;
            throw std::system_error(error, std::generic_category(), "Cannot replace UF JSON file: " + path.string());
        }
    } catch (...) {
        unlink_temporary();
        throw;
    }
    unlink_temporary();
}

std::string python_json_repr(const nlohmann::json& value) {
    std::string output;
    append_repr(output, value);
    return output;
}

std::string python_json_repr(const nlohmann::ordered_json& value) {
    std::string output;
    append_repr(output, value);
    return output;
}

std::string read_python_text_file(const std::filesystem::path& path,
    const std::string& encoding, bool check_interrupt) {
    return read_text_file<std::string>(path, encoding, check_interrupt,
        [](std::string text, const auto& close_input) {
            close_input();
            return text;
        });
}

nlohmann::json read_python_json_file(const std::filesystem::path& path,
    const std::string& encoding, bool close_before_parse, bool check_interrupt) {
    return read_json_file<nlohmann::json>(path, encoding, close_before_parse,
                                         check_interrupt);
}

nlohmann::ordered_json read_python_ordered_json_file(const std::filesystem::path& path,
    const std::string& encoding, bool close_before_parse, bool check_interrupt) {
    return read_json_file<nlohmann::ordered_json>(path, encoding, close_before_parse,
                                                 check_interrupt);
}

}  // namespace uf
