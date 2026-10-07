#pragma once

#include "uf/python_integer.hpp"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace uf {

// Integers outside Int64/UInt64 use a private typed binary payload internally.
// These are numbers on disk: use the Python serializers, never json::dump.
// A text JSON string/object cannot impersonate this internal value type.
inline constexpr std::uint64_t python_json_integer_subtype = 0x55465059494e54ull;
template<class Json>
bool python_json_extended_integer(const Json& value) {
    if (!value.is_binary()) return false;
    const auto& bytes = value.get_binary();
    if (!bytes.has_subtype() || bytes.subtype() != python_json_integer_subtype || bytes.empty())
        return false;
    const auto start = bytes.front() == '-' ? std::size_t{1} : std::size_t{0};
    if (start == bytes.size() || (bytes[start] == '0' && (start || bytes.size() > 1))) return false;
    for (auto index = start; index < bytes.size(); ++index)
        if (bytes[index] < '0' || bytes[index] > '9') return false;
    return true;
}
template<class Json>
std::string_view python_json_extended_integer_decimal(const Json& value) {
    const auto& bytes = value.get_binary();
    return {reinterpret_cast<const char*>(bytes.data()), bytes.size()};
}
nlohmann::json python_json_integer(const PythonInteger& value);

// Python int/float scalar conversion, including bool, decimal strings and
// extended integer payloads. int(float) truncates toward zero without Int64
// narrowing; nonfinite int conversions and overflowing float(int) raise.
// Wrong receiver types raise PythonJsonTypeError (defined in json_compat.hpp).
PythonInteger python_json_int(const nlohmann::json& value);
double python_json_as_double(const nlohmann::json& value);

// Integer/index receiver policy (bool is an int); strings and floats are not
// coerced. Check the native signed 64-bit interface before returning, so an
// exact JSON integer cannot silently wrap or truncate at get<int64_t>().
std::int64_t python_json_index_int64(const nlohmann::json& value);

// Lossless count conversion for native archive interfaces. Accept represented
// numbers equal to a nonnegative integer (including bool/integral floats),
// rejecting text, fractions and values outside UInt64 without wrapping.
std::uint64_t python_json_exact_uint64(const nlohmann::json& value);

// json.loads numeric policies: exact integer tokens, Python decimal digit
// limits, binary64 overflow/underflow and NaN/Infinity/-Infinity constants.
// Strings use Python escape/surrogate-pair policies; lone surrogates are stored
// internally as WTF-8 and re-emitted as escapes by ASCII serializers. Native
// string consumers must retain that representation or apply their own boundary
// conversion. Container syntax is still validated by nlohmann.
nlohmann::json parse_python_json(std::string_view text);
nlohmann::ordered_json parse_python_ordered_json(std::string_view text);

// CPython str/repr notation for a binary64 value: shortest round-trip digits,
// fixed notation for decimal exponents -4..15, a .0 for fixed integers,
// signed zero and a signed exponent with at least two digits. Nonfinite values
// use nan/inf/-inf. No Python runtime is used.
std::string python_float_repr(double value);

// json.dumps float notation: finite values share str/repr formatting;
// nonfinite values use NaN/Infinity/-Infinity.
std::string python_json_float(double value);

// Python JSON string escaping. ASCII output accepts internal WTF-8 surrogate
// codepoints; UTF-8 output rejects them as Python's subsequent strict encode()
// does. Invalid byte sequences are rejected in either mode.
std::string python_json_string(std::string_view value, bool ensure_ascii = true);

// json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
// for represented JSON values. Shared-work hashes depend on these exact bytes,
// including float notation. Nonfinite floats follow Python's default allow_nan.
std::string python_canonical_json(const nlohmann::json& value);

// Compact sorted JSON with Python's default ensure_ascii=True. Prepared model
// identity hashing uses these bytes for headers and tensor descriptors.
std::string python_ascii_canonical_json(const nlohmann::json& value);

// Compact numeric-preserving JSON for native worker/event messages. Sorted
// object keys, ASCII string escaping and Python number notation preserve values;
// replacement mode retains nlohmann's existing invalid-UTF-8 byte policy.
std::string python_compact_json(const nlohmann::json& value,
                                bool replace_invalid = false);

// json.dumps(value, sort_keys=True), with default spacing, ASCII escaping
// and allow_nan, for represented JSON values. UF image headers use these bytes.
std::string python_sorted_json(const nlohmann::json& value);

// json.dumps(value, indent=2, sort_keys=True), with its default ensure_ascii
// and allow_nan policies, for represented JSON values. No final newline.
std::string python_pretty_json(const nlohmann::json& value);

// json.dumps(value, indent=2) for ordered objects, preserving insertion order.
std::string python_pretty_json(const nlohmann::ordered_json& value);

// storage.atomic_json: create parents, write Python pretty JSON and a newline
// to an owned noninheritable temporary descriptor, fsync and close it before
// replacing the destination, then unlink the temporary in finally. This does
// not sync the parent directory; callers apply their maintained sync sequence.
void write_python_json_file_atomic(const std::filesystem::path& path,
                                   const nlohmann::json& value);

// Python repr for represented JSON values. Strings use CPython 3.10 / Unicode
// 13.0.0 quote, escape and printability rules. Objects retain their native
// iteration order; the ordered overload preserves dictionaries read from disk.
// WTF-8 surrogate codepoints are escaped; other invalid bytes and non-JSON
// native types are rejected.
std::string python_json_repr(const nlohmann::json& value);
std::string python_json_repr(const nlohmann::ordered_json& value);

// Path.read_text: native POSIX path, strict decoding and universal newlines,
// with checked close before returning text. BOMs are retained as text data.
std::string read_python_text_file(const std::filesystem::path& path,
    const std::string& encoding, bool check_interrupt = true);

// Owned POSIX text reads with strict decoding, universal newlines and complete
// JSON parsing. Text BOMs are rejected as json.loads(str) does. Set
// close_before_parse for json.loads(Path.read_text()); clear it for json.load()
// inside a with block. Checked close errors can replace a read/parse error.
// Cleanup ownership checks may disable interruption after a signal. These use
// the shared string/number parser and nlohmann's remaining container limits.
nlohmann::json read_python_json_file(const std::filesystem::path& path,
    const std::string& encoding, bool close_before_parse,
    bool check_interrupt = true);
nlohmann::ordered_json read_python_ordered_json_file(const std::filesystem::path& path,
    const std::string& encoding, bool close_before_parse,
    bool check_interrupt = true);

}  // namespace uf
