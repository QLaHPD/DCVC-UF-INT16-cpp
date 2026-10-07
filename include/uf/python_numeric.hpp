#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace uf {

// Numeric str input is interpreted as UTF-8, using the reference Python 3.10
// Unicode 13 decimal/space properties. Unsupported characters remain invalid.
std::string python_numeric_ascii(std::string_view value);

// A nonempty run of Python regex Unicode decimal digits (\d+), without
// signs, separators, whitespace or integer conversion/digit-limit checks.
// Invalid Unix filename bytes act as nondigits, as with surrogateescape.
bool python_decimal_digits(std::string_view value);

// UTF-8 text helpers share the same Unicode tables but preserve the original
// bytes, including digits. Text files use strict decoding, as Path.read_text
// does; strip/splitlines follow Python's whitespace and line boundaries.
std::string python_strip(std::string_view value);
// str.split() without a separator or maxsplit: Unicode whitespace, no empty
// tokens. Preserve the original decoded spelling of each token.
std::vector<std::string> python_split_whitespace(std::string_view value);
std::vector<std::string> python_splitlines(std::string_view value);

// Strip text, then retain leading Python isdigit() characters and one optional
// ASCII sign. Return raw UTF-8; the caller converts the prefix to an integer.
// Nondecimal digits therefore remain conversion errors. Invalid Unix bytes
// behave as nondigits (the environment's Python surrogate escapes).
std::string python_signed_digit_prefix(std::string_view value);

// Python float(str) decimal syntax and binary64 conversion. Unicode numeric
// normalization precedes whitespace/separator handling. Hexadecimal floats
// and trailing junk are rejected; infinity, NaN, overflow and underflow retain
// Python's behavior. The conversion uses a private C numeric locale.
std::string python_float_text(std::string_view value);
double python_float(std::string_view value);

// Python 3.10 Fraction(str) syntax, normalization and integer-ratio conversion.
// Decimal parts apply their own integer digit limits; arithmetic adds no limit.
double python_fraction_double(std::string_view value);
std::string python_fraction_text(std::string_view value);
// str(Fraction(float)): exact binary ratio, reduced; nonfinite input throws.
std::string python_fraction_text(double value);

// argparse's ^-\d+$|^-\d*\.\d+$ matcher, including Unicode decimal digits
// and the regular expression's allowance for one final newline.
bool python_negative_number(std::string_view value);

// Reference interpreter policy: 4300 by default; PYTHONINTMAXSTRDIGITS is
// either zero (unlimited) or at least 640. Read once per process. Calling this
// at executable startup also rejects a malformed environment before dispatch.
std::size_t python_integer_digit_limit();
void check_python_integer_digits(std::size_t count);

}  // namespace uf
