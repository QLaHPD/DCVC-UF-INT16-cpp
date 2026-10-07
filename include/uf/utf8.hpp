#pragma once

#include <cstddef>
#include <string>
#include <string_view>

namespace uf {

// Decode UTF-8 with CPython's errors='replace' invalid-sequence boundaries.
// Return UTF-8, preserving valid characters and replacing errors with U+FFFD.
std::string utf8_replace(std::string_view bytes);

// Python 3.10 process-text encoding under the environment's locale/UTF-8 policy.
// Uses private locale objects; does not change the process locale/environment.
std::string python_process_text_encoding();

// POSIX filesystem str/bytes conversion with Python's locale/UTF-8 policy and
// surrogateescape. Text uses WTF-8 for U+DC80..U+DCFF; those codepoints encode
// back to the original bytes. Other surrogate/invalid text points raise.
std::string python_filesystem_decode(std::string_view bytes);
std::string python_filesystem_encode(std::string_view text);

// Python str.encode('utf-8', 'strict'): WTF-8 surrogate points are errors.
std::string python_utf8_encode(std::string_view text);
// Python str.encode('latin-1', 'strict'): emit one byte per U+0000..U+00FF;
// other codepoints, surrogates and malformed UTF-8 text raise.
std::string python_latin1_encode(std::string_view text);
// Python bytes.decode('latin-1'): every input byte maps to U+0000..U+00FF.
std::string python_latin1_decode(std::string_view bytes);

// Strictly decode into UTF-8, then normalize CRLF and CR to LF like text=True.
std::string decode_process_text(std::string bytes, const std::string& encoding);

// json.loads(bytes) detection: UTF-8/16/32, optional BOM and null-byte prefix.
// Preserve newlines and surrogatepass code points (WTF-8 for lone surrogates).
// The downstream JSON value/parser must separately support those surrogates.
std::string decode_python_json_bytes(std::string_view bytes);

// Last Unicode characters of already valid UTF-8 text, counted by code point.
std::string utf8_tail(std::string_view text, std::size_t characters);

// Last decoded characters of a seekable diagnostic file, without reading the
// entire file or changing its position. Inspection/read failures propagate;
// interrupted calls retry with foreground interruption checks.
std::string utf8_file_tail(int descriptor, std::size_t characters);

}  // namespace uf
