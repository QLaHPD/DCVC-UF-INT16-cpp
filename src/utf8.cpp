#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "uf/utf8.hpp"
#include "uf/process.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iconv.h>
#include <langinfo.h>
#include <limits>
#include <locale.h>
#include <optional>
#include <stdexcept>
#include <system_error>
#include <utility>
#include <sys/stat.h>
#include <unistd.h>

namespace uf {
namespace {

bool continuation(unsigned char byte) { return byte >= 0x80 && byte < 0xc0; }

struct Unit { std::size_t bytes; bool valid; };

struct TextLocale {
    locale_t value{};
    explicit TextLocale(const char* name) : value(newlocale(LC_CTYPE_MASK, name, nullptr)) {}
    ~TextLocale() { if (value) freelocale(value); }
    TextLocale(const TextLocale&) = delete;
    TextLocale& operator=(const TextLocale&) = delete;
};

const char* environment(const char* name) {
    const auto* value = std::getenv(name);
    return value && *value ? value : nullptr;
}

struct TextConverter {
    iconv_t value;
    explicit TextConverter(const std::string& encoding, bool encode = false)
        : value(iconv_open(encode ? encoding.c_str() : "UTF-8",
                           encode ? "UTF-8" : encoding.c_str())) {
        if (value == reinterpret_cast<iconv_t>(-1)) {
            throw std::runtime_error("Cannot open process text encoding: " + encoding);
        }
    }
    ~TextConverter() { iconv_close(value); }
    TextConverter(const TextConverter&) = delete;
    TextConverter& operator=(const TextConverter&) = delete;
};

// CPython 3.10.20 Objects/stringlib/codecs.h and unicodeobject.c consume
// the valid prefix of an incomplete/invalid sequence for one replacement,
// leaving an offending continuation or next start byte for the next step.
Unit unit(std::string_view input, std::size_t offset, bool surrogatepass = false) {
    const auto first = static_cast<unsigned char>(input[offset]);
    if (first < 0x80) return {1, true};
    std::size_t length;
    if (first >= 0xc2 && first <= 0xdf) length = 2;
    else if (first >= 0xe0 && first <= 0xef) length = 3;
    else if (first >= 0xf0 && first <= 0xf4) length = 4;
    else return {1, false};
    for (std::size_t i = 1; i < length; ++i) {
        if (i >= input.size() - offset) return {i, false};
        const auto byte = static_cast<unsigned char>(input[offset + i]);
        if (!continuation(byte)) return {i, false};
        if (i == 1 && ((first == 0xe0 && byte < 0xa0) ||
                       (first == 0xed && byte >= 0xa0 && !surrogatepass) ||
                       (first == 0xf0 && byte < 0x90) ||
                       (first == 0xf4 && byte >= 0x90))) return {1, false};
    }
    return {length, true};
}

void append_code_point(std::string& output, std::uint32_t point) {
    if (point <= 0x7f) output.push_back(static_cast<char>(point));
    else if (point <= 0x7ff) {
        output.push_back(static_cast<char>(0xc0 | (point >> 6)));
        output.push_back(static_cast<char>(0x80 | (point & 0x3f)));
    } else if (point <= 0xffff) {
        output.push_back(static_cast<char>(0xe0 | (point >> 12)));
        output.push_back(static_cast<char>(0x80 | ((point >> 6) & 0x3f)));
        output.push_back(static_cast<char>(0x80 | (point & 0x3f)));
    } else {
        output.push_back(static_cast<char>(0xf0 | (point >> 18)));
        output.push_back(static_cast<char>(0x80 | ((point >> 12) & 0x3f)));
        output.push_back(static_cast<char>(0x80 | ((point >> 6) & 0x3f)));
        output.push_back(static_cast<char>(0x80 | (point & 0x3f)));
    }
}

}  // namespace

std::string utf8_replace(std::string_view bytes) {
    std::string output;
    output.reserve(bytes.size());
    for (std::size_t position = 0; position < bytes.size();) {
        const auto next = unit(bytes, position);
        if (next.valid) output.append(bytes.data() + position, next.bytes);
        else output.append("\xef\xbf\xbd", 3);
        position += next.bytes;
    }
    return output;
}

std::string python_process_text_encoding() {
    const auto* utf8_mode = environment("PYTHONUTF8");
    if (utf8_mode && std::strcmp(utf8_mode, "0") && std::strcmp(utf8_mode, "1")) {
        throw std::invalid_argument("invalid PYTHONUTF8 environment variable value");
    }
    if (utf8_mode && !std::strcmp(utf8_mode, "1")) return "UTF-8";
    // Python starts LC_CTYPE from the environment, retaining C if that locale
    // cannot be installed. Keep this lookup private in the threaded native app.
    TextLocale preferred("");
    TextLocale fallback("C");
    const auto locale = preferred.value ? preferred.value : fallback.value;
    if (!locale) throw std::runtime_error("Cannot create process text locale");
    const auto* name = nl_langinfo_l(NL_LOCALE_NAME(LC_CTYPE), locale);
    const bool legacy = name && (!std::strcmp(name, "C") || !std::strcmp(name, "POSIX"));
    const auto* coerce = environment("PYTHONCOERCECLOCALE");
    if (legacy && !environment("LC_ALL") && (!coerce || std::strcmp(coerce, "0"))) {
        for (const auto* candidate : {"C.UTF-8", "C.utf8", "UTF-8"}) {
            TextLocale target(candidate);
            if (target.value) {
                const auto* codeset = nl_langinfo_l(CODESET, target.value);
                if (codeset && *codeset) return codeset;
            }
        }
    }
    if (legacy && !utf8_mode) return "UTF-8";
    const auto* codeset = nl_langinfo_l(CODESET, locale);
    return codeset && *codeset ? codeset : "UTF-8";
}

std::string python_filesystem_decode(std::string_view bytes) {
    const auto encoding = python_process_text_encoding();
    std::string output;
    output.reserve(bytes.size());
    if (encoding == "UTF-8" || encoding == "UTF8") {
        for (std::size_t position = 0; position < bytes.size();) {
            const auto next = unit(bytes, position);
            if (next.valid) output.append(bytes.data() + position, next.bytes);
            else for (std::size_t index = 0; index < next.bytes; ++index)
                append_code_point(output, 0xdc00 + static_cast<unsigned char>(bytes[position + index]));
            position += next.bytes;
        }
        return output;
    }
    TextConverter converter(encoding);
    auto* input = const_cast<char*>(bytes.data());
    auto remaining = bytes.size();
    std::array<char, 8192> buffer{};
    bool flush = remaining == 0;
    for (;;) {
        char* destination = buffer.data();
        auto available = buffer.size();
        const auto converted = flush
            ? iconv(converter.value, nullptr, nullptr, &destination, &available)
            : iconv(converter.value, &input, &remaining, &destination, &available);
        output.append(buffer.data(), buffer.size() - available);
        if (converted != static_cast<std::size_t>(-1)) {
            if (flush) break;
            flush = remaining == 0;
        } else if (errno == E2BIG) continue;
        else if (remaining && (errno == EILSEQ || errno == EINVAL) &&
                 static_cast<unsigned char>(*input) >= 0x80) {
            append_code_point(output, 0xdc00 + static_cast<unsigned char>(*input));
            ++input; --remaining;
            flush = remaining == 0;
        } else throw std::invalid_argument("Cannot decode filesystem bytes with surrogateescape");
    }
    return output;
}

std::string python_filesystem_encode(std::string_view text) {
    const auto encoding = python_process_text_encoding();
    const bool utf8 = encoding == "UTF-8" || encoding == "UTF8";
    std::optional<TextConverter> converter;
    if (!utf8) converter.emplace(encoding, true);
    std::string output;
    output.reserve(text.size());
    const auto emit = [&](std::string_view run) {
        if (utf8) { output.append(run.data(), run.size()); return; }
        auto* input = const_cast<char*>(run.data());
        auto remaining = run.size();
        std::array<char, 8192> buffer{};
        while (remaining) {
            char* destination = buffer.data();
            auto available = buffer.size();
            const auto converted = iconv(converter->value, &input, &remaining, &destination, &available);
            output.append(buffer.data(), buffer.size() - available);
            if (converted == static_cast<std::size_t>(-1) && errno != E2BIG)
                throw std::invalid_argument("Cannot encode filesystem text");
        }
    };
    std::size_t start = 0;
    for (std::size_t position = 0; position < text.size();) {
        const auto next = unit(text, position, true);
        if (!next.valid) throw std::invalid_argument("Invalid WTF-8 filesystem text");
        const auto first = static_cast<unsigned char>(text[position]);
        std::uint32_t point = next.bytes == 1 ? first : first & (next.bytes == 2 ? 0x1f : next.bytes == 3 ? 0xf : 0x7);
        for (std::size_t index = 1; index < next.bytes; ++index)
            point = (point << 6) | (static_cast<unsigned char>(text[position + index]) & 0x3f);
        if (point >= 0xd800 && point <= 0xdfff) {
            if (point < 0xdc80 || point > 0xdcff)
                throw std::invalid_argument("Cannot encode surrogate in filesystem text");
            emit(text.substr(start, position - start));
            output.push_back(static_cast<char>(point - 0xdc00));
            start = position + next.bytes;
        }
        position += next.bytes;
    }
    emit(text.substr(start));
    if (converter) {
        std::array<char, 8192> buffer{};
        for (;;) {
            char* destination = buffer.data();
            auto available = buffer.size();
            const auto converted = iconv(converter->value, nullptr, nullptr, &destination, &available);
            output.append(buffer.data(), buffer.size() - available);
            if (converted != static_cast<std::size_t>(-1)) break;
            if (errno != E2BIG) throw std::invalid_argument("Cannot finish filesystem text encoding");
        }
    }
    return output;
}

std::string python_utf8_encode(std::string_view text) {
    for (std::size_t position = 0; position < text.size();) {
        const auto next = unit(text, position);
        if (!next.valid) throw std::invalid_argument("Cannot encode filesystem text as strict UTF-8");
        position += next.bytes;
    }
    return std::string(text);
}

std::string python_latin1_encode(std::string_view text) {
    std::string output;
    output.reserve(text.size());
    for (std::size_t position = 0; position < text.size();) {
        const auto next = unit(text, position);
        if (!next.valid) throw std::invalid_argument("Cannot encode invalid text as Latin-1");
        const auto first = static_cast<unsigned char>(text[position]);
        if (next.bytes == 1) output.push_back(static_cast<char>(first));
        else if (next.bytes == 2 && (first == 0xc2 || first == 0xc3)) {
            output.push_back(static_cast<char>(((first & 0x1f) << 6) |
                (static_cast<unsigned char>(text[position + 1]) & 0x3f)));
        } else throw std::invalid_argument("Cannot encode text outside Latin-1");
        position += next.bytes;
    }
    return output;
}

std::string python_latin1_decode(std::string_view bytes) {
    std::string output;
    output.reserve(bytes.size());
    for (const unsigned char byte : bytes) append_code_point(output, byte);
    return output;
}

std::string decode_process_text(std::string bytes, const std::string& encoding) {
    if (encoding != "UTF-8" && encoding != "UTF8") {
        TextConverter converter(encoding);
        std::string converted;
        converted.reserve(bytes.size());
        char* input = bytes.data();
        std::size_t remaining = bytes.size();
        std::array<char, 8192> block{};
        // Flush once input is consumed, including an empty input's initial state.
        bool flush = remaining == 0;
        for (;;) {
            char* output = block.data();
            std::size_t available = block.size();
            const auto count = iconv(converter.value, flush ? nullptr : &input,
                                     flush ? nullptr : &remaining, &output, &available);
            const int error = errno;
            converted.append(block.data(), block.size() - available);
            if (count == static_cast<std::size_t>(-1)) {
                if (error == E2BIG) continue;
                throw std::invalid_argument("Invalid " + encoding + " process text");
            }
            if (count) throw std::invalid_argument("Non-reversible " + encoding + " process text");
            if (flush) break;
            flush = true;
        }
        bytes = std::move(converted);
    }
    std::size_t written = 0;
    for (std::size_t position = 0; position < bytes.size();) {
        const auto next = unit(bytes, position);
        if (!next.valid) throw std::invalid_argument("Invalid UTF-8 process text");
        if (bytes[position] == '\r') {
            ++position;
            if (position < bytes.size() && bytes[position] == '\n') ++position;
            bytes[written++] = '\n';
        } else {
            for (std::size_t i = 0; i < next.bytes; ++i) bytes[written++] = bytes[position++];
        }
    }
    bytes.resize(written);
    return bytes;
}

std::string decode_python_json_bytes(std::string_view bytes) {
    // Adapted from CPython 3.10 json.detect_encoding and surrogatepass Unicode
    // decoding policies. See licenses/CPython-3.10.20.txt. No Python is invoked.
    enum class Encoding { utf8, utf16_be, utf16_le, utf32_be, utf32_le };
    Encoding encoding = Encoding::utf8;
    std::size_t cursor = 0;
    const auto starts = [&](std::string_view prefix) {
        return bytes.size() >= prefix.size() && bytes.substr(0, prefix.size()) == prefix;
    };
    const auto byte = [&](std::size_t index) {
        return static_cast<std::uint32_t>(static_cast<unsigned char>(bytes[index]));
    };
    if (starts(std::string_view("\0\0\xfe\xff", 4))) {
        encoding = Encoding::utf32_be; cursor = 4;
    } else if (starts(std::string_view("\xff\xfe\0\0", 4))) {
        encoding = Encoding::utf32_le; cursor = 4;
    } else if (starts(std::string_view("\xfe\xff", 2))) {
        encoding = Encoding::utf16_be; cursor = 2;
    } else if (starts(std::string_view("\xff\xfe", 2))) {
        encoding = Encoding::utf16_le; cursor = 2;
    } else if (starts(std::string_view("\xef\xbb\xbf", 3))) {
        cursor = 3;
    } else if (bytes.size() >= 4) {
        if (!byte(0)) encoding = byte(1) ? Encoding::utf16_be : Encoding::utf32_be;
        else if (!byte(1)) encoding = byte(2) || byte(3) ? Encoding::utf16_le : Encoding::utf32_le;
    } else if (bytes.size() == 2) {
        if (!byte(0)) encoding = Encoding::utf16_be;
        else if (!byte(1)) encoding = Encoding::utf16_le;
    }
    if (encoding == Encoding::utf8) {
        for (std::size_t position = cursor; position < bytes.size();) {
            const auto next = unit(bytes, position, true);
            if (!next.valid) throw std::invalid_argument("Invalid UTF-8 JSON bytes");
            position += next.bytes;
        }
        return std::string(bytes.substr(cursor));
    }
    std::string output;
    output.reserve(bytes.size() - cursor);
    const bool utf16 = encoding == Encoding::utf16_be || encoding == Encoding::utf16_le;
    const bool little = encoding == Encoding::utf16_le || encoding == Encoding::utf32_le;
    const auto code_unit = [&](std::size_t index) {
        return little ? byte(index) | (byte(index + 1) << 8)
                      : (byte(index) << 8) | byte(index + 1);
    };
    while (cursor < bytes.size()) {
        const std::size_t width = utf16 ? 2 : 4;
        if (bytes.size() - cursor < width)
            throw std::invalid_argument(utf16 ? "Truncated UTF-16 JSON bytes" : "Truncated UTF-32 JSON bytes");
        std::uint32_t point;
        if (utf16) {
            point = code_unit(cursor);
            cursor += 2;
            if (point >= 0xd800 && point <= 0xdbff && bytes.size() - cursor >= 2) {
                const auto next = code_unit(cursor);
                if (next >= 0xdc00 && next <= 0xdfff) {
                    point = 0x10000 + ((point - 0xd800) << 10) + next - 0xdc00;
                    cursor += 2;
                }
            }
        } else {
            point = little ? byte(cursor) | (byte(cursor + 1) << 8) |
                    (byte(cursor + 2) << 16) | (byte(cursor + 3) << 24)
                : (byte(cursor) << 24) | (byte(cursor + 1) << 16) |
                    (byte(cursor + 2) << 8) | byte(cursor + 3);
            cursor += 4;
            if (point > 0x10ffff) throw std::invalid_argument("UTF-32 JSON code point out of range");
        }
        append_code_point(output, point);
    }
    return output;
}

std::string utf8_tail(std::string_view text, std::size_t characters) {
    auto begin = text.size();
    for (std::size_t i = 0; i < characters && begin; ++i) {
        --begin;
        while (begin && continuation(static_cast<unsigned char>(text[begin]))) --begin;
    }
    return std::string(text.substr(begin));
}

std::string utf8_file_tail(int descriptor, std::size_t characters) {
    if (!characters) return {};
    if (characters > (std::numeric_limits<std::size_t>::max() - 3) / 4)
        throw std::length_error("UF diagnostic character limit is too large");
    struct stat info{};
    for (;;) {
        ProcessInterrupt::check();
        if (fstat(descriptor, &info) == 0) break;
        if (errno != EINTR) {
            throw std::system_error(errno, std::generic_category(), "Cannot inspect diagnostic file");
        }
    }
    if (info.st_size < 0) throw std::length_error("Negative diagnostic file size");
    // Four bytes per character plus three bytes of boundary context are enough
    // to keep a partially read leading sequence outside the retained suffix.
    const auto count = static_cast<std::size_t>(std::min<std::uintmax_t>(
        static_cast<std::uintmax_t>(info.st_size), characters * 4 + 3));
    std::string raw(count, '\0');
    const auto offset = info.st_size - static_cast<off_t>(count);
    std::size_t received = 0;
    while (received < raw.size()) {
        ProcessInterrupt::check();
        const auto size = std::min<std::size_t>(raw.size() - received,
            static_cast<std::size_t>(std::numeric_limits<ssize_t>::max()));
        const auto result = pread(descriptor, raw.data() + received, size,
            offset + static_cast<off_t>(received));
        if (result < 0 && errno == EINTR) continue;
        if (result < 0) {
            throw std::system_error(errno, std::generic_category(), "Cannot read diagnostic file");
        }
        if (!result) break;
        received += static_cast<std::size_t>(result);
    }
    raw.resize(received);
    return utf8_tail(utf8_replace(raw), characters);
}

}  // namespace uf
