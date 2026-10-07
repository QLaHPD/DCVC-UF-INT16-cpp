#include "uf/image_pixels.hpp"

#include "uf/media.hpp"
#include "uf/process.hpp"
#include "image_native.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <csetjmp>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#ifdef UF_HAS_JPEG
#include <jpeglib.h>
#include <jerror.h>
#endif
#ifdef UF_HAS_PNG
#include <png.h>
#endif
#ifdef UF_HAS_WEBP
#include <webp/decode.h>
#include <webp/demux.h>
#endif
#ifdef UF_HAS_ZLIB
#include <zlib.h>
#endif

namespace uf {
namespace {

// Twice Pillow's default MAX_IMAGE_PIXELS, where its warning becomes an error.
constexpr std::uint64_t max_pixels = image_native::max_pixels;
constexpr std::uint64_t max_rgb_bytes = max_pixels * 3;

std::size_t rgb_size(std::uint64_t width, std::uint64_t height) {
    if (!width || !height || width >= std::numeric_limits<std::uint32_t>::max() ||
        height >= std::numeric_limits<std::uint32_t>::max() ||
        width * height > max_pixels ||
        width * height > std::numeric_limits<std::size_t>::max() / 3) {
        throw std::runtime_error("Invalid or excessive UF image dimensions");
    }
    return static_cast<std::size_t>(width * height * 3);
}

std::uint16_t value16(const std::uint8_t* p, bool little) {
    return little ? std::uint16_t(p[0] | (std::uint16_t(p[1]) << 8))
                  : std::uint16_t((std::uint16_t(p[0]) << 8) | p[1]);
}

std::uint32_t value32(const std::uint8_t* p, bool little) {
    if (little) {
        return std::uint32_t(p[0]) | (std::uint32_t(p[1]) << 8) |
               (std::uint32_t(p[2]) << 16) | (std::uint32_t(p[3]) << 24);
    }
    return (std::uint32_t(p[0]) << 24) | (std::uint32_t(p[1]) << 16) |
           (std::uint32_t(p[2]) << 8) | std::uint32_t(p[3]);
}

class ImageFile {
public:
    explicit ImageFile(const std::filesystem::path& path) {
        if (path.native().find('\0') != std::string::npos)
            throw std::invalid_argument("embedded null byte");
        input_.open(path, std::ios::binary | std::ios::ate);
        if (!input_) throw std::runtime_error("Cannot open UF image input: " + path.string());
        const auto end = input_.tellg();
        if (end < 0) throw std::runtime_error("Cannot determine UF image input size");
        size_ = static_cast<std::uint64_t>(end);
    }
    std::uint64_t size() const { return size_; }
    void close_checked() {
        if (!input_.is_open()) return;
        input_.clear();
        input_.close();
        if (input_.fail()) throw std::runtime_error("Cannot close UF image input");
    }
    bool read(std::uint64_t offset, void* destination, std::size_t count) {
        if (offset > size_ || count > size_ - offset ||
            offset > std::uint64_t(std::numeric_limits<std::streamoff>::max()) ||
            count > std::size_t(std::numeric_limits<std::streamsize>::max())) return false;
        input_.clear();
        input_.seekg(static_cast<std::streamoff>(offset));
        if (!input_) return false;
        input_.read(static_cast<char*>(destination), static_cast<std::streamsize>(count));
        return input_.gcount() == static_cast<std::streamsize>(count);
    }
    std::vector<std::uint8_t> bytes(std::uint64_t offset, std::uint64_t count) {
        if (count > std::numeric_limits<std::size_t>::max() || offset > size_ || count > size_ - offset) {
            throw std::runtime_error("Truncated or excessive UF image input");
        }
        std::vector<std::uint8_t> result(static_cast<std::size_t>(count));
        std::size_t done = 0;
        while (done < result.size()) {
            ProcessInterrupt::check();
            const auto block = std::min<std::size_t>(65536, result.size() - done);
            if (!read(offset + done, result.data() + done, block)) {
                throw std::runtime_error("Truncated UF image input");
            }
            done += block;
        }
        return result;
    }
private:
    std::ifstream input_;
    std::uint64_t size_{};
};

std::uint64_t value64(const std::uint8_t* p, bool little) {
    return little ? std::uint64_t(value32(p, true)) | (std::uint64_t(value32(p + 4, true)) << 32)
                  : (std::uint64_t(value32(p, false)) << 32) | value32(p + 4, false);
}

// Read just IFD0. This works with file regions and libavif's in-memory EXIF,
// including BigTIFF; unrelated metadata and compressed pixels are skipped.
template<class Reader>
std::optional<unsigned> tiff_orientation_at(Reader&& read, std::uint64_t size) {
    std::array<std::uint8_t, 16> header{};
    if (size < 8 || !read(0, header.data(), 8)) return {};
    std::uint64_t base = 0;
    while (std::memcmp(header.data(), "Exif\0\0", 6) == 0) {
        ProcessInterrupt::check();
        base += 6;
        if (size - base < 8 || !read(base, header.data(), 8)) return {};
    }
    const auto length = size - base;
    const bool little = header[0] == 'I' && header[1] == 'I';
    if (!little && !(header[0] == 'M' && header[1] == 'M')) return {};
    const auto version = value16(header.data() + 2, little);
    const bool big = version == 43;
    if (version != 42 && !big) return {};
    std::uint64_t offset = value32(header.data() + 4, little);
    if (big) {
        if (length < 16 || value16(header.data() + 4, little) != 8 ||
            value16(header.data() + 6, little) != 0 ||
            !read(base + 8, header.data() + 8, 8)) return {};
        offset = value64(header.data() + 8, little);
    }
    const unsigned count_size = big ? 8 : 2, entry_size = big ? 20 : 12;
    if (offset > length || length - offset < count_size ||
        !read(base + offset, header.data(), count_size)) return {};
    const std::uint64_t count = big ? value64(header.data(), little) : value16(header.data(), little);
    if (count > (length - offset - count_size) / entry_size) return {};
    std::array<std::uint8_t, 20> entry{};
    for (std::uint64_t i = 0; i < count; ++i) {
        ProcessInterrupt::check();
        if (!read(base + offset + count_size + i * entry_size, entry.data(), entry_size)) return {};
        if (value16(entry.data(), little) != 0x0112) continue;
        const auto values = big ? value64(entry.data() + 4, little) : value32(entry.data() + 4, little);
        if (values != 1) continue;
        const auto type = value16(entry.data() + 2, little);
        const auto* stored = entry.data() + (big ? 12 : 8);
        // Preserve even an invalid numeric tag: its presence suppresses XMP,
        // and ImageOps leaves unknown orientation numbers unchanged.
        if (type == 3) return value16(stored, little);
        if (type == 4) return value32(stored, little);
    }
    return {};
}

std::optional<unsigned> tiff_orientation(ImageFile& file, std::uint64_t start,
                                         std::uint64_t length) {
    if (start > file.size() || length > file.size() - start) return {};
    return tiff_orientation_at([&](std::uint64_t offset, void* bytes, std::size_t count) {
        return offset <= length && count <= length - offset && file.read(start + offset, bytes, count);
    }, length);
}

std::optional<unsigned> xmp_orientation(std::string_view text) {
    constexpr std::string_view marker = "tiff:Orientation";
    std::size_t position = 0;
    while ((position = text.find(marker, position)) != std::string_view::npos) {
        position += marker.size();
        auto digit = position;
        if (text.substr(position, 2) == "=\"") digit += 2;
        else if (text.substr(position, 1) == ">") ++digit;
        else continue;
        if (digit < text.size() && text[digit] >= '0' && text[digit] <= '9') {
            return static_cast<unsigned>(text[digit] - '0');
        }
    }
    return {};
}

std::optional<unsigned> xmp_orientation(ImageFile& file, std::uint64_t start, std::uint64_t length) {
    if (start > file.size() || length > file.size() - start) return {};
    std::array<std::uint8_t, 65536 + 64> bytes{};
    std::size_t carry = 0;
    std::uint64_t done = 0;
    while (done < length) {
        ProcessInterrupt::check();
        const auto block = static_cast<std::size_t>(std::min<std::uint64_t>(65536, length - done));
        if (!file.read(start + done, bytes.data() + carry, block)) return {};
        if (auto result = xmp_orientation(std::string_view(
                reinterpret_cast<const char*>(bytes.data()), carry + block))) return result;
        const auto total = carry + block;
        carry = std::min<std::size_t>(64, total);
        std::memmove(bytes.data(), bytes.data() + total - carry, carry);
        done += block;
    }
    return {};
}

constexpr std::size_t png_text_chunk_limit = 1024 * 1024;
constexpr std::size_t png_text_memory_limit = 64 * png_text_chunk_limit;

// Pillow counts decoded Unicode characters, rather than UTF-8 bytes, for
// iTXt. Reject overlong encodings, surrogates and incomplete sequences.
std::optional<std::size_t> utf8_characters(std::string_view value) {
    std::size_t count = 0;
    for (std::size_t at = 0; at < value.size(); ++count) {
        if ((count & 65535) == 0) ProcessInterrupt::check();
        const auto first = static_cast<unsigned char>(value[at]);
        const auto width = first < 0x80 ? 1u : first >= 0xc2 && first <= 0xdf ? 2u :
                           first >= 0xe0 && first <= 0xef ? 3u :
                           first >= 0xf0 && first <= 0xf4 ? 4u : 0u;
        if (!width || width > value.size() - at) return {};
        for (unsigned index = 1; index < width; ++index) {
            const auto next = static_cast<unsigned char>(value[at + index]);
            if (next < 0x80 || next > 0xbf) return {};
        }
        if (width >= 3) {
            const auto second = static_cast<unsigned char>(value[at + 1]);
            if ((first == 0xe0 && second < 0xa0) || (first == 0xed && second >= 0xa0) ||
                (first == 0xf0 && second < 0x90) || (first == 0xf4 && second > 0x8f)) return {};
        }
        at += width;
    }
    return count;
}

std::optional<std::string> png_inflate_text(std::string_view compressed) {
#ifdef UF_HAS_ZLIB
    struct InflateState {
        z_stream stream{};
        InflateState() {
            if (inflateInit(&stream) != Z_OK) throw std::runtime_error("Cannot create PNG text inflater");
        }
        ~InflateState() { inflateEnd(&stream); }
    } inflater;
    std::string output(16 * 1024, '\0');
    inflater.stream.next_in = reinterpret_cast<Bytef*>(const_cast<char*>(compressed.data()));
    inflater.stream.avail_in = static_cast<uInt>(compressed.size());
    inflater.stream.next_out = reinterpret_cast<Bytef*>(output.data());
    inflater.stream.avail_out = static_cast<uInt>(output.size());
    int status;
    for (;;) {
        ProcessInterrupt::check();
        status = inflate(&inflater.stream, Z_SYNC_FLUSH);
        if (status == Z_MEM_ERROR) throw std::bad_alloc();
        if (status != Z_OK && status != Z_STREAM_END && status != Z_BUF_ERROR) return {};
        if (status == Z_STREAM_END || inflater.stream.avail_out ||
            output.size() == png_text_chunk_limit) break;
        const auto written = static_cast<std::size_t>(inflater.stream.total_out);
        output.resize(std::min(png_text_chunk_limit, output.size() * 2));
        inflater.stream.next_out = reinterpret_cast<Bytef*>(output.data() + written);
        inflater.stream.avail_out = static_cast<uInt>(output.size() - written);
    }
    // decompressobj.decompress(..., MAX_TEXT_CHUNK) rejects unconsumed
    // compressed input when its output limit stops decoding. Bytes after a
    // completed stream are unused_data and do not trigger that error.
    if (status != Z_STREAM_END && inflater.stream.avail_in) {
        throw std::runtime_error("Decompressed PNG text exceeds Pillow's chunk limit");
    }
    output.resize(inflater.stream.total_out);
    return output;
#else
    (void)compressed;
    return {};
#endif
}

std::vector<std::uint8_t> png_raw_exif(std::string_view text) {
    // Image.getexif() drops the profile's first three lines, joins the rest,
    // then uses bytes.fromhex(). Whitespace is permitted between byte pairs.
    std::size_t at = 0;
    for (unsigned line = 0; line < 3; ++line) {
        const auto newline = text.find('\n', at);
        if (newline == std::string_view::npos) return {};
        at = newline + 1;
    }
    std::string hex;
    hex.reserve(text.size() - at);
    for (; at < text.size(); ++at) {
        if ((at & 65535) == 0) ProcessInterrupt::check();
        if (text[at] != '\n') hex.push_back(text[at]);
    }
    const auto digit = [](unsigned char value) -> int {
        if (value >= '0' && value <= '9') return value - '0';
        if (value >= 'a' && value <= 'f') return value - 'a' + 10;
        if (value >= 'A' && value <= 'F') return value - 'A' + 10;
        return -1;
    };
    std::vector<std::uint8_t> result;
    for (at = 0; at < hex.size();) {
        if ((at & 65535) == 0) ProcessInterrupt::check();
        if (std::string_view(" \t\n\r\f\v").find(hex[at]) != std::string_view::npos) {
            ++at;
            continue;
        }
        const auto high = digit(static_cast<unsigned char>(hex[at]));
        const auto low = at + 1 < hex.size() ? digit(static_cast<unsigned char>(hex[at + 1])) : -1;
        if (high < 0 || low < 0) throw std::runtime_error("Invalid hexadecimal PNG EXIF profile");
        result.push_back(static_cast<std::uint8_t>((high << 4) | low));
        at += 2;
    }
    return result;
}

struct PngMetadata {
    std::size_t text_memory{};
    std::optional<std::string> exif, raw_exif, xml, xmp;
    bool exif_is_text{}, xmp_is_text{};

    void count_text(std::size_t count) {
        if (count > png_text_memory_limit - text_memory) {
            throw std::runtime_error("PNG text exceeds Pillow's total text memory limit");
        }
        text_memory += count;
    }
    void icc(std::string_view bytes) {
        const auto separator = bytes.find('\0');
        if (separator == std::string_view::npos || separator + 1 >= bytes.size()) {
            throw std::runtime_error("Invalid PNG iCCP compression method");
        }
        if (static_cast<unsigned char>(bytes[separator + 1]) != 0) {
            throw std::runtime_error("Unknown PNG iCCP compression method");
        }
        // Pillow accepts zlib errors as an absent profile, but enforces the
        // same per-chunk decompression limit as compressed text. ICC bytes do
        // not count toward text_memory and convert(RGB) does not apply them.
        (void)png_inflate_text(bytes.substr(separator + 2));
    }
    void store_text(std::string_view key, std::string_view value, bool exif_bytes = false) {
        if (key == "exif") { exif = value; exif_is_text = !exif_bytes; }
        else if (key == "Raw profile type exif") raw_exif = value;
        else if (key == "XML:com.adobe.xmp") xml = value;
        else if (key == "xmp") { xmp = value; xmp_is_text = true; }
    }
    void text(std::string_view bytes, bool international, bool ztext) {
        const auto key_end = bytes.find('\0');
        const auto key = bytes.substr(0, key_end);
        auto value = key_end == std::string_view::npos ? std::string_view{} : bytes.substr(key_end + 1);
        std::optional<std::string> inflated;
        if (international) {
            if (key_end == std::string_view::npos || value.size() < 2) return;
            const auto compressed = static_cast<unsigned char>(value[0]);
            const auto method = static_cast<unsigned char>(value[1]);
            value.remove_prefix(2);
            const auto language_end = value.find('\0');
            if (language_end == std::string_view::npos) return;
            const auto language = value.substr(0, language_end);
            value.remove_prefix(language_end + 1);
            const auto translated_end = value.find('\0');
            if (translated_end == std::string_view::npos) return;
            const auto translated = value.substr(0, translated_end);
            value.remove_prefix(translated_end + 1);
            if (compressed) {
                if (method != 0) return;
                inflated = png_inflate_text(value);
                if (!inflated) return;
                value = *inflated;
            }
            // Pillow saves raw iTXt XMP before attempting the UTF-8 fields.
            if (key == "XML:com.adobe.xmp") { xmp = value; xmp_is_text = false; }
            if (!utf8_characters(language) || !utf8_characters(translated)) return;
            const auto characters = utf8_characters(value);
            if (!characters) return;
            store_text(key, value);
            count_text(*characters);
        } else {
            if (ztext) {
                const auto method = value.empty() ? 0 : static_cast<unsigned char>(value.front());
                if (method != 0) throw std::runtime_error("Unknown PNG zTXt compression method");
                if (!value.empty()) value.remove_prefix(1);
                inflated = png_inflate_text(value);
                value = inflated ? std::string_view(*inflated) : std::string_view{};
            }
            if (!key.empty()) {
                // Latin-1 characters each occupy one byte. tEXt's special
                // 'exif' value remains bytes; zTXt stores ordinary text.
                store_text(key, value, !ztext);
                count_text(value.size());
            }
        }
    }
    unsigned orientation() const {
        std::optional<unsigned> tag;
        const auto inspect = [&](const std::uint8_t* data, std::size_t size) {
            return tiff_orientation_at([&](std::uint64_t offset, void* destination, std::size_t count) {
                if (offset > size || count > size - offset) return false;
                std::memcpy(destination, data + offset, count);
                return true;
            }, size);
        };
        if (exif) {
            if (exif_is_text && !exif->empty()) throw std::runtime_error("PNG EXIF metadata must be bytes");
            tag = inspect(reinterpret_cast<const std::uint8_t*>(exif->data()), exif->size());
        } else if (raw_exif) {
            const auto bytes = png_raw_exif(*raw_exif);
            tag = inspect(bytes.data(), bytes.size());
        }
        if (tag) return *tag;
        if (xml && !xml->empty()) return xmp_orientation(*xml).value_or(1);
        if (xmp && !xmp->empty()) {
            if (xmp_is_text) throw std::runtime_error("PNG fallback XMP metadata must be bytes");
            return xmp_orientation(*xmp).value_or(1);
        }
        return 1;
    }
};

void png_header_crc(ImageFile& file, std::uint64_t start, std::uint32_t length) {
    static const auto table = [] {
        std::array<std::uint32_t, 256> result{};
        for (std::uint32_t index = 0; index < result.size(); ++index) {
            auto value = index;
            for (unsigned bit = 0; bit < 8; ++bit) value = (value >> 1) ^ ((value & 1) ? 0xedb88320u : 0);
            result[index] = value;
        }
        return result;
    }();
    std::uint32_t crc = 0xffffffffu;
    std::array<std::uint8_t, 65536> block{};
    const auto size = std::uint64_t(length) + 4;  // Chunk type and payload.
    for (std::uint64_t done = 0; done < size;) {
        ProcessInterrupt::check();
        const auto count = static_cast<std::size_t>(std::min<std::uint64_t>(block.size(), size - done));
        if (!file.read(start + done, block.data(), count)) throw std::runtime_error("Truncated PNG header");
        for (std::size_t index = 0; index < count; ++index) crc = table[(crc ^ block[index]) & 255] ^ (crc >> 8);
        done += count;
    }
    if (!file.read(start + size, block.data(), 4) || value32(block.data(), false) != (crc ^ 0xffffffffu)) {
        throw std::runtime_error("Invalid PNG header checksum");
    }
}

struct PngRegion {
    std::uint64_t offset{}, size{};
};

struct PngFirstFrame {
    unsigned orientation{1};
    std::uint8_t header_depth{}, header_color{};
    std::array<std::uint8_t, 13> ihdr{};
    std::array<std::uint8_t, 3> background{};
    std::uint32_t canvas_width{}, canvas_height{}, width{}, height{}, x{}, y{};
    std::uint64_t data_begin{}, data_end{};
    std::vector<PngRegion> headers;
    bool stream{}, header_mode{};
};

PngFirstFrame png_metadata(ImageFile& file, bool transpose, bool verify_header = false) {
    PngMetadata metadata;
    PngFirstFrame first;
    std::array<std::uint8_t, 8> header{};
    std::optional<std::uint32_t> frames;
    std::optional<std::uint32_t> sequence;
    bool image_data = false, frame_control = false, animated = false;
    bool collect_data = true;
    for (std::uint64_t position = 8; position <= file.size() && file.size() - position >= 8;) {
        ProcessInterrupt::check();
        if (!file.read(position, header.data(), header.size())) break;
        const auto length = value32(header.data(), false);
        const std::string_view kind(reinterpret_cast<const char*>(header.data() + 4), 4);
        const auto word = [](unsigned char value) {
            return (value >= 'a' && value <= 'z') || (value >= 'A' && value <= 'Z') ||
                   (value >= '0' && value <= '9') || value == '_';
        };
        if (!std::all_of(kind.begin(), kind.end(), word)) {
            if (image_data) break;  // Pillow's load_end catches bad chunk IDs.
            throw std::runtime_error("Invalid PNG chunk identifier");
        }
        if (kind == "IEND" || (image_data && animated && kind == "fcTL")) break;
        if (length > file.size() - position - 8) throw std::runtime_error("Truncated PNG chunk");
        const auto data_chunk = kind == "IDAT" || kind == "fdAT" || (image_data && kind == "DDAT");
        if (kind == "fdAT") {
            if (length < 4) throw std::runtime_error("Truncated PNG animation data");
            std::array<std::uint8_t, 4> value{};
            if (!file.read(position + 8, value.data(), value.size())) throw std::runtime_error("Truncated PNG animation data");
            const auto number = value32(value.data(), false);
            if (!sequence || std::uint64_t(*sequence) + 1 != number) {
                throw std::runtime_error("PNG animation frame sequence error");
            }
            sequence = number;
        }
        if (data_chunk) {
            if (!image_data) {
                rgb_size(first.canvas_width, first.canvas_height);
                if (std::uint64_t(first.x) + first.width > first.canvas_width ||
                    std::uint64_t(first.y) + first.height > first.canvas_height) {
                    throw std::runtime_error("PNG animation frame exceeds its final canvas");
                }
                animated = frames.has_value() &&
                    (std::uint64_t(*frames) + (frame_control ? 0u : 1u) > 1);
                first.stream = first.stream || frames.has_value() || frame_control || kind == "fdAT";
                first.data_begin = position;
                image_data = true;
                // Image.open stops at this first tile. verify subsequently
                // checks chunk CRCs without processing/decompressing pixels or
                // interpreting metadata after this point.
                if (verify_header) {
                    if (!first.header_mode) throw std::runtime_error("Unsupported PNG image mode");
                    // fdAT's tile position follows its four-byte sequence.
                    if (kind == "fdAT") first.data_begin += 4;
                    return first;
                }
            }
            if (collect_data) first.data_end = position + std::uint64_t(length) + 12;
        } else if (kind == "iCCP") {
            if (image_data) collect_data = false;
            const auto bytes = file.bytes(position + 8, length);
            metadata.icc(std::string_view(bytes.empty() ? "" : reinterpret_cast<const char*>(bytes.data()), bytes.size()));
            // The synthesized stream omits profiles already handled here.
            // libpng's ICC parsing/validation differs from Pillow's reader.
            first.stream = true;
        } else if (kind == "IHDR") {
            if (image_data) collect_data = false;
            std::array<std::uint8_t, 13> ihdr{};
            if (length < 13 || !file.read(position + 8, ihdr.data(), ihdr.size())) {
                throw std::runtime_error("Truncated PNG image header");
            }
            if (!image_data) {
                first.ihdr = ihdr;
                if (verify_header) {
                    const auto depth = ihdr[8], color = ihdr[9];
                    const auto supported =
                        (color == 0 && (depth == 1 || depth == 2 || depth == 4 || depth == 8 || depth == 16)) ||
                        (color == 3 && (depth == 1 || depth == 2 || depth == 4 || depth == 8)) ||
                        ((color == 2 || color == 4 || color == 6) && (depth == 8 || depth == 16));
                    if (supported) {
                        first.header_mode = true;
                        first.header_depth = depth;
                        first.header_color = color;
                    }
                    if (ihdr[11]) throw std::runtime_error("Unknown PNG filter category");
                }
                first.canvas_width = value32(ihdr.data(), false);
                first.canvas_height = value32(ihdr.data() + 4, false);
                if (!frame_control) { first.width = first.canvas_width; first.height = first.canvas_height; }
            }
        } else if (kind == "tEXt" || kind == "zTXt" || kind == "iTXt" || kind == "eXIf") {
            if (image_data) collect_data = false;
            const auto bytes = file.bytes(position + 8, length);
            const std::string_view data(bytes.empty() ? "" : reinterpret_cast<const char*>(bytes.data()), bytes.size());
            if (kind == "eXIf") {
                metadata.exif = data;
                metadata.exif_is_text = false;
            } else metadata.text(data, kind == "iTXt", kind == "zTXt");
        } else if (kind == "acTL") {
            if (image_data) collect_data = false;
            if (length < 8) throw std::runtime_error("Truncated PNG animation control");
            if (!image_data) {
                std::array<std::uint8_t, 4> count{};
                if (!file.read(position + 8, count.data(), count.size())) throw std::runtime_error("Truncated PNG animation control");
                const auto value = value32(count.data(), false);
                if (frames) frames.reset();
                else if (value && value <= 0x80000000u) frames = value;
            }
        } else if (kind == "fcTL") {
            if (image_data) collect_data = false;
            std::array<std::uint8_t, 26> control{};
            if (length < control.size() || !file.read(position + 8, control.data(), control.size())) {
                throw std::runtime_error("Truncated PNG animation frame control");
            }
            const auto number = value32(control.data(), false);
            if ((!sequence && number != 0) || (sequence && std::uint64_t(*sequence) + 1 != number)) {
                throw std::runtime_error("PNG animation frame sequence error");
            }
            sequence = number;
            const auto width = value32(control.data() + 4, false), height = value32(control.data() + 8, false);
            const auto x = value32(control.data() + 12, false), y = value32(control.data() + 16, false);
            if (std::uint64_t(x) + width > first.canvas_width || std::uint64_t(y) + height > first.canvas_height) {
                throw std::runtime_error("PNG animation frame exceeds its canvas");
            }
            if (!image_data) {
                frame_control = true;
                first.width = width; first.height = height; first.x = x; first.y = y;
            }
        } else {
            if (image_data) collect_data = false;
            if (verify_header && !image_data) {
                // Pillow's initial chunk handlers read these scalar fields.
                const auto depth = first.header_depth, color = first.header_color;
                const bool short_transparency = kind == "tRNS" &&
                    (((color == 0 && depth) && length < 2) || (color == 2 && length < 6));
                if (short_transparency || (kind == "gAMA" && length < 4) ||
                    (kind == "cHRM" && length % 4) || (kind == "sRGB" && length < 1) ||
                    (kind == "pHYs" && length < 9)) {
                    throw std::runtime_error("Truncated PNG header metadata");
                }
            }
            if (!image_data && kind == "PLTE" && first.ihdr[9] == 3) {
                first.background.fill(0);
                if (length >= 3 && !file.read(position + 8, first.background.data(), first.background.size())) {
                    throw std::runtime_error("Truncated PNG palette");
                }
            }
        }
        // Pillow verifies initial metadata CRCs; loading the pixels and the
        // trailing metadata skips CRCs. Do not stop at the first EXIF tag:
        // later chunks replace the corresponding metadata dictionary entry.
        if (!image_data) {
            png_header_crc(file, position + 4, length);
            if (kind != "IHDR" && kind != "acTL" && kind != "fcTL" &&
                kind != "tEXt" && kind != "zTXt" && kind != "iTXt" && kind != "eXIf" && kind != "iCCP") {
                first.headers.push_back({position, std::uint64_t(length) + 12});
            }
        }
        position += std::uint64_t(length) + 12;
    }
    if (verify_header) throw std::runtime_error("PNG contains no image data tile");
    first.orientation = transpose ? metadata.orientation() : 1u;
    return first;
}

void verify_png(ImageFile& file) {
    const auto first = png_metadata(file, false, true);
    std::array<std::uint8_t, 8> header{};
    for (auto position = first.data_begin;;) {
        ProcessInterrupt::check();
        if (!file.read(position, header.data(), header.size()))
            throw std::runtime_error("Truncated PNG file");
        const auto length = value32(header.data(), false);
        const std::string_view kind(reinterpret_cast<const char*>(header.data() + 4), 4);
        if (!std::all_of(kind.begin(), kind.end(), [](unsigned char value) {
                return (value >= 'a' && value <= 'z') || (value >= 'A' && value <= 'Z') ||
                       (value >= '0' && value <= '9') || value == '_';
            })) throw std::runtime_error("Invalid PNG chunk identifier");
        // PngStream.verify stops on the IEND header; it does not read or check
        // that chunk's payload/CRC or any trailing bytes.
        if (kind == "IEND") return;
        if (position > file.size() || file.size() - position < 12 ||
            length > file.size() - position - 12)
            throw std::runtime_error("Truncated PNG file");
        png_header_crc(file, position + 4, length);
        position += std::uint64_t(length) + 12;
    }
}

// Header-only JPEG acceptance follows Pillow's JpegImagePlugin/ImageFile;
// licenses/Pillow.txt. Its verify() does not decode the compressed scan.
void verify_jpeg(ImageFile& file) {
    std::array<std::uint8_t, 3> signature{};
    if (!file.read(0, signature.data(), signature.size()) ||
        signature[0] != 0xff || signature[1] != 0xd8 || signature[2] != 0xff)
        throw std::runtime_error("Invalid JPEG signature");
    std::uint64_t position = 3;
    std::uint16_t width{}, height{};
    bool frame_mode = false, marker_prefix = true;
    std::size_t icc_count{};
    std::vector<std::uint8_t> first_icc;
    const auto byte = [&]() {
        ProcessInterrupt::check();
        std::uint8_t value{};
        if (!file.read(position, &value, 1)) throw std::runtime_error("Truncated JPEG header");
        ++position;
        return value;
    };
    for (;;) {
        ProcessInterrupt::check();
        if (!marker_prefix) {
            // Pillow skips non-marker bytes between header markers.
            if (byte() != 0xff) continue;
        }
        const auto marker = byte();
        marker_prefix = false;
        if (marker == 0xff) { marker_prefix = true; continue; }
        if (marker == 0) continue;  // Escaped FF/junk in the header.
        if (marker < 0xc0 || marker > 0xfe)
            throw std::runtime_error("Unknown JPEG header marker");
        // Pillow's marker table assigns no segment handler to these markers.
        if (marker == 0xc8 || (marker >= 0xd0 && marker <= 0xd9) ||
            (marker >= 0xf0 && marker <= 0xfd)) continue;

        std::array<std::uint8_t, 2> encoded_length{};
        if (!file.read(position, encoded_length.data(), encoded_length.size()))
            throw std::runtime_error("Truncated JPEG segment length");
        position += encoded_length.size();
        const auto length = value16(encoded_length.data(), false);
        // ImageFile._safe_read returns empty for a nonpositive requested size.
        const auto payload = file.bytes(position, length > 2 ? length - 2 : 0);
        position += payload.size();
        const auto starts = [&](std::string_view prefix) {
            return payload.size() >= prefix.size() &&
                std::memcmp(payload.data(), prefix.data(), prefix.size()) == 0;
        };
        const auto frame = (marker >= 0xc0 && marker <= 0xcf &&
                            marker != 0xc4 && marker != 0xc8 && marker != 0xcc) || marker == 0xde;
        if (frame) {
            if (payload.size() < 5)
                throw std::runtime_error("Truncated JPEG frame header");
            width = value16(payload.data() + 3, false);
            height = value16(payload.data() + 1, false);
            if (payload[0] != 8)
                throw std::runtime_error("Unsupported JPEG sample precision");
            if (payload.size() < 6)
                throw std::runtime_error("Truncated JPEG component count");
            if (payload[5] != 1 && payload[5] != 3 && payload[5] != 4)
                throw std::runtime_error("Unsupported JPEG component count");
            frame_mode = true;
            // SOF sorts ICC fragments lexicographically and reads the total
            // fragment field of the first one. Incomplete fields fail here,
            // while a mismatched fragment count just yields no profile.
            if (icc_count && first_icc.size() < 14)
                throw std::runtime_error("Truncated JPEG ICC profile header");
            icc_count = 0;
            first_icc.clear();
            if ((payload.size() - 6) % 3)
                throw std::runtime_error("Truncated JPEG frame components");
        } else if (marker == 0xdb) {
            for (std::size_t offset = 0; offset < payload.size();) {
                ProcessInterrupt::check();
                const std::size_t precision = payload[offset] / 16 == 0 ? 1 : 2;
                const auto table_bytes = 1 + precision * 64;
                if (table_bytes > payload.size() - offset)
                    throw std::runtime_error("Invalid JPEG quantization table length");
                offset += table_bytes;
            }
        } else if (marker >= 0xe0 && marker <= 0xef) {
            if ((marker == 0xe0 && starts("JFIF")) ||
                (marker == 0xee && starts("Adobe"))) {
                if (payload.size() < 7)
                    throw std::runtime_error("Truncated JPEG application version");
            } else if (marker == 0xe2 && starts(std::string_view("ICC_PROFILE\0", 12))) {
                if (!icc_count || payload < first_icc) first_icc = payload;
                ++icc_count;
            } else if (marker == 0xed && starts(std::string_view("Photoshop 3.0\0", 14))) {
                // The resource parser ignores short integer fields, but its
                // unguarded Pascal-name byte must be present.
                for (std::uint64_t offset = 14; offset <= payload.size() &&
                     payload.size() - offset >= 4 &&
                     std::memcmp(payload.data() + offset, "8BIM", 4) == 0;) {
                    ProcessInterrupt::check();
                    offset += 4;
                    if (payload.size() - offset < 2) break;
                    const auto code = value16(payload.data() + offset, false);
                    offset += 2;
                    if (offset == payload.size())
                        throw std::runtime_error("Truncated JPEG Photoshop resource name");
                    offset += 1 + payload[static_cast<std::size_t>(offset)];
                    offset += offset & 1;
                    if (offset > payload.size() || payload.size() - offset < 4) break;
                    const auto size = value32(payload.data() + offset, false);
                    offset += 4;
                    if (code == 0x03ed && std::min<std::uint64_t>(size, payload.size() - offset) < 14) break;
                    offset += size;
                    offset += offset & 1;
                }
            }
        }
        if (marker == 0xda) {
            if (!frame_mode) throw std::runtime_error("JPEG contains no image frame");
            (void)rgb_size(width, height);
            return;
        }
    }
}

#ifdef UF_HAS_WEBP
void verify_webp(ImageFile& file) {
    const auto bytes = file.bytes(0, file.size());
    const WebPData data{bytes.data(), bytes.size()};
    // Pillow opens WebP with the animation decoder's default options. Its
    // inherited verify does not request the first decoded frame.
    const std::unique_ptr<WebPAnimDecoder, decltype(&WebPAnimDecoderDelete)> decoder(
        WebPAnimDecoderNew(&data, nullptr), WebPAnimDecoderDelete);
    if (!decoder) throw std::runtime_error("Cannot create UF WebP decoder");
    WebPAnimInfo info{};
    if (!WebPAnimDecoderGetInfo(decoder.get(), &info))
        throw std::runtime_error("Invalid UF WebP animation metadata");
    (void)rgb_size(info.canvas_width, info.canvas_height);
}
#endif

enum class ImageKind { jpeg, png, webp, bmp, dib, tiff, avif, other };

ImageKind identify(const std::array<std::uint8_t, 16>& prefix) {
    static constexpr std::uint8_t png_magic[]{137, 80, 78, 71, 13, 10, 26, 10};
    if (prefix[0] == 0xff && prefix[1] == 0xd8) return ImageKind::jpeg;
    if (std::memcmp(prefix.data(), png_magic, sizeof(png_magic)) == 0) return ImageKind::png;
    if (std::memcmp(prefix.data(), "RIFF", 4) == 0 &&
        std::memcmp(prefix.data() + 8, "WEBP", 4) == 0) return ImageKind::webp;
    if (prefix[0] == 'B' && prefix[1] == 'M') return ImageKind::bmp;
    const auto dib_size = value32(prefix.data(), true);
    if (dib_size == 12 || dib_size == 40 || dib_size == 52 || dib_size == 56 ||
        dib_size == 64 || dib_size == 108 || dib_size == 124) return ImageKind::dib;
    if ((prefix[0] == 'I' && prefix[1] == 'I') || (prefix[0] == 'M' && prefix[1] == 'M')) {
        return ImageKind::tiff;
    }
    if (std::memcmp(prefix.data() + 4, "ftyp", 4) == 0 &&
        (std::memcmp(prefix.data() + 8, "avif", 4) == 0 || std::memcmp(prefix.data() + 8, "avis", 4) == 0 ||
         std::memcmp(prefix.data() + 8, "mif1", 4) == 0 || std::memcmp(prefix.data() + 8, "msf1", 4) == 0)) {
        return ImageKind::avif;
    }
    return ImageKind::other;
}

unsigned image_orientation(ImageFile& file, ImageKind kind) {
    if (kind == ImageKind::tiff) return tiff_orientation(file, 0, file.size()).value_or(1);
    std::array<std::uint8_t, 8> header{};
    std::optional<unsigned> xmp;
    if (kind == ImageKind::jpeg) {
        std::uint64_t position = 2;
        while (position < file.size()) {
            ProcessInterrupt::check();
            std::uint8_t marker{};
            if (!file.read(position++, &marker, 1) || marker != 0xff) break;
            do {
                if (!file.read(position++, &marker, 1)) return xmp.value_or(1);
            } while (marker == 0xff);
            if (marker == 0xda || marker == 0xd9 || marker == 0) break;
            if (marker == 0x01 || (marker >= 0xd0 && marker <= 0xd8)) continue;
            if (!file.read(position, header.data(), 2)) break;
            const auto length = value16(header.data(), false);
            if (length < 2 || length > file.size() - position) break;
            if (marker == 0xe1) {
                // APP1 may also contain XMP, so require the EXIF signature.
                std::array<std::uint8_t, 6> signature{};
                if (length >= 8 && file.read(position + 2, signature.data(), signature.size()) &&
                    std::memcmp(signature.data(), "Exif\0\0", 6) == 0) {
                    if (auto value = tiff_orientation(file, position + 2, length - 2)) return *value;
                }
                static constexpr char xmp_marker[] = "http://ns.adobe.com/xap/1.0/";
                std::array<char, sizeof(xmp_marker)> xmp_signature{};
                if (!xmp && length >= 2 + xmp_signature.size() &&
                    file.read(position + 2, xmp_signature.data(), xmp_signature.size()) &&
                    std::memcmp(xmp_signature.data(), xmp_marker, sizeof(xmp_marker)) == 0) {
                    xmp = xmp_orientation(file, position + 2 + xmp_signature.size(),
                                          length - 2 - xmp_signature.size());
                }
            }
            position += length;
        }
    } else if (kind == ImageKind::png) {
        return png_metadata(file, true).orientation;
    } else if (kind == ImageKind::webp) {
        if (!file.read(4, header.data(), 4)) return 1;
        const auto end = std::min(file.size(), std::uint64_t(value32(header.data(), true)) + 8);
        std::uint64_t position = 12;
        while (position <= end && end - position >= 8) {
            ProcessInterrupt::check();
            if (!file.read(position, header.data(), header.size())) break;
            const auto length = value32(header.data() + 4, true);
            if (length > end - position - 8) break;
            if (std::memcmp(header.data(), "EXIF", 4) == 0) {
                if (auto value = tiff_orientation(file, position + 8, length)) return *value;
            }
            if (std::memcmp(header.data(), "XMP ", 4) == 0) {
                xmp = xmp_orientation(file, position + 8, length);
            }
            position += std::uint64_t(length) + 8 + (length & 1);
        }
    }
    return xmp.value_or(1);
}

ImagePixels orient(ImagePixels image, unsigned orientation) {
    if (orientation <= 1 || orientation > 8) return image;
    ImagePixels result;
    const bool transpose = orientation >= 5;
    result.width = transpose ? image.height : image.width;
    result.height = transpose ? image.width : image.height;
    result.rgb.resize(image.rgb.size());
    for (std::uint32_t y = 0; y < image.height; ++y) {
        ProcessInterrupt::check();
        for (std::uint32_t x = 0; x < image.width; ++x) {
            std::uint32_t out_x{}, out_y{};
            switch (orientation) {
                case 2: out_x = image.width - 1 - x; out_y = y; break;
                case 3: out_x = image.width - 1 - x; out_y = image.height - 1 - y; break;
                case 4: out_x = x; out_y = image.height - 1 - y; break;
                case 5: out_x = y; out_y = x; break;
                case 6: out_x = image.height - 1 - y; out_y = x; break;
                case 7: out_x = image.height - 1 - y; out_y = image.width - 1 - x; break;
                case 8: out_x = y; out_y = image.width - 1 - x; break;
                default: return image;
            }
            const auto source = (static_cast<std::size_t>(y) * image.width + x) * 3;
            const auto target = (static_cast<std::size_t>(out_y) * result.width + out_x) * 3;
            std::copy_n(image.rgb.data() + source, 3, result.rgb.data() + target);
        }
    }
    return result;
}

#ifdef UF_HAS_JPEG
struct JpegError {
    jpeg_error_mgr manager{};
    std::jmp_buf jump;
    char message[JMSG_LENGTH_MAX]{};
};

void jpeg_failure(j_common_ptr decoder) {
    auto* error = reinterpret_cast<JpegError*>(decoder->err);
    decoder->err->format_message(decoder, error->message);
    std::longjmp(error->jump, 1);
}

void jpeg_warning(j_common_ptr decoder, int level) {
    if (level < 0) {
        // stdio_src otherwise inserts an EOI into truncated input. Pillow's
        // normal image loader does not enable LOAD_TRUNCATED_IMAGES.
        if (decoder->err->msg_code == JWRN_JPEG_EOF) jpeg_failure(decoder);
        ++decoder->err->num_warnings;
    }
}

struct JpegState {
    jpeg_decompress_struct decoder{};
    JpegError error{};
    std::FILE* input{};
    std::uint8_t* pixels{};
    bool created{}, cmyk{};
    std::uint32_t width{}, height{};
    std::size_t stride{}, bytes{};
    ~JpegState() {
        if (created) jpeg_destroy_decompress(&decoder);
        if (input) std::fclose(input);
        std::free(pixels);
    }
};

ImagePixels jpeg_rgb(const std::filesystem::path& source) {
    // All objects owning resources exist before setjmp. Library error jumps
    // never cross a newly constructed C++ container/destructor.
    const auto state = std::make_unique<JpegState>();
    state->input = std::fopen(source.c_str(), "rb");
    if (!state->input) throw std::runtime_error("Cannot open UF JPEG input");
    state->decoder.err = jpeg_std_error(&state->error.manager);
    state->error.manager.error_exit = jpeg_failure;
    state->error.manager.emit_message = jpeg_warning;
    if (setjmp(state->error.jump)) {
        throw std::runtime_error(std::string("Cannot decode UF JPEG: ") + state->error.message);
    }
    state->created = true;
    jpeg_create_decompress(&state->decoder);
    jpeg_stdio_src(&state->decoder, state->input);
    if (jpeg_read_header(&state->decoder, TRUE) != JPEG_HEADER_OK) {
        throw std::runtime_error("Invalid UF JPEG header");
    }
    rgb_size(state->decoder.image_width, state->decoder.image_height);
    state->cmyk = state->decoder.jpeg_color_space == JCS_CMYK ||
                  state->decoder.jpeg_color_space == JCS_YCCK;
    state->decoder.out_color_space = state->cmyk ? JCS_CMYK
        : state->decoder.jpeg_color_space == JCS_GRAYSCALE ? JCS_GRAYSCALE : JCS_RGB;
    state->decoder.dct_method = JDCT_ISLOW;
    state->decoder.do_fancy_upsampling = TRUE;
    state->decoder.scale_num = state->decoder.scale_denom = 1;
    if (!jpeg_start_decompress(&state->decoder)) throw std::runtime_error("Incomplete UF JPEG header");
    state->width = state->decoder.output_width;
    state->height = state->decoder.output_height;
    rgb_size(state->width, state->height);
    if (state->decoder.output_components != (state->cmyk ? 4 :
        state->decoder.out_color_space == JCS_GRAYSCALE ? 1 : 3)) {
        throw std::runtime_error("Invalid UF JPEG output channels");
    }
    state->stride = static_cast<std::size_t>(state->width) * state->decoder.output_components;
    state->bytes = state->stride * state->height;
    state->pixels = static_cast<std::uint8_t*>(std::malloc(state->bytes));
    if (!state->pixels) throw std::bad_alloc();
    while (state->decoder.output_scanline < state->decoder.output_height) {
        ProcessInterrupt::check();
        JSAMPROW row = state->pixels + state->decoder.output_scanline * state->stride;
        if (jpeg_read_scanlines(&state->decoder, &row, 1) != 1) {
            throw std::runtime_error("Incomplete UF JPEG pixels");
        }
    }
    if (!jpeg_finish_decompress(&state->decoder)) throw std::runtime_error("Incomplete UF JPEG pixels");
    jpeg_destroy_decompress(&state->decoder);
    state->created = false;
    ImagePixels result{state->width, state->height, {}};
    result.rgb.resize(rgb_size(result.width, result.height));
    const auto area = result.rgb.size() / 3;
    if (state->cmyk) {
        // Pillow's CMYK;I unpack inverts libjpeg's channels before convert(RGB).
        // Combining the two steps gives rounded (inverted C/M/Y * inverted K)/255.
        for (std::size_t i = 0; i < area; ++i) {
            if (i % result.width == 0) ProcessInterrupt::check();
            const auto k = state->pixels[i * 4 + 3];
            for (std::size_t c = 0; c < 3; ++c) {
                result.rgb[i * 3 + c] = static_cast<std::uint8_t>(
                    (unsigned(state->pixels[i * 4 + c]) * k + 127) / 255);
            }
        }
    } else if (state->stride == static_cast<std::size_t>(result.width)) {
        for (std::size_t i = 0; i < area; ++i) {
            if (i % result.width == 0) ProcessInterrupt::check();
            std::fill_n(result.rgb.data() + i * 3, 3, state->pixels[i]);
        }
    } else {
        std::copy_n(state->pixels, result.rgb.size(), result.rgb.data());
    }
    return result;
}
#endif

#ifdef UF_HAS_PNG
// Present the first tile as an ordinary PNG without copying compressed pixels.
// Used for APNG and PNG metadata that libpng must not reinterpret. The metadata
// pass has already checked its original header CRCs.
// Data CRCs are skipped by Pillow, so synthesized data chunks use a quiet
// CRC policy in libpng as well.
class PngFrameStream {
public:
    PngFrameStream(ImageFile& file, const PngFirstFrame& frame)
        : file_(file), frame_(frame), next_chunk_(frame.data_begin) {
        static constexpr std::array<std::uint8_t, 8> magic{137, 80, 78, 71, 13, 10, 26, 10};
        std::copy(magic.begin(), magic.end(), prefix_.begin());
        store32(prefix_.data() + 8, 13);
        std::memcpy(prefix_.data() + 12, "IHDR", 4);
        std::copy(frame.ihdr.begin(), frame.ihdr.end(), prefix_.begin() + 16);
        store32(prefix_.data() + 16, frame.width);
        store32(prefix_.data() + 20, frame.height);
    }
    void read(std::uint8_t* destination, std::size_t size) {
        while (size) {
            ProcessInterrupt::check();
            std::size_t used = 0;
            if (prefix_done_ < prefix_.size()) {
                used = std::min(size, prefix_.size() - prefix_done_);
                std::memcpy(destination, prefix_.data() + prefix_done_, used);
                prefix_done_ += used;
            } else if (header_index_ < frame_.headers.size()) {
                const auto& region = frame_.headers[header_index_];
                used = static_cast<std::size_t>(std::min<std::uint64_t>(
                    std::min<std::size_t>(size, 65536), region.size - header_done_));
                if (!file_.read(region.offset + header_done_, destination, used)) {
                    throw std::runtime_error("Cannot read PNG frame header");
                }
                header_done_ += used;
                if (header_done_ == region.size) { header_done_ = 0; ++header_index_; }
            } else if (chunk_loaded_ || next_chunk_ < frame_.data_end) {
                if (!chunk_loaded_) load_chunk();
                if (chunk_header_done_ < chunk_header_.size()) {
                    used = std::min(size, chunk_header_.size() - chunk_header_done_);
                    std::memcpy(destination, chunk_header_.data() + chunk_header_done_, used);
                    chunk_header_done_ += used;
                } else if (payload_done_ < payload_size_) {
                    used = static_cast<std::size_t>(std::min<std::uint64_t>(
                        std::min<std::size_t>(size, 65536), payload_size_ - payload_done_));
                    if (!file_.read(payload_offset_ + payload_done_, destination, used)) {
                        throw std::runtime_error("Cannot read PNG frame pixels");
                    }
                    payload_done_ += used;
                } else {
                    used = std::min(size, std::size_t(4) - crc_done_);
                    std::memset(destination, 0, used);
                    crc_done_ += used;
                    if (crc_done_ == 4) chunk_loaded_ = false;
                }
            } else {
                static constexpr std::array<std::uint8_t, 12> end{
                    0, 0, 0, 0, 'I', 'E', 'N', 'D', 0, 0, 0, 0
                };
                if (end_done_ == end.size()) throw std::runtime_error("Unexpected PNG frame read after IEND");
                used = std::min(size, end.size() - end_done_);
                std::memcpy(destination, end.data() + end_done_, used);
                end_done_ += used;
            }
            destination += used;
            size -= used;
        }
    }
private:
    static void store32(std::uint8_t* destination, std::uint32_t value) {
        for (unsigned index = 0; index < 4; ++index) {
            destination[index] = static_cast<std::uint8_t>(value >> (24 - index * 8));
        }
    }
    void load_chunk() {
        std::array<std::uint8_t, 8> source{};
        if (!file_.read(next_chunk_, source.data(), source.size())) {
            throw std::runtime_error("Truncated PNG frame data chunk");
        }
        const auto length = value32(source.data(), false);
        const std::string_view kind(reinterpret_cast<const char*>(source.data() + 4), 4);
        const bool fd_at = kind == "fdAT";
        if ((kind != "IDAT" && kind != "DDAT" && !fd_at) || (fd_at && length < 4) ||
            std::uint64_t(length) + 12 > frame_.data_end - next_chunk_) {
            throw std::runtime_error("PNG frame data changed while reading");
        }
        payload_offset_ = next_chunk_ + (fd_at ? 12 : 8);
        payload_size_ = length - (fd_at ? 4 : 0);
        next_chunk_ += std::uint64_t(length) + 12;
        store32(chunk_header_.data(), static_cast<std::uint32_t>(payload_size_));
        std::memcpy(chunk_header_.data() + 4, "IDAT", 4);
        chunk_header_done_ = 0; payload_done_ = 0; crc_done_ = 0;
        chunk_loaded_ = true;
    }
    ImageFile& file_;
    const PngFirstFrame& frame_;
    std::array<std::uint8_t, 33> prefix_{};
    std::array<std::uint8_t, 8> chunk_header_{};
    std::size_t prefix_done_{}, header_index_{}, chunk_header_done_{}, crc_done_{}, end_done_{};
    std::uint64_t header_done_{}, next_chunk_{}, payload_offset_{}, payload_size_{}, payload_done_{};
    bool chunk_loaded_{};
};

struct PngState {
    png_structp decoder{};
    png_infop info{};
    std::FILE* input{};
    std::uint8_t* pixels{};
    std::jmp_buf jump;
    char message[256]{};
    std::uint32_t width{}, height{};
    std::size_t stride{}, bytes{};
    int passes{};
    bool gray16{};
    std::unique_ptr<PngFrameStream> frame_input;
    std::exception_ptr read_error;
    ~PngState() {
        if (decoder) png_destroy_read_struct(&decoder, &info, nullptr);
        if (input) std::fclose(input);
        std::free(pixels);
    }
};

void png_failure(png_structp decoder, png_const_charp message) {
    auto* state = static_cast<PngState*>(png_get_error_ptr(decoder));
    std::snprintf(state->message, sizeof(state->message), "%s", message ? message : "PNG error");
    std::longjmp(state->jump, 1);
}

void png_warning(png_structp, png_const_charp) {}

void png_frame_read(png_structp decoder, png_bytep destination, png_size_t size) {
    auto* state = static_cast<PngState*>(png_get_io_ptr(decoder));
    try {
        state->frame_input->read(destination, size);
    } catch (...) {
        state->read_error = std::current_exception();
    }
    // Leave the C++ catch scope before jumping through libpng's C frames.
    if (state->read_error) png_error(decoder, "Cannot read first PNG frame");
}

ImagePixels png_rgb(const std::filesystem::path& source, ImageFile& file, const PngFirstFrame& frame) {
    const auto state = std::make_unique<PngState>();
    if (frame.stream) {
        rgb_size(frame.width, frame.height);
        state->frame_input = std::make_unique<PngFrameStream>(file, frame);
    } else {
        state->input = std::fopen(source.c_str(), "rb");
        if (!state->input) throw std::runtime_error("Cannot open UF PNG input");
    }
    if (setjmp(state->jump)) {
        if (state->read_error) std::rethrow_exception(state->read_error);
        throw std::runtime_error(std::string("Cannot decode UF PNG: ") + state->message);
    }
    state->decoder = png_create_read_struct(PNG_LIBPNG_VER_STRING, state.get(), png_failure, png_warning);
    if (!state->decoder) throw std::runtime_error("Cannot create UF PNG decoder");
    state->info = png_create_info_struct(state->decoder);
    if (!state->info) throw std::bad_alloc();
    if (state->frame_input) {
        png_set_read_fn(state->decoder, state.get(), png_frame_read);
        png_set_crc_action(state->decoder, PNG_CRC_QUIET_USE, PNG_CRC_QUIET_USE);
    } else png_init_io(state->decoder, state->input);
#ifdef PNG_HANDLE_AS_UNKNOWN_SUPPORTED
    // Our metadata pass supplies Pillow's text handling and limits. Disable
    // libpng's second text/EXIF parse, which has different limits and decoding.
    static const png_byte metadata_chunks[][5] = {"tEXt", "zTXt", "iTXt", "eXIf", "iCCP"};
    png_set_keep_unknown_chunks(state->decoder, PNG_HANDLE_CHUNK_NEVER,
                                metadata_chunks[0], 5);
#endif
#ifdef PNG_SET_USER_LIMITS_SUPPORTED
    png_set_user_limits(state->decoder, std::numeric_limits<std::uint32_t>::max() - 1,
                        std::numeric_limits<std::uint32_t>::max() - 1);
#endif
    png_read_info(state->decoder, state->info);
    state->width = png_get_image_width(state->decoder, state->info);
    state->height = png_get_image_height(state->decoder, state->info);
    rgb_size(state->width, state->height);
    const auto color = png_get_color_type(state->decoder, state->info);
    const auto depth = png_get_bit_depth(state->decoder, state->info);
    state->gray16 = color == PNG_COLOR_TYPE_GRAY && depth == 16;
    if (color == PNG_COLOR_TYPE_PALETTE) png_set_palette_to_rgb(state->decoder);
    if (color == PNG_COLOR_TYPE_GRAY && depth < 8) png_set_expand_gray_1_2_4_to_8(state->decoder);
    if (depth == 16 && !state->gray16) png_set_strip_16(state->decoder);
    // Expansion can create alpha from tRNS even when the original palette or
    // low-bit grayscale header has no alpha flag. Pillow convert(RGB) drops
    // that alpha while preserving the stored color samples.
    png_set_strip_alpha(state->decoder);
    if ((color == PNG_COLOR_TYPE_GRAY || color == PNG_COLOR_TYPE_GRAY_ALPHA) && !state->gray16) {
        png_set_gray_to_rgb(state->decoder);
    }
    // No gamma/ICC operation or alpha compositing: preserve stored samples.
    state->passes = png_set_interlace_handling(state->decoder);
    png_read_update_info(state->decoder, state->info);
    state->stride = png_get_rowbytes(state->decoder, state->info);
    if (state->stride != static_cast<std::size_t>(state->width) * (state->gray16 ? 2 : 3) ||
        png_get_channels(state->decoder, state->info) != (state->gray16 ? 1 : 3) ||
        png_get_bit_depth(state->decoder, state->info) != (state->gray16 ? 16 : 8)) {
        throw std::runtime_error("Unsupported UF PNG pixel layout");
    }
    state->bytes = state->stride * state->height;
    state->pixels = static_cast<std::uint8_t*>(std::calloc(state->bytes, 1));
    if (!state->pixels) throw std::bad_alloc();
    for (int pass = 0; pass < state->passes; ++pass) {
        for (std::uint32_t row = 0; row < state->height; ++row) {
            ProcessInterrupt::check();
            png_read_row(state->decoder, state->pixels + static_cast<std::size_t>(row) * state->stride, nullptr);
        }
    }
    png_read_end(state->decoder, nullptr);
    png_destroy_read_struct(&state->decoder, &state->info, nullptr);
    ImagePixels result{state->width, state->height, {}};
    result.rgb.resize(rgb_size(result.width, result.height));
    if (state->gray16) {
        for (std::size_t i = 0; i < result.rgb.size() / 3; ++i) {
            if (i % result.width == 0) ProcessInterrupt::check();
            // PNG stores network-order samples. Pillow's I;16 -> RGB clamps
            // the integer to 255, unlike truncation of RGB/RGBA 16-bit samples.
            const auto value = std::min<unsigned>(value16(state->pixels + i * 2, false), 255);
            std::fill_n(result.rgb.data() + i * 3, 3, static_cast<std::uint8_t>(value));
        }
    } else {
        std::copy_n(state->pixels, result.rgb.size(), result.rgb.data());
    }
    if (!frame.stream || (frame.x == 0 && frame.y == 0 && result.width == frame.canvas_width &&
                          result.height == frame.canvas_height)) return result;
    if (std::uint64_t(frame.x) + result.width > frame.canvas_width ||
        std::uint64_t(frame.y) + result.height > frame.canvas_height) {
        throw std::runtime_error("Decoded PNG frame exceeds its canvas");
    }
    ImagePixels canvas{frame.canvas_width, frame.canvas_height, {}};
    canvas.rgb.resize(rgb_size(canvas.width, canvas.height));
    if (std::any_of(frame.background.begin(), frame.background.end(), [](auto value) { return value != 0; })) {
        for (std::size_t pixel = 0; pixel < canvas.rgb.size() / 3; ++pixel) {
            if ((pixel & 65535) == 0) ProcessInterrupt::check();
            std::copy_n(frame.background.data(), 3, canvas.rgb.data() + pixel * 3);
        }
    }
    for (std::uint32_t row = 0; row < result.height; ++row) {
        ProcessInterrupt::check();
        const auto destination = (static_cast<std::size_t>(frame.y + row) * canvas.width + frame.x) * 3;
        const auto bytes = static_cast<std::size_t>(result.width) * 3;
        std::copy_n(result.rgb.data() + static_cast<std::size_t>(row) * bytes, bytes, canvas.rgb.data() + destination);
    }
    return canvas;
}
#endif

#ifdef UF_HAS_WEBP
ImagePixels webp_rgb(ImageFile& file) {
    const auto bytes = file.bytes(0, file.size());
    WebPBitstreamFeatures features{};
    if (WebPGetFeatures(bytes.data(), bytes.size(), &features) != VP8_STATUS_OK) {
        throw std::runtime_error("Invalid UF WebP header");
    }
    rgb_size(features.width, features.height);
    WebPAnimDecoderOptions options{};
    if (!WebPAnimDecoderOptionsInit(&options)) throw std::runtime_error("Incompatible native WebP ABI");
    options.color_mode = MODE_RGBA;
    options.use_threads = 1;
    const WebPData data{bytes.data(), bytes.size()};
    const std::unique_ptr<WebPAnimDecoder, decltype(&WebPAnimDecoderDelete)> decoder(
        WebPAnimDecoderNew(&data, &options), WebPAnimDecoderDelete);
    if (!decoder) throw std::runtime_error("Cannot create UF WebP decoder");
    WebPAnimInfo info{};
    if (!WebPAnimDecoderGetInfo(decoder.get(), &info) || !info.frame_count) {
        throw std::runtime_error("Invalid UF WebP animation metadata");
    }
    ImagePixels result{info.canvas_width, info.canvas_height, {}};
    result.rgb.resize(rgb_size(result.width, result.height));
    std::uint8_t* rgba{};
    int timestamp{};
    ProcessInterrupt::check();
    // AnimDecoder also handles still images, and returns the composited first
    // animation canvas as requested by the Python WebP plugin.
    if (!WebPAnimDecoderGetNext(decoder.get(), &rgba, &timestamp) || !rgba) {
        throw std::runtime_error("Cannot decode first UF WebP frame");
    }
    for (std::size_t i = 0; i < result.rgb.size() / 3; ++i) {
        if (i % result.width == 0) ProcessInterrupt::check();
        std::copy_n(rgba + i * 4, 3, result.rgb.data() + i * 3);
    }
    return result;
}
#endif

class BmpStream {
public:
    BmpStream(ImageFile& file, std::uint64_t position) : file_(file), position_(position) {}
    std::optional<std::uint8_t> byte() {
        if (position_ >= file_.size()) return {};
        if (!available_ || position_ < cached_ || position_ - cached_ >= available_) {
            ProcessInterrupt::check();
            cached_ = position_;
            available_ = static_cast<std::size_t>(std::min<std::uint64_t>(buffer_.size(), file_.size() - position_));
            if (!file_.read(cached_, buffer_.data(), available_)) throw std::runtime_error("Cannot read BMP RLE data");
        }
        return buffer_[static_cast<std::size_t>(position_++ - cached_)];
    }
    std::size_t read(std::uint8_t* destination, std::size_t count) {
        std::size_t done = 0;
        while (done < count) {
            const auto value = byte();
            if (!value) break;
            destination[done++] = *value;
        }
        return done;
    }
    void align_word() { if (position_ & 1) ++position_; }
private:
    ImageFile& file_;
    std::uint64_t position_, cached_{};
    std::size_t available_{};
    std::array<std::uint8_t, 65536> buffer_{};
};

std::vector<std::uint8_t> bmp_rle(ImageFile& file, std::uint64_t offset,
                                 std::uint32_t width, std::uint32_t height, bool rle4) {
    const auto size = static_cast<std::size_t>(std::uint64_t(width) * height);
    std::vector<std::uint8_t> decoded;
    decoded.reserve(std::min<std::size_t>(size, 65536));
    BmpStream input(file, offset);
    std::uint64_t x = 0;
    const auto append = [&](std::uint64_t count, std::uint8_t value) {
        // Only the prefix passed to the raw image decoder is observable.
        // Keep delta/run overshoot bounded to the required image length.
        const auto used = static_cast<std::size_t>(std::min<std::uint64_t>(count, size - decoded.size()));
        decoded.insert(decoded.end(), used, value);
    };
    while (decoded.size() < size) {
        ProcessInterrupt::check();
        const auto encoded = input.byte(), command = input.byte();
        if (!encoded || !command) break;
        if (*encoded) {
            const auto count = std::min<std::uint64_t>(*encoded, x < width ? width - x : 0);
            if (rle4) {
                for (std::uint64_t index = 0; index < count; ++index) {
                    append(1, (index & 1) ? *command & 15 : *command >> 4);
                }
            } else append(count, *command);
            x += count;
        } else if (*command == 0) {
            const auto remainder = decoded.size() % width;
            if (remainder) append(width - remainder, 0);
            x = 0;
        } else if (*command == 1) {
            break;
        } else if (*command == 2) {
            // The installed Pillow 12.1.1 decoder reads two delta pairs,
            // discarding the first pair. Preserve its stream consumption.
            std::array<std::uint8_t, 2> ignored{}, delta{};
            if (input.read(ignored.data(), ignored.size()) != ignored.size()) break;
            if (input.read(delta.data(), delta.size()) != delta.size()) {
                throw std::runtime_error("Incomplete BMP RLE delta coordinates");
            }
            append(std::uint64_t(delta[0]) + std::uint64_t(delta[1]) * width, 0);
            x = decoded.size() % width;
        } else {
            // RLE4's installed decoder consumes floor(n/2) literal bytes;
            // an odd final nibble is not appended.
            const auto count = rle4 ? *command / 2u : *command;
            std::array<std::uint8_t, 255> bytes{};
            const auto read = input.read(bytes.data(), count);
            for (std::size_t index = 0; index < read; ++index) {
                if (rle4) {
                    append(1, bytes[index] >> 4);
                    append(1, bytes[index] & 15);
                } else append(1, bytes[index]);
            }
            if (read < count) break;
            x += *command;
            input.align_word();
        }
    }
    if (decoded.size() < size) throw std::runtime_error("Not enough decoded BMP image data");
    return decoded;
}

ImagePixels bmp_rgb(ImageFile& file, bool dib = false) {
    std::array<std::uint8_t, 18> prefix{};
    if (dib) {
        if (!file.read(0, prefix.data() + 14, 4)) throw std::runtime_error("Truncated UF DIB header");
    } else if (!file.read(0, prefix.data(), prefix.size())) throw std::runtime_error("Truncated UF BMP header");
    std::uint64_t pixel_offset = dib ? 0 : value32(prefix.data() + 10, true);
    const auto header_size = value32(prefix.data() + 14, true);
    if (header_size != 12 && header_size != 40 && header_size != 52 && header_size != 56 &&
        header_size != 64 && header_size != 108 && header_size != 124) {
        throw std::runtime_error("Unsupported UF BMP header type");
    }
    const auto header_start = dib ? 0u : 14u;
    const auto header = file.bytes(header_start, header_size);
    std::uint64_t metadata_end = std::uint64_t(header_start) + header_size;
    const bool core = header_size == 12;
    const auto width = core ? value16(header.data() + 4, true) : value32(header.data() + 4, true);
    const auto encoded_height = core ? value16(header.data() + 6, true) : value32(header.data() + 8, true);
    const bool top_down = !core && header[11] == 0xff;
    const std::uint32_t height = top_down ? (0u - encoded_height) : encoded_height;
    const auto depth = value16(header.data() + (core ? 10 : 14), true);
    const auto compression = core ? 0 : value32(header.data() + 16, true);
    // Pillow reads the plane count without using it to reject the image.
    rgb_size(width, height);
    if (depth != 1 && depth != 4 && depth != 8 && depth != 16 && depth != 24 && depth != 32) {
        throw std::runtime_error("Unsupported UF BMP pixel depth");
    }
    if (compression > 3) throw std::runtime_error("Unsupported UF BMP compression");
    std::array<unsigned, 3> byte_order{2, 1, 0};
    bool rgb565 = false;
    if (compression == 3) {
        std::array<std::uint32_t, 4> masks{};
        if (header_size >= 52) {
            for (unsigned index = 0; index < 3; ++index) masks[index] = value32(header.data() + 40 + index * 4, true);
            if (header_size >= 56) masks[3] = value32(header.data() + 52, true);
        } else {
            const auto stored = file.bytes(metadata_end, 12);
            for (unsigned index = 0; index < 3; ++index) masks[index] = value32(stored.data() + index * 4, true);
            metadata_end += 12;
        }
        const std::array<std::uint32_t, 3> rgb_masks{masks[0], masks[1], masks[2]};
        if (depth == 32) {
            static constexpr std::array<std::array<std::uint32_t, 4>, 8> supported{{
                {0xff0000, 0xff00, 0xff, 0},
                {0xff000000, 0xff0000, 0xff00, 0},
                {0xff000000, 0xff00, 0xff, 0},
                {0xff000000, 0xff0000, 0xff00, 0xff},
                {0xff, 0xff00, 0xff0000, 0xff000000},
                {0xff0000, 0xff00, 0xff, 0xff000000},
                {0xff000000, 0xff00, 0xff, 0xff0000},
                {0, 0, 0, 0}
            }};
            if (std::find(supported.begin(), supported.end(), masks) == supported.end()) {
                throw std::runtime_error("Unsupported UF BMP bitfields layout");
            }
            if (masks[0]) {
                const auto byte = [](std::uint32_t mask) {
                    return mask == 0xff ? 0u : mask == 0xff00 ? 1u : mask == 0xff0000 ? 2u : 3u;
                };
                for (unsigned index = 0; index < 3; ++index) byte_order[index] = byte(masks[index]);
            }  // All-zero masks select BGRA; RGB conversion drops its alpha.
        } else if (depth == 24 && rgb_masks == std::array<std::uint32_t, 3>{0xff0000, 0xff00, 0xff}) {
            // Standard BGR triplets.
        } else if (depth == 16 && rgb_masks == std::array<std::uint32_t, 3>{0xf800, 0x7e0, 0x1f}) {
            rgb565 = true;
        } else if (depth != 16 || rgb_masks != std::array<std::uint32_t, 3>{0x7c00, 0x3e0, 0x1f}) {
            throw std::runtime_error("Unsupported UF BMP bitfields layout");
        }
    }
    std::array<std::array<std::uint8_t, 3>, 256> palette{};
    bool grayscale = false, monochrome = false;
    if (depth <= 8) {
        const auto declared = core ? 0 : value32(header.data() + 32, true);
        const auto colors = declared ? declared : (1u << depth);
        if (colors > 256) throw std::runtime_error("Unsupported UF BMP palette size");
        if (!dib && pixel_offset == std::uint64_t(14) + header_size) pixel_offset += std::uint64_t(colors) * 4;
        const auto padding = core ? 3u : 4u;
        const auto read = std::min<std::uint64_t>(std::uint64_t(colors) * padding, file.size() - metadata_end);
        const auto entries = file.bytes(metadata_end, read);
        metadata_end += read;
        grayscale = true;
        for (unsigned index = 0; index < colors; ++index) {
            const auto expected = colors == 2 ? index * 255u : index;
            const auto begin = std::size_t(index) * padding;
            if (begin > entries.size() || entries.size() - begin < 3 ||
                entries[begin] != expected || entries[begin + 1] != expected || entries[begin + 2] != expected) {
                grayscale = false;
            }
            if (begin <= entries.size() && entries.size() - begin >= padding) {
                palette[index] = {entries[begin + 2], entries[begin + 1], entries[begin]};
            }
        }
        if (grayscale) {
            monochrome = colors == 2;
            for (unsigned index = 0; index < palette.size(); ++index) {
                palette[index].fill(static_cast<std::uint8_t>(monochrome ? (index ? 255 : 0) : index));
            }
        }
    }
    if (!pixel_offset) pixel_offset = metadata_end;
    if (pixel_offset > file.size()) throw std::runtime_error("Truncated UF BMP pixels");
    ImagePixels result{width, height, {}};
    result.rgb.resize(rgb_size(width, height));
    if (compression == 1 || compression == 2) {
        if (depth > 8 || monochrome) throw std::runtime_error("Unsupported UF BMP RLE output mode");
        const auto indices = bmp_rle(file, pixel_offset, width, height, compression == 2);
        for (std::uint32_t input_y = 0; input_y < height; ++input_y) {
            ProcessInterrupt::check();
            const auto y = top_down ? input_y : height - 1 - input_y;
            for (std::uint32_t x = 0; x < width; ++x) {
                if ((x & 65535) == 0) ProcessInterrupt::check();
                const auto index = indices[static_cast<std::size_t>(input_y) * width + x];
                auto* target = result.rgb.data() + (static_cast<std::size_t>(y) * width + x) * 3;
                std::copy_n(palette[index].data(), 3, target);
            }
        }
        return result;
    }
    const auto unpacked_bits = depth <= 8 && grayscale ? (monochrome ? 1u : 8u) : depth;
    const std::uint64_t stride = ((std::uint64_t(width) * depth + 31) / 32) * 4;
    const std::uint64_t row_bytes = (std::uint64_t(width) * unpacked_bits + 7) / 8;
    if (stride < row_bytes) throw std::runtime_error("Invalid UF BMP raw row stride");
    // Pillow's raw decoder returns after the last pixel row, before padding.
    if ((std::uint64_t(height) - 1) * stride + row_bytes > file.size() - pixel_offset ||
        row_bytes > std::numeric_limits<std::size_t>::max()) throw std::runtime_error("Truncated UF BMP pixels");
    std::vector<std::uint8_t> row(static_cast<std::size_t>(row_bytes));
    for (std::uint32_t input_y = 0; input_y < height; ++input_y) {
        ProcessInterrupt::check();
        if (!file.read(pixel_offset + input_y * stride, row.data(), row.size())) {
            throw std::runtime_error("Truncated UF BMP pixels");
        }
        const auto y = top_down ? input_y : height - 1 - input_y;
        for (std::uint32_t x = 0; x < width; ++x) {
            if ((x & 65535) == 0) ProcessInterrupt::check();
            auto* target = result.rgb.data() + (static_cast<std::size_t>(y) * width + x) * 3;
            if (depth <= 8) {
                const auto index = unpacked_bits == 8 ? row[x] : unpacked_bits == 4
                    ? (row[x / 2] >> ((x % 2) ? 0 : 4)) & 15
                    : (row[x / 8] >> (7 - x % 8)) & 1;
                std::copy_n(palette[index].data(), 3, target);
            } else if (depth == 16) {
                const auto value = value16(row.data() + std::size_t(x) * 2, true);
                target[0] = static_cast<std::uint8_t>(((value >> (rgb565 ? 11 : 10)) & 31) * 255 / 31);
                target[1] = static_cast<std::uint8_t>(((value >> 5) & (rgb565 ? 63 : 31)) * 255 / (rgb565 ? 63 : 31));
                target[2] = static_cast<std::uint8_t>((value & 31) * 255 / 31);
            } else {
                const auto* bgr = row.data() + static_cast<std::size_t>(x) * (depth / 8);
                for (unsigned component = 0; component < 3; ++component) target[component] = bgr[byte_order[component]];
            }
        }
    }
    return result;
}

std::uint64_t decimal(const std::string& token) {
    if (token.empty()) throw std::runtime_error("Invalid FFmpeg PPM number");
    std::uint64_t result = 0;
    for (const unsigned char ch : token) {
        if (ch < '0' || ch > '9' || result > (std::numeric_limits<std::uint64_t>::max() - (ch - '0')) / 10) {
            throw std::runtime_error("Invalid FFmpeg PPM number");
        }
        result = result * 10 + ch - '0';
    }
    return result;
}

ImagePixels ffmpeg_rgb(const std::filesystem::path& source, bool autorotate = false) {
    std::vector<std::string> command{"ffmpeg", "-v", "error", "-nostdin"};
    if (!autorotate) command.push_back("-noautorotate");
    command.insert(command.end(), {"-i", source.string(), "-frames:v", "1", "-pix_fmt", "rgb24",
                                   "-f", "image2pipe", "-vcodec", "ppm", "pipe:1"});
    ChildPipe ffmpeg(command);
    std::vector<std::uint8_t> bytes;
    std::array<std::uint8_t, 65536> chunk{};
    // The fixed allowance covers a PPM header without reducing Pillow's limit.
    constexpr auto capacity = max_rgb_bytes + 65536;
    while (const auto count = ffmpeg.read(chunk.data(), chunk.size())) {
        if (count > capacity - bytes.size()) throw std::runtime_error("Excessive UF image pixels");
        bytes.insert(bytes.end(), chunk.begin(), chunk.begin() + count);
    }
    ffmpeg.finish();
    std::size_t cursor = 0;
    const auto token = [&]() {
        for (;;) {
            while (cursor < bytes.size() && std::isspace(bytes[cursor])) ++cursor;
            if (cursor < bytes.size() && bytes[cursor] == '#') {
                while (cursor < bytes.size() && bytes[cursor] != '\n') ++cursor;
                continue;
            }
            break;
        }
        const auto start = cursor;
        while (cursor < bytes.size() && !std::isspace(bytes[cursor]) && bytes[cursor] != '#') ++cursor;
        if (start == cursor) throw std::runtime_error("Invalid FFmpeg PPM image header");
        return std::string(bytes.begin() + start, bytes.begin() + cursor);
    };
    if (token() != "P6") throw std::runtime_error("FFmpeg did not return RGB PPM pixels");
    const auto width = decimal(token());
    const auto height = decimal(token());
    const auto size = rgb_size(width, height);
    if (token() != "255" || cursor >= bytes.size() || !std::isspace(bytes[cursor])) {
        throw std::runtime_error("Invalid UF PPM image depth");
    }
    ++cursor;
    if (bytes.size() - cursor != size) throw std::runtime_error("Incomplete FFmpeg UF image pixels");
    ImagePixels result{static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height), {}};
    result.rgb.assign(bytes.begin() + cursor, bytes.end());
    return result;
}

}  // namespace

void verify_image(const std::filesystem::path& source) {
    ProcessInterrupt::check();
    if (source.native().find('\0') != std::string::npos)
        throw std::invalid_argument("embedded null byte");
    ImageFile file(source);
    try {
        std::array<std::uint8_t, 16> prefix{};
        const auto count = std::min<std::uint64_t>(prefix.size(), file.size());
        if (!count || !file.read(0, prefix.data(), static_cast<std::size_t>(count)))
            throw std::runtime_error("Empty or unreadable UF image input");
        switch (identify(prefix)) {
            case ImageKind::jpeg: verify_jpeg(file); break;
            case ImageKind::png: verify_png(file); break;
#ifdef UF_HAS_WEBP
            case ImageKind::webp: verify_webp(file); break;
#endif
#ifdef UF_HAS_TIFF
            case ImageKind::tiff: {
                // Validate with the selected reader: FFmpeg rejects numeric
                // TIFF modes that our native decoder can read. Retain full
                // decode validation rather than accepting only the header.
                file.close_checked();
                (void)read_image_rgb(source, false);
                break;
            }
#endif
#ifdef UF_HAS_AVIF
            case ImageKind::avif: {
                // Older FFmpeg demuxers reject AVIF containers even when a
                // native decoder is available. Validate with that reader.
                file.close_checked();
                (void)read_image_rgb(source, false);
                break;
            }
#endif
            default: {
                file.close_checked();
                ChildPipe validation({"ffmpeg", "-v", "error", "-xerror", "-nostdin",
                    "-i", source.string(), "-map", "0:v:0", "-frames:v", "1", "-f", "null", "-"});
                validation.drain();
                break;
            }
        }
    } catch (...) {
        file.close_checked();
        throw;
    }
    file.close_checked();
}

ImagePixels read_image_rgb(const std::filesystem::path& source, bool exif_transpose) {
    ProcessInterrupt::check();
    ImageFile file(source);
    std::array<std::uint8_t, 16> prefix{};
    const auto count = std::min<std::uint64_t>(prefix.size(), file.size());
    if (!count || !file.read(0, prefix.data(), static_cast<std::size_t>(count))) {
        throw std::runtime_error("Empty or unreadable UF image input");
    }
    const auto kind = identify(prefix);
    // TiffImagePlugin.load_end() transposes and clears its orientation tag.
    // It therefore also applies when callers only request convert("RGB").
    const auto orient_pixels = exif_transpose || kind == ImageKind::tiff;
    // Text limits apply to convert("RGB") callers too. Only EXIF/profile
    // interpretation depends on whether the caller requests transposition.
    std::optional<PngFirstFrame> png_first;
    if (kind == ImageKind::png) png_first.emplace(png_metadata(file, orient_pixels));
    auto orientation = png_first ? png_first->orientation
        : orient_pixels ? image_orientation(file, kind) : 1u;
    ImagePixels image;
    switch (kind) {
#ifdef UF_HAS_JPEG
        case ImageKind::jpeg: image = jpeg_rgb(source); break;
#endif
#ifdef UF_HAS_PNG
        case ImageKind::png: image = png_rgb(source, file, *png_first); break;
#endif
#ifdef UF_HAS_WEBP
        case ImageKind::webp: image = webp_rgb(file); break;
#endif
        case ImageKind::bmp:
        case ImageKind::dib:
            image = bmp_rgb(file, kind == ImageKind::dib);
            break;
        case ImageKind::tiff:
            if (auto native = image_native::tiff_rgb(source, orientation)) image = std::move(*native);
            else image = ffmpeg_rgb(source);
            break;
        case ImageKind::avif:
#ifdef UF_HAS_AVIF
            image = image_native::avif_rgb(file.bytes(0, file.size()), orientation);
#else
            image = ffmpeg_rgb(source, exif_transpose);
#endif
            break;
        default: image = ffmpeg_rgb(source, exif_transpose && kind == ImageKind::other); break;
    }
    if (image.rgb.size() != rgb_size(image.width, image.height)) {
        throw std::runtime_error("Invalid UF RGB image layout");
    }
    return orient_pixels ? orient(std::move(image), orientation) : std::move(image);
}

namespace {
#if defined(UF_HAS_PNG) && defined(PNG_WRITE_SUPPORTED)
struct PngWriteState {
    png_structp encoder{};
    png_infop info{};
    std::jmp_buf jump;
    char message[256]{};
    int descriptor{-1};
    std::exception_ptr write_error;
    ~PngWriteState() {
        if (encoder) png_destroy_write_struct(&encoder, &info);
    }
};

void png_write_failure(png_structp encoder, png_const_charp message) {
    auto* state = static_cast<PngWriteState*>(png_get_error_ptr(encoder));
    std::snprintf(state->message, sizeof(state->message), "%s", message ? message : "PNG error");
    std::longjmp(state->jump, 1);
}

void png_descriptor_write(png_structp encoder, png_bytep bytes, png_size_t size) {
    auto* state = static_cast<PngWriteState*>(png_get_io_ptr(encoder));
    try {
        write_descriptor(state->descriptor, bytes, size);
    } catch (...) {
        state->write_error = std::current_exception();
    }
    // Exit the C++ catch scope before jumping through libpng's C frames.
    if (state->write_error) png_error(encoder, "Cannot write UF PNG output");
}

void png_descriptor_flush(png_structp) {}  // write(2) has no userspace buffer.

void png_write_rgb(int descriptor, const ImagePixels& pixels) {
    // Resource owners exist before setjmp; mutable state lives on the heap.
    const auto state = std::make_unique<PngWriteState>();
    state->descriptor = descriptor;
    if (setjmp(state->jump)) {
        if (state->write_error) std::rethrow_exception(state->write_error);
        throw std::runtime_error(std::string("Cannot encode UF PNG: ") + state->message);
    }
    state->encoder = png_create_write_struct(PNG_LIBPNG_VER_STRING, state.get(), png_write_failure, png_warning);
    if (!state->encoder) throw std::bad_alloc();
    state->info = png_create_info_struct(state->encoder);
    if (!state->info) throw std::bad_alloc();
    png_set_write_fn(state->encoder, state.get(), png_descriptor_write, png_descriptor_flush);
#ifdef PNG_SET_USER_LIMITS_SUPPORTED
    png_set_user_limits(state->encoder, PNG_UINT_31_MAX, PNG_UINT_31_MAX);
#endif
    // Image.fromarray(RGB).save(PNG) writes a minimal header and no implicit
    // sRGB/gAMA/profile chunks. Keep compression/filter decisions with libpng.
    png_set_IHDR(state->encoder, state->info, pixels.width, pixels.height, 8,
                 PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_BASE,
                 PNG_FILTER_TYPE_BASE);
    png_write_info(state->encoder, state->info);
    const auto stride = static_cast<std::size_t>(pixels.width) * 3;
    for (std::uint32_t row = 0; row < pixels.height; ++row) {
        ProcessInterrupt::check();
        png_write_row(state->encoder, pixels.rgb.data() + static_cast<std::size_t>(row) * stride);
    }
    png_write_end(state->encoder, state->info);
}
#else
void png_write_rgb(int descriptor, const ImagePixels& pixels) {
    auto pipe = process_pipe();
    // The child writes encoded PNG bytes to the already-owned output fd. RGB
    // input uses a pipe; neither side creates a raw-pixel staging file.
    ChildProcess writer({"ffmpeg", "-v", "error", "-nostdin", "-f", "rawvideo",
                         "-pix_fmt", "rgb24", "-video_size",
                         std::to_string(pixels.width) + "x" + std::to_string(pixels.height),
                         "-i", "pipe:0", "-frames:v", "1", "-f", "image2pipe", "-c:v", "png", "pipe:1"},
                        {{pipe[0].get(), 0}, {descriptor, 1}});
    pipe[0].close_checked("Cannot close PNG writer input reader");
    write_descriptor(pipe[1].get(), pixels.rgb.data(), pixels.rgb.size());
    pipe[1].close_checked("Cannot close PNG writer input");
    if (writer.wait() != 0) throw std::runtime_error("FFmpeg PNG output failed");
}
#endif
}  // namespace

void write_image_png(const std::filesystem::path& destination, const ImagePixels& pixels) {
    ProcessInterrupt::check();
    if (pixels.rgb.size() != rgb_size(pixels.width, pixels.height)) {
        throw std::invalid_argument("Invalid RGB pixels for PNG output");
    }
    if (destination.native().find('\0') != std::string::npos)
        throw std::invalid_argument("embedded null byte");
    struct stat status{};
    const bool created = stat(destination.c_str(), &status) != 0;
    FileDescriptor output;
    for (;;) {
        ProcessInterrupt::check();
        const int descriptor = open(destination.c_str(), O_RDWR | O_CREAT | O_TRUNC | O_CLOEXEC, 0666);
        if (descriptor >= 0) { output.reset(descriptor); break; }
        const int error = errno;
        if (error != EINTR)
            throw std::system_error(error, std::generic_category(), "Cannot open UF PNG output");
    }
    try {
        png_write_rgb(output.get(), pixels);
    } catch (...) {
        output.close_checked("Cannot close UF PNG output");
        // Pillow removes newly created files after an encoder exception,
        // suppressing PermissionError only. Existing outputs stay truncated.
        if (created && unlink(destination.c_str()) != 0) {
            const int error = errno;
            if (error != EACCES && error != EPERM)
                throw std::system_error(error, std::generic_category(), "Cannot remove failed UF PNG output");
        }
        throw;
    }
    output.close_checked("Cannot close UF PNG output");
    ProcessInterrupt::check();
}

namespace image_native {

std::size_t rgb_bytes(std::uint64_t width, std::uint64_t height) {
    return rgb_size(width, height);
}

std::optional<unsigned> orientation_from_exif(const std::uint8_t* bytes, std::size_t size) {
    if (!bytes || !size) return {};
    return tiff_orientation_at([&](std::uint64_t offset, void* out, std::size_t count) {
        if (offset > size || count > size - offset) return false;
        std::memcpy(out, bytes + offset, count);
        return true;
    }, size);
}

std::optional<unsigned> orientation_from_xmp(const std::uint8_t* bytes, std::size_t size) {
    if (!bytes || !size) return {};
    return xmp_orientation(std::string_view(reinterpret_cast<const char*>(bytes), size));
}

}  // namespace image_native

}  // namespace uf
