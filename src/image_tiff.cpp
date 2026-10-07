#include "image_native.hpp"

#include "uf/process.hpp"

#include <optional>

#ifdef UF_HAS_TIFF
#include <tiffio.h>
#ifdef UF_HAS_LCMS2
#include <lcms2.h>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>
#endif

namespace uf::image_native {
#ifdef UF_HAS_TIFF
namespace {

struct Diagnostics {
    char message[512]{};
    bool failed{};
};

int tiff_error(TIFF*, void* context, const char*, const char* format, va_list arguments) {
    auto* diagnostics = static_cast<Diagnostics*>(context);
    std::vsnprintf(diagnostics->message, sizeof(diagnostics->message), format, arguments);
    diagnostics->failed = true;
    return 1;
}

int tiff_warning(TIFF*, void*, const char*, const char*, va_list) { return 1; }

void check(bool success, const Diagnostics& diagnostics, const char* operation) {
    if (!success || diagnostics.failed) {
        throw std::runtime_error(std::string("UF TIFF ") + operation + ": " +
                                 (diagnostics.message[0] ? diagnostics.message : "invalid image"));
    }
}

std::size_t allocation(std::uint64_t value) {
    if (!value || value > std::numeric_limits<std::size_t>::max() ||
        value > std::uint64_t(std::numeric_limits<tmsize_t>::max())) {
        throw std::runtime_error("Invalid or excessive UF TIFF buffer");
    }
    return static_cast<std::size_t>(value);
}

struct RgbaState {
    TIFFRGBAImage image{};
    bool active{};
    ~RgbaState() { if (active) TIFFRGBAImageEnd(&image); }
};

ImagePixels ycbcr_rgb(TIFF* file, const Diagnostics& diagnostics, std::uint32_t width,
                      std::uint32_t height) {
    RgbaState state;
    char error[1024]{};
    check(TIFFRGBAImageOK(file, error) != 0, diagnostics, error);
    state.active = TIFFRGBAImageBegin(&state.image, file, 0, error) != 0;
    check(state.active, diagnostics, error);
    // The generic EXIF transform is applied by the caller after decoding.
    state.image.orientation = ORIENTATION_TOPLEFT;
    state.image.req_orientation = ORIENTATION_TOPLEFT;
    state.image.col_offset = 0;
    if (state.image.width != width || state.image.height != height) {
        throw std::runtime_error("Inconsistent UF TIFF RGB dimensions");
    }
    std::uint32_t block_height = height;
    TIFFGetFieldDefaulted(file, TIFFIsTiled(file) ? TIFFTAG_TILELENGTH : TIFFTAG_ROWSPERSTRIP,
                          &block_height);
    if (!block_height || block_height > height) block_height = height;
    std::vector<std::uint32_t> block(allocation(std::uint64_t(width) * block_height * 4) / 4);
    ImagePixels result{width, height, {}};
    result.rgb.resize(rgb_bytes(width, height));
    for (std::uint32_t y = 0; y < height;) {
        ProcessInterrupt::check();
        const auto rows = std::min(block_height, height - y);
        state.image.row_offset = static_cast<int>(y);
        check(TIFFRGBAImageGet(&state.image, block.data(), width, rows) != 0, diagnostics, "YCbCr pixels");
        for (std::uint32_t row = 0; row < rows; ++row) {
            ProcessInterrupt::check();
            for (std::uint32_t x = 0; x < width; ++x) {
                const auto rgba = block[static_cast<std::size_t>(row) * width + x];
                auto* destination = result.rgb.data() + (static_cast<std::size_t>(y + row) * width + x) * 3;
                destination[0] = static_cast<std::uint8_t>(TIFFGetR(rgba));
                destination[1] = static_cast<std::uint8_t>(TIFFGetG(rgba));
                destination[2] = static_cast<std::uint8_t>(TIFFGetB(rgba));
            }
        }
        y += rows;
    }
    return result;
}

struct Layout {
    std::uint32_t width{}, height{};
    std::uint16_t depth{1}, samples{1}, format{SAMPLEFORMAT_UINT}, planar{PLANARCONFIG_CONTIG};
    std::uint16_t photo{PHOTOMETRIC_MINISWHITE}, channels{};
    bool associated{};
    const std::uint16_t* red{};
    const std::uint16_t* green{};
    const std::uint16_t* blue{};
};

std::uint32_t sample(const std::uint8_t* row, std::uint64_t index, unsigned depth) {
    if (depth == 8) return row[index];
    if (depth == 16) {
        std::uint16_t value;
        std::memcpy(&value, row + index * 2, sizeof(value));
        return value;
    }
    if (depth == 32) {
        std::uint32_t value;
        std::memcpy(&value, row + index * 4, sizeof(value));
        return value;
    }
    std::uint32_t result = 0;
    for (unsigned bit = 0; bit < depth; ++bit) {
        const auto position = index * depth + bit;
        result = (result << 1) | ((row[position / 8] >> (7 - position % 8)) & 1);
    }
    return result;
}

std::uint8_t stored_byte(const Layout& layout, const std::uint8_t* row, std::uint64_t index) {
    const auto value = sample(row, index, layout.depth);
    if (layout.photo == PHOTOMETRIC_PALETTE) return static_cast<std::uint8_t>(value);
    if (layout.photo == PHOTOMETRIC_RGB || layout.photo == PHOTOMETRIC_SEPARATED) {
        return static_cast<std::uint8_t>(layout.depth == 16 ? value >> 8 : value);
    }
    if (layout.depth <= 8) {
        const auto maximum = (1u << layout.depth) - 1;
        const auto byte = value * 255 / maximum;
        return static_cast<std::uint8_t>(layout.photo == PHOTOMETRIC_MINISWHITE ? 255 - byte : byte);
    }
    // Pillow's integer/float grayscale modes clamp numeric samples to 0..255
    // on RGB conversion; they do not rescale the entire range to eight bits.
    if (layout.depth == 32 && layout.format == SAMPLEFORMAT_IEEEFP) {
        float number;
        std::memcpy(&number, &value, sizeof(number));
        if (std::isnan(number) || number <= 0) return 0;
        if (number >= 255) return 255;
        return static_cast<std::uint8_t>(number);
    }
    if (layout.depth == 32) {
        // Mode I stores both signed/unsigned TIFF words in a signed INT32.
        std::int32_t number;
        std::memcpy(&number, &value, sizeof(number));
        return static_cast<std::uint8_t>(std::clamp<std::int32_t>(number, 0, 255));
    }
    if (layout.depth == 16 && layout.format == SAMPLEFORMAT_INT) {
        const auto word = static_cast<std::uint16_t>(value);
        std::int16_t number;
        std::memcpy(&number, &word, sizeof(number));
        return static_cast<std::uint8_t>(std::clamp<int>(number, 0, 255));
    }
    return static_cast<std::uint8_t>(std::min<std::uint32_t>(value, 255));
}

void copy_row(const Layout& layout, const std::uint8_t* row, std::uint32_t x, std::uint32_t y,
              std::uint32_t count, std::optional<unsigned> plane, std::vector<std::uint8_t>& pixels) {
    for (std::uint32_t column = 0; column < count; ++column) {
        auto* destination = pixels.data() +
            (static_cast<std::size_t>(y) * layout.width + x + column) * layout.channels;
        if (plane) destination[*plane] = stored_byte(layout, row, column);
        else {
            for (unsigned channel = 0; channel < layout.channels; ++channel) {
                destination[channel] = stored_byte(layout, row, std::uint64_t(column) * layout.samples + channel);
            }
        }
    }
}

void read_pixels(TIFF* file, const Diagnostics& diagnostics, const Layout& layout,
                 std::vector<std::uint8_t>& pixels) {
    const bool separate = layout.planar == PLANARCONFIG_SEPARATE;
    if (!TIFFIsTiled(file)) {
        const auto stride = allocation(TIFFScanlineSize64(file));
        const std::uint64_t row_bits = std::uint64_t(layout.width) * layout.depth *
                                       (separate ? 1 : layout.samples);
        check((row_bits + 7) / 8 <= stride, diagnostics, "scanline layout");
        std::vector<std::uint8_t> row(stride);
        // Plane-major reads keep compressed strips sequential.
        for (unsigned plane = 0; plane < (separate ? layout.samples : 1); ++plane) {
            for (std::uint32_t y = 0; y < layout.height; ++y) {
                ProcessInterrupt::check();
                check(TIFFReadScanline(file, row.data(), y, static_cast<std::uint16_t>(plane)) == 1,
                      diagnostics, "scanline pixels");
                if (!separate || plane < layout.channels) {
                    copy_row(layout, row.data(), 0, y, layout.width,
                             separate ? std::optional<unsigned>(plane) : std::nullopt, pixels);
                }
            }
        }
        return;
    }
    std::uint32_t tile_width{}, tile_height{};
    check(TIFFGetField(file, TIFFTAG_TILEWIDTH, &tile_width) &&
          TIFFGetField(file, TIFFTAG_TILELENGTH, &tile_height) && tile_width && tile_height,
          diagnostics, "tile dimensions");
    const auto stride = allocation(TIFFTileRowSize64(file));
    const auto bytes = allocation(TIFFTileSize64(file));
    const std::uint64_t row_bits = std::uint64_t(tile_width) * layout.depth * (separate ? 1 : layout.samples);
    check((row_bits + 7) / 8 <= stride && tile_height <= bytes / stride, diagnostics, "tile layout");
    std::vector<std::uint8_t> tile(bytes);
    for (unsigned plane = 0; plane < (separate ? layout.samples : 1); ++plane) {
        for (std::uint64_t y = 0; y < layout.height; y += tile_height) {
            for (std::uint64_t x = 0; x < layout.width; x += tile_width) {
                ProcessInterrupt::check();
                const auto decoded = TIFFReadTile(file, tile.data(), static_cast<std::uint32_t>(x),
                                                  static_cast<std::uint32_t>(y), 0,
                                                  static_cast<std::uint16_t>(plane));
                check(decoded >= 0 && static_cast<std::uint64_t>(decoded) >= std::uint64_t(stride) * tile_height,
                      diagnostics, "tile pixels");
                const auto rows = std::min<std::uint64_t>(tile_height, layout.height - y);
                const auto columns = std::min<std::uint64_t>(tile_width, layout.width - x);
                for (std::uint32_t row = 0; row < rows; ++row) {
                    ProcessInterrupt::check();
                    if (!separate || plane < layout.channels) {
                        copy_row(layout, tile.data() + static_cast<std::size_t>(row) * stride,
                                 static_cast<std::uint32_t>(x), static_cast<std::uint32_t>(y + row),
                                 static_cast<std::uint32_t>(columns),
                                 separate ? std::optional<unsigned>(plane) : std::nullopt, pixels);
                    }
                }
            }
        }
    }
}

#ifdef UF_HAS_LCMS2
ImagePixels lab_rgb(const Layout& layout, const std::vector<std::uint8_t>& pixels) {
    // Image.convert("RGB") creates these profiles in this order, with the
    // default D50 Lab v2 white point, perceptual intent and no transform flags.
    const std::unique_ptr<void, decltype(&cmsCloseProfile)> srgb(
        cmsCreate_sRGBProfile(), cmsCloseProfile);
    if (!srgb) throw std::runtime_error("UF TIFF cannot create sRGB profile");
    const std::unique_ptr<void, decltype(&cmsCloseProfile)> lab(
        cmsCreateLab2Profile(nullptr), cmsCloseProfile);
    if (!lab) throw std::runtime_error("UF TIFF cannot create Lab profile");
    // Pillow's LAB/RGB imaging storage has an auxiliary byte per pixel.
    // PT_LabV2 is required here; TYPE_Lab_8 describes the v4 encoding.
    constexpr cmsUInt32Number lab_format =
        COLORSPACE_SH(PT_LabV2) | CHANNELS_SH(3) | BYTES_SH(1) | EXTRA_SH(1);
    const std::unique_ptr<void, decltype(&cmsDeleteTransform)> transform(
        cmsCreateTransform(lab.get(), lab_format, srgb.get(), TYPE_RGBA_8,
                           INTENT_PERCEPTUAL, 0), cmsDeleteTransform);
    if (!transform) throw std::runtime_error("UF TIFF cannot create Lab-to-RGB transform");

    ImagePixels result{layout.width, layout.height, {}};
    result.rgb.resize(rgb_bytes(layout.width, layout.height));
    const auto pixel_count = result.rgb.size() / 3;
    constexpr std::size_t block_pixels = 16384;
    const auto capacity = std::min(block_pixels, pixel_count);
    std::vector<std::uint8_t> input(capacity * 4);
    std::vector<std::uint8_t> output(capacity * 4);
    // The contiguous LAB unpacker shifts the signed TIFF a/b bytes. Pillow's
    // separate-plane unpackers copy each band directly without this shift.
    const std::uint8_t bias = layout.planar == PLANARCONFIG_CONTIG ? 128 : 0;
    for (std::size_t start = 0; start < pixel_count;) {
        ProcessInterrupt::check();
        const auto count = std::min(capacity, pixel_count - start);
        for (std::size_t i = 0; i < count; ++i) {
            const auto* source = pixels.data() + (start + i) * 3;
            auto* destination = input.data() + i * 4;
            destination[0] = source[0];
            destination[1] = source[1] ^ bias;
            destination[2] = source[2] ^ bias;
            destination[3] = 255;
        }
        cmsDoTransform(transform.get(), input.data(), output.data(),
                       static_cast<cmsUInt32Number>(count));
        for (std::size_t i = 0; i < count; ++i) {
            std::copy_n(output.data() + i * 4, 3, result.rgb.data() + (start + i) * 3);
        }
        start += count;
    }
    return result;
}
#endif

}  // namespace
#endif

std::optional<ImagePixels> tiff_rgb(const std::filesystem::path& source, unsigned& orientation) {
#ifdef UF_HAS_TIFF
    Diagnostics diagnostics;
    const std::unique_ptr<TIFFOpenOptions, decltype(&TIFFOpenOptionsFree)> options(
        TIFFOpenOptionsAlloc(), TIFFOpenOptionsFree);
    if (!options) throw std::bad_alloc();
    TIFFOpenOptionsSetErrorHandlerExtR(options.get(), tiff_error, &diagnostics);
    TIFFOpenOptionsSetWarningHandlerExtR(options.get(), tiff_warning, &diagnostics);
    // B asks the decoder for MSB-first packed samples; words are swapped by
    // libtiff into native endian order before the scanline/tile APIs return.
    const std::unique_ptr<TIFF, decltype(&TIFFClose)> file(
        TIFFOpenExt(source.c_str(), "rB", options.get()), TIFFClose);
    check(bool(file), diagnostics, "input");
    Layout layout;
    check(TIFFGetField(file.get(), TIFFTAG_IMAGEWIDTH, &layout.width) &&
          TIFFGetField(file.get(), TIFFTAG_IMAGELENGTH, &layout.height), diagnostics, "dimensions");
    const auto size = rgb_bytes(layout.width, layout.height);
    TIFFGetFieldDefaulted(file.get(), TIFFTAG_BITSPERSAMPLE, &layout.depth);
    TIFFGetFieldDefaulted(file.get(), TIFFTAG_SAMPLESPERPIXEL, &layout.samples);
    TIFFGetFieldDefaulted(file.get(), TIFFTAG_SAMPLEFORMAT, &layout.format);
    TIFFGetFieldDefaulted(file.get(), TIFFTAG_PLANARCONFIG, &layout.planar);
    TIFFGetField(file.get(), TIFFTAG_PHOTOMETRIC, &layout.photo);
    std::uint16_t exif_orientation{};
    if (TIFFGetField(file.get(), TIFFTAG_ORIENTATION, &exif_orientation)) orientation = exif_orientation;
    else {
        std::uint32_t xmp_size{};
        const void* xmp{};
        if (TIFFGetField(file.get(), TIFFTAG_XMLPACKET, &xmp_size, &xmp)) {
            orientation = orientation_from_xmp(static_cast<const std::uint8_t*>(xmp), xmp_size).value_or(1);
        }
    }
    if (!layout.samples || layout.samples > 6 ||
        (layout.planar != PLANARCONFIG_CONTIG && layout.planar != PLANARCONFIG_SEPARATE)) return {};

    std::uint16_t compression{COMPRESSION_NONE};
    TIFFGetFieldDefaulted(file.get(), TIFFTAG_COMPRESSION, &compression);
    if (layout.photo == PHOTOMETRIC_YCBCR && layout.samples == 3 && layout.depth == 8) {
        if (compression == COMPRESSION_JPEG && layout.planar == PLANARCONFIG_CONTIG) {
            // Pillow also delegates this case's color conversion to libjpeg.
            check(TIFFSetField(file.get(), TIFFTAG_JPEGCOLORMODE, JPEGCOLORMODE_RGB) != 0,
                  diagnostics, "JPEG RGB mode");
            layout.photo = PHOTOMETRIC_RGB;
        } else {
            return ycbcr_rgb(file.get(), diagnostics, layout.width, layout.height);
        }
    }
    std::uint16_t extra_count{};
    // Match the accessor's const uint16_t** variadic output argument.
    const std::uint16_t* extra{};
    // GetField tests the directory's tag-presence bit. GetFieldDefaulted also
    // exposes unspecified extra channels synthesized by libtiff's reader.
    const bool has_extra_tag = TIFFGetField(file.get(), TIFFTAG_EXTRASAMPLES,
                                           &extra_count, &extra) != 0;
    TIFFGetFieldDefaulted(file.get(), TIFFTAG_EXTRASAMPLES, &extra_count, &extra);
    if (layout.photo == PHOTOMETRIC_RGB && (layout.depth == 8 || layout.depth == 16) &&
        layout.format == SAMPLEFORMAT_UINT && layout.samples >= 3) {
        // Pillow opens four RGB samples without an ExtraSamples tag as RGBA.
        // Its compressed separate-plane decoder then unpremultiplies the
        // synthesized unspecified alpha, just as it does associated alpha.
        const bool synthesized_planar_alpha = layout.samples == 4 && !has_extra_tag &&
            compression != COMPRESSION_NONE && layout.planar == PLANARCONFIG_SEPARATE &&
            extra_count && extra && extra[0] == EXTRASAMPLE_UNSPECIFIED;
        layout.associated = layout.samples >= 4 && extra_count && extra &&
            (extra[0] == EXTRASAMPLE_ASSOCALPHA || synthesized_planar_alpha);
        layout.channels = layout.associated ? 4 : 3;
    } else if (layout.photo == PHOTOMETRIC_SEPARATED && layout.samples >= 4 &&
               layout.format == SAMPLEFORMAT_UINT && (layout.depth == 8 || layout.depth == 16)) {
        layout.channels = 4;
#ifdef UF_HAS_LCMS2
    } else if (layout.photo == PHOTOMETRIC_CIELAB && layout.depth == 8 && layout.samples == 3 &&
               layout.format == SAMPLEFORMAT_UINT && extra_count == 0) {
        std::uint16_t fill_order{FILLORDER_MSB2LSB};
        TIFFGetFieldDefaulted(file.get(), TIFFTAG_FILLORDER, &fill_order);
        if (fill_order != FILLORDER_MSB2LSB) return {};
        layout.channels = 3;
#endif
    } else if (layout.photo == PHOTOMETRIC_PALETTE && layout.format == SAMPLEFORMAT_UINT &&
               (layout.depth == 1 || layout.depth == 2 || layout.depth == 4 || layout.depth == 8)) {
        const std::uint16_t* red{};
        const std::uint16_t* green{};
        const std::uint16_t* blue{};
        check(TIFFGetField(file.get(), TIFFTAG_COLORMAP, &red, &green, &blue) && red && green && blue,
              diagnostics, "palette");
        layout.red = red; layout.green = green; layout.blue = blue;
        layout.channels = 1;
    } else if (layout.photo == PHOTOMETRIC_MINISWHITE || layout.photo == PHOTOMETRIC_MINISBLACK) {
        const bool supported = (layout.format == SAMPLEFORMAT_UINT &&
            (layout.depth == 1 || layout.depth == 2 || layout.depth == 4 || layout.depth == 8 ||
             layout.depth == 12 || layout.depth == 16 || layout.depth == 32)) ||
            (layout.format == SAMPLEFORMAT_INT && (layout.depth == 8 || layout.depth == 16 || layout.depth == 32)) ||
            (layout.format == SAMPLEFORMAT_IEEEFP && layout.depth == 32);
        if (!supported) return {};
        layout.channels = 1;
    } else return {};

    std::vector<std::uint8_t> pixels(size / 3 * layout.channels);
    read_pixels(file.get(), diagnostics, layout, pixels);
#ifdef UF_HAS_LCMS2
    if (layout.photo == PHOTOMETRIC_CIELAB) return lab_rgb(layout, pixels);
#endif
    ImagePixels result{layout.width, layout.height, {}};
    result.rgb.resize(size);
    for (std::size_t i = 0; i < size / 3; ++i) {
        if (i % layout.width == 0) ProcessInterrupt::check();
        const auto* value = pixels.data() + i * layout.channels;
        auto* destination = result.rgb.data() + i * 3;
        if (layout.photo == PHOTOMETRIC_PALETTE) {
            destination[0] = static_cast<std::uint8_t>(layout.red[value[0]] >> 8);
            destination[1] = static_cast<std::uint8_t>(layout.green[value[0]] >> 8);
            destination[2] = static_cast<std::uint8_t>(layout.blue[value[0]] >> 8);
        } else if (layout.photo == PHOTOMETRIC_SEPARATED) {
            const unsigned black = 255 - value[3];
            for (unsigned c = 0; c < 3; ++c) {
                destination[c] = static_cast<std::uint8_t>(((255 - value[c]) * black + 127) / 255);
            }
        } else if (layout.photo == PHOTOMETRIC_RGB) {
            if (!layout.associated) std::copy_n(value, 3, destination);
            else if (!value[3]) std::fill_n(destination, 3, 0);
            else {
                for (unsigned c = 0; c < 3; ++c) {
                    destination[c] = static_cast<std::uint8_t>(
                        std::min<unsigned>(255, unsigned(value[c]) * 255 / value[3]));
                }
            }
        } else std::fill_n(destination, 3, value[0]);
    }
    return result;
#else
    (void)source;
    (void)orientation;
    return {};
#endif
}

}  // namespace uf::image_native
