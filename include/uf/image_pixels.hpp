#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

namespace uf {

struct ImagePixels {
    std::uint32_t width{}, height{};
    std::vector<std::uint8_t> rgb;
};

// First image/frame, oriented like ImageOps.exif_transpose().convert("RGB").
// Native JPEG/PNG/WebP/TIFF/AVIF backends are optional; other formats use FFmpeg.
// BMP/DIB raw, bitfield and RLE readers are native C++ without decoder libraries.
// False matches Image.open(...).convert("RGB") without an extra transpose.
// Pillow's TIFF decoder itself applies orientation, so TIFF still does so.
// Native filename bytes; embedded NUL is rejected before opening the input.
ImagePixels read_image_rgb(const std::filesystem::path& source, bool exif_transpose = true);
// Image.open(...).verify() policy for JPEG/PNG and native WebP: inspect headers
// and PNG chunk CRCs without decoding pixels. Other formats (and optional WebP
// without its library) retain FFmpeg validation; those paths may differ.
// With native TIFF enabled, validation uses the selected full image reader,
// including numeric grayscale modes that FFmpeg does not support.
void verify_image(const std::filesystem::path& source);
// Write packed RGB as PNG with owned mode0666 output (subject to umask) and
// checked close. Native libpng emits RGB8/noninterlaced data without implicit
// colorspace/profile metadata. FFmpeg is the fallback; bytes/metadata may differ.
void write_image_png(const std::filesystem::path& destination, const ImagePixels& pixels);

}  // namespace uf
