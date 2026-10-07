#include "image_native.hpp"

#include "uf/process.hpp"

#ifdef UF_HAS_AVIF
#include <avif/avif.h>

#include <algorithm>
#include <array>
#include <climits>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
#include <sched.h>

namespace uf::image_native {
namespace {

void avif_check(avifResult result, const char* operation) {
    if (result != AVIF_RESULT_OK) {
        throw std::runtime_error(std::string("UF AVIF ") + operation + ": " + avifResultToString(result));
    }
}

int decoder_threads() {
    cpu_set_t cpus;
    CPU_ZERO(&cpus);
    if (sched_getaffinity(0, sizeof(cpus), &cpus) == 0 && CPU_COUNT(&cpus) > 0) return CPU_COUNT(&cpus);
    return static_cast<int>(std::max(1u, std::min<unsigned>(INT_MAX, std::thread::hardware_concurrency())));
}

unsigned transform_orientation(const avifImage& image) {
    const auto angle = (image.transformFlags & AVIF_TRANSFORM_IROT) && image.irot.angle < 4
                     ? image.irot.angle : 0;
    // AVIF rotates anticlockwise, then reflects on the signalled axis.
    constexpr std::array<unsigned, 4> rotation{1, 8, 3, 6};
    constexpr std::array<unsigned, 4> top_bottom{4, 5, 2, 7};
    constexpr std::array<unsigned, 4> left_right{2, 7, 4, 5};
    if (!(image.transformFlags & AVIF_TRANSFORM_IMIR)) return rotation[angle];
    return image.imir.axis ? left_right[angle] : top_bottom[angle];
}

}  // namespace

ImagePixels avif_rgb(const std::vector<std::uint8_t>& bytes, unsigned& orientation) {
    const std::unique_ptr<avifDecoder, decltype(&avifDecoderDestroy)> decoder(
        avifDecoderCreate(), avifDecoderDestroy);
    if (!decoder) throw std::bad_alloc();
    decoder->maxThreads = decoder_threads();
    // Match Pillow's acceptance of older libheif metadata and its decision
    // to leave clean-aperture cropping unapplied.
    decoder->strictFlags &= ~(AVIF_STRICT_CLAP_VALID | AVIF_STRICT_PIXI_REQUIRED);
    decoder->codecChoice = AVIF_CODEC_CHOICE_AUTO;
    decoder->imageSizeLimit = static_cast<std::uint32_t>(max_pixels);
    avif_check(avifDecoderSetIOMemory(decoder.get(), bytes.data(), bytes.size()), "input");
    ProcessInterrupt::check();
    avif_check(avifDecoderParse(decoder.get()), "metadata");
    if (!decoder->image || !decoder->imageCount) throw std::runtime_error("Empty UF AVIF image");
    rgb_bytes(decoder->image->width, decoder->image->height);

    const auto intrinsic = transform_orientation(*decoder->image);
    const auto exif_tag = orientation_from_exif(decoder->image->exif.data, decoder->image->exif.size);
    // Pillow rewrites an existing EXIF orientation to the intrinsic AVIF
    // orientation, including 1. XMP is considered only if no tag is supplied.
    orientation = (intrinsic != 1 || exif_tag.has_value()) ? intrinsic
        : orientation_from_xmp(decoder->image->xmp.data, decoder->image->xmp.size).value_or(1);

    ProcessInterrupt::check();
    avif_check(avifDecoderNthImage(decoder.get(), 0), "first frame");
    const auto* image = decoder->image;
    if (!image) throw std::runtime_error("Missing UF AVIF frame");
    ImagePixels result{image->width, image->height, {}};
    result.rgb.resize(rgb_bytes(result.width, result.height));

    avifRGBImage rgb{};
    avifRGBImageSetDefaults(&rgb, image);
    rgb.depth = 8;
    // Keep alpha through YUV conversion so the library can unpremultiply an
    // associated-alpha source. The application's RGB conversion then drops it.
    const unsigned channels = decoder->alphaPresent ? 4 : 3;
    rgb.format = channels == 4 ? AVIF_RGB_FORMAT_RGBA : AVIF_RGB_FORMAT_RGB;
    rgb.rowBytes = result.width * channels;
    std::vector<std::uint8_t> converted(static_cast<std::size_t>(rgb.rowBytes) * result.height);
    rgb.pixels = converted.data();
    ProcessInterrupt::check();
    avif_check(avifImageYUVToRGB(image, &rgb), "RGB conversion");
    for (std::uint32_t y = 0; y < result.height; ++y) {
        ProcessInterrupt::check();
        const auto* row = converted.data() + static_cast<std::size_t>(y) * rgb.rowBytes;
        auto* destination = result.rgb.data() + static_cast<std::size_t>(y) * result.width * 3;
        if (channels == 3) std::copy_n(row, static_cast<std::size_t>(result.width) * 3, destination);
        else {
            for (std::uint32_t x = 0; x < result.width; ++x) {
                std::copy_n(row + static_cast<std::size_t>(x) * 4, 3,
                            destination + static_cast<std::size_t>(x) * 3);
            }
        }
    }
    return result;
}

}  // namespace uf::image_native
#endif
