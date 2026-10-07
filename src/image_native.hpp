#pragma once

#include "uf/image_pixels.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <vector>

namespace uf::image_native {

inline constexpr std::uint64_t max_pixels = 178956970;
std::size_t rgb_bytes(std::uint64_t width, std::uint64_t height);
std::optional<unsigned> orientation_from_exif(const std::uint8_t* bytes, std::size_t size);
std::optional<unsigned> orientation_from_xmp(const std::uint8_t* bytes, std::size_t size);

std::optional<ImagePixels> tiff_rgb(const std::filesystem::path& source, unsigned& orientation);
ImagePixels avif_rgb(const std::vector<std::uint8_t>& bytes, unsigned& orientation);

}  // namespace uf::image_native
