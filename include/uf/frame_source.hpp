#pragma once

#include "uf/codec.hpp"

#include <functional>
#include <array>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>

namespace uf {

struct IndexedPacket {
    std::streamoff offset{};
    std::uint64_t first{}, count{};
    NalType type{};
    Sps sps;
    std::uint8_t qp{};
};

struct DecodedFrame {
    std::uint64_t index{};
    std::uint32_t width{}, height{};
    NalType type{};
    std::uint8_t qp{};
    std::vector<std::uint8_t> rgb, raw;
    // Image metadata keeps Python's width/height objects after slicing. Pixel
    // storage uses the actual cropped dimensions above; the UI uses these
    // recorded values for scaling and status text.
    std::optional<std::array<nlohmann::json, 2>> recorded_dimensions;
};

using ViewerCodecLoader = std::function<std::shared_ptr<Codec>(const nlohmann::json& pipeline,
                                                              bool image_only)>;

// Length of array[:stop] for a native image axis. Python integer endpoints
// (including bool and unbounded JSON integers) clip; None selects the axis.
std::uint32_t image_crop_extent(const nlohmann::json& stop, std::uint32_t extent);

std::vector<std::uint8_t> display_rgb(const std::vector<std::uint8_t>& raw,
                                    std::uint32_t width, std::uint32_t height,
                                    std::uint32_t crop_width, std::uint32_t crop_height);

// Validates storage and indexes packets without loading neural models. A single
// decode worker owns the lazy codec, temporal state and one decoded packet cache.
class IndexedFrameSource {
public:
    IndexedFrameSource(const std::filesystem::path& input, ViewerCodecLoader loader);
    ~IndexedFrameSource();
    IndexedFrameSource(const IndexedFrameSource&) = delete;
    IndexedFrameSource& operator=(const IndexedFrameSource&) = delete;
    const nlohmann::json& metadata() const;
    // Gallery key: repr of the original ordered pipeline object.
    std::string pipeline_repr() const;
    const std::vector<IndexedPacket>& packets() const;
    std::uint64_t frame_count() const;
    DecodedFrame seek(std::uint64_t frame, const std::function<bool()>& cancelled = {});
    // Close the owned payload before dropping codec/cache handles. Explicit
    // file-close failures propagate; destruction remains a fallback.
    void close_checked();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace uf
