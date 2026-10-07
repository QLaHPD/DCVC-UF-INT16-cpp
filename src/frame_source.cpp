#include "uf/frame_source.hpp"
#include "uf/archive.hpp"
#include "uf/image_archive.hpp"
#include "uf/json_compat.hpp"
#include "uf/process.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace uf {
namespace {

std::string identity(const std::array<std::uint8_t, 32>& bytes) {
    std::ostringstream result;
    result << std::hex << std::setfill('0');
    for (const auto value : bytes) result << std::setw(2) << static_cast<unsigned>(value);
    return result.str();
}

std::uint8_t byte_pixel(float value) {
    // NumPy's display transform rounds ties to the even integer.
    value = std::clamp(value, 0.0f, 255.0f);
    auto base = static_cast<unsigned>(std::floor(value));
    const float fraction = value - base;
    if (fraction > 0.5f || (fraction == 0.5f && base % 2)) ++base;
    return static_cast<std::uint8_t>(base);
}

}  // namespace

std::uint32_t image_crop_extent(const nlohmann::json& stop, std::uint32_t extent) {
    if (stop.is_null()) return extent;
    if (!stop.is_number_integer() && !stop.is_boolean() && !python_json_extended_integer(stop))
        throw PythonJsonTypeError("slice indices must be integers or None or have an __index__ method");
    if (json_python_less(stop, nlohmann::json(0))) {
        const auto negative_extent = -static_cast<std::int64_t>(extent);
        if (json_python_less(stop, nlohmann::json(negative_extent))) return 0;
        return static_cast<std::uint32_t>(static_cast<std::int64_t>(extent) +
                                          python_json_index_int64(stop));
    }
    if (!json_python_less(stop, nlohmann::json(extent))) return extent;
    return static_cast<std::uint32_t>(python_json_index_int64(stop));
}

std::vector<std::uint8_t> display_rgb(const std::vector<std::uint8_t>& raw,
                                    std::uint32_t width, std::uint32_t height,
                                    std::uint32_t crop_width, std::uint32_t crop_height) {
    const auto area = static_cast<std::uint64_t>(width) * height;
    if (!width || !height || width % 2 || height % 2 ||
        crop_width > width || crop_height > height ||
        area > std::numeric_limits<std::size_t>::max() / 3 || raw.size() != area + area / 2)
        throw std::invalid_argument("Invalid UF display frame dimensions");
    std::vector<std::uint8_t> rgb(static_cast<std::size_t>(crop_width) * crop_height * 3);
    if (!crop_width || !crop_height) return rgb;
    for (std::uint32_t y = 0; y < crop_height; ++y) {
        ProcessInterrupt::check();
        for (std::uint32_t x = 0; x < crop_width; ++x) {
            const auto chroma = static_cast<std::size_t>(y / 2) * (width / 2) + x / 2;
            const float luminance = raw[static_cast<std::size_t>(y) * width + x];
            const float u = static_cast<float>(raw[area + chroma]) - 128.0f;
            const float v = static_cast<float>(raw[area + area / 4 + chroma]) - 128.0f;
            const auto offset = (static_cast<std::size_t>(y) * crop_width + x) * 3;
            // Each NumPy ufunc produces a Float32 array before the next step.
            // uf_codec also disables contraction so these steps cannot fuse.
            const float red_delta = 1.5748f * v;
            const float green_u = 0.187324f * u;
            const float green_v = 0.468124f * v;
            const float green_base = luminance - green_u;
            const float blue_delta = 1.8556f * u;
            rgb[offset] = byte_pixel(luminance + red_delta);
            rgb[offset + 1] = byte_pixel(green_base - green_v);
            rgb[offset + 2] = byte_pixel(luminance + blue_delta);
        }
    }
    return rgb;
}

struct IndexedFrameSource::Impl {
    nlohmann::json metadata;
    nlohmann::ordered_json recorded_pipeline;
    std::ifstream input;
    std::vector<IndexedPacket> packets;
    std::vector<std::uint64_t> starts;
    std::vector<std::size_t> anchors;
    std::uint64_t frame_count{};
    ViewerCodecLoader loader;
    std::shared_ptr<Codec> codec;
    std::size_t next_packet{}, cached_packet{std::numeric_limits<std::size_t>::max()};
    std::vector<std::vector<std::uint8_t>> cached;
    bool image{}, close_attempted{};

    Impl(const std::filesystem::path& path, ViewerCodecLoader factory) : loader(std::move(factory)) {
        try {
            auto extension = path.extension().string();
            std::transform(extension.begin(), extension.end(), extension.begin(),
                [](unsigned char c) { return std::tolower(c); });
            image = extension == ".dcvci";
            std::filesystem::path bitstream;
            std::uint64_t offset = 0;
            if (image) {
                auto archive = read_intra_image(path, true, false);
                metadata = std::move(archive.metadata);
                recorded_pipeline = std::move(archive.recorded_pipeline);
                bitstream = std::move(archive.path);
                offset = archive.payload_offset;
            } else {
                const auto archive = load_archive(path);
                metadata = archive.metadata;
                bitstream = artifact_path(archive, "video.bin");
            }
            input.open(bitstream, std::ios::binary);
            if (!input || offset > static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max()))
                throw std::runtime_error("Cannot open UF viewer payload");
            input.seekg(static_cast<std::streamoff>(offset));
            const auto preamble = read_preamble(input);
            const auto& pipeline = metadata.at("pipeline");
            const auto runtime = pipeline.value("runtime", std::string("fp16"));
            if (runtime != "int16" && runtime != "fp16") throw std::runtime_error("Unknown UF archive runtime");
            if ((runtime == "int16") != preamble.integer) throw std::runtime_error("UF runtime/model identity mismatch");
            if (preamble.integer) {
                const auto& ids = pipeline.at("integer");
                if (identity(preamble.image_identity) != ids.at("image").get<std::string>() ||
                    identity(preamble.video_identity) != (image ? std::string(64, '0') : ids.at("video").get<std::string>()))
                    throw std::runtime_error("UF runtime/model identity mismatch");
            }
            const auto& video = metadata.at("video");
            frame_count = python_json_exact_uint64(video.at("frames"));
            const auto frames = frame_count;
            const auto variant = pipeline.at("variant").get<std::string>();
            if (variant != "ld" && variant != "hts" && variant != "htl" && !(image && variant == "image"))
                throw std::runtime_error("Unknown UF viewer model variant");
            const std::uint64_t chunk = variant == "ld" ? 1 : 8;
            std::array<std::optional<Sps>, 16> table{};
            std::uint64_t count = 0;
            for (;;) {
                ProcessInterrupt::check();
                const auto packet_offset = input.tellg();
                const auto header = read_optional_header(input);
                if (!header) break;
                const auto type = header->type;
                const auto id = header->sps_id;
                if (type == NalType::sps) {
                    const auto sps = read_sps(input, id);
                    if (!json_python_equal(nlohmann::json(sps.width), video.at("width")) ||
                        !json_python_equal(nlohmann::json(sps.height), video.at("height")) ||
                        sps.width % 2 || sps.height % 2)
                        throw std::runtime_error("UF bitstream dimensions differ from metadata");
                    table[id] = sps;
                    continue;
                }
                if ((type != NalType::intra && type != NalType::predicted) || !table[id] || count >= frames ||
                    (packets.empty() && type != NalType::intra) || packet_offset == std::streampos(-1))
                    throw std::runtime_error("Unexpected UF packet or unknown SPS");
                const auto packet = read_packet(input, type, id);
                if (packet.qp >= 64) throw std::runtime_error("Invalid UF packet QP");
                const auto length = std::min(type == NalType::intra ? 1 : chunk, frames - count);
                if (type == NalType::intra) anchors.push_back(packets.size());
                starts.push_back(count);
                packets.push_back({static_cast<std::streamoff>(packet_offset), count, length, type, *table[id], packet.qp});
                count += length;
            }
            if (!count || count != frames ||
                !json_python_equal(nlohmann::json(packets.size()), video.at("packets")))
                throw std::runtime_error("Incomplete UF viewer bitstream");
            if (image && (packets.size() != 1 || packets[0].type != NalType::intra))
                throw std::runtime_error(".dcvci must contain exactly one I-frame");
        } catch (...) {
            // Source constructor closes a successfully opened stream even
            // when preamble/index validation fails. Close can replace it.
            if (input.is_open()) close_checked();
            throw;
        }
    }

    void close_checked() {
        if (!close_attempted) {
            close_attempted = true;
            input.clear();
            input.close();
            if (input.fail()) throw std::runtime_error("Cannot close UF viewer payload");
        }
        codec.reset();
        cached.clear();
    }

    ~Impl() { try { close_checked(); } catch (...) {} }

    DecodedFrame seek(std::uint64_t frame, const std::function<bool()>& cancelled) {
        const auto check = [&]() {
            ProcessInterrupt::check();
            if (cancelled && cancelled()) throw std::runtime_error("UF viewer decoding cancelled");
        };
        if (frame >= frame_count)
            throw std::out_of_range("UF frame index is outside the bitstream");
        check();
        if (!codec) {
            if (!loader) throw std::runtime_error("UF viewer has no model loader");
            codec = loader(metadata.at("pipeline"), image || metadata.at("pipeline").value("image_only", false));
            if (!codec) throw std::runtime_error("UF viewer model loader returned no codec");
        }
        const auto target = static_cast<std::size_t>(std::upper_bound(starts.begin(), starts.end(), frame) - starts.begin() - 1);
        if (target != cached_packet) {
            const auto anchor = *(std::upper_bound(anchors.begin(), anchors.end(), target) - 1);
            if (target < next_packet || next_packet < anchor) next_packet = anchor;
            try {
                while (next_packet <= target) {
                    check();
                    const auto& row = packets[next_packet];
                    input.clear();
                    input.seekg(row.offset);
                    if (!input) throw std::runtime_error("Cannot seek UF packet");
                    const auto header = read_optional_header(input);
                    if (!header || header->type != row.type || header->sps_id != row.sps.id)
                        throw std::runtime_error("UF packet changed after indexing");
                    const auto packet = read_packet(input, row.type, row.sps.id);
                    if (packet.qp != row.qp) throw std::runtime_error("UF packet changed after indexing");
                    auto frames = codec->decode_packet(packet, row.sps, row.count);
                    if (frames.size() != row.count) throw std::runtime_error("Unexpected UF decoder chunk length");
                    cached = std::move(frames);
                    cached_packet = next_packet++;
                }
            } catch (...) {
                // A failed packet may have altered the DPB. Restart the next
                // request at an I anchor instead of reusing partial state.
                next_packet = 0;
                cached_packet = std::numeric_limits<std::size_t>::max();
                cached.clear();
                throw;
            }
        }
        check();
        const auto& packet = packets[target];
        DecodedFrame result;
        result.index = frame;
        result.width = image ? image_crop_extent(metadata.at("image_width"), packet.sps.width) : packet.sps.width;
        result.height = image ? image_crop_extent(metadata.at("image_height"), packet.sps.height) : packet.sps.height;
        if (image)
            result.recorded_dimensions = std::array<nlohmann::json, 2>{
                metadata.at("image_width"), metadata.at("image_height")};
        result.type = packet.type;
        result.qp = packet.qp;
        result.raw = cached.at(static_cast<std::size_t>(frame - packet.first));
        result.rgb = display_rgb(result.raw, packet.sps.width, packet.sps.height, result.width, result.height);
        return result;
    }
};

IndexedFrameSource::IndexedFrameSource(const std::filesystem::path& input, ViewerCodecLoader loader)
    : impl_(std::make_unique<Impl>(input, std::move(loader))) {}
IndexedFrameSource::~IndexedFrameSource() = default;
const nlohmann::json& IndexedFrameSource::metadata() const { return impl_->metadata; }
std::string IndexedFrameSource::pipeline_repr() const {
    return impl_->image ? python_json_repr(impl_->recorded_pipeline)
                        : python_json_repr(impl_->metadata.at("pipeline"));
}
const std::vector<IndexedPacket>& IndexedFrameSource::packets() const { return impl_->packets; }
std::uint64_t IndexedFrameSource::frame_count() const { return impl_->frame_count; }
DecodedFrame IndexedFrameSource::seek(std::uint64_t frame, const std::function<bool()>& cancelled) {
    return impl_->seek(frame, cancelled);
}
void IndexedFrameSource::close_checked() { impl_->close_checked(); }

}  // namespace uf
