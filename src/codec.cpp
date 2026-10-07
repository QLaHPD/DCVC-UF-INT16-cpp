#include "uf/codec.hpp"
#include "uf/archive.hpp"
#include "uf/fp16_codec.hpp"

#include <stdexcept>
#include <utility>

namespace uf {

CodecModels CodecModels::load(std::string runtime,
                               const std::filesystem::path& image,
                               const std::optional<std::filesystem::path>& video,
                               std::string variant, const ExpectedModel& expected_image,
                               const ExpectedModel& expected_video) {
    if ((runtime != "int16" && runtime != "fp16") ||
        (variant != "hts" && variant != "htl" && variant != "ld")) {
        throw std::invalid_argument("Unknown UF codec runtime or variant");
    }
    CodecModels result;
    result.runtime = std::move(runtime);
    result.variant = std::move(variant);
    result.image_path = image;
    result.video_path = video;
    if (result.runtime == "int16") {
        result.integer_image = load_prepared(image, expected_image.source_sha256,
                                              std::string("image"), expected_image.prepared_identity);
        result.image_sha256 = result.integer_image->source_sha256;
        if (video) {
            result.integer_video = load_prepared(*video, expected_video.source_sha256,
                                                  result.variant, expected_video.prepared_identity);
            result.video_sha256 = result.integer_video->source_sha256;
        }
    } else {
        if (expected_image.prepared_identity || expected_video.prepared_identity) {
            throw std::invalid_argument("FP16 models cannot use prepared INT16 identities");
        }
        result.image_sha256 = sha256_file(image);
        if (expected_image.source_sha256 && result.image_sha256 != *expected_image.source_sha256) {
            throw std::runtime_error("UF FP16 image checkpoint differs from archive identity");
        }
        if (video) {
            result.video_sha256 = sha256_file(*video);
            if (expected_video.source_sha256 && result.video_sha256 != expected_video.source_sha256) {
                throw std::runtime_error("UF FP16 video checkpoint differs from archive identity");
            }
        }
    }
    return result;
}

struct Codec::Impl {
    std::unique_ptr<IntegerCodec> integer;
    std::unique_ptr<Fp16Codec> fp16;
};

Codec::Codec(CodecModels models, torch::Device device, double skip_threshold)
    : impl_(std::make_unique<Impl>()) {
    // CLI validation belongs to its parser. Direct codec callers retain each
    // Python backend's threshold conversion/range policy.
    // Prepared containers are local to construction, as in Python. Each
    // integer model already retains its device tensors and copied CDF values.
    const auto& loaded = models;
    if (loaded.runtime == "int16") {
        if (!loaded.integer_image || (loaded.video_path && !loaded.integer_video)) {
            throw std::invalid_argument("UF INT16 codec requires prepared models");
        }
        impl_->integer = std::make_unique<IntegerCodec>(*loaded.integer_image,
            loaded.integer_video ? &*loaded.integer_video : nullptr, loaded.variant,
            std::move(device), skip_threshold);
    } else if (loaded.runtime == "fp16") {
        impl_->fp16 = std::make_unique<Fp16Codec>(loaded.image_path, loaded.video_path,
            loaded.variant, std::move(device), skip_threshold,
            loaded.image_sha256, loaded.video_sha256);
    } else {
        throw std::invalid_argument("Unknown UF codec runtime");
    }
}

Codec::~Codec() = default;

CodecStatistics Codec::encode(FrameSource source, const std::filesystem::path& output,
                               std::uint32_t width, std::uint32_t height,
                               int qp_i, int qp_p, int reset_interval, int intra_period,
                               std::optional<std::uint64_t> max_frames, EncodeProgress progress) {
    if (impl_->integer) return impl_->integer->encode(std::move(source), output,
        width, height, qp_i, qp_p, reset_interval, intra_period, max_frames, std::move(progress));
    return impl_->fp16->encode(std::move(source), output, width, height,
        qp_i, qp_p, reset_interval, intra_period, max_frames, std::move(progress));
}

CodecStatistics Codec::decode(const std::filesystem::path& input,
                               std::uint64_t total_frames, std::uint64_t total_packets,
                               FrameSink sink, std::optional<Sps> dimensions,
                               DecodeProgress progress) {
    if (impl_->integer) return impl_->integer->decode(input, total_frames, total_packets,
        std::move(sink), dimensions, std::move(progress));
    return impl_->fp16->decode(input, total_frames, total_packets,
        std::move(sink), dimensions, std::move(progress));
}

CodecStatistics Codec::decode_stream(std::istream& input,
                                      std::uint64_t total_frames, std::uint64_t total_packets,
                                      FrameSink sink, std::optional<Sps> dimensions,
                                      DecodeProgress progress) {
    if (impl_->integer) return impl_->integer->decode_stream(input, total_frames, total_packets,
        std::move(sink), dimensions, std::move(progress));
    return impl_->fp16->decode_stream(input, total_frames, total_packets,
        std::move(sink), dimensions, std::move(progress));
}

std::vector<std::vector<std::uint8_t>> Codec::decode_packet(const Packet& packet, const Sps& sps,
    std::optional<std::uint64_t> valid_frames) {
    if (impl_->integer) return impl_->integer->decode_packet(packet, sps, valid_frames);
    return impl_->fp16->decode_packet(packet, sps, valid_frames);
}

}  // namespace uf
