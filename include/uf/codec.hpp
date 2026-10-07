#pragma once

#include "uf/integer_codec.hpp"

#include <filesystem>
#include <memory>
#include <optional>
#include <string>

namespace uf {

struct ExpectedModel {
    std::optional<std::string> source_sha256;
    std::optional<std::string> prepared_identity;
};

// Loading result that callers may reuse. Integer codecs retain device tensor
// handles and entropy tables without borrowing these prepared containers;
// FP16 verifies the source hashes again when loading the CUDA proxies.
struct CodecModels {
    std::string runtime;
    std::string variant;
    std::filesystem::path image_path;
    std::optional<std::filesystem::path> video_path;
    std::string image_sha256;
    std::optional<std::string> video_sha256;
    std::optional<PreparedModel> integer_image;
    std::optional<PreparedModel> integer_video;

    static CodecModels load(std::string runtime,
                             const std::filesystem::path& image,
                             const std::optional<std::filesystem::path>& video,
                             std::string variant,
                             const ExpectedModel& expected_image = {},
                             const ExpectedModel& expected_video = {});
};

class Codec {
public:
    // Construction retains model tensor handles/entropy values, not prepared
    // containers. Transfer a one-use loading result with std::move to release
    // its CPU artifact state when construction returns.
    Codec(CodecModels models, torch::Device device, double skip_threshold = 0);
    ~Codec();
    Codec(const Codec&) = delete;
    Codec& operator=(const Codec&) = delete;

    CodecStatistics encode(FrameSource source, const std::filesystem::path& output,
                           std::uint32_t width, std::uint32_t height,
                           int qp_i, int qp_p, int reset_interval = 32,
                           int intra_period = -1,
                           std::optional<std::uint64_t> max_frames = std::nullopt,
                           EncodeProgress progress = {});
    CodecStatistics decode(const std::filesystem::path& input,
                           std::uint64_t total_frames, std::uint64_t total_packets,
                           FrameSink sink = {}, std::optional<Sps> dimensions = std::nullopt,
                           DecodeProgress progress = {});
    CodecStatistics decode_stream(std::istream& input,
                                  std::uint64_t total_frames, std::uint64_t total_packets,
                                  FrameSink sink = {}, std::optional<Sps> dimensions = std::nullopt,
                                  DecodeProgress progress = {});
    // Validate the complete decoded chunk, then convert only the requested
    // prefix to raw frames. The indexed viewer excludes padded tail frames.
    std::vector<std::vector<std::uint8_t>> decode_packet(const Packet& packet, const Sps& sps,
        std::optional<std::uint64_t> valid_frames = std::nullopt);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace uf
