#pragma once

#include "uf/fp16_model.hpp"
#include "uf/integer_codec.hpp"

namespace uf {

class Fp16Codec {
public:
    Fp16Codec(const std::filesystem::path& image,
              const std::optional<std::filesystem::path>& video,
              std::string variant, torch::Device device,
              double skip_threshold = 0,
              const std::optional<std::string>& image_sha256 = std::nullopt,
              const std::optional<std::string>& video_sha256 = std::nullopt);
    ~Fp16Codec();
    Fp16Codec(const Fp16Codec&) = delete;
    Fp16Codec& operator=(const Fp16Codec&) = delete;

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
    // Finite checks cover the full chunk before raw conversion of its prefix.
    std::vector<std::vector<std::uint8_t>> decode_packet(const Packet& packet, const Sps& sps,
        std::optional<std::uint64_t> valid_frames = std::nullopt);
    // Python crop endpoints, shared YUV420 pooling, round/truncate byte casts
    // and squeeze(0); tensor operators own rank/channel/batch/crop behavior.
    static std::vector<std::uint8_t> raw_frame(const torch::Tensor& frame,
                                              std::int64_t width, std::int64_t height);
private:
    CodecStatistics decode_stream_impl(std::istream& input,
        std::uint64_t total_frames, std::uint64_t total_packets,
        FrameSink sink, std::optional<Sps> dimensions, DecodeProgress progress,
        const std::function<void()>& close_input);
    struct Impl;
    std::unique_ptr<Impl> impl_;
    torch::Tensor frame_tensor(const std::vector<std::vector<std::uint8_t>>& frames,
                                std::uint32_t width, std::uint32_t height) const;
};

}  // namespace uf
