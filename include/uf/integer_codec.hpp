#pragma once

#include "uf/integer_model.hpp"
#include "uf/bitstream.hpp"

#include <cstddef>
#include <filesystem>
#include <functional>
#include <fstream>
#include <istream>
#include <memory>
#include <optional>
#include <string>

namespace uf {

struct CodecStatistics {
    std::uint64_t frames{};
    std::uint64_t packets{};
    std::string decoded_yuv_sha256;
    double encode_seconds{};
    double decode_seconds{};
};

// One frame is planar, byte-valued YUV444 in C,H,W order, with UV nearest
// neighbor expanded from YUV420 as done by the Python FrameReader.
using FrameSource = std::function<std::optional<std::vector<std::uint8_t>>() >;
using FrameSink = std::function<void(const std::vector<std::uint8_t>&)>;
using EncodeProgress = std::function<void(std::uint64_t frames, double frames_per_second)>;
// Called after each coded packet's valid frames reach the digest/sink, before
// final frame/packet-count validation. Callback exceptions propagate.
using DecodeProgress = std::function<void(std::uint64_t frames, std::uint64_t total_frames)>;

class RawYuv420Reader {
public:
    // Validate both planar/expanded buffer lengths and the stream read count
    // before allocation. Input is expanded from planar YUV420 to CHW YUV444.
    RawYuv420Reader(const std::filesystem::path& path, std::uint32_t width,
                    std::uint32_t height);
    ~RawYuv420Reader();
    std::optional<std::vector<std::uint8_t>> read();
    // Close the owned input before publication. One explicit attempt; close
    // failure can replace an active read/encode error. Destruction is fallback.
    void close_checked();

private:
    std::ifstream input_;
    std::uint32_t width_, height_;
    std::size_t pixels_{}, planar_bytes_{}, expanded_bytes_{};
    bool close_attempted_{};
};

class IntegerCodec {
public:
    // Apply deterministic Torch settings before model construction. CUDA
    // operations use a pooled codec stream and restore the caller's stream
    // on return; CPU execution needs no CUDA stream/backend headers.
    // Prepared containers need only survive construction. Models retain their
    // device tensor handles and entropy tables; no full CPU tensor clone is made.
    IntegerCodec(const PreparedModel& image, const PreparedModel* video,
                 std::string variant, torch::Device device = torch::kCPU,
                 double skip_threshold = 0);
    ~IntegerCodec();
    IntegerCodec(IntegerCodec&&) noexcept;
    IntegerCodec& operator=(IntegerCodec&&) noexcept;
    IntegerCodec(const IntegerCodec&) = delete;
    IntegerCodec& operator=(const IntegerCodec&) = delete;

    void set_skip_threshold(double threshold);
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
    // Check finite pixels on valid frames before raw conversion/digest/sink;
    // the padded tail is excluded, following the shared Python archive decoder.
    CodecStatistics decode_stream(std::istream& input,
                                  std::uint64_t total_frames,
                                  std::uint64_t total_packets,
                                  FrameSink sink = {}, std::optional<Sps> dimensions = std::nullopt,
                                  DecodeProgress progress = {});
    // Stateful packet decoding for random access. Start at an I packet and
    // validate the stream/model identity before using this entry point.
    // Validate the full chunk; only its requested prefix becomes raw buffers.
    std::vector<std::vector<std::uint8_t>> decode_packet(const Packet& packet, const Sps& sps,
        std::optional<std::uint64_t> valid_frames = std::nullopt);

    // Python's first-batch slicing, Long conversion and four-slice chroma sum;
    // signed crop endpoints and shape/dtype/broadcast behavior belong to Torch.
    static std::vector<std::uint8_t> raw_frame(const torch::Tensor& frame,
                                                std::int64_t width,
                                                std::int64_t height);

private:
    CodecStatistics decode_stream_impl(std::istream& input,
        std::uint64_t total_frames, std::uint64_t total_packets,
        FrameSink sink, std::optional<Sps> dimensions, DecodeProgress progress,
        const std::function<void()>& close_input);
    struct RuntimeState;
    std::string variant_;
    torch::Device device_;
    std::unique_ptr<RuntimeState> runtime_;
    std::unique_ptr<IntegerModel> image_;
    std::unique_ptr<IntegerModel> video_;
    Preamble preamble_;
    int chunk_size_;

    torch::Tensor frame_tensor(const std::vector<std::vector<std::uint8_t>>& frames,
                                std::uint32_t width, std::uint32_t height) const;
};

}  // namespace uf
