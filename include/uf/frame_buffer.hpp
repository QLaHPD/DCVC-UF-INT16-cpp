#pragma once

#include "uf/media.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace uf {

class BufferedFrames {
public:
    using Read = std::function<std::optional<std::vector<std::uint8_t>>() >;
    using Cancel = std::function<void()>;
    // Callbacks must own any resources they use: a stopped worker can retain
    // them after close reports the reference's ten-second join timeout.
    BufferedFrames(Read source, Cancel cancel, std::size_t requested_frames,
                    std::size_t frame_bytes, bool adaptive = false);
    ~BufferedFrames();
    BufferedFrames(const BufferedFrames&) = delete;
    BufferedFrames& operator=(const BufferedFrames&) = delete;
    std::optional<std::vector<std::uint8_t>> read();
    void prime();
    // As in FrameReader.set_prefetch_frames, any threaded reader accepts a
    // positive replacement capacity. The local byte cap applies at creation.
    void set_capacity(std::size_t frames);
    void close_checked();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Own the media reader through worker callbacks, including a close timeout.
// Explicit close stops/joins prefetch before closing media descriptors.
class BufferedFrameReader {
public:
    BufferedFrameReader(const std::filesystem::path& source,
        std::uint32_t width, std::uint32_t height, const MediaInfo& original,
        std::optional<double> fps, int threads, std::optional<int> input_fd,
        std::string hwaccel, std::size_t requested_frames, bool adaptive = false);
    ~BufferedFrameReader();
    BufferedFrameReader(const BufferedFrameReader&) = delete;
    BufferedFrameReader& operator=(const BufferedFrameReader&) = delete;
    std::optional<std::vector<std::uint8_t>> read();
    void prime();
    void set_capacity(std::size_t frames);
    void close_checked();
private:
    std::shared_ptr<FfmpegFrameReader> reader_;
    std::unique_ptr<BufferedFrames> buffer_;
    bool closed_{};
};

}  // namespace uf
