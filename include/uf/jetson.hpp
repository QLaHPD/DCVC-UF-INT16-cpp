#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace uf {

// GStreamer NVDEC emits timestamped I420 Matroska through a private descriptor;
// FFmpeg keeps the software path's scaling, FPS and raw YUV420 conversion.
class JetsonFramePipe {
public:
    JetsonFramePipe(const std::filesystem::path& source,
                    std::vector<std::string> ffmpeg_command);
    ~JetsonFramePipe();
    JetsonFramePipe(const JetsonFramePipe&) = delete;
    JetsonFramePipe& operator=(const JetsonFramePipe&) = delete;
    std::size_t read(std::uint8_t* destination, std::size_t size);
    void finish();
    void cancel() noexcept;
    void stop_for_cleanup();
    void close_checked();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace uf
