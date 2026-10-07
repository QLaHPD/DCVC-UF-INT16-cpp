#pragma once

#include <cstdint>
#include <cstddef>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <nlohmann/json.hpp>

namespace uf {

class ChildProcess;
struct Sps;

class DecoderError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct MediaInfo {
    std::uint32_t width{};
    std::uint32_t height{};
    double fps{};
    double duration{};
    bool audio{};
};

struct CapturedProcessOutput {
    std::string output;
    std::string errors;
    int returncode{};
};

// Spawn with explicit argument vectors. No shell command interpolation.
// With capture_stderr, communicate() drains both pipes together.
class ChildPipe {
public:
    explicit ChildPipe(const std::vector<std::string>& command,
                       std::optional<int> input_fd = std::nullopt,
                       bool own_process_group = false,
                       std::optional<int> error_fd = std::nullopt,
                       bool download_teardown = false,
                       bool start_new_session = false,
                       bool capture_stderr = false);
    ~ChildPipe();
    ChildPipe(const ChildPipe&) = delete;
    ChildPipe& operator=(const ChildPipe&) = delete;

    std::size_t read(std::uint8_t* destination, std::size_t size,
                     const std::function<bool()>& cancelled = {});
    std::string read_all(const std::function<bool()>& cancelled = {});
    // Capture both raw streams once and reap, without a size cap or exit-code check.
    // Text decoding and failure diagnostics belong to the caller.
    CapturedProcessOutput communicate(const std::function<bool()>& cancelled = {});
    void drain(const std::function<bool()>& cancelled = {});
    // yt-dlp completion after EOF: wait 15 seconds, then TERM/5 seconds/KILL.
    void drain_download();
    // download_pipe finally: close output, wait (15 seconds normally, one
    // second exceptionally), then TERM/5 seconds/KILL, and close diagnostics.
    // Only normal completion rejects a nonzero child result.
    void close_download_checked(bool normal);
    // Wait without rejecting a nonzero result; the caller supplies diagnostics.
    // Return the subprocess convention (ordinary exit status or negative signal).
    int wait_result(const std::function<bool()>& cancelled = {});
    // First 8192 bytes of a seekable diagnostic file, with UTF-8 replacement;
    // does not read a capture_stderr pipe. Download
    // callers can request their existing 2500-character tail instead.
    std::string error_text(bool download_tail = false);
    void finish(const std::function<bool()>& cancelled = {});
    void cancel() noexcept;
    // Stop/reap while keeping pipe descriptors open for a prefetch worker.
    void stop_for_cleanup();
    // Input cleanup: kill/reap an unfinished child, then close output and
    // diagnostics explicitly. Repeated calls after an attempt are harmless.
    void close_checked();
    int output_fd() const { return stdout_fd_; }

private:
    int stdout_fd_{-1};
    int stderr_fd_{-1};
    std::unique_ptr<ChildProcess> process_;
    bool finished_{};
    bool download_teardown_{};
    bool stderr_pipe_{};
    bool cleanup_attempted_{};
    void check_cancelled(const std::function<bool()>& cancelled);
};

// First video, Python int/float/Fraction conversions and final audio scan.
// Dimensions must fit UInt32; text capture is strict and uncapped.
MediaInfo probe_media(const std::filesystem::path& source);
// Python int conversion followed by the native media interface's UInt32 bound.
// Used for local probing and selected remote dimensions; zero is retained.
std::uint32_t python_media_dimension(const nlohmann::json& value);
// Python's None/even-padding and scale/round-to-even formulas, within UInt32.
// A supplied zero short edge clamps to 2x2; CLI validation is separate.
std::pair<std::uint32_t, std::uint32_t> scaled_dimensions(
    std::uint32_t width, std::uint32_t height,
    std::optional<std::uint32_t> short_edge);

// Planar/expanded byte counts are validated and cached before starting a child.
class FfmpegFrameReader {
public:
    FfmpegFrameReader(const std::filesystem::path& source,
                      std::uint32_t width, std::uint32_t height,
                      const MediaInfo& original,
                      std::optional<double> fps, int threads = 1,
                      std::optional<int> input_fd = std::nullopt,
                      std::string hwaccel = "none");
    ~FfmpegFrameReader();
    FfmpegFrameReader(const FfmpegFrameReader&) = delete;
    FfmpegFrameReader& operator=(const FfmpegFrameReader&) = delete;
    std::optional<std::vector<std::uint8_t>> read();
    void cancel() noexcept;
    void stop_for_cleanup();
    // Call only after the prefetch worker has stopped using this reader.
    void close_checked();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Both export formats use FFmpeg. Raw YUV omits audio; failures decode the
// first 8192 diagnostic bytes with UTF-8 replacement, as run_decode does.
class FfmpegVideoWriter {
public:
    enum class Format { matroska, raw_yuv };

    FfmpegVideoWriter(const std::filesystem::path& output,
                      std::uint32_t width, std::uint32_t height, double fps,
                      std::optional<std::filesystem::path> opus_audio = std::nullopt,
                      Format format = Format::matroska);
    // Archive FPS goes to FFmpeg as Python str(value), preserving integer,
    // float and string spellings. FFmpeg reports invalid rate arguments.
    FfmpegVideoWriter(const std::filesystem::path& output,
                      std::uint32_t width, std::uint32_t height,
                      const nlohmann::json& fps,
                      std::optional<std::filesystem::path> opus_audio = std::nullopt,
                      Format format = Format::matroska);
    // Preserve recorded width/height/FPS spellings in FFmpeg arguments while
    // checking dimensions against the native frame-buffer interface.
    FfmpegVideoWriter(const std::filesystem::path& output,
                      const nlohmann::json& video_metadata,
                      std::optional<std::filesystem::path> opus_audio = std::nullopt,
                      Format format = Format::matroska);
    ~FfmpegVideoWriter();
    FfmpegVideoWriter(const FfmpegVideoWriter&) = delete;
    FfmpegVideoWriter& operator=(const FfmpegVideoWriter&) = delete;

    void write(const std::vector<std::uint8_t>& frame);
    void finish();
    // run_decode finally: kill/reap an unfinished child, then close input and
    // diagnostics with errors propagated. Safe to call again after an attempt.
    void close_checked();

private:
    FfmpegVideoWriter(const std::filesystem::path& output,
                      const Sps& dimensions, const nlohmann::json& fps,
                      std::optional<std::filesystem::path> opus_audio,
                      Format format, const std::string& video_size);
    int stdin_fd_{-1};
    int stderr_fd_{-1};
    std::unique_ptr<ChildProcess> process_;
    bool finished_{};
    bool cleanup_attempted_{};
    std::size_t frame_size_{};
    std::string error_text();
};

// Keep the Python helper's integer 20 default and explicit numeric JSON type.
// Audio commands inherit stdout/stderr and, without an input fd, stdin.
// CLI choices are checked by the parser; duration/complexity/VBR go to FFmpeg.
void encode_opus(const std::filesystem::path& source,
                 const std::filesystem::path& output,
                 double duration, std::string channels = "mono",
                 std::string bitrate = "6k", const nlohmann::json& frame_ms = 20,
                 int complexity = 10, std::string vbr = "on",
                 std::optional<int> input_fd = std::nullopt);
void verify_opus(const std::filesystem::path& path);

}  // namespace uf
