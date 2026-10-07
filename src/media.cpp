#include "uf/media.hpp"
#include "uf/archive.hpp"
#include "uf/process.hpp"
#include "uf/jetson.hpp"
#include "uf/utf8.hpp"
#include "uf/python_numeric.hpp"
#include "uf/python_integer.hpp"
#include "uf/python_json.hpp"
#include "uf/json_compat.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <csignal>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <poll.h>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

namespace uf {
namespace {

struct FrameSizes {
    std::size_t pixels{};
    std::size_t planar{};
    std::size_t expanded{};
};

FrameSizes frame_sizes(std::uint32_t width, std::uint32_t height, bool expand) {
    constexpr auto maximum = std::numeric_limits<std::size_t>::max();
    const auto w = static_cast<std::size_t>(width);
    const auto h = static_cast<std::size_t>(height);
    if (!h || w > maximum / h) throw std::overflow_error("Media pixel count exceeds native range");
    FrameSizes sizes;
    sizes.pixels = w * h;
    if (sizes.pixels > maximum - sizes.pixels / 2)
        throw std::overflow_error("Media planar byte count exceeds native range");
    sizes.planar = sizes.pixels + sizes.pixels / 2;
    if (expand) {
        if (sizes.pixels > maximum / 3)
            throw std::overflow_error("Media expanded byte count exceeds native range");
        sizes.expanded = sizes.pixels * 3;
    }
    const auto buffer_limit = std::vector<std::uint8_t>{}.max_size();
    if (sizes.planar > buffer_limit || sizes.expanded > buffer_limit)
        throw std::length_error("Media frame exceeds native buffer range");
    return sizes;
}

std::uint32_t probe_dimension(const nlohmann::json& value) {
    const auto integer = python_json_int(value);
    const auto& digits = integer.decimal();
    // Python int conversion precedes this native interface's UInt32 bound.
    if (digits.front() == '-' || digits.size() > 10 ||
        (digits.size() == 10 && digits > "4294967295"))
        throw std::overflow_error("Media dimension exceeds native UInt32 range");
    std::uint32_t result = 0;
    for (const auto digit : digits) result = result * 10 + static_cast<unsigned>(digit - '0');
    return result;
}

double probe_float(const nlohmann::json& value) {
    return python_json_as_double(value);
}

double probe_fraction(const nlohmann::json& value) {
    if (value.is_string()) return python_fraction_double(value.get_ref<const std::string&>());
    // Fraction(float) retains the exact finite binary ratio; converting that
    // ratio back to float returns the input. Fraction normalizes signed zero.
    const auto result = probe_float(value);
    if (std::isinf(result)) throw std::overflow_error("cannot convert Infinity to integer ratio");
    if (std::isnan(result)) throw std::invalid_argument("cannot convert NaN to integer ratio");
    return result == 0.0 ? 0.0 : result;
}

std::string python_argument_text(const nlohmann::json& value) {
    return python_filesystem_encode(
        value.is_string() ? value.get<std::string>() : python_json_repr(value));
}

void run_audio_command(const std::vector<std::string>& command,
                       std::optional<int> input_fd = std::nullopt) {
    // subprocess.run(check=True) inherits stdout/stderr and stdin unless an
    // input pipe is supplied. No output pipe or diagnostic scratch file.
    std::vector<std::pair<int, int>> descriptors;
    if (input_fd) descriptors.emplace_back(*input_fd, STDIN_FILENO);
    ChildProcess process(command, descriptors);
    const auto status = process.wait();
    if (status != 0) {
        const auto returncode = process.returncode().value_or(status);
        if (returncode < 0) {
            throw std::runtime_error("FFmpeg audio command terminated by signal " +
                                     std::to_string(-returncode));
        }
        throw std::runtime_error("FFmpeg audio command returned non-zero exit status " +
                                 std::to_string(returncode));
    }
}

std::vector<std::uint8_t> expand_420(const std::vector<std::uint8_t>& planar,
                                  std::uint32_t width, std::uint32_t height,
                                  const FrameSizes& sizes) {
    const auto pixels = sizes.pixels;
    std::vector<std::uint8_t> full(sizes.expanded);
    std::copy_n(planar.data(), pixels, full.data());
    for (int component = 0; component < 2; ++component) {
        const auto* source = planar.data() + pixels + component * pixels / 4;
        auto* destination = full.data() + pixels * (component + 1);
        for (std::uint32_t y = 0; y < height; ++y) {
            for (std::uint32_t x = 0; x < width; ++x) {
                destination[static_cast<std::size_t>(y) * width + x] =
                    source[static_cast<std::size_t>(y / 2) * width / 2 + x / 2];
            }
        }
    }
    return full;
}

}  // namespace

std::uint32_t python_media_dimension(const nlohmann::json& value) {
    return probe_dimension(value);
}

ChildPipe::ChildPipe(const std::vector<std::string>& command,
                     std::optional<int> input_fd, bool own_process_group,
                     std::optional<int> error_fd, bool download_teardown,
                     bool start_new_session, bool capture_stderr)
    : download_teardown_(download_teardown), stderr_pipe_(capture_stderr) {
    if (capture_stderr && error_fd) {
        throw std::invalid_argument("Cannot capture an inherited diagnostic file as a pipe");
    }
    auto pipes = process_pipe();
    std::array<FileDescriptor, 2> error_pipe{};
    FileDescriptor errors;
    if (capture_stderr) error_pipe = process_pipe();
    else {
        errors = error_fd ? FileDescriptor(fcntl(*error_fd, F_DUPFD_CLOEXEC, 3))
            : python_temporary_anonymous_file();
        if (errors.get() < 0) throw std::runtime_error("Cannot duplicate media diagnostics");
    }
    FileDescriptor null_input(input_fd ? -1 : open("/dev/null", O_RDONLY | O_CLOEXEC));
    if (!input_fd && null_input.get() < 0) throw std::runtime_error("Cannot open null media input");
    process_ = std::make_unique<ChildProcess>(command, std::vector<std::pair<int, int>>{
        {input_fd.value_or(null_input.get()), STDIN_FILENO},
        {pipes[1].get(), STDOUT_FILENO},
        {capture_stderr ? error_pipe[1].get() : errors.get(), STDERR_FILENO}},
        own_process_group, start_new_session);
    stdout_fd_ = pipes[0].release();
    stderr_fd_ = capture_stderr ? error_pipe[0].release() : errors.release();
}

ChildPipe::~ChildPipe() {
    if (download_teardown_ && !cleanup_attempted_) {
        try { close_download_checked(false); }
        catch (...) { if (process_) process_->signal(SIGKILL); }
    }
    process_.reset();
    if (stdout_fd_ >= 0) close(stdout_fd_);
    if (stderr_fd_ >= 0) close(stderr_fd_);
}

void ChildPipe::cancel() noexcept { if (process_) process_->signal(SIGKILL); }

void ChildPipe::stop_for_cleanup() {
    if (process_ && !process_->poll_for_cleanup()) {
        process_->signal(SIGKILL);
        process_->wait_for_cleanup();
    }
}

void ChildPipe::close_checked() {
    if (cleanup_attempted_) return;
    cleanup_attempted_ = true;
    stop_for_cleanup();
    FileDescriptor output(std::exchange(stdout_fd_, -1));
    output.close_checked("Cannot close media output");
    FileDescriptor errors(std::exchange(stderr_fd_, -1));
    errors.close_checked("Cannot close media diagnostics");
}

void ChildPipe::check_cancelled(const std::function<bool()>& cancelled) {
    if (ProcessInterrupt::requested() || (cancelled && cancelled())) {
        if (!download_teardown_) cancel();
        throw std::runtime_error("Media process cancelled");
    }
}

std::size_t ChildPipe::read(std::uint8_t* destination, std::size_t size,
                           const std::function<bool()>& cancelled) {
    while (true) {
        check_cancelled(cancelled);
        // An interrupt can arrive on another thread while this pipe is idle.
        // Poll even without a caller-supplied cancellation callback.
        pollfd descriptor{stdout_fd_, POLLIN, 0};
        const int status = ::poll(&descriptor, 1, 100);
        if (status < 0 && errno == EINTR) continue;
        if (status < 0 || (status > 0 && (descriptor.revents & (POLLERR | POLLNVAL)))) {
            throw std::runtime_error("Cannot poll media output");
        }
        if (!status) continue;
        check_cancelled(cancelled);
        const auto count = ::read(stdout_fd_, destination, size);
        if (count >= 0) return static_cast<std::size_t>(count);
        if (errno != EINTR) throw std::runtime_error("Media pipe read failed");
    }
}

std::string ChildPipe::read_all(const std::function<bool()>& cancelled) {
    std::string result;
    std::array<std::uint8_t, 8192> buffer{};
    while (const auto count = read(buffer.data(), buffer.size(), cancelled)) {
        if (result.size() + count > 256 * 1024 * 1024) {
            throw std::runtime_error("Media process output exceeds 256 MiB");
        }
        result.append(reinterpret_cast<const char*>(buffer.data()), count);
    }
    finish(cancelled);
    return result;
}

CapturedProcessOutput ChildPipe::communicate(const std::function<bool()>& cancelled) {
    CapturedProcessOutput result;
    std::array<std::uint8_t, 8192> buffer{};
    if (stderr_pipe_) {
        std::array<pollfd, 2> descriptors{{{stdout_fd_, POLLIN, 0}, {stderr_fd_, POLLIN, 0}}};
        std::array<std::string*, 2> streams{{&result.output, &result.errors}};
        std::size_t open = descriptors.size();
        while (open) {
            check_cancelled(cancelled);
            const auto ready = ::poll(descriptors.data(), descriptors.size(), 100);
            if (ready < 0 && errno == EINTR) continue;
            if (ready < 0) throw std::runtime_error("Cannot poll captured process output");
            if (!ready) continue;
            for (std::size_t i = 0; i < descriptors.size(); ++i) {
                auto& descriptor = descriptors[i];
                if (descriptor.fd < 0) continue;
                if (descriptor.revents & (POLLERR | POLLNVAL)) {
                    throw std::runtime_error("Cannot poll captured process stream");
                }
                if (!(descriptor.revents & (POLLIN | POLLHUP))) continue;
                check_cancelled(cancelled);
                const auto count = ::read(descriptor.fd, buffer.data(), buffer.size());
                if (count < 0 && errno == EINTR) continue;
                if (count < 0) throw std::runtime_error("Cannot read captured process stream");
                if (!count) { descriptor.fd = -1; --open; }
                else streams[i]->append(reinterpret_cast<const char*>(buffer.data()),
                                        static_cast<std::size_t>(count));
            }
        }
        result.returncode = wait_result(cancelled);
        return result;
    }
    while (const auto count = read(buffer.data(), buffer.size(), cancelled)) {
        result.output.append(reinterpret_cast<const char*>(buffer.data()), count);
    }
    result.returncode = wait_result(cancelled);
    // Ordinary media callers retain their anonymous diagnostic file.
    off_t offset = 0;
    for (;;) {
        check_cancelled(cancelled);
        const auto count = pread(stderr_fd_, buffer.data(), buffer.size(), offset);
        if (count < 0 && errno == EINTR) continue;
        if (count < 0) throw std::runtime_error("Cannot read captured process diagnostics");
        if (!count) break;
        if (count > std::numeric_limits<off_t>::max() - offset) {
            throw std::length_error("Captured process diagnostic offset overflow");
        }
        result.errors.append(reinterpret_cast<const char*>(buffer.data()),
                             static_cast<std::size_t>(count));
        offset += count;
    }
    return result;
}

void ChildPipe::drain(const std::function<bool()>& cancelled) {
    std::array<std::uint8_t, 8192> buffer{};
    while (read(buffer.data(), buffer.size(), cancelled)) {}
    finish(cancelled);
}

void ChildPipe::drain_download() {
    if (cleanup_attempted_) return;
    try {
        // Source uses one-MiB read requests while discarding the remainder.
        std::vector<std::uint8_t> buffer(1024u * 1024u);
        while (read(buffer.data(), buffer.size())) {}
    } catch (...) {
        close_download_checked(false);
        throw;
    }
    close_download_checked(true);
}

void ChildPipe::close_download_checked(bool normal) {
    if (cleanup_attempted_) return;
    cleanup_attempted_ = true;
    const auto close_errors = [&] {
        FileDescriptor errors(std::exchange(stderr_fd_, -1));
        errors.close_checked("Cannot close yt-dlp diagnostics");
    };
    try {
        FileDescriptor output(std::exchange(stdout_fd_, -1));
        output.close_checked("Cannot close yt-dlp stream");
        const auto wait_for = [&](std::chrono::seconds timeout) -> std::optional<int> {
            const auto deadline = std::chrono::steady_clock::now() + timeout;
            for (;;) {
                // Cleanup must proceed after a latched body interrupt.
                if (normal) check_cancelled({});
                if (const auto status = process_->poll_for_cleanup()) return status;
                if (std::chrono::steady_clock::now() >= deadline) return std::nullopt;
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
        };
        auto status = wait_for(std::chrono::seconds(normal ? 15 : 1));
        if (!status) {
            process_->signal(SIGTERM);
            status = wait_for(std::chrono::seconds(5));
            if (!status) {
                process_->signal(SIGKILL);
                status = normal ? process_->wait() : process_->wait_for_cleanup();
            }
        }
        finished_ = true;
        if (normal && *status != 0) {
            throw std::runtime_error("yt-dlp stream failed: " + error_text(true));
        }
    } catch (...) {
        // The TemporaryFile context closes even when the inner finally fails.
        // Its close error can replace the stream/wait/diagnostic failure.
        close_errors();
        throw;
    }
    close_errors();
}

std::string ChildPipe::error_text(bool download_tail) {
    if (download_tail) return utf8_file_tail(stderr_fd_, 2500);
    std::array<char, 8192> buffer{};
    std::size_t size = 0;
    while (size < buffer.size()) {
        const auto count = pread(stderr_fd_, buffer.data() + size, buffer.size() - size,
                                 static_cast<off_t>(size));
        if (count < 0 && errno == EINTR) continue;
        if (count < 0) throw std::runtime_error("Cannot read media process diagnostics");
        if (!count) break;
        size += static_cast<std::size_t>(count);
    }
    return utf8_replace(std::string_view(buffer.data(), size));
}

int ChildPipe::wait_result(const std::function<bool()>& cancelled) {
    if (finished_) return process_->returncode().value();
    if (cancelled) {
        while (!process_->poll()) {
            check_cancelled(cancelled);
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        check_cancelled(cancelled);
    }
    const int status = process_->wait();
    finished_ = true;
    return process_->returncode().value_or(status);
}

void ChildPipe::finish(const std::function<bool()>& cancelled) {
    if (finished_) return;
    const int status = wait_result(cancelled);
    if (status != 0) {
        throw std::runtime_error("Media process failed: " + error_text());
    }
}

MediaInfo probe_media(const std::filesystem::path& source) {
    const auto encoding = python_process_text_encoding();
    ChildPipe process({"ffprobe", "-v", "error", "-show_streams", "-show_format",
                       "-of", "json", source.string()}, std::nullopt, false,
                      std::nullopt, false, false, true);
    auto captured = process.communicate();
    captured.output = decode_process_text(std::move(captured.output), encoding);
    captured.errors = decode_process_text(std::move(captured.errors), encoding);
    if (captured.returncode != 0)
        throw std::runtime_error("ffprobe failed with exit status " + std::to_string(captured.returncode));
    const auto data = parse_python_json(captured.output);
    const auto& streams = data.at("streams");
    if (!streams.is_array()) {
        if (streams.is_object() || streams.is_string()) {
            if (!json_truthy(streams)) throw std::invalid_argument("No video stream: " + source.string());
            throw PythonJsonTypeError("string indices must be integers");
        }
        throw PythonJsonTypeError("'" + std::string(json_compat_detail::python_type_name(streams)) +
                                  "' object is not iterable");
    }
    MediaInfo info;
    const nlohmann::json* video = nullptr;
    for (const auto& item : streams) {
        if (item.at("codec_type") == "video") { video = &item; break; }
    }
    if (!video) throw std::invalid_argument("No video stream: " + source.string());
    auto rate = python_json_dict_get(*video, "avg_frame_rate");
    if (!json_truthy(rate)) rate = python_json_dict_get(*video, "r_frame_rate");
    if (!json_truthy(rate) || rate == "0/0")
        rate = python_json_dict_get(*video, "r_frame_rate", nlohmann::json("30/1"));
    info.width = probe_dimension(video->at("width"));
    info.height = probe_dimension(video->at("height"));
    info.fps = probe_fraction(rate);
    static const nlohmann::json empty_format = nlohmann::json::object();
    static const nlohmann::json zero(0);
    const auto& format = python_json_dict_get(data, "format", empty_format);
    info.duration = probe_float(python_json_dict_get(format, "duration", zero));
    for (const auto& item : streams) {
        if (item.at("codec_type") == "audio") { info.audio = true; break; }
    }
    return info;
}

std::pair<std::uint32_t, std::uint32_t> scaled_dimensions(
    std::uint32_t width, std::uint32_t height,
    std::optional<std::uint32_t> short_edge) {
    if (!short_edge) {
        const auto even = [](std::uint32_t value) {
            if (value == std::numeric_limits<std::uint32_t>::max()) {
                throw std::overflow_error("Media dimension exceeds native UInt32 range");
            }
            return value + value % 2;
        };
        return {even(width), even(height)};
    }
    if (!width || !height) throw std::invalid_argument("Cannot scale zero source dimensions");
    const auto scale = static_cast<double>(*short_edge) / std::min(width, height);
    const auto even = [scale](std::uint32_t value) {
        const auto half = value * scale / 2.0;
        // Python round(x) is ties-to-even regardless of the active rounding
        // mode. nearbyint follows that mode instead.
        const auto lower = std::floor(half);
        const auto fraction = half - lower;
        const auto rounded = lower + (fraction > 0.5 ||
            (fraction == 0.5 && std::fmod(lower, 2.0) != 0) ? 1.0 : 0.0);
        if (rounded < 1.0) return std::uint32_t{2};
        if (rounded > std::numeric_limits<std::uint32_t>::max() / 2) {
            throw std::overflow_error("Media dimension exceeds native UInt32 range");
        }
        return static_cast<std::uint32_t>(rounded) * 2;
    };
    return {even(width), even(height)};
}

struct FfmpegFrameReader::Impl {
    std::uint32_t width, height;
    FrameSizes buffer_sizes;
    std::unique_ptr<ChildPipe> software;
    std::unique_ptr<JetsonFramePipe> jetson;
    bool ended{}, closed{};

    Impl(const std::filesystem::path& source, std::uint32_t w, std::uint32_t h,
          const MediaInfo& original, std::optional<double> fps, int threads,
          std::optional<int> input_fd, const std::string& hwaccel) : width(w), height(h) {
        if (threads < 1 || width < 2 || height < 2 || width % 2 || height % 2) {
            throw std::invalid_argument("Invalid FFmpeg frame reader settings");
        }
        if (hwaccel != "none" && hwaccel != "auto" && hwaccel != "jetson") {
            throw std::invalid_argument("Unsupported FFmpeg decoder");
        }
        buffer_sizes = frame_sizes(width, height, true);
        std::string filters;
        if (width != original.width || height != original.height) {
            filters += "scale=" + std::to_string(width) + ":" + std::to_string(height) + ":flags=bicubic,";
        }
        if (fps) filters += "fps=" + python_float_repr(*fps) + ",";
        filters += "format=yuv420p,setsar=1/1";
        std::vector<std::string> command{"ffmpeg", "-v", "error", "-nostdin", "-threads",
            std::to_string(threads), "-noautorotate", "-hwaccel", hwaccel == "jetson" ? "none" : hwaccel, "-i",
            input_fd ? "pipe:0" : source.string(),
            "-map", "0:v:0", "-an", "-sn", "-dn", "-vf", filters, "-pix_fmt", "yuv420p",
            "-vsync", "0", "-f", "rawvideo", "pipe:1"};
        if (hwaccel == "jetson") {
            if (input_fd) throw DecoderError("Jetson decoding requires a local file and GStreamer");
            jetson = std::make_unique<JetsonFramePipe>(source, std::move(command));
        } else software = std::make_unique<ChildPipe>(command, input_fd);
    }

    void cancel() noexcept {
        if (software) software->cancel();
        if (jetson) jetson->cancel();
    }

    void close_checked() {
        if (closed) return;
        closed = true;
        if (software) software->close_checked();
        if (jetson) jetson->close_checked();
    }

    void stop_for_cleanup() {
        if (software) software->stop_for_cleanup();
        if (jetson) jetson->stop_for_cleanup();
    }

    ~Impl() {
        try { close_checked(); } catch (...) {}
        cancel();
    }
};

FfmpegFrameReader::FfmpegFrameReader(const std::filesystem::path& source,
                                     std::uint32_t width, std::uint32_t height,
                                     const MediaInfo& original,
                                     std::optional<double> fps, int threads,
                                     std::optional<int> input_fd, std::string hwaccel)
    : impl_(std::make_unique<Impl>(source, width, height, original, fps, threads, input_fd, hwaccel)) {}
FfmpegFrameReader::~FfmpegFrameReader() = default;

std::optional<std::vector<std::uint8_t>> FfmpegFrameReader::read() {
    if (impl_->closed) throw std::invalid_argument("Frame reader is closed");
    if (impl_->ended) return std::nullopt;
    const auto size = impl_->buffer_sizes.planar;
    std::vector<std::uint8_t> planar(size);
    std::size_t received = 0;
    while (received < size) {
        const auto count = impl_->jetson ? impl_->jetson->read(planar.data() + received, size - received)
            : impl_->software->read(planar.data() + received, size - received);
        if (!count) {
            if (received) throw std::invalid_argument("FFmpeg produced a truncated frame");
            if (impl_->jetson) {
                impl_->jetson->finish();
            } else {
                const auto status = impl_->software->wait_result();
                impl_->ended = true;
                if (status != 0) throw std::runtime_error(impl_->software->error_text());
            }
            impl_->ended = true;
            return std::nullopt;
        }
        received += count;
    }
    return expand_420(planar, impl_->width, impl_->height, impl_->buffer_sizes);
}

void FfmpegFrameReader::cancel() noexcept { impl_->cancel(); }
void FfmpegFrameReader::stop_for_cleanup() { impl_->stop_for_cleanup(); }
void FfmpegFrameReader::close_checked() { impl_->close_checked(); }

FfmpegVideoWriter::FfmpegVideoWriter(
    const std::filesystem::path& output, std::uint32_t width,
    std::uint32_t height, double fps,
    std::optional<std::filesystem::path> opus_audio, Format format)
    : FfmpegVideoWriter(output, width, height, nlohmann::json(fps),
                        std::move(opus_audio), format) {}

FfmpegVideoWriter::FfmpegVideoWriter(
    const std::filesystem::path& output, std::uint32_t width,
    std::uint32_t height, const nlohmann::json& fps,
    std::optional<std::filesystem::path> opus_audio, Format format)
    : FfmpegVideoWriter(output, Sps{0, width, height}, fps,
                        std::move(opus_audio), format,
                        std::to_string(width) + "x" + std::to_string(height)) {}

FfmpegVideoWriter::FfmpegVideoWriter(
    const std::filesystem::path& output, const nlohmann::json& video_metadata,
    std::optional<std::filesystem::path> opus_audio, Format format)
    : FfmpegVideoWriter(output, archive_video_dimensions(video_metadata),
                        video_metadata.at("fps"), std::move(opus_audio), format,
                        python_argument_text(video_metadata.at("width")) + "x" +
                        python_argument_text(video_metadata.at("height"))) {}

FfmpegVideoWriter::FfmpegVideoWriter(
    const std::filesystem::path& output, const Sps& dimensions,
    const nlohmann::json& fps, std::optional<std::filesystem::path> opus_audio,
    Format format, const std::string& video_size) {
    const auto width = dimensions.width;
    const auto height = dimensions.height;
    if (width < 2 || height < 2 || width % 2 || height % 2) {
        throw std::invalid_argument("Invalid reconstructed video settings");
    }
    frame_size_ = frame_sizes(width, height, false).planar;
    auto pipes = process_pipe();
    auto errors = python_temporary_anonymous_file();
    std::vector<std::string> command = {"ffmpeg", "-v", "error", "-nostdin", "-y",
        "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s",
        video_size, "-r", python_argument_text(fps),
        "-i", "pipe:0"};
    if (opus_audio && format != Format::raw_yuv) {
        command.insert(command.end(), {"-i", opus_audio->string(), "-map", "0:v:0",
                                       "-map", "1:a:0", "-c:a", "copy"});
    }
    if (format == Format::raw_yuv) {
        command.insert(command.end(), {"-c:v", "rawvideo", "-pix_fmt", "yuv420p",
                                       "-f", "rawvideo", output.string()});
    } else {
        command.insert(command.end(), {"-c:v", "ffv1", "-f", "matroska", output.string()});
    }
    process_ = std::make_unique<ChildProcess>(command, std::vector<std::pair<int, int>>{
        {pipes[0].get(), STDIN_FILENO}, {errors.get(), STDERR_FILENO}});
    stdin_fd_ = pipes[1].release();
    stderr_fd_ = errors.release();
}

FfmpegVideoWriter::~FfmpegVideoWriter() {
    try { close_checked(); } catch (...) {}
    // Fallback releases any resources left after a checked cleanup failure.
    process_.reset();
    if (stdin_fd_ >= 0) close(stdin_fd_);
    if (stderr_fd_ >= 0) close(stderr_fd_);
}

std::string FfmpegVideoWriter::error_text() {
    for (;;) {
        ProcessInterrupt::check();
        if (lseek(stderr_fd_, 0, SEEK_SET) >= 0) break;
        if (errno != EINTR)
            throw std::runtime_error("Cannot seek reconstructed-video diagnostics");
    }
    std::array<char, 8192> buffer{};
    std::size_t size = 0;
    while (size < buffer.size()) {
        ProcessInterrupt::check();
        const auto count = ::read(stderr_fd_, buffer.data() + size, buffer.size() - size);
        if (count < 0 && errno == EINTR) continue;
        if (count < 0) throw std::runtime_error("Cannot read reconstructed-video diagnostics");
        if (!count) break;
        size += static_cast<std::size_t>(count);
    }
    return utf8_replace(std::string_view(buffer.data(), size));
}

void FfmpegVideoWriter::write(const std::vector<std::uint8_t>& frame) {
    if (frame.size() != frame_size_ || stdin_fd_ < 0) {
        throw std::invalid_argument("Invalid FFmpeg output frame");
    }
    std::size_t offset = 0;
    while (offset < frame.size()) {
        ProcessInterrupt::check();
        const auto count = ::write(stdin_fd_, frame.data() + offset, frame.size() - offset);
        if (count > 0) offset += count;
        else if (count < 0 && errno == EINTR) continue;
        else throw std::runtime_error("FFmpeg reconstructed-video pipe failed");
    }
}

void FfmpegVideoWriter::finish() {
    if (finished_) return;
    FileDescriptor input(std::exchange(stdin_fd_, -1));
    input.close_checked("Cannot close reconstructed-video input");
    const int status = process_->wait();
    finished_ = true;
    if (status != 0) {
        throw std::runtime_error(error_text());
    }
}

void FfmpegVideoWriter::close_checked() {
    if (cleanup_attempted_) return;
    cleanup_attempted_ = true;
    if (process_ && !process_->poll_for_cleanup()) {
        process_->signal(SIGKILL);
        process_->wait_for_cleanup();
    }
    finished_ = true;
    FileDescriptor input(std::exchange(stdin_fd_, -1));
    input.close_checked("Cannot close reconstructed-video input");
    FileDescriptor diagnostics(std::exchange(stderr_fd_, -1));
    diagnostics.close_checked("Cannot close reconstructed-video diagnostics");
}

void encode_opus(const std::filesystem::path& source,
                 const std::filesystem::path& output, double duration,
                 std::string channels, std::string bitrate, const nlohmann::json& frame_ms,
                 int complexity, std::string vbr, std::optional<int> input_fd) {
    run_audio_command({"ffmpeg", "-v", "error", "-nostdin", "-i",
                       input_fd ? "pipe:0" : source.string(),
                       "-map", "0:a:0", "-vn", "-c:a", "libopus", "-ac",
                       channels == "mono" ? "1" : "2", "-b:a", bitrate,
                       "-frame_duration", python_argument_text(frame_ms), "-compression_level",
                       std::to_string(complexity), "-vbr", vbr, "-t", python_float_repr(duration),
                       "-f", "opus", output.string()}, input_fd);
}

void verify_opus(const std::filesystem::path& path) {
    run_audio_command({"ffmpeg", "-v", "error", "-nostdin", "-i", path.string(),
                       "-map", "0:a:0", "-f", "null", "-"});
}

}  // namespace uf
