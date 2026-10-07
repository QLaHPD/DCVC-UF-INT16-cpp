#include "uf/jetson.hpp"
#include "uf/media.hpp"
#include "uf/process.hpp"
#include "uf/utf8.hpp"
#include "uf/python_numeric.hpp"
#include "uf/python_json.hpp"
#include "uf/python_path.hpp"
#include "uf/json_compat.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <csignal>
#include <iterator>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <utility>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace uf {
namespace {

bool unaligned(const nlohmann::json& value, int multiple) {
    if (python_json_extended_integer(value)) {
        const auto decimal = python_json_extended_integer_decimal(value);
        int remainder = 0;
        for (const auto digit : decimal)
            if (digit != '-') remainder = (remainder * 10 + digit - '0') % multiple;
        return remainder != 0;
    }
    if (value.is_boolean()) return static_cast<int>(value.get<bool>()) % multiple != 0;
    if (value.is_number_unsigned()) return value.get<std::uint64_t>() % multiple != 0;
    if (value.is_number_integer()) return value.get<std::int64_t>() % multiple != 0;
    if (value.is_number_float()) return std::fmod(value.get<double>(), multiple) != 0.0;
    throw PythonJsonTypeError("Hardware source dimension must be numeric");
}

std::string time_base(const nlohmann::json& value) {
    if (value.is_string()) return python_fraction_text(value.get_ref<const std::string&>());
    if (python_json_extended_integer(value)) {
        const auto decimal = python_json_extended_integer_decimal(value);
        check_python_integer_digits(decimal.size() - (decimal.front() == '-'));
        return std::string(decimal);
    }
    if (value.is_boolean()) return value.get<bool>() ? "1" : "0";
    if (value.is_number_unsigned()) return std::to_string(value.get<std::uint64_t>());
    if (value.is_number_integer()) return std::to_string(value.get<std::int64_t>());
    if (value.is_number_float()) return python_fraction_text(value.get<double>());
    throw PythonJsonTypeError("Hardware source time base must be a string or rational number");
}

std::string source_time_base(const std::filesystem::path& source) {
    if (!python_path_is_file(source) || !find_executable("gst-launch-1.0")) {
        throw DecoderError("Jetson decoding requires a local file and GStreamer");
    }
    try {
        const auto encoding = python_process_text_encoding();
        ChildPipe probe({"ffprobe", "-v", "error", "-show_streams", "-show_format",
                          "-of", "json", source.string()}, std::nullopt, false,
                         std::nullopt, false, false, true);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        auto captured = probe.communicate([&] {
            if (std::chrono::steady_clock::now() >= deadline)
                throw std::runtime_error("ffprobe timed out after 30 seconds");
            return false;
        });
        captured.output = decode_process_text(std::move(captured.output), encoding);
        captured.errors = decode_process_text(std::move(captured.errors), encoding);
        if (captured.returncode != 0)
            throw std::runtime_error("ffprobe failed with exit status " + std::to_string(captured.returncode));
        const auto data = parse_python_json(captured.output);
        const auto& source_streams = data.at("streams");
        std::vector<nlohmann::json> streams;
        if (source_streams.is_array()) {
            for (const auto& stream : source_streams)
                if (python_json_dict_get(stream, "codec_type") == "video") streams.push_back(stream);
        } else if (source_streams.is_object() || source_streams.is_string()) {
            // Dictionary keys and string characters are str receivers. Empty
            // iterables select no video; a nonempty one fails its first .get.
            if (json_truthy(source_streams))
                throw PythonJsonAttributeError("'str' object has no attribute 'get'");
        } else {
            throw PythonJsonTypeError("'" +
                std::string(json_compat_detail::python_type_name(source_streams)) + "' object is not iterable");
        }
        if (streams.size() != 1) throw DecoderError("hardware path requires exactly one video stream");
        const auto& stream = streams.front();
        if (python_json_dict_get(stream, "codec_name") != "h264" ||
            python_json_dict_get(stream, "pix_fmt") != "yuv420p") {
            throw DecoderError("hardware path currently supports 8-bit H.264 only");
        }
        if (unaligned(stream.at("width"), 8) || unaligned(stream.at("height"), 2)) {
            throw DecoderError("input dimensions require software plane packing");
        }
        const auto fields = python_json_dict_get(stream, "field_order", nlohmann::json("unknown"));
        if (fields != "unknown" && fields != "progressive") throw DecoderError("interlaced input requires software decoding");
        const auto range = python_json_dict_get(stream, "color_range", nlohmann::json("unknown"));
        if (range != "unknown" && range != "tv") throw DecoderError("full-range input requires software decoding");
        static const nlohmann::json empty_tags = nlohmann::json::object();
        if (json_truthy(python_json_dict_get(stream, "side_data_list")) ||
            json_truthy(python_json_dict_get(python_json_dict_get(stream, "tags", empty_tags), "rotate"))) {
            throw DecoderError("video side data requires software decoding");
        }
        const auto& format = data.at("format");
        if (!json_python_equal(python_json_dict_get(format, "start_time"),
                               python_json_dict_get(stream, "start_time"))) {
            throw DecoderError("different container/video start times require software decoding");
        }
        const auto base = time_base(stream.at("time_base"));
        std::istringstream names(format.at("format_name").get<std::string>());
        std::string name;
        bool supported = false;
        while (std::getline(names, name, ',')) if (name == "mov" || name == "mp4") supported = true;
        if (!supported) throw DecoderError("unsupported hardware input container");
        return base;
    } catch (const DecoderError&) {
        throw;
    } catch (const PythonJsonTypeError&) {
        throw;
    } catch (const PythonJsonAttributeError&) {
        throw;
    } catch (const std::overflow_error&) {
        throw;
    } catch (const std::invalid_argument& error) {
        ProcessInterrupt::check();
        throw DecoderError(std::string("cannot probe hardware input: ") + error.what());
    } catch (const nlohmann::json::parse_error& error) {
        ProcessInterrupt::check();
        throw DecoderError(std::string("cannot probe hardware input: ") + error.what());
    } catch (const nlohmann::json::out_of_range& error) {
        ProcessInterrupt::check();
        throw DecoderError(std::string("cannot probe hardware input: ") + error.what());
    } catch (const std::runtime_error& error) {
        ProcessInterrupt::check();
        throw DecoderError(std::string("cannot probe hardware input: ") + error.what());
    }
}

struct DiagnosticReport {
    bool failure{};
    std::string tail;
};

DiagnosticReport diagnostics(int descriptor) {
    DiagnosticReport result;
    std::array<char, 8192> buffer{};
    off_t offset = 0;
    std::string overlap;
    for (;;) {
        ProcessInterrupt::check();
        const auto count = pread(descriptor, buffer.data(), buffer.size(), offset);
        if (count < 0 && errno == EINTR) continue;
        if (count < 0) {
            throw std::system_error(errno, std::generic_category(), "Cannot read Jetson diagnostics");
        }
        if (!count) break;
        if (count > std::numeric_limits<off_t>::max() - offset) {
            throw std::length_error("Jetson diagnostic offset overflow");
        }
        offset += count;
        const auto text = overlap + std::string(buffer.data(), static_cast<std::size_t>(count));
        for (const char* token : {"ERROR", "NvMapMem", "Failed"}) {
            if (text.find(token) != std::string::npos) result.failure = true;
        }
        overlap = text.substr(text.size() > 8 ? text.size() - 8 : 0);
    }
    // Python checks tokens in the full replacement-decoded log, then slices
    // decoded characters. ASCII tokens cannot be created by UTF-8 replacement,
    // so scanning raw blocks preserves that check without retaining the log.
    result.tail = utf8_file_tail(descriptor, 2000);
    return result;
}

}  // namespace

struct JetsonFramePipe::Impl {
    FileDescriptor errors;
    std::unique_ptr<ChildProcess> decoder;
    std::unique_ptr<ChildPipe> ffmpeg;
    bool finished{}, cleanup_attempted{};

    Impl(const std::filesystem::path& source, std::vector<std::string> command) {
        const auto base = source_time_base(source);
        auto bridge = process_pipe();
        errors = python_temporary_anonymous_file();
        FileDescriptor null_input(open("/dev/null", O_RDONLY | O_CLOEXEC));
        if (null_input.get() < 0) throw DecoderError("Cannot open Jetson null input");
        const auto input = std::find(command.begin(), command.end(), "-i");
        const auto hardware = std::find(command.begin(), command.end(), "-hwaccel");
        const auto filters = std::find(command.begin(), command.end(), "-vf");
        if (input == command.end() || std::next(input) == command.end() ||
            hardware == command.end() || std::next(hardware) == command.end() ||
            filters == command.end() || std::next(filters) == command.end()) {
            throw std::invalid_argument("Incomplete Jetson FFmpeg command");
        }
        *std::next(input) = "pipe:0";
        *std::next(hardware) = "none";
        *std::next(filters) = "settb=expr=" + base + "," + *std::next(filters);
        try {
            decoder = std::make_unique<ChildProcess>(std::vector<std::string>{
                "gst-launch-1.0", "-q", "filesrc", "location=" + source.string(), "!",
                "qtdemux", "name=demux", "demux.video_0", "!", "queue", "!", "h264parse", "!",
                "nvv4l2decoder", "!", "nvvidconv", "!", "video/x-raw,format=I420", "!",
                "matroskamux", "streamable=true", "timecodescale=1", "!", "fdsink", "fd=3"},
                std::vector<std::pair<int, int>>{{null_input.get(), 0}, {errors.get(), 1},
                    {errors.get(), 2}, {bridge[1].get(), 3}});
            ffmpeg = std::make_unique<ChildPipe>(command, bridge[0].get(), false, errors.get());
        } catch (const std::exception& error) {
            ProcessInterrupt::check();
            throw DecoderError(std::string("cannot start Jetson decoder: ") + error.what());
        }
        // Neither end remains open in the parent; downstream failure must
        // reach the GStreamer producer, and EOF must reach FFmpeg.
    }

    void cancel() noexcept {
        if (ffmpeg) ffmpeg->cancel();
        if (decoder) decoder->signal(SIGKILL);
    }

    void stop_for_cleanup() {
        cancel();
        if (ffmpeg) ffmpeg->stop_for_cleanup();
        if (decoder && !decoder->poll_for_cleanup()) decoder->wait_for_cleanup();
    }

    void close_checked() {
        if (cleanup_attempted) return;
        cleanup_attempted = true;
        // FrameReader abandons input with KILL before joining its worker.
        // Reap without checking an already latched foreground interrupt.
        stop_for_cleanup();
        if (ffmpeg) ffmpeg->close_checked();
        errors.close_checked("Cannot close Jetson diagnostics");
    }

    ~Impl() {
        try { close_checked(); } catch (...) {}
        cancel();
    }

    void finish() {
        if (finished) return;
        int ffmpeg_code, code;
        try {
            // A nonzero FFmpeg result still requires the decoder wait and the
            // complete shared diagnostic log, as in JetsonPipe.wait().
            ffmpeg_code = ffmpeg->wait_result();
            const int decoder_status = decoder->wait();
            code = decoder->returncode().value_or(decoder_status);
        } catch (...) {
            cancel();
            throw;
        }
        // Read diagnostics only after both waits succeed. A failed read is
        // propagated directly instead of being retried while wrapping it.
        const auto report = diagnostics(errors.get());
        if (ffmpeg_code || code || report.failure) {
            throw DecoderError("Jetson decoder failed (" + std::to_string(code) +
                "), FFmpeg (" + std::to_string(ffmpeg_code) + "): " + report.tail);
        }
        finished = true;
    }
};

JetsonFramePipe::JetsonFramePipe(const std::filesystem::path& source, std::vector<std::string> command)
    : impl_(std::make_unique<Impl>(source, std::move(command))) {}
JetsonFramePipe::~JetsonFramePipe() = default;
std::size_t JetsonFramePipe::read(std::uint8_t* destination, std::size_t size) { return impl_->ffmpeg->read(destination, size); }
void JetsonFramePipe::finish() { impl_->finish(); }
void JetsonFramePipe::cancel() noexcept { impl_->cancel(); }
void JetsonFramePipe::stop_for_cleanup() { impl_->stop_for_cleanup(); }
void JetsonFramePipe::close_checked() { impl_->close_checked(); }

}  // namespace uf
