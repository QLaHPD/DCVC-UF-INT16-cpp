#include "uf/viewer.hpp"
#include "uf/process.hpp"
#include "uf/python_path.hpp"
#include "uf/json_compat.hpp"
#include "uf/python_numeric.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <initializer_list>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

#ifdef UF_HAS_TK
#include <tcl.h>
#include <tk.h>
#endif

namespace uf {
#ifdef UF_HAS_TK
namespace {

struct FilterWeights {
    int first{};
    std::vector<std::int32_t> values;
};

// Separable, widened triangle filtering, with 22-bit coefficients and an
// 8-bit intermediate, follows Pillow's RGB BILINEAR display-resize arithmetic.
// Reference: https://github.com/python-pillow/Pillow/blob/12.1.1/src/libImaging/Resample.c
std::vector<FilterWeights> weights(int input, int output) {
    std::vector<FilterWeights> result(static_cast<std::size_t>(output));
    // Pillow's box coordinates enter precompute_coeffs as floats, even for
    // the default whole-image box, before the Double scale calculation.
    const double scale = static_cast<double>(static_cast<float>(input)) / output;
    const double support = std::max(1.0, scale);
    const double inverse_scale = 1.0 / support;
    for (int x = 0; x < output; ++x) {
        const double center = (x + 0.5) * scale;
        auto& row = result[x];
        row.first = static_cast<int>(std::max(0.0, center - support + 0.5));
        const int end = static_cast<int>(std::min(static_cast<double>(input), center + support + 0.5));
        std::vector<double> coefficients;
        double sum = 0;
        for (int i = row.first; i < end; ++i) {
            // Preserve Pillow's rounded reciprocal followed by multiplication.
            const double value = std::max(0.0, 1.0 - std::abs((i - center + 0.5) * inverse_scale));
            coefficients.push_back(value); sum += value;
        }
        for (auto value : coefficients) {
            if (sum != 0.0) value /= sum;
            row.values.push_back(static_cast<std::int32_t>(0.5 + value * (1 << 22)));
        }
    }
    return result;
}

std::vector<std::uint8_t> resize_rgb(const DecodedFrame& frame, int width, int height) {
    auto pixels = frame.rgb;
    int current_width = static_cast<int>(frame.width);
    int current_height = static_cast<int>(frame.height);
    for (const bool horizontal : {true, false}) {
        const int input = horizontal ? current_width : current_height;
        const int output = horizontal ? width : height;
        if (input == output) continue;
        const auto table = weights(input, output);
        const int next_width = horizontal ? width : current_width;
        const int next_height = horizontal ? current_height : height;
        std::vector<std::uint8_t> next(static_cast<std::size_t>(next_width) * next_height * 3);
        for (int y = 0; y < next_height; ++y) {
            for (int x = 0; x < next_width; ++x) {
                const auto& row = table[horizontal ? x : y];
                for (int c = 0; c < 3; ++c) {
                    std::int64_t sum = 1 << 21;
                    for (std::size_t i = 0; i < row.values.size(); ++i) {
                        const auto source_x = horizontal ? row.first + static_cast<int>(i) : x;
                        const auto source_y = horizontal ? y : row.first + static_cast<int>(i);
                        sum += static_cast<std::int64_t>(pixels[
                            (static_cast<std::size_t>(source_y) * current_width + source_x) * 3 + c]) * row.values[i];
                    }
                    next[(static_cast<std::size_t>(y) * next_width + x) * 3 + c] =
                        static_cast<std::uint8_t>(std::clamp<std::int64_t>(sum >> 22, 0, 255));
                }
            }
        }
        pixels = std::move(next);
        current_width = next_width; current_height = next_height;
    }
    return pixels;
}

struct DecodeRequest { std::uint64_t id{}, index{}; };
struct DecodeResult {
    std::uint64_t id{};
    std::optional<DecodedFrame> frame;
    std::string error;
};

class DecodeWorker {
public:
    DecodeWorker(std::unique_ptr<IndexedFrameSource> source,
                 std::vector<std::filesystem::path> images, ViewerCodecLoader loader)
        : source_(std::move(source)), images_(std::move(images)), loader_(std::move(loader)),
          thread_([this] { loop(); }) {}
    ~DecodeWorker() { close(); }
    void request(std::uint64_t id, std::uint64_t index) {
        std::lock_guard<std::mutex> lock(mutex_);
        latest_id_ = id;
        pending_ = DecodeRequest{id, index};
        condition_.notify_one();
    }
    std::optional<DecodeResult> poll() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (worker_failed_ && !result_)
            return DecodeResult{latest_id_, std::nullopt, "UF viewer decode worker stopped"};
        auto result = std::move(result_); result_.reset();
        return result;
    }
    void close() noexcept {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_.store(true);
            pending_.reset();
        }
        condition_.notify_one();
        if (thread_.joinable()) thread_.join();
    }
private:
    std::unique_ptr<IndexedFrameSource> source_;
    std::vector<std::filesystem::path> images_;
    ViewerCodecLoader loader_;
    std::mutex mutex_;
    std::condition_variable condition_;
    std::atomic<bool> stopping_{false};
    std::optional<DecodeRequest> pending_;
    std::optional<DecodeResult> result_;
    std::uint64_t latest_id_{};
    bool worker_failed_{};
    // Last member: every object the thread uses is initialized before launch.
    std::thread thread_;

    void loop() noexcept {
        try {
            std::string cached_pipeline_key, current_pipeline_key;
            std::shared_ptr<Codec> cached_image;
            const ViewerCodecLoader image_loader = [&](const nlohmann::json& pipeline, bool image_only) {
                if (!image_only) throw std::runtime_error("Image gallery requires an intra image");
                if (!cached_image || cached_pipeline_key != current_pipeline_key) {
                    cached_image.reset();
                    cached_image = loader_(pipeline, true);
                    cached_pipeline_key = current_pipeline_key;
                }
                return cached_image;
            };
            while (!stopping_.load()) {
                DecodeRequest request;
                {
                    std::unique_lock<std::mutex> lock(mutex_);
                    condition_.wait(lock, [&] { return stopping_.load() || pending_.has_value(); });
                    if (stopping_.load()) break;
                    request = *pending_; pending_.reset();
                }
                DecodeResult result;
                result.id = request.id;
                std::unique_ptr<IndexedFrameSource> image;
                try {
                    if (source_) result.frame = source_->seek(request.index, [&] { return stopping_.load(); });
                    else {
                        image = std::make_unique<IndexedFrameSource>(
                            images_.at(static_cast<std::size_t>(request.index)), image_loader);
                        current_pipeline_key = image->pipeline_repr();
                        result.frame = image->seek(0, [&] { return stopping_.load(); });
                    }
                } catch (const std::exception& error) { result.error = error.what(); }
                catch (...) { result.error = "Unknown UF viewer decode error"; }
                if (!stopping_.load()) {
                    std::lock_guard<std::mutex> lock(mutex_);
                    result_ = std::move(result);
                }
                // Gallery finally closes after enqueuing the frame/error.
                // A close failure stops this worker outside the decode catch.
                if (image) image->close_checked();
            }
            if (source_) source_->close_checked();
        } catch (...) {
            std::lock_guard<std::mutex> lock(mutex_);
            worker_failed_ = true;
        }
        source_.reset();
    }
};

class ViewerApp {
public:
    ViewerApp(const ViewerOptions& options, const std::vector<std::filesystem::path>& images,
              std::unique_ptr<IndexedFrameSource> source, ViewerCodecLoader loader)
        : options_(options), images_(images), gallery_(!images.empty()) {
        try {
            total_ = gallery_ ? images_.size() : source->frame_count();
            if (!total_ || total_ > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
                throw std::runtime_error("Invalid UF viewer frame count");
            runtime_ = gallery_ ? "UF / recorded runtime"
                : source->metadata().at("pipeline").value("runtime", std::string("fp16"));
            if (gallery_) fps_ = 1.0;
            else if (options.fps && *options.fps != 0) fps_ = *options.fps;
            else fps_ = source->metadata().at("video").at("fps");
            // max(0.1, fps) compares before converting: NaN keeps the first
            // argument; exact integer metadata remains integer for the timer.
            if (!json_python_less(nlohmann::json(0.1), fps_)) fps_ = 0.1;
            // Tcl/Tk is owned entirely by the UI thread; codec loading is deferred
            // until a request reaches the worker.
            Tcl_FindExecutable(std::filesystem::read_symlink("/proc/self/exe").c_str());
            interpreter_.reset(Tcl_CreateInterp());
            if (!interpreter_ || Tcl_Init(interpreter_.get()) != TCL_OK || Tk_Init(interpreter_.get()) != TCL_OK)
                throw std::runtime_error("Cannot initialize the UF viewer: " +
                    std::string(interpreter_ ? Tcl_GetStringResult(interpreter_.get()) : "no Tcl interpreter"));
            command({"wm", "withdraw", "."});
            Tcl_CreateObjCommand(interpreter_.get(), "uf_viewer", callback, this, nullptr);
            command({"wm", "title", ".", gallery_ ? "DCVC-UF Intra Image Viewer"
                : "DCVC-UF Viewer - " + options.input.filename().string()});
            command({"wm", "protocol", ".", "WM_DELETE_WINDOW", "uf_viewer close"});
            command({"label", ".image", "-bg", "black"});
            command({"pack", ".image", "-fill", "both", "-expand", "1"});
            command({"frame", ".controls"});
            command({"pack", ".controls", "-fill", "x", "-padx", "8", "-pady", "8"});
            if (!gallery_) {
                command({"button", ".controls.play", "-text", "Play", "-width", "8", "-command", "uf_viewer play"});
                command({"pack", ".controls.play", "-side", "left", "-padx", "0 6"});
            }
            command({"button", ".controls.previous", "-text", gallery_ ? "Prev Image" : "Prev",
                "-width", gallery_ ? "10" : "8", "-command", "uf_viewer previous"});
            command({"pack", ".controls.previous", "-side", "left", "-padx", "0 6"});
            command({"button", ".controls.next", "-text", gallery_ ? "Next Image" : "Next",
                "-width", gallery_ ? "10" : "8", "-command", "uf_viewer next"});
            command({"pack", ".controls.next", "-side", "left", "-padx", "0 10"});
            command({"scale", ".controls.seek", "-from", "0", "-to", std::to_string(total_ - 1),
                "-orient", "horizontal", "-showvalue", "0"});
            command({"pack", ".controls.seek", "-side", "left", "-fill", "x", "-expand", "1"});
            command({"bind", ".controls.seek", "<ButtonRelease-1>", "uf_viewer slider"});
            command({"label", ".controls.count", "-width", "18", "-anchor", "e"});
            command({"pack", ".controls.count", "-side", "left", "-padx", "10 0"});
            command({"frame", ".meta"});
            command({"pack", ".meta", "-fill", "x", "-padx", "8", "-pady", "0 8"});
            command({"label", ".meta.status", "-anchor", "w"});
            command({"pack", ".meta.status", "-fill", "x"});
            command({"image", "create", "photo", "uf_frame"});
            if (gallery_) {
                static const std::pair<const char*, const char*> bindings[] = {
                    {"<Left>", "previous"}, {"<Right>", "next"}, {"<Home>", "first"}, {"<End>", "last"}};
                for (const auto& binding : bindings)
                    command({"bind", ".", binding.first, std::string("uf_viewer ") + binding.second});
            }
            worker_ = std::make_unique<DecodeWorker>(std::move(source), images_, std::move(loader));
            set_status(gallery_ ? "Loading image..." : "Loading frame...");
            request(options.start_frame <= 0 ? 0 : static_cast<std::uint64_t>(options.start_frame));
            poll_timer_ = Tcl_CreateTimerHandler(30, tick, this);
            command({"wm", "deiconify", "."});
        } catch (...) {
            cancel_timers();
            if (interpreter_) {
                Tcl_DeleteCommand(interpreter_.get(), "uf_viewer");
                try { command({"destroy", "."}); } catch (...) {}
            }
            if (worker_) worker_->close();
            throw;
        }
    }

    ~ViewerApp() {
        cancel_timers();
        if (interpreter_) {
            Tcl_DeleteCommand(interpreter_.get(), "uf_viewer");
            try { command({"destroy", "."}); } catch (...) {}
        }
        if (worker_) worker_->close();
    }

    int run() {
        while (!closed_ && Tk_GetNumMainWindows() > 0) {
            ProcessInterrupt::check();
            for (int i = 0; i < 100 && !closed_ && Tcl_DoOneEvent(TCL_ALL_EVENTS | TCL_DONT_WAIT); ++i) {}
            if (!closed_) std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        ProcessInterrupt::check();
        return 0;
    }
private:
    struct DeleteInterpreter { void operator()(Tcl_Interp* value) const { Tcl_DeleteInterp(value); } };
    ViewerOptions options_;
    std::vector<std::filesystem::path> images_;
    bool gallery_, playing_{false}, closed_{false};
    std::uint64_t total_{}, current_{}, request_id_{};
    std::string runtime_;
    nlohmann::json fps_ = 1.0;
    std::unique_ptr<Tcl_Interp, DeleteInterpreter> interpreter_;
    std::unique_ptr<DecodeWorker> worker_;
    Tcl_TimerToken poll_timer_{}, play_timer_{};

    std::string command(std::initializer_list<std::string> arguments) {
        struct Objects {
            std::vector<Tcl_Obj*> values;
            ~Objects() { for (const auto value : values) Tcl_DecrRefCount(value); }
        } objects;
        objects.values.reserve(arguments.size());
        for (const auto& argument : arguments) {
            if (argument.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
                throw std::runtime_error("UF viewer argument is too large");
            auto* value = Tcl_NewStringObj(argument.data(), static_cast<int>(argument.size()));
            Tcl_IncrRefCount(value); objects.values.push_back(value);
        }
        const auto code = Tcl_EvalObjv(interpreter_.get(), static_cast<int>(objects.values.size()),
                                       objects.values.data(), TCL_EVAL_GLOBAL);
        const std::string result = Tcl_GetStringResult(interpreter_.get());
        if (code != TCL_OK) throw std::runtime_error(result);
        return result;
    }

    void set_status(const std::string& value) { command({".meta.status", "configure", "-text", value}); }
    void request(std::uint64_t index) {
        current_ = std::min(index, total_ - 1);
        command({".controls.seek", "set", std::to_string(current_)});
        command({".controls.count", "configure", "-text", std::to_string(current_ + 1) + "/" +
            std::to_string(total_) + (gallery_ ? " images" : "")});
        if (gallery_) set_status("Loading " + images_[current_].filename().string() + "...");
        worker_->request(++request_id_, current_);
    }
    void cancel_timers() noexcept {
        if (poll_timer_) { Tcl_DeleteTimerHandler(poll_timer_); poll_timer_ = nullptr; }
        if (play_timer_) { Tcl_DeleteTimerHandler(play_timer_); play_timer_ = nullptr; }
    }
    void pause() {
        playing_ = false;
        if (play_timer_) { Tcl_DeleteTimerHandler(play_timer_); play_timer_ = nullptr; }
        if (!gallery_) command({".controls.play", "configure", "-text", "Play"});
    }
    void schedule_play() {
        if (play_timer_) Tcl_DeleteTimerHandler(play_timer_);
        const auto delay = fps_.is_number_float()
            ? 1000.0 / python_json_as_double(fps_)
            : python_fraction_double("1000/" + python_json_int(fps_).decimal());
        play_timer_ = Tcl_CreateTimerHandler(std::max(1, static_cast<int>(delay)), play_tick, this);
    }
    void display(const DecodedFrame& frame) {
        if (!frame.width || !frame.height || frame.width > static_cast<std::uint32_t>(std::numeric_limits<int>::max() / 4) ||
            frame.height > static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
            frame.rgb.size() != static_cast<std::uint64_t>(frame.width) * frame.height * 3)
            throw std::runtime_error("Invalid UF viewer RGB frame");
        const auto recorded_width = frame.recorded_dimensions
            ? (*frame.recorded_dimensions)[0] : nlohmann::json(frame.width);
        const auto recorded_height = frame.recorded_dimensions
            ? (*frame.recorded_dimensions)[1] : nlohmann::json(frame.height);
        const auto axis_ratio = [](int maximum, const nlohmann::json& recorded) {
            if (recorded.is_null())
                throw PythonJsonTypeError("unsupported operand type(s) for /: 'int' and 'NoneType'");
            auto divisor = python_json_int(recorded).decimal();
            const bool negative = !divisor.empty() && divisor.front() == '-';
            if (negative) divisor.erase(0, 1);
            // Integer true division rounds the ratio, not each integer to
            // double first. Reuse the exact rational conversion boundary.
            const auto numerator = negative ? -static_cast<std::int64_t>(maximum)
                                            : static_cast<std::int64_t>(maximum);
            return python_fraction_double(std::to_string(numerator) + "/" + divisor);
        };
        const double width_ratio = axis_ratio(options_.max_width, recorded_width);
        const double height_ratio = axis_ratio(options_.max_height, recorded_height);
        const double ratio = std::min({width_ratio, height_ratio, 1.0});
        int width = static_cast<int>(frame.width), height = static_cast<int>(frame.height);
        if (ratio < 1.0) {
            const auto scaled_axis = [ratio](const nlohmann::json& recorded) {
                const auto scaled = python_json_as_double(recorded) * ratio;
                if (!std::isfinite(scaled))
                    throw std::overflow_error("Nonfinite UF viewer resize dimension");
                if (scaled < 1) return 1;
                if (scaled >= std::ldexp(1.0, std::numeric_limits<int>::digits))
                    throw std::overflow_error("UF viewer resize dimension exceeds native Int range");
                return static_cast<int>(scaled);
            };
            width = scaled_axis(recorded_width);
            height = scaled_axis(recorded_height);
        }
        if (width > std::numeric_limits<int>::max() / 4 ||
            static_cast<std::uint64_t>(width) * height > std::numeric_limits<std::size_t>::max() / 4)
            throw std::length_error("UF viewer resize exceeds native RGB storage");
        const auto rgb = resize_rgb(frame, width, height);
        std::vector<std::uint8_t> rgba(static_cast<std::size_t>(width) * height * 4, 255);
        for (std::size_t i = 0; i < rgb.size() / 3; ++i)
            std::copy_n(rgb.data() + 3 * i, 3, rgba.data() + 4 * i);
        const auto photo = Tk_FindPhoto(interpreter_.get(), "uf_frame");
        Tk_PhotoImageBlock block{rgba.data(), width, height, width * 4, 4, {0, 1, 2, 3}};
        if (!photo || Tk_PhotoSetSize(interpreter_.get(), photo, width, height) != TCL_OK ||
            Tk_PhotoPutBlock(interpreter_.get(), photo, &block, 0, 0, width, height, TK_PHOTO_COMPOSITE_SET) != TCL_OK)
            throw std::runtime_error(Tcl_GetStringResult(interpreter_.get()));
        command({".image", "configure", "-image", "uf_frame"});
        const auto path = gallery_ ? images_[current_] : options_.input;
        set_status(path.filename().string() + " | " + python_json_repr(recorded_width) + "x" + python_json_repr(recorded_height) +
            " | " + (frame.type == NalType::intra ? "I" : "P") + "-frame | QP " + std::to_string(frame.qp) + " | " + runtime_);
        if (playing_) schedule_play();
    }
    void poll_results() {
        if (const auto result = worker_->poll(); result && result->id == request_id_) {
            if (!result->error.empty()) { pause(); set_status(result->error); }
            else if (result->frame) display(*result->frame);
        }
    }
    void action(const std::string& kind) {
        if (closed_) return;
        if (kind == "close") {
            closed_ = true; cancel_timers(); command({"destroy", "."});
        } else if (kind == "play" && !gallery_) {
            if (playing_) pause();
            else { playing_ = true; command({".controls.play", "configure", "-text", "Pause"}); schedule_play(); }
        } else if (kind == "previous") { pause(); request(current_ ? current_ - 1 : 0); }
        else if (kind == "next") { pause(); request(std::min(current_ + 1, total_ - 1)); }
        else if (kind == "first") { pause(); request(0); }
        else if (kind == "last") { pause(); request(total_ - 1); }
        else if (kind == "slider") {
            pause();
            const auto text = command({".controls.seek", "get"});
            std::size_t used = 0;
            const double index = std::stod(text, &used);
            if (used != text.size() || !std::isfinite(index)) throw std::runtime_error("Invalid UF slider position");
            request(index <= 0 ? 0 : index >= static_cast<double>(total_ - 1) ? total_ - 1 : static_cast<std::uint64_t>(index));
        }
    }

    static int callback(ClientData data, Tcl_Interp* interpreter, int count, Tcl_Obj* const arguments[]) noexcept {
        try {
            if (count != 2) { Tcl_WrongNumArgs(interpreter, 1, arguments, "action"); return TCL_ERROR; }
            static_cast<ViewerApp*>(data)->action(Tcl_GetString(arguments[1]));
            return TCL_OK;
        } catch (const std::exception& error) {
            Tcl_SetObjResult(interpreter, Tcl_NewStringObj(error.what(), -1));
            return TCL_ERROR;
        } catch (...) { return TCL_ERROR; }
    }
    static void tick(ClientData data) noexcept {
        auto& app = *static_cast<ViewerApp*>(data);
        app.poll_timer_ = nullptr;
        if (app.closed_) return;
        try { app.poll_results(); }
        catch (const std::exception& error) {
            try { app.pause(); app.set_status(error.what()); } catch (...) { app.closed_ = true; }
        } catch (...) { app.closed_ = true; }
        if (!app.closed_) app.poll_timer_ = Tcl_CreateTimerHandler(30, tick, &app);
    }
    static void play_tick(ClientData data) noexcept {
        auto& app = *static_cast<ViewerApp*>(data);
        app.play_timer_ = nullptr;
        if (!app.playing_ || app.closed_) return;
        try {
            if (app.current_ + 1 >= app.total_) app.pause();
            else app.request(app.current_ + 1);
        } catch (const std::exception& error) {
            try { app.pause(); app.set_status(error.what()); } catch (...) { app.closed_ = true; }
        } catch (...) { app.closed_ = true; }
    }
};

}  // namespace
#endif

int view_archive(const ViewerOptions& options, ViewerCodecLoader loader) {
    if (options.fps && (!std::isfinite(*options.fps) || *options.fps <= 0))
        throw std::invalid_argument("UF viewer playback FPS must be positive");
#ifdef UF_HAS_TK
    std::vector<std::filesystem::path> images;
    const bool directory = std::filesystem::is_directory(options.input);
    const auto is_image = [](const std::filesystem::path& path) {
        auto extension = path.extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(),
            [](unsigned char c) { return std::tolower(c); });
        return extension == ".dcvci";
    };
    if (directory && !std::filesystem::exists(options.input / "manifest.json")) {
        const auto add = [&](const auto& entry) {
            if (entry.is_regular_file() && is_image(entry.path())) images.push_back(entry.path());
        };
        if (options.recursive) {
            for (const auto& entry : std::filesystem::recursive_directory_iterator(options.input)) add(entry);
        } else {
            for (const auto& entry : std::filesystem::directory_iterator(options.input)) add(entry);
        }
        std::sort(images.begin(), images.end(), python_path_less);
        if (images.empty()) throw std::runtime_error("No .dcvci images found");
    } else if (is_image(options.input)) images.push_back(options.input);
    std::unique_ptr<IndexedFrameSource> source;
    if (images.empty()) source = std::make_unique<IndexedFrameSource>(options.input, loader);
    ViewerApp app(options, images, std::move(source), std::move(loader));
    return app.run();
#else
    throw std::runtime_error("Native Tcl/Tk viewer is unavailable; enable UF_BUILD_VIEWER with Tcl/Tk headers and libraries");
#endif
}

}  // namespace uf
