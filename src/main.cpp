#include "uf/archive.hpp"
#include "uf/image_archive.hpp"
#include "uf/image_workflow.hpp"
#include "uf/image_pixels.hpp"
#include "uf/integer_codec.hpp"
#include "uf/codec.hpp"
#include "uf/cli.hpp"
#include "uf/decode_workers.hpp"
#include "uf/json_compat.hpp"
#include "uf/python_numeric.hpp"
#include "uf/python_json.hpp"
#include "uf/python_path.hpp"
#include "uf/utf8.hpp"
#include "uf/int16_ops.hpp"
#include "uf/int16_validation.hpp"
#include "uf/lifecycle.hpp"
#include "uf/media.hpp"
#include "uf/pareto.hpp"
#include "uf/prepare.hpp"
#include "uf/presets.hpp"
#include "uf/prepared.hpp"
#include "uf/workflow.hpp"
#include "uf/workers.hpp"
#include "uf/shared_work.hpp"
#include "uf/setup.hpp"
#include "uf/remote.hpp"
#include "uf/reference_paths.hpp"
#include "uf/stream_pipeline.hpp"
#include "uf/process.hpp"
#include "uf/events.hpp"
#include "uf/dashboard.hpp"
#include "uf/viewer.hpp"
#ifdef UF_HAS_DCVC2021
#include "uf/dcvc2021_evaluation.hpp"
#endif
#ifdef UF_HAS_DCVC_TCM
#include "uf/dcvc_tcm_evaluation.hpp"
#endif
#ifdef UF_HAS_DCVC_HEM
#include "uf/dcvc_hem_evaluation.hpp"
#endif
#ifdef UF_HAS_DCVC_DC
#include "uf/dcvc_dc_evaluation.hpp"
#endif
#ifdef UF_HAS_EVALUATION
#include "uf/bd_metrics.hpp"
#include "uf/evaluation.hpp"
#endif
#ifdef UF_HAS_TRAINING
#include "uf/training_loader.hpp"
#include "uf/training_workflow.hpp"
#endif

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cctype>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <unordered_map>
#include <vector>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sched.h>
#include <unistd.h>

namespace {

template<class Handler>
int dashboard_backend(Handler&& handler) {
    // UF dashboard._backend reports handler failures and then completion while
    // presentation remains installed. Keep event failures outside a retry loop.
    int code = 0;
    try {
        code = std::forward<Handler>(handler)();
    } catch (const std::exception& error) {
        uf::event("backend_error", {{"error", error.what()}});
        code = uf::ProcessInterrupt::requested()
            ? 128 + uf::ProcessInterrupt::requested() : 1;
    } catch (...) {
        uf::event("backend_error", {{"error", "Unknown native backend exception"}});
        code = uf::ProcessInterrupt::requested()
            ? 128 + uf::ProcessInterrupt::requested() : 1;
    }
    uf::event("backend_finished", {{"code", code}});
    return code;
}

void usage(std::ostream& output = std::cerr) {
    output << "Usage:\n"
                 "  dcvc-uf check-setup [--require-native] [--require-int16-native]\n"
                 "  dcvc-uf check-int16 --output FILE [--variant hts|htl|ld --size 32]"
                 " [--cpu-reference --compare FILE --prepared_i FILE --prepared_p FILE]\n"
                 "  dcvc-uf inspect|check-archive|scan-bitstream|inspect-prepared PATH"
                 " [--variant hts|htl|ld]\n"
                 "  dcvc-uf verify --input ARCHIVE [--runtime auto|fp16|int16]"
                 " [--device cpu|cuda:0]\n"
                 "  dcvc-uf decode --input ARCHIVE --output_file FILE"
                 " [--model_path_i PATH --model_path_p PATH --device cpu|cuda:0]\n"
                 "  dcvc-uf decode --input_folder DIR --output_folder DIR"
                 " [--recursive --worker N --input_kind videos|images|all]\n"
                 "  dcvc-uf view ARCHIVE|IMAGE_FOLDER [--recursive --start_frame N]"
                 " [--fps FPS --max_width 1280 --max_height 720]\n"
                 "  dcvc-uf encode-raw --input_file FILE.yuv --output_file FILE.bin"
                 " --width W --height H --prepared_i PATH --prepared_p PATH"
                 " [--variant hts|htl|ld --qp_i 36 --qp_p 30]\n"
                 "  dcvc-uf encode --input_file VIDEO --output_root DIR"
                 " [--runtime fp16|int16 --model_structure hts|htl|ld]\n"
                 "  dcvc-uf encode --base_root DIR --output_root DIR"
                 " [--channel_ids ID... --procs N --cuda_idx GPU...] [--no-recursive]\n"
                 "  dcvc-uf stream --source_urls URL... --output_root DIR"
                 " [--source-max-height 480 --max-videos N --yt-dlp PATH --procs N]\n"
                 "  dcvc-uf encode-image --input_file IMAGE --output_file IMAGE.dcvci"
                 " [--thumbnail_qp 45 --runtime fp16|int16 --model_path_i PATH]\n"
                 "  dcvc-uf cleanup --input ARCHIVE [--base_root DIR]"
                 " [--yes --prepared_i PATH --prepared_p PATH]\n"
                 "  dcvc-uf manage --output_root DIR [--channel_ids ID...]"
                 " [--ui auto|plain|tui --auto-delete|--cleanup-dry-run|--keep-originals]\n"
                 "  dcvc-uf pareto-search --input_file VIDEO --output_json FILE"
                 " [--trials 2000 --initial_samples 800 --resolution 144]\n"
                 "  dcvc-uf prepare-int16 [--model_structure hts|htl|ld]"
                 " [--model_path_i PATH --model_path_p PATH]\n";
#ifdef UF_HAS_EVALUATION
    output << "  dcvc-uf compare-bd-rate --base_method METHOD --log_paths METHOD FILE... "
                 "[--compare_between class|sequence --plot_rd_curve 0|1]\n"
                 "  dcvc-uf test-video --test_config FILE --output_path FILE "
                 "[--model_path_i FILE --model_path_p FILE --worker N]\n"
                 "  dcvc-uf test-compress-time [--img --model_structure ld|hts|htl --output_path FILE]\n";
#endif
#ifdef UF_HAS_DCVC2021
    output << "  dcvc-uf dcvc2021-download-checkpoints [--output_directory DIR]\n";
#ifdef UF_HAS_EVALUATION
    output << "  dcvc-uf dcvc2021-test-video --test_config FILE --output_json_result_path FILE"
                 " --i_frame_model_path FILE... --model_path FILE... [--cuda --worker N]\n";
#endif
#endif
#if defined(UF_HAS_DCVC_TCM) && defined(UF_HAS_EVALUATION)
    output << "  dcvc-uf dcvc-tcm-test-video --test_config FILE --output_path FILE"
                 " --i_frame_model_path FILE... [--model_path FILE... --force_intra --cuda --worker N]\n";
#endif
#ifdef UF_HAS_DCVC_HEM
    output << "  dcvc-uf dcvc-hem-download-checkpoints [--output_directory DIR]\n";
#ifdef UF_HAS_EVALUATION
    output << "  dcvc-uf dcvc-hem-test-video --test_config FILE --output_path FILE"
                 " --i_frame_model_path FILE [--model_path FILE --rate_num N --force_intra --cuda --worker N]\n";
#endif
#endif
#ifdef UF_HAS_DCVC_DC
    output << "  dcvc-uf dcvc-dc-download-checkpoints [--output_directory DIR]\n"
                 "  dcvc-uf dcvc-dc-data-to-png [--input FILE --output DIR --width N --height N]\n";
#ifdef UF_HAS_EVALUATION
    output << "  dcvc-uf dcvc-dc-test-video --test_config FILE --output_path FILE"
                 " --i_frame_model_path FILE [--p_frame_model_path FILE --rate_num N --force_intra --cuda --worker N]\n";
#endif
#endif
#ifdef UF_HAS_TRAINING
    output << "  dcvc-uf train-image --train_dataset DIR --save_dir DIR --lambdas MIN MAX\n"
                 "  dcvc-uf train-video --train_dataset DIR --save_dir DIR --lambdas MIN MAX"
                 " --model_path_i PATH [--model_structure ld|hts|htl]\n";
#endif
    if (!output) throw std::runtime_error("Cannot write UF usage");
}

std::unordered_map<std::string, std::string> options(int argc, char** argv) {
    return uf::cli::parse(argc, argv);
}

std::string required(const std::unordered_map<std::string, std::string>& values,
                     const std::string& name) {
    const auto it = values.find(name);
    if (it == values.end()) throw std::invalid_argument("Missing required option " + name);
    return it->second;
}

bool nonempty_option(const std::unordered_map<std::string, std::string>& values,
                     const char* name) {
    const auto found = values.find(name);
    return found != values.end() && !found->second.empty();
}

std::string lower_extension(const std::filesystem::path& path) {
    auto extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char letter) { return static_cast<char>(std::tolower(letter)); });
    return extension;
}

std::filesystem::path checkpoint_root() {
    return uf::checkpoint_root();
}

std::filesystem::path resolve_prepared_path(
    const std::unordered_map<std::string, std::string>& values,
    const std::string& explicit_name, const std::string& model_name,
    const std::string& variant, const std::string& source_sha256) {
    const auto model_option = explicit_name == "__prepared_i"
        ? "__model_path_i" : "__model_path_p";
    const auto checkpoint = values.count(model_option)
        ? std::filesystem::path(values.at(model_option))
        : checkpoint_root() / model_name;
    return uf::prepared_artifact_path(checkpoint, variant,
        values.count(explicit_name)
            ? std::optional<std::filesystem::path>(values.at(explicit_name)) : std::nullopt,
        source_sha256);
}

std::string encode_runtime(const std::unordered_map<std::string, std::string>& values) {
    const auto runtime = values.count("__runtime") ? values.at("__runtime")
        : (values.count("__target_psnr") ? "int16" : "fp16");
    if (runtime != "int16" && runtime != "fp16") {
        throw std::invalid_argument("--runtime must be fp16 or int16 for encode");
    }
    return runtime;
}

int integer_option(const std::string& text, const std::string& name) {
    return uf::cli::integer(text, name);
}

std::vector<int> cuda_indices(const std::unordered_map<std::string, std::string>& values) {
    std::vector<int> result;
    if (!values.count("__cuda_idx")) return result;
    for (const auto& text : uf::cli::list(values, "__cuda_idx")) {
        const int index = integer_option(text, "--cuda_idx");
        if (index < 0) throw std::invalid_argument("--cuda_idx must be nonnegative");
        result.push_back(index);
    }
    if (result.empty()) throw std::invalid_argument("--cuda_idx requires a CUDA index");
    return result;
}

std::string codec_device(const std::unordered_map<std::string, std::string>& values,
                         const std::string& runtime) {
    std::string device = values.count("__device") ? values.at("__device") : "cuda:0";
    if (values.count("__cuda")) {
        const auto& enabled = values.at("__cuda");
        if (enabled == "false" || enabled == "0") device = "cpu";
        else if (enabled == "true" || enabled == "1") {
            if (!values.count("__device")) device = "cuda:0";
        } else throw std::invalid_argument("--cuda must be true or false");
    }
    if (device == "cuda" || (values.count("__cuda_idx") && device.rfind("cuda", 0) == 0)) {
        const auto indexes = cuda_indices(values);
        // Python's single-archive decode uses cuda_idx[0]. Encode image passes
        // also use the first device; worker_devices retains the full list for
        // workflows that schedule independent processes across several GPUs.
        device = "cuda:" + std::to_string(indexes.empty() ? 0 : indexes.front());
    }
    if (runtime == "fp16" && torch::Device(device).type() != torch::kCUDA) {
        throw std::invalid_argument("UF FP16 requires CUDA; select --runtime int16 for CPU execution");
    }
    return device;
}

std::vector<std::string> worker_devices(
    const std::unordered_map<std::string, std::string>& values, const std::string& device) {
    if (torch::Device(device).type() != torch::kCUDA || !values.count("__cuda_idx")) return {device};
    std::vector<std::string> result;
    for (const int index : cuda_indices(values)) result.push_back("cuda:" + std::to_string(index));
    return result;
}

std::optional<nlohmann::json> apply_target_psnr_options(uf::cli::Options& values) {
    if (!values.count("__target_psnr")) return {};
    nlohmann::json constraints = nlohmann::json::object();
    for (const auto* field : {"qp_i", "qp_p", "reset_interval", "intra_period"}) {
        const auto name = std::string("__") + field;
        if (values.count(name)) constraints[field] = uf::cli::integer(values.at(name), "--" + std::string(field));
    }
    if (values.count("__skip_thres")) {
        constraints["skip_threshold"] = uf::cli::number(values.at("__skip_thres"), "--skip_thres");
    }
    nlohmann::json requirements = nlohmann::json::object();
    for (const auto* field : {"runtime", "model_structure"}) {
        const auto name = std::string("__") + field;
        if (values.count(name)) requirements[field] = values.at(name);
    }
    if (values.count("__resolution")) {
        requirements["resolution"] = uf::cli::integer(values.at("__resolution"), "--resolution");
    }
    const auto benchmark = uf::bundled_benchmark();
    const auto target = uf::cli::number(values.at("__target_psnr"), "--target-psnr");
    const auto selection = uf::select_target_psnr(benchmark, target, constraints, requirements);
    values["__runtime"] = selection.runtime;
    values["__model_structure"] = selection.variant;
    values["__resolution"] = std::to_string(selection.resolution);
    const auto& selected = selection.trial.at("config");
    for (const auto* field : {"qp_i", "qp_p", "reset_interval", "intra_period"}) {
        values[std::string("__") + field] = uf::python_json_int(nlohmann::json(selected.at(field))).decimal();
    }
    values["__skip_thres"] = uf::python_float_repr(
        uf::python_json_as_double(nlohmann::json(selected.at("skip_threshold"))));

    const auto configuration = uf::python_json_repr(selected);
    std::ostringstream message;
    message.imbue(std::locale::classic());
    const auto& trial_id = selection.trial.at("trial");
    message << "Benchmark trial " << (trial_id.is_string() ? trial_id.get<std::string>()
                                                        : uf::python_json_repr(trial_id)) << ": "
            << std::fixed << std::setprecision(3)
            << uf::python_json_as_double(nlohmann::json(selection.trial.at("psnr_db")))
            << " dB, " << uf::python_json_as_double(nlohmann::json(selection.trial.at("encode_seconds"))) << "s, "
            << uf::python_json_repr(selection.trial.at("bitstream_bytes")) << " bytes; "
            << configuration << ". Reference measurements, not a per-video quality guarantee.";
    return nlohmann::json{{"target_psnr", target}, {"benchmark", uf::python_path_text(benchmark.filename())},
        {"trial", selection.trial.at("trial")}, {"measured_psnr", selection.trial.at("psnr_db")},
        {"encode_seconds", selection.trial.at("encode_seconds")},
        {"bitstream_bytes", selection.trial.at("bitstream_bytes")}, {"config", selected},
        {"message", message.str()}};
}

uf::CodecModels archive_models(const std::unordered_map<std::string, std::string>& values,
                               const nlohmann::json& pipeline, bool image_only) {
    const auto runtime = pipeline.value("runtime", std::string("fp16"));
    if (values.count("__runtime") && values.at("__runtime") != "auto" &&
        values.at("__runtime") != runtime) {
        throw std::invalid_argument("Selected runtime does not match UF archive");
    }
    if (runtime == "int16" &&
        pipeline.at("integer").at("arithmetic") != uf::int16::arithmetic_id) {
        throw std::invalid_argument("Unsupported integer arithmetic version");
    }
    const auto variant = image_only ? std::string("hts")
        : pipeline.at("variant").get<std::string>();
    uf::ExpectedModel expected_image, expected_video;
    expected_image.source_sha256 = pipeline.at("image_sha256").get<std::string>();
    std::filesystem::path image;
    std::optional<std::filesystem::path> video;
    if (!image_only) expected_video.source_sha256 = pipeline.at("video_sha256").get<std::string>();
    if (runtime == "int16") {
        expected_image.prepared_identity = pipeline.at("integer").at("image").get<std::string>();
        image = resolve_prepared_path(values, "__prepared_i", "cvpr2026_image.pth.tar",
                                      "image", *expected_image.source_sha256);
        if (!image_only) {
            expected_video.prepared_identity = pipeline.at("integer").at("video").get<std::string>();
            video = resolve_prepared_path(values, "__prepared_p",
                "cvpr2026_video_" + variant + ".pth.tar", variant, *expected_video.source_sha256);
        }
    } else if (runtime == "fp16") {
        image = values.count("__model_path_i") ? std::filesystem::path(values.at("__model_path_i"))
            : checkpoint_root() / "cvpr2026_image.pth.tar";
        if (!image_only) video = values.count("__model_path_p")
            ? std::filesystem::path(values.at("__model_path_p"))
            : checkpoint_root() / ("cvpr2026_video_" + variant + ".pth.tar");
    } else throw std::invalid_argument("Unsupported UF archive runtime");
    return uf::CodecModels::load(runtime, image, video, variant, expected_image, expected_video);
}

int run_view(int argc, char** argv) {
    uf::ProcessInterrupt interruption;
    const auto values = options(argc, argv);
    for (const auto& [key, _] : values) {
        if (key != "__input" && key != "__input_file" && key != "__recursive" &&
            key != "__fps" && key != "__start_frame" && key != "__max_width" && key != "__max_height" &&
            key != "__prepared_i" && key != "__prepared_p" && key != "__model_path_i" && key != "__model_path_p" &&
            key != "__runtime" && key != "__device" && key != "__cuda" && key != "__cuda_idx")
            throw std::invalid_argument("Unknown viewer option: " + key);
    }
    if (values.count("__input") && values.count("__input_file"))
        throw std::invalid_argument("Select one viewer input");
    if (values.count("__runtime") && values.at("__runtime") != "auto" &&
        values.at("__runtime") != "int16" && values.at("__runtime") != "fp16")
        throw std::invalid_argument("--runtime must be auto, fp16, or int16");
    uf::ViewerOptions config;
    config.input = values.count("__input") ? values.at("__input") : required(values, "__input_file");
    config.recursive = values.count("__recursive") && values.at("__recursive") == "true";
    if (values.count("__fps")) {
        config.fps = uf::cli::number(values.at("__fps"), "--fps");
    }
    if (values.count("__start_frame")) {
        config.start_frame = uf::cli::signed_integer(values.at("__start_frame"), "--start_frame");
    }
    if (values.count("__max_width")) config.max_width = integer_option(values.at("__max_width"), "--max_width");
    if (values.count("__max_height")) config.max_height = integer_option(values.at("__max_height"), "--max_height");
    return uf::view_archive(config, [values](const nlohmann::json& pipeline, bool image_only) {
        uf::ProcessInterrupt::check();
        const auto runtime = pipeline.value("runtime", std::string("fp16"));
        auto models = archive_models(values, pipeline, image_only);
        uf::ProcessInterrupt::check();
        return std::make_shared<uf::Codec>(std::move(models), torch::Device(codec_device(values, runtime)),
                                         pipeline.value("skip_threshold", 0.0));
    });
}

class StagedOutput {
public:
    enum class Kind { reserved_file, bitstream_directory, image_directory };

    explicit StagedOutput(std::filesystem::path target, Kind kind = Kind::reserved_file)
        : target_(std::move(target)), kind_(kind) {
        if (kind != Kind::image_directory && std::filesystem::exists(target_)) {
            throw std::runtime_error("Refusing to overwrite: " + target_.string());
        }
        const auto parent = target_.parent_path().empty()
            ? std::filesystem::path(".") : target_.parent_path();
        std::filesystem::create_directories(parent);
        if (kind == Kind::reserved_file) {
            // Python 3.10 Path.suffix excludes a leading or trailing dot.
            const auto name = target_.filename().native();
            const auto dot = name.rfind('.');
            const auto suffix = dot != std::string::npos && dot > 0 && dot + 1 < name.size()
                ? name.substr(dot) : std::string{};
            auto temporary = uf::python_temporary_file(
                "." + name.substr(0, name.size() - suffix.size()), suffix, parent);
            try {
                temporary_ = std::move(temporary.path);
                temporary.descriptor.close_checked("Cannot close staged output reservation");
            } catch (...) {
                const auto& path = temporary_.empty() ? temporary.path : temporary_;
                if (unlink(path.c_str()) != 0) {
                    const int error = errno;
                    if (error != ENOENT)
                        throw std::system_error(error, std::generic_category(),
                                                "Cannot remove video stage reservation");
                }
                throw;
            }
        } else {
            directory_ = uf::python_temporary_mkdtemp(kind == Kind::image_directory
                ? std::string(".uf-png-") : "." + target_.filename().native() + ".tmp-", {}, parent);
            try {
                temporary_ = directory_ / (kind == Kind::image_directory
                    ? std::filesystem::path("image.png") : target_.filename());
            } catch (...) {
                rmdir(directory_.c_str());
                throw;
            }
        }
    }
    ~StagedOutput() {
        if (kind_ == Kind::image_directory) {
            if (!cleanup_attempted_ && !directory_.empty()) {
                try { uf::python_temporary_rmtree(directory_, false, false); } catch (...) {}
            }
            return;
        }
        if (kind_ == Kind::reserved_file && cleanup_attempted_) return;
        if (!temporary_.empty()) {
            std::error_code ignored;
            std::filesystem::remove(temporary_, ignored);
        }
        if (!directory_.empty()) {
            std::error_code ignored;
            std::filesystem::remove_all(directory_, ignored);
        }
    }
    const std::filesystem::path& temporary() const { return temporary_; }
    void close_image_stage() {
        if (kind_ != Kind::image_directory || cleanup_attempted_) return;
        cleanup_attempted_ = true;
        uf::python_temporary_rmtree(directory_, false, false);
        directory_.clear();
        temporary_.clear();
    }
    void close_video_stage() {
        if (kind_ != Kind::reserved_file || cleanup_attempted_) return;
        cleanup_attempted_ = true;
        if (temporary_.empty()) return;
        // Path.unlink(missing_ok=True) in run_decode's finally.
        if (unlink(temporary_.c_str()) != 0) {
            const int error = errno;
            if (error != ENOENT)
                throw std::system_error(error, std::generic_category(),
                                        "Cannot remove reconstructed-video stage");
        }
        temporary_.clear();
    }
    void publish() {
        uf::ProcessInterrupt::check();
        // run_decode publishes the completed FFmpeg file with link/unlink.
        // Image and native raw-bitstream stages retain their sync boundaries.
        if (kind_ != Kind::reserved_file) sync_path(temporary_, false);
        uf::ProcessInterrupt::check();
        std::filesystem::create_hard_link(temporary_, target_);
        if (kind_ != Kind::reserved_file) {
            const auto parent = target_.parent_path().empty()
                ? std::filesystem::path(".") : target_.parent_path();
            sync_path(parent, true);
        }
        if (kind_ == Kind::image_directory) {
            close_image_stage();
            return;
        }
        if (kind_ == Kind::reserved_file) {
            // Publication's os.unlink is strict; finally removal tolerates a
            // missing name after an interrupted or partially completed publish.
            if (unlink(temporary_.c_str()) != 0) {
                const int error = errno;
                throw std::system_error(error, std::generic_category(),
                                        "Cannot remove published video stage");
            }
        } else {
            std::filesystem::remove(temporary_);
        }
        temporary_.clear();
        if (!directory_.empty()) {
            std::filesystem::remove(directory_);
            directory_.clear();
        }
    }

private:
    static void sync_path(const std::filesystem::path& path, bool directory) {
        uf::ProcessInterrupt::check();
        uf::FileDescriptor file(open(path.c_str(), O_RDONLY | O_CLOEXEC | (directory ? O_DIRECTORY : 0)));
        if (file.get() < 0) {
            const int error = errno;
            throw std::system_error(error, std::generic_category(), "Cannot open staged output for sync");
        }
        try {
            for (;;) {
                uf::ProcessInterrupt::check();
                if (fsync(file.get()) == 0) break;
                const int error = errno;
                if (error != EINTR)
                    throw std::system_error(error, std::generic_category(), "Cannot sync staged output");
            }
            file.close_checked("Cannot close staged output sync descriptor");
        } catch (...) {
            file.close_checked("Cannot close staged output sync descriptor");
            throw;
        }
    }
    std::filesystem::path target_, temporary_, directory_;
    Kind kind_;
    bool cleanup_attempted_{};
};

// read_image verifies by chunks, then FrameSource reopens the container at its
// payload offset. Keep the full compressed payload out of direct decode and
// structure-inspection scratch storage as well.
template <typename Function>
auto with_intra_payload(const uf::IntraImageArchive& archive, Function&& function) {
    uf::ProcessInterrupt::check();
    std::ifstream input(archive.path, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot open UF intra image payload");
    bool close_attempted = false;
    const auto close_input = [&] {
        if (close_attempted) return;
        close_attempted = true;
        input.clear();
        input.close();
        if (input.fail()) throw std::runtime_error("Cannot close UF intra image payload");
    };
    try {
        if (archive.payload_offset >
            static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max()))
            throw std::overflow_error("UF intra image payload offset exceeds native stream range");
        input.seekg(static_cast<std::streamoff>(archive.payload_offset));
        if (!input) throw std::runtime_error("Cannot seek UF intra image payload");
        auto result = function(input);
        close_input();
        return result;
    } catch (...) {
        // A context close failure may replace the active body failure. Never
        // retry a close already attempted by the successful-body path.
        close_input();
        throw;
    }
}


int run_intra_decode_values(const std::unordered_map<std::string, std::string>& values,
                            bool write_output) {
    uf::ProcessInterrupt::check();
    const auto input = values.count("__input") ? values.at("__input") : required(values, "__input_file");
    const auto archive = uf::read_intra_image(input, true, false);
    return with_intra_payload(archive, [&](std::istream& payload) {
        const auto& data = archive.metadata;
        const auto& pipeline = data.at("pipeline");
        const auto runtime = pipeline.value("runtime", std::string("fp16"));
        const auto& video = data.at("video");
        const auto scan = uf::scan_bitstream(payload, "hts", 1);
        if (scan.packets != 1 || scan.intra_packets != 1 ||
            scan.preamble.integer != (runtime == "int16") ||
            !uf::json_python_equal(nlohmann::json(scan.dimensions.width), video.at("width")) ||
            !uf::json_python_equal(nlohmann::json(scan.dimensions.height), video.at("height"))) {
            throw std::runtime_error("UF intra image bitstream differs from metadata");
        }
        const auto width = scan.dimensions.width;
        const auto height = scan.dimensions.height;
        uf::ProcessInterrupt::check();
        auto models = archive_models(values, pipeline, true);
        uf::ProcessInterrupt::check();
        const auto device = codec_device(values, runtime);
        torch::NoGradGuard no_grad;
        uf::Codec codec(std::move(models), torch::Device(device), pipeline.value("skip_threshold", 0.0));
        payload.clear();
        payload.seekg(static_cast<std::streamoff>(archive.payload_offset));
        if (!payload) throw std::runtime_error("Cannot seek UF intra image payload");
        std::vector<std::uint8_t> raw;
        const auto stats = codec.decode_stream(payload, 1, 1,
            [&](const std::vector<std::uint8_t>& frame) {
                uf::ProcessInterrupt::check();
                raw = frame;
            }, uf::Sps{0, width, height});
        uf::ProcessInterrupt::check();
        // FrameSource.seek crops once even for verification. decode_image applies
        // the same slice again before saving; negative stops therefore crop twice.
        const auto crop_width = uf::image_crop_extent(data.at("image_width"), width);
        const auto crop_height = uf::image_crop_extent(data.at("image_height"), height);
        const auto& validation = data.at("validation");
        if (validation.contains("decoded_yuv_sha256") &&
            uf::json_truthy(validation.at("decoded_yuv_sha256")) &&
            validation.at("decoded_yuv_sha256") != stats.decoded_yuv_sha256) {
            throw std::runtime_error("Decoded UF intra image differs from archive validation");
        }
        if (write_output) {
            const auto output_path = std::filesystem::path(required(values, "__output_file"));
            StagedOutput stage(output_path, StagedOutput::Kind::image_directory);
            try {
                // Python decode_image always saves PNG, independent of target suffix.
                const auto output_width = uf::image_crop_extent(data.at("image_width"), crop_width);
                const auto output_height = uf::image_crop_extent(data.at("image_height"), crop_height);
                auto rgb = uf::display_rgb(raw, width, height, output_width, output_height);
                uf::ImagePixels pixels{output_width, output_height, std::move(rgb)};
                uf::write_image_png(stage.temporary(), pixels);
                stage.publish();
            } catch (...) {
                stage.close_image_stage();
                throw;
            }
        }
        uf::event(write_output ? "decode_completed" : "verification_completed",
            {{"archive", input}, {"frames", stats.frames}, {"runtime", runtime}});
        return 0;
    });
}

uf::ScanResult scan_intra_payload(const uf::IntraImageArchive& archive) {
    return with_intra_payload(archive, [&](std::istream& stream) {
        const auto& video = archive.metadata.at("video");
        const auto scan = uf::scan_bitstream(stream, "hts", 1);
        if (scan.packets != 1 || scan.intra_packets != 1 ||
            !uf::json_python_equal(nlohmann::json(scan.dimensions.width), video.at("width")) ||
            !uf::json_python_equal(nlohmann::json(scan.dimensions.height), video.at("height")) ||
            scan.preamble.integer !=
                (archive.metadata.at("pipeline").value("runtime", std::string("fp16")) == "int16")) {
            throw std::runtime_error("UF intra image bitstream differs from metadata");
        }
        return scan;
    });
}

int run_archive_decode_values(const std::unordered_map<std::string, std::string>& values,
                              bool write_output) {
    uf::ProcessInterrupt::check();
    for (const auto& [key, _] : values) {
        if (key != "__input" && key != "__input_file" && key != "__output_file" &&
            key != "__prepared_i" && key != "__prepared_p" && key != "__device" &&
            key != "__model_path_i" && key != "__model_path_p" &&
            key != "__cuda_idx" && key != "__cuda" && key != "__runtime") {
            throw std::invalid_argument("Unknown option: " + key);
        }
    }
    const auto input = values.count("__input") ? values.at("__input") : required(values, "__input_file");
    if (lower_extension(input) == ".dcvci") {
        return run_intra_decode_values(values, write_output);
    }
    const auto archive = uf::load_archive(input);
    uf::ProcessInterrupt::check();
    const auto& data = archive.metadata;
    const auto& pipeline = data.at("pipeline");
    const bool image_only = pipeline.value("image_only", false);
    const auto runtime = pipeline.value("runtime", std::string("fp16"));
    auto models = archive_models(values, pipeline, image_only);
    uf::ProcessInterrupt::check();
    const auto device = codec_device(values, runtime);
    torch::NoGradGuard no_grad;
    uf::Codec codec(std::move(models), torch::Device(device), pipeline.value("skip_threshold", 0.0));
    const auto& metadata = data.at("video");
    // Writers must close/terminate before stage destruction removes their path.
    std::unique_ptr<StagedOutput> stage;
    std::unique_ptr<uf::FfmpegVideoWriter> writer;
    bool cleanup_attempted = false;
    const auto close_decode_resources = [&] {
        if (cleanup_attempted) return;
        cleanup_attempted = true;
        if (writer) writer->close_checked();
        if (stage) stage->close_video_stage();
    };
    try {
        if (write_output) {
            const auto path = uf::python_path_resolve(required(values, "__output_file"));
            stage = std::make_unique<StagedOutput>(path);
            const bool raw_yuv = lower_extension(path) == ".yuv";
            std::optional<std::filesystem::path> audio;
            if (!raw_yuv && data["artifacts"].contains("audio.opus")) {
                audio = uf::artifact_path(archive, "audio.opus");
            }
            writer = std::make_unique<uf::FfmpegVideoWriter>(
                stage->temporary(), metadata, audio,
                raw_yuv ? uf::FfmpegVideoWriter::Format::raw_yuv
                        : uf::FfmpegVideoWriter::Format::matroska);
        }
        const auto dimensions = uf::archive_video_dimensions(metadata);
        const auto result = codec.decode(uf::artifact_path(archive, "video.bin"),
                                         uf::python_json_exact_uint64(metadata.at("frames")),
                                         uf::python_json_exact_uint64(metadata.at("packets")),
                                         [&](const std::vector<std::uint8_t>& frame) {
            uf::ProcessInterrupt::check();
            if (writer) writer->write(frame);
        }, dimensions);
        uf::ProcessInterrupt::check();
        if (writer) writer->finish();
        const auto& validation = data["validation"];
        if (validation.contains("decoded_yuv_sha256") &&
            uf::json_truthy(validation.at("decoded_yuv_sha256")) &&
            validation.at("decoded_yuv_sha256") != result.decoded_yuv_sha256) {
            throw std::runtime_error("Decoded UF pixels differ from archive validation");
        }
        if (data["artifacts"].contains("audio.opus")) {
            uf::verify_opus(uf::artifact_path(archive, "audio.opus"));
        }
        if (stage) stage->publish();
        uf::event(write_output ? "decode_completed" : "verification_completed",
            {{"archive", uf::python_path_text(archive.root)}, {"decoded_frames", result.frames},
             {"decode_seconds", result.decode_seconds},
             {"decoded_yuv_sha256", result.decoded_yuv_sha256}});
        close_decode_resources();
        return 0;
    } catch (...) {
        close_decode_resources();
        throw;
    }
}

int run_archive_decode(int argc, char** argv, bool write_output) {
    uf::ProcessInterrupt interruption;
    return run_archive_decode_values(options(argc, argv), write_output);
}

int run_batch_decode(const std::unordered_map<std::string, std::string>& values) {
    for (const auto& [key, _] : values) {
        if (key != "__input" && key != "__input_file" &&
            key != "__input_folder" && key != "__output_folder" &&
            key != "__output_file" &&
            key != "__worker" && key != "__recursive" &&
            key != "__input_kind" && key != "__device" &&
            key != "__cuda_idx" && key != "__cuda" && key != "__runtime" &&
            key != "__prepared_i" && key != "__prepared_p" &&
            key != "__model_path_i" && key != "__model_path_p") {
            throw std::invalid_argument("Unknown batch decode option: " + key);
        }
    }
    if (!values.count("__output_folder")) {
        throw std::invalid_argument("Batch decode requires --output_folder");
    }
    const bool folder = nonempty_option(values, "__input_folder");
    const bool single = nonempty_option(values, "__input") || nonempty_option(values, "__input_file");
    if (folder == single) {
        throw std::invalid_argument("Select one of --input and --input_folder");
    }
    const auto kind = values.count("__input_kind") ? values.at("__input_kind") : "all";
    if (kind != "all" && kind != "videos" && kind != "images") {
        throw std::invalid_argument("--input_kind must be all, videos, or images");
    }
    if (values.count("__runtime") && values.at("__runtime") != "auto" &&
        values.at("__runtime") != "int16" && values.at("__runtime") != "fp16") {
        throw std::invalid_argument("--runtime must be auto, fp16, or int16 for decode");
    }
    const auto workers = values.count("__worker") ? integer_option(values.at("__worker"), "--worker") : 1;
    if (workers < 1) throw std::invalid_argument("--worker must be positive");
    const auto& output_folder = values.at("__output_folder");
    // Path('') denotes the current directory when batch input selects this
    // path; a merely present empty output option does not select batch mode.
    const auto output_root = std::filesystem::absolute(output_folder.empty() ? "." : output_folder);
    std::filesystem::path base;
    std::vector<std::filesystem::path> inputs;
    if (folder) {
        base = std::filesystem::canonical(values.at("__input_folder"));
        if (!std::filesystem::is_directory(base)) {
            throw std::invalid_argument("--input_folder must be a directory");
        }
        const bool recursive = values.count("__recursive") && values.at("__recursive") == "true";
        const auto add_input = [&](const std::filesystem::path& path) {
            uf::ProcessInterrupt::check();
            if (!std::filesystem::is_regular_file(path)) return;
            const auto relative = path.lexically_relative(base);
            for (const auto& part : relative) {
                if (!part.empty() && part.string()[0] == '.') return;
            }
            const auto extension = lower_extension(path);
            if (extension != ".bin" && extension != ".dcvci") return;
            if ((kind == "videos" && extension == ".dcvci") ||
                (kind == "images" && extension == ".bin")) return;
            inputs.push_back(path);
        };
        if (recursive) {
            for (const auto& entry : std::filesystem::recursive_directory_iterator(base)) {
                add_input(entry.path());
            }
        } else {
            for (const auto& entry : std::filesystem::directory_iterator(base)) {
                add_input(entry.path());
            }
        }
        std::sort(inputs.begin(), inputs.end(), uf::python_path_less);
    } else {
        inputs.push_back(std::filesystem::absolute(nonempty_option(values, "__input")
            ? values.at("__input") : values.at("__input_file")));
        base = inputs.front().parent_path();
    }
    const bool cpu = (values.count("__cuda") && values.at("__cuda") == "false") ||
        (values.count("__device") && values.at("__device") == "cpu");
    std::vector<std::string> devices;
    if (cpu) devices.push_back("cpu");
    else {
        const auto indexes = cuda_indices(values);
        if (indexes.empty()) {
            devices.push_back(values.count("__device") && values.at("__device").rfind("cuda:", 0) == 0
                ? values.at("__device") : "cuda:0");
        } else {
            for (const auto index : indexes) devices.push_back("cuda:" + std::to_string(index));
        }
    }
    auto decode_options = values;
    for (const auto* key : {"__input", "__input_file", "__input_folder", "__output_folder",
                             "__output_file", "__worker", "__recursive", "__input_kind", "__cuda_idx"}) {
        decode_options.erase(key);
    }
    std::vector<uf::DecodeJob> jobs;
    jobs.reserve(inputs.size());
    for (const auto& input : inputs) {
        uf::ProcessInterrupt::check();
        const bool image = lower_extension(input) == ".dcvci";
        if ((image && kind == "videos") || (!image && kind == "images")) continue;
        auto relative = input.lexically_relative(base);
        relative.replace_extension(image ? ".png" : ".yuv");
        jobs.push_back({input, output_root / relative, decode_options});
    }
    const auto report = uf::decode_jobs(jobs, workers, devices);
    std::cout << "batch_decoded=" << report.decoded << " failed=" << report.failed << '\n';
    return report.failed ? 1 : 0;
}

void write_cleanup_record(const std::filesystem::path& path,
                          const nlohmann::json& data) {
    uf::write_python_json_file_atomic(path, data);
}

void unlink_cleanup_source(const std::filesystem::path& path) {
    uf::ProcessInterrupt::check();
    if (path.native().find('\0') != std::string::npos)
        throw std::invalid_argument("embedded null byte");
    if (unlink(path.c_str()) != 0) {
        const int error = errno;
        throw std::system_error(error, std::generic_category(),
                                "Cannot unlink UF original: " + path.string());
    }
}

void sync_cleanup_directory(const std::filesystem::path& input) {
    const auto path = uf::python_path_normalize(input);
    if (path.native().find('\0') != std::string::npos)
        throw std::invalid_argument("embedded null byte");
    int descriptor;
    for (;;) {
        uf::ProcessInterrupt::check();
        descriptor = open(path.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
        if (descriptor >= 0) break;
        const int error = errno;
        if (error == EINTR) continue;
        throw std::system_error(error, std::generic_category(),
                                "Cannot open UF original directory: " + path.string());
    }
    uf::FileDescriptor directory(descriptor);
    const auto close_directory = [&] {
        const int owned = directory.release();
        if (close(owned) != 0) {
            const int error = errno;
            throw std::system_error(error, std::generic_category(),
                                    "Cannot close UF original directory: " + path.string());
        }
    };
    try {
        for (;;) {
            uf::ProcessInterrupt::check();
            if (fsync(directory.get()) == 0) break;
            const int error = errno;
            if (error == EINTR) continue;
            throw std::system_error(error, std::generic_category(),
                                    "Cannot sync UF original directory: " + path.string());
        }
        close_directory();
    } catch (...) {
        if (directory.get() >= 0) close_directory();
        throw;
    }
}

int run_image_cleanup_values(const std::unordered_map<std::string, std::string>& values) {
    const auto input = values.count("__input") ? values.at("__input") : required(values, "__input_file");
    const auto target = uf::python_path_normalize(input);
    const auto initial = uf::read_intra_image(target, true, false);
    uf::SharedWorkPool shared(uf::WorkKind::image, 900, "cleanup");
    shared.ensure_pipeline(target.parent_path(), initial.metadata.at("pipeline"));
    auto lease = shared.try_acquire(target.parent_path(), target.filename().string());
    if (!lease) throw std::runtime_error("Thumbnail owned by another instance");
    return uf::with_shared_lease(*lease, [&]() -> int {
        // cleanup_image reads the container again after acquiring ownership.
        const auto archive = uf::read_intra_image(target, true, false);
        const auto& data = archive.metadata;
        auto source = uf::python_path_normalize(uf::python_path_from_text(data.at("source").at("path").get_ref<const std::string&>()));
        if (values.count("__base_root") && !values.at("__base_root").empty()) {
            const auto base = uf::python_path_resolve(values.at("__base_root"));
            source = uf::python_path_normalize(
                base / uf::python_path_from_text(data.at("source").at("relative_path").get_ref<const std::string&>()));
            if (!uf::python_path_is_relative_to(uf::python_path_resolve(source), base))
                throw std::invalid_argument("Original path escapes base_root");
        }
        if (!uf::python_path_exists(source)) return 0;
        const auto before = uf::python_path_identity(source);
        if (uf::python_path_is_symlink(source) ||
            !uf::json_python_equal(uf::sha256_file(source), data.at("source").at("sha256")))
            throw std::invalid_argument("Original thumbnail changed; retaining it");
        if (!values.count("__yes")) {
            uf::event("cleanup_preview", {{"source", uf::python_path_text(source)},
                                         {"bytes", uf::python_path_size(source)}});
            return 0;
        }
        uf::ProcessInterrupt::check();
        run_intra_decode_values(values, false);
        uf::ProcessInterrupt::check();
        if (!(uf::python_path_identity(source) == before) ||
            !uf::json_python_equal(uf::sha256_file(source), data.at("source").at("sha256")))
            throw std::invalid_argument("Original thumbnail changed during verification");
        lease->check();
        unlink_cleanup_source(source);
        sync_cleanup_directory(source.parent_path());
        uf::event("original_deleted", {{"source", uf::python_path_text(source)}, {"archive", uf::python_path_text(target)}});
        return 0;
    });
}

int run_cleanup_values(const std::unordered_map<std::string, std::string>& values) {
    for (const auto& [key, _] : values) {
        if (key != "__input" && key != "__input_file" && key != "__base_root" &&
            key != "__yes" && key != "__prepared_i" && key != "__prepared_p" &&
            key != "__model_path_i" && key != "__model_path_p" &&
            key != "__device" && key != "__cuda_idx" &&
            key != "__cuda" && key != "__runtime") {
            throw std::invalid_argument("Unknown cleanup option: " + key);
        }
    }
    const auto input = values.count("__input") ? values.at("__input") : required(values, "__input_file");
    if (lower_extension(input) == ".dcvci") return run_image_cleanup_values(values);
    const auto archive = uf::load_archive(input);
    const auto& data = archive.metadata;
    const auto flat = uf::python_json_dict_get(data, "layout") == "flat";
    const auto stem = flat ? uf::python_filesystem_encode(data.at("archive_stem").get_ref<const std::string&>())
                           : archive.root.filename().string();
    const auto pool_root = flat ? archive.root : archive.root.parent_path();
    const nlohmann::json default_key(uf::python_filesystem_decode(stem + ".uf"));
    const auto lease_key = flat
        ? uf::python_filesystem_encode(uf::python_json_dict_get(data, "lease_key", default_key).get_ref<const std::string&>()) : stem;
    uf::SharedWorkPool shared(uf::WorkKind::video, 900, "cleanup");
    shared.ensure_pipeline(pool_root, data.at("pipeline"));
    auto lease = shared.try_acquire(pool_root, lease_key);
    if (!lease) throw std::runtime_error("Archive is owned by another worker; retry cleanup later");
    return uf::with_shared_lease(*lease, [&]() -> int {
        if (!uf::python_json_dict_get(data.at("pipeline"), "max_frames").is_null())
            throw std::invalid_argument("Partial archives cannot authorize original deletion");
        if (uf::json_truthy(data.at("source").at("media").at("audio")) &&
            !uf::python_json_string_in("audio.opus", data.at("artifacts")))
            throw std::invalid_argument("Archive omits source audio; refusing original deletion");
        if (uf::python_json_dict_get(data.at("source"), "kind") == "youtube")
            throw std::invalid_argument("Streamed archives have no stored original video to delete");
        std::filesystem::path source;
        if (values.count("__base_root") && !values.at("__base_root").empty()) {
            const auto base = uf::python_path_resolve(values.at("__base_root"));
            source = uf::python_path_normalize(
                base / uf::python_path_from_text(data.at("source").at("relative_path").get_ref<const std::string&>()));
            if (!uf::python_path_is_relative_to(uf::python_path_resolve(source), base))
                throw std::invalid_argument("Original path escapes base_root");
        } else {
            source = uf::python_path_normalize(uf::python_path_from_text(data.at("source").at("path").get_ref<const std::string&>()));
        }
        if (!uf::python_path_exists(source)) {
            uf::event("original_already_absent", {{"source", uf::python_path_text(source)}});
            return 0;
        }
        if (uf::python_path_is_symlink(source) || !uf::python_path_is_file(source) ||
            !uf::json_python_equal(uf::sha256_file(source), data.at("source").at("sha256")))
            throw std::invalid_argument("Original differs from the archived source; refusing deletion");
        if (!values.count("__yes")) {
            uf::event("cleanup_preview", {{"source", uf::python_path_text(source)},
                {"bytes", uf::python_path_size(source)},
                {"message", "Pass --yes to verify decoding and delete only this original video"}});
            return 0;
        }
        const auto before = uf::python_path_identity(source);
        auto verify_values = values;
        verify_values.erase("__base_root");
        verify_values.erase("__yes");
        uf::ProcessInterrupt::check();
        run_archive_decode_values(verify_values, false);
        uf::ProcessInterrupt::check();
        if (!(uf::python_path_identity(source) == before) ||
            !uf::json_python_equal(uf::sha256_file(source), data.at("source").at("sha256")))
            throw std::invalid_argument("Source changed during validation");
        const auto control = [&](const std::string& name) {
            return archive.root / (flat ? stem + "." + name : name);
        };
        const auto now_seconds = [] {
            return std::chrono::duration<double>(
                std::chrono::system_clock::now().time_since_epoch()).count();
        };
        lease->check();
        write_cleanup_record(control("cleanup-intent.json"),
            {{"source", uf::python_path_text(source)}, {"sha256", data.at("source").at("sha256")},
             {"requested_at", now_seconds()}});
        unlink_cleanup_source(source);
        sync_cleanup_directory(source.parent_path());
        write_cleanup_record(control("cleanup.json"),
            {{"source", uf::python_path_text(source)}, {"sha256", data.at("source").at("sha256")},
             {"deleted_at", now_seconds()}});
        uf::event("original_deleted", {{"source", uf::python_path_text(source)}, {"archive", uf::python_path_text(archive.root)}});
        return 0;
    });
}

int run_cleanup(int argc, char** argv) {
    uf::ProcessInterrupt interruption;
    return run_cleanup_values(options(argc, argv));
}

std::vector<std::string> list_option(const std::unordered_map<std::string, std::string>& values,
                                     const std::string& key) {
    return uf::cli::list(values, key);
}

using OptionValues = std::unordered_map<std::string, std::string>;

struct ValidationJob {
    std::filesystem::path final;
    std::optional<std::filesystem::path> legacy;
};

void check_cleanup_policy(const OptionValues& values) {
    if (values.count("__auto_delete") + values.count("__keep_originals") +
        values.count("__cleanup_dry_run") > 1) {
        throw std::invalid_argument("Select one cleanup policy");
    }
    if (values.count("__ui") && values.at("__ui") != "auto" &&
        values.at("__ui") != "plain" && values.at("__ui") != "tui") {
        throw std::invalid_argument("--ui must be auto, plain, or tui");
    }
}

OptionValues validation_options(const OptionValues& values,
                                const std::filesystem::path& archive) {
    OptionValues result = {{"__input", archive.string()}, {"__runtime", "auto"}};
    for (const auto* key : {"__prepared_i", "__prepared_p", "__model_path_i",
                           "__model_path_p", "__device", "__cuda"}) {
        if (values.count(key)) result[key] = values.at(key);
    }
    const auto indices = list_option(values, "__cuda_idx");
    if (!indices.empty()) result["__cuda_idx"] = indices.front();
    return result;
}

void finish_lifecycle(const OptionValues& values, uf::Dashboard& dashboard,
                      std::set<std::filesystem::path> blocked = {},
                      const std::optional<std::set<std::filesystem::path>>& sources = std::nullopt,
                      const std::vector<ValidationJob>& validation_jobs = {}) {
    const auto output = std::filesystem::path(required(values, "__output_root"));
    const auto channel_ids = list_option(values, "__channel_ids");
    const auto base = values.count("__base_root")
        ? std::optional<std::filesystem::path>(values.at("__base_root")) : std::nullopt;
    std::set<std::filesystem::path> normalized, kept;
    for (const auto& path : blocked) normalized.insert(uf::python_path_resolve(path));
    blocked = std::move(normalized);
    const auto refresh = [&]() {
        auto inventory = uf::inventory_channels(output, channel_ids, base,
            values.count("__allow_missing_metadata") != 0, sources);
        for (const auto& error : inventory.errors) {
            uf::event("inventory_error", {{"manifest", uf::python_path_text(error.manifest)}, {"error", error.message}});
        }
        nlohmann::json rows = nlohmann::json::array();
        for (auto& row : inventory.channels) {
            if (kept.count(row.channel)) {
                row.ineligible += row.originals;
                row.originals = 0;
            }
            nlohmann::json archives = nlohmann::json::array();
            for (const auto& item : row.archives) {
                archives.push_back({{"input", uf::python_path_text(item.manifest)}, {"source", uf::python_path_text(item.source)},
                    {"eligible", item.eligible}, {"remote", item.remote}, {"exists", item.exists},
                    {"kind", item.kind}, {"metadata", item.metadata}});
            }
            std::string error;
            for (const auto& reason : row.blockers) {
                if (!error.empty()) error += "; ";
                error += reason;
            }
            rows.push_back({{"channel", uf::python_path_text(row.channel)}, {"archives", archives},
                {"originals", row.originals}, {"bytes", row.bytes}, {"streamed", row.streamed},
                {"absent", row.absent}, {"ineligible", row.ineligible},
                {"blocked", blocked.count(row.channel) != 0 || !row.blockers.empty()},
                {"blockers", row.blockers}, {"last_error", error}});
        }
        uf::event("lifecycle_inventory", {{"channels", rows}});
        return inventory.channels;
    };
    const auto is_blocked = [&](const uf::ChannelInventory& row) {
        return blocked.count(row.channel) || !row.blockers.empty();
    };
    const auto cleanup = [&](const uf::ChannelInventory& row) {
        uf::event("channel_deletion_started", {{"channel", uf::python_path_text(row.channel)},
            {"originals", row.originals}, {"bytes", row.bytes}});
        std::size_t failures = 0;
        for (const auto& item : row.archives) {
            uf::ProcessInterrupt::check();
            if (!item.eligible) continue;
            auto options = validation_options(values, item.manifest);
            if (base) options["__base_root"] = base->string();
            if (!values.count("__cleanup_dry_run")) options["__yes"] = "true";
            uf::event("original_validation_started", {{"channel", uf::python_path_text(row.channel)},
                {"source", uf::python_path_text(item.source)}});
            try {
                run_cleanup_values(options);
            } catch (const std::exception& error) {
                uf::ProcessInterrupt::check();
                ++failures;
                uf::event("cleanup_failed", {{"channel", uf::python_path_text(row.channel)},
                    {"source", uf::python_path_text(item.source)}, {"error", error.what()}});
            }
        }
        uf::event("channel_deletion_finished", {{"channel", uf::python_path_text(row.channel)},
            {"failed", failures}, {"dry_run", values.count("__cleanup_dry_run") != 0}});
    };
    const auto revalidate = [&](const std::filesystem::path& channel) {
        std::vector<ValidationJob> jobs;
        for (const auto& job : validation_jobs) {
            if (uf::python_path_normalize(job.final.parent_path()).native() == channel.native())
                jobs.push_back(job);
        }
        if (jobs.empty()) {
            uf::event("validation_failed", {{"channel", uf::python_path_text(channel)},
                {"message", "No completed-job evidence; rerun encoding to recover missing work"}});
            return false;
        }
        try {
            for (const auto& job : jobs) {
                uf::ProcessInterrupt::check();
                const auto final = !uf::python_path_exists(job.final) && job.legacy ? *job.legacy : job.final;
                uf::load_archive(final);
                uf::event("channel_validation_started", {{"channel", uf::python_path_text(channel)}, {"archive", uf::python_path_text(final)}});
                run_archive_decode_values(validation_options(values, final), false);
            }
            uf::event("channel_validation_completed", {{"channel", uf::python_path_text(channel)}});
            return true;
        } catch (const std::exception& error) {
            uf::ProcessInterrupt::check();
            uf::event("validation_failed", {{"channel", uf::python_path_text(channel)}, {"error", error.what()}});
            return false;
        }
    };
    auto rows = refresh();
    if (values.count("__auto_delete") || values.count("__cleanup_dry_run")) {
        for (const auto& row : rows) {
            uf::ProcessInterrupt::check();
            if (row.originals && !is_blocked(row)) cleanup(row);
        }
        refresh();
        return;
    }
    if (!dashboard.interactive() || values.count("__keep_originals")) {
        for (const auto& row : rows) if (row.originals) {
            uf::event("cleanup_available", {{"channel", uf::python_path_text(row.channel)},
                {"originals", row.originals}, {"blocked", is_blocked(row)},
                {"message", "Use --ui tui to review deletion, or --auto-delete for verified automatic cleanup"}});
        }
        return;
    }
    uf::event("lifecycle_waiting", {{"message",
        "Approvals: d reviews deletion, y confirms, k keeps originals, r retries validation, x exits after work"}});
    auto refreshed = std::chrono::steady_clock::now();
    while (dashboard.interactive()) {
        uf::ProcessInterrupt::check();
        for (const auto& action : dashboard.poll_actions()) {
            if (action.kind == "exit-approvals") return;
            if (!action.channel) continue;
            const auto channel = uf::python_path_from_text(*action.channel);
            // Actions can remain queued while encoding or validation runs. Use
            // a fresh inventory and require a channel inside the selected roots.
            rows = refresh();
            const auto found = std::find_if(rows.begin(), rows.end(), [&](const auto& row) {
                return row.channel.native() == channel.native();
            });
            if (found == rows.end()) continue;
            if (action.kind == "keep") kept.insert(channel);
            else if (action.kind == "retry") {
                if (revalidate(channel)) blocked.erase(channel);
            } else if (action.kind == "approve") {
                if (is_blocked(*found)) {
                    uf::event("cleanup_blocked", {{"channel", uf::python_path_text(channel)},
                        {"message", "This channel has failed, busy, or incomplete jobs; retain originals"}});
                } else if (found->originals) cleanup(*found);
            }
            rows = refresh();
            refreshed = std::chrono::steady_clock::now();
        }
        if (std::chrono::steady_clock::now() - refreshed >= std::chrono::seconds(5)) {
            rows = refresh();
            refreshed = std::chrono::steady_clock::now();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    uf::ProcessInterrupt::check();
}

int run_manage(int argc, char** argv) {
    uf::ProcessInterrupt interruption;
    const auto values = options(argc, argv);
    for (const auto& [key, _] : values) {
        if (key != "__output_root" && key != "__base_root" &&
            key != "__channel_ids" && key != "__allow_missing_metadata" &&
            key != "__auto_delete" && key != "__keep_originals" &&
            key != "__cleanup_dry_run" && key != "__ui" &&
            key != "__prepared_i" && key != "__prepared_p" &&
            key != "__model_path_i" && key != "__model_path_p" &&
            key != "__device" && key != "__cuda_idx" && key != "__cuda") {
            throw std::invalid_argument("Unknown manage option: " + key);
        }
    }
    check_cleanup_policy(values);
    uf::Dashboard dashboard(required(values, "__output_root"), 1,
        values.count("__ui") ? values.at("__ui") : "auto");
    return dashboard_backend([&] {
        finish_lifecycle(values, dashboard);
        return 0;
    });
}

int run_raw_encode(int argc, char** argv) {
    const auto values = options(argc, argv);
    for (const auto& [key, _] : values) {
        if (key != "__input_file" && key != "__output_file" && key != "__width" &&
            key != "__height" && key != "__prepared_i" && key != "__prepared_p" &&
            key != "__variant" && key != "__qp_i" && key != "__qp_p" &&
            key != "__reset_interval" && key != "__intra_period" &&
            key != "__max_frames" && key != "__skip_thres" && key != "__device" &&
            key != "__runtime" && key != "__model_path_i" && key != "__model_path_p" &&
            key != "__cuda" && key != "__cuda_idx") {
            throw std::invalid_argument("Unknown option: " + key);
        }
    }
    const auto variant = values.count("__variant") ? values.at("__variant") : "hts";
    const auto dimension = [&](const std::string& name) {
        const auto value = uf::cli::unsigned_integer(required(values, name), name);
        if (!value || value > std::numeric_limits<std::uint32_t>::max()) {
            throw std::invalid_argument(name + " must fit a positive uint32 dimension");
        }
        return static_cast<std::uint32_t>(value);
    };
    const auto width = dimension("__width"), height = dimension("__height");
    const auto qp_i = values.count("__qp_i") ? integer_option(values.at("__qp_i"), "--qp_i") : 36;
    const auto qp_p = values.count("__qp_p") ? integer_option(values.at("__qp_p"), "--qp_p") : 30;
    const auto reset = values.count("__reset_interval") ? integer_option(values.at("__reset_interval"), "--reset_interval") : 32;
    const auto intra = values.count("__intra_period") ? integer_option(values.at("__intra_period"), "--intra_period") : -1;
    const auto skip = values.count("__skip_thres") ? uf::cli::number(values.at("__skip_thres"), "--skip_thres") : 0.0;
    const auto runtime = encode_runtime(values);
    const auto device = codec_device(values, runtime);
    const auto image_checkpoint = values.count("__model_path_i")
        ? std::filesystem::path(values.at("__model_path_i"))
        : checkpoint_root() / "cvpr2026_image.pth.tar";
    const auto video_checkpoint = values.count("__model_path_p")
        ? std::filesystem::path(values.at("__model_path_p"))
        : checkpoint_root() / ("cvpr2026_video_" + variant + ".pth.tar");
    const auto image_path = runtime == "int16"
        ? uf::prepared_artifact_path(image_checkpoint, "image", values.count("__prepared_i")
            ? std::optional<std::filesystem::path>(values.at("__prepared_i")) : std::nullopt)
        : image_checkpoint;
    const auto video_path = runtime == "int16"
        ? uf::prepared_artifact_path(video_checkpoint, variant, values.count("__prepared_p")
            ? std::optional<std::filesystem::path>(values.at("__prepared_p")) : std::nullopt)
        : video_checkpoint;
    auto models = uf::CodecModels::load(runtime, image_path, video_path, variant);
    uf::RawYuv420Reader reader(required(values, "__input_file"), width, height);
    try {
        uf::Codec codec(std::move(models), torch::Device(device), skip);
        std::optional<std::uint64_t> maximum;
        if (values.count("__max_frames")) {
            maximum = uf::cli::unsigned_integer(values.at("__max_frames"), "--max_frames");
            if (!*maximum) throw std::invalid_argument("--max_frames must be positive");
        }
        StagedOutput stage(required(values, "__output_file"), StagedOutput::Kind::bitstream_directory);
        torch::NoGradGuard no_grad;
        const auto result = codec.encode([&] { return reader.read(); }, stage.temporary(),
                                          width, height, qp_i, qp_p, reset, intra, maximum);
        reader.close_checked();
        stage.publish();
        std::cout << "frames=" << result.frames << " packets=" << result.packets << '\n';
        if (!std::cout) throw std::runtime_error("Cannot report raw UF encode statistics");
        return 0;
    } catch (...) {
        reader.close_checked();
        throw;
    }
}

bool image_source(const std::filesystem::path& path) {
    auto extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    static const std::vector<std::string> extensions = {
        ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".avif"};
    return std::find(extensions.begin(), extensions.end(), extension) != extensions.end();
}

bool video_source(const std::filesystem::path& path) {
    auto extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    static const std::vector<std::string> extensions = {
        ".mkv", ".mp4", ".webm", ".mov", ".avi", ".m4v", ".ts"};
    return std::find(extensions.begin(), extensions.end(), extension) != extensions.end();
}

uf::ImageRun run_image_encode_values(const std::unordered_map<std::string, std::string>& values) {
    if (values.count("__thumbnail_codec") &&
        values.at("__thumbnail_codec") != "dcvc-intra") {
        throw std::invalid_argument("Image inputs require --thumbnail_codec dcvc-intra");
    }
    uf::ImageEncodeOptions config;
    config.runtime = encode_runtime(values);
    config.source = required(values, "__input_file");
    if (!image_source(config.source)) {
        throw std::invalid_argument("Unsupported UF intra image source extension");
    }
    config.qp = values.count("__thumbnail_qp")
        ? integer_option(values.at("__thumbnail_qp"), "--thumbnail_qp") : 45;
    config.skip_threshold = values.count("__skip_thres")
        ? uf::cli::number(values.at("__skip_thres"), "--skip_thres") : 0.0;
    if (values.count("__shared_instance")) {
        config.shared_instance = values.at("__shared_instance");
    }
    if (values.count("__shared_lease_seconds")) {
        config.shared_lease_seconds = uf::cli::number(values.at("__shared_lease_seconds"), "--shared-lease-seconds");
    }
    config.device = codec_device(values, config.runtime);
    const auto source = uf::python_path_absolute(config.source);
    config.source_base = values.count("__base_root") && !values.at("__base_root").empty()
        ? uf::python_path_resolve(values.at("__base_root"))
        : uf::python_path_normalize(source.parent_path());
    const auto relative = uf::python_path_relative_to(source, *config.source_base);
    config.target = values.count("__output_file")
        ? std::filesystem::path(values.at("__output_file"))
        : uf::python_path_resolve(required(values, "__output_root")) /
          relative.parent_path() /
          (source.filename().string() + "_qI" + std::to_string(config.qp) + ".dcvci");
    if (config.target.extension() != ".dcvci") {
        throw std::invalid_argument("UF intra image output must end in .dcvci");
    }
    const auto checkpoint = values.count("__model_path_i")
        ? std::filesystem::path(values.at("__model_path_i"))
        : checkpoint_root() / "cvpr2026_image.pth.tar";
    config.checkpoint_image = checkpoint;
    if (config.runtime == "int16") {
        config.prepared_image = uf::prepared_artifact_path(checkpoint, "image", values.count("__prepared_i")
            ? std::optional<std::filesystem::path>(values.at("__prepared_i")) : std::nullopt);
    }
    return uf::encode_image_pass({config});
}

std::vector<std::filesystem::path> folder_sources(
    const std::filesystem::path& base_root,
    const std::vector<std::string>& channel_ids,
    bool recursive, bool images) {
    const auto base = uf::python_path_resolve(base_root);
    std::vector<std::filesystem::path> roots;
    if (channel_ids.empty()) roots.push_back(base);
    else {
        for (const auto& id : channel_ids) {
            const auto name = std::filesystem::path(id);
            roots.push_back(uf::python_path_normalize(base / name));
        }
    }
    for (const auto& root : roots) {
        if (!uf::python_path_is_relative_to(uf::python_path_resolve(root), base) ||
            (!images && !uf::python_path_is_directory(root))) {
            throw std::invalid_argument(images ? "Channel escapes base_root"
                : "Invalid channel/input directory: " + root.string());
        }
    }
    std::set<std::filesystem::path, decltype(&uf::python_path_less)> sources(uf::python_path_less);
    const auto add = [&](const std::filesystem::path& path) {
        if (!uf::python_path_is_file(path) || uf::python_path_is_symlink(path) ||
            !(images ? image_source(path) : video_source(path))) return;
        const auto relative = uf::python_path_relative_to(path, base);
        for (const auto& part : relative) {
            const auto name = part.string();
            if (!name.empty() && (name[0] == '.' ||
                (name.size() >= 3 && name.compare(name.size() - 3, 3, ".uf") == 0))) return;
        }
        sources.insert(path);
    };
    for (const auto& root : roots) {
        if (recursive) {
            if (!uf::python_path_is_directory(root)) continue;
            for (const auto& entry : std::filesystem::recursive_directory_iterator(
                    root, std::filesystem::directory_options::skip_permission_denied)) {
                add(entry.path());
            }
        } else {
            for (const auto& entry : std::filesystem::directory_iterator(root)) {
                add(entry.path());
            }
        }
    }
    return {sources.begin(), sources.end()};
}

int run_image_encode(int argc, char** argv) {
    uf::ProcessInterrupt interruption;
    const auto values = options(argc, argv);
    for (const auto& [key, _] : values) {
        if (key != "__input_file" && key != "__output_file" &&
            key != "__output_root" && key != "__thumbnail_qp" &&
            key != "__thumbnail_codec" && key != "__prepared_i" &&
            key != "__model_path_i" && key != "__base_root" && key != "__device" &&
            key != "__cuda" && key != "__cuda_idx" &&
            key != "__runtime" && key != "__skip_thres" &&
            key != "__shared_instance" && key != "__shared_lease_seconds" &&
            key != "__shared_work") {
            throw std::invalid_argument("Unknown intra image encode option: " + key);
        }
    }
    const auto report = run_image_encode_values(values);
    return report.failed || report.busy || report.worker_failed ? 1 : 0;
}

int run_local_encode(int argc, char** argv) {
    uf::ProcessInterrupt interruption;
    auto values = options(argc, argv);
    // Apply the benchmark before discovery and the early single-image path,
    // as Python main() does before dispatching encode/stream to the dashboard.
    std::optional<nlohmann::json> target_selection;
    try {
        target_selection = apply_target_psnr_options(values);
    } catch (const std::invalid_argument& error) {
        throw uf::cli::ParseError(error.what());
    } catch (const std::out_of_range& error) {
        throw uf::cli::ParseError(error.what());
    }
    // Python adds Twitch names to a local mount's channel selection. YouTube
    // channels, or Twitch without a mount, select remote streaming instead.
    if (values.count("__youtube_channels") ||
        (values.count("__twitch_channels") && !values.count("__base_root"))) {
        if (values.count("__input_file") || values.count("__base_root")) {
            throw uf::cli::ParseError("Remote channel inputs cannot be combined with local input");
        }
        uf::RemoteOptions expansion;
        expansion.source_urls = list_option(values, "__source_urls");
        expansion.youtube_channels = list_option(values, "__youtube_channels");
        expansion.twitch_channels = list_option(values, "__twitch_channels");
        auto urls = uf::remote_source_urls(expansion);
        for (auto& url : urls) url = uf::python_filesystem_encode(url);
        uf::cli::set_list(values, "__source_urls", urls);
        // Pass the expanded URLs onwards; the producer must not reread files.
        values.erase("__youtube_channels");
        values.erase("__twitch_channels");
    } else if (values.count("__base_root") && values.count("__twitch_channels")) {
        auto channels = list_option(values, "__channel_ids");
        const auto twitch = list_option(values, "__twitch_channels");
        channels.insert(channels.end(), twitch.begin(), twitch.end());
        uf::cli::set_list(values, "__channel_ids", channels);
        values.erase("__twitch_channels");
    }
    uf::cli::validate_encode_options(values);
    const bool remote_input = values.count("__source_urls") ||
        values.count("__youtube_channels") || values.count("__twitch_channels");
    if ((values.count("__input_file") || values.count("__base_root")) && remote_input) {
        throw std::invalid_argument("Remote URLs/channels cannot be combined with local input");
    }
    for (const auto& [key, _] : values) {
        if (key != "__input_file" && key != "__output_root" &&
            key != "__base_root" && key != "__channel_ids" &&
            key != "__recursive" && key != "__procs" &&
            key != "__prepared_i" && key != "__prepared_p" &&
            key != "__model_path_i" && key != "__model_path_p" &&
            key != "__model_structure" && key != "__runtime" &&
            key != "__device" && key != "__qp_i" && key != "__qp_p" &&
            key != "__cuda" && key != "__cuda_idx" &&
            key != "__fps" &&
            key != "__resolution" && key != "__reset_interval" &&
            key != "__intra_period" &&
            key != "__max_frames" && key != "__skip_thres" &&
            key != "__audio" && key != "__opus_channels" &&
            key != "__opus_bitrate" && key != "__opus_frame_ms" &&
            key != "__opus_complexity" && key != "__opus_vbr" &&
            key != "__input_threads" && key != "__prefetch_frames" &&
            key != "__ff_hwaccel" && key != "__target_psnr") {
            if (key == "__source_urls" || key == "__youtube_channels" ||
                key == "__twitch_channels" || key == "__cookies" ||
                key == "__ytdlp_bin" ||
                key == "__source_max_height" ||
                key == "__max_videos" || key == "__no_thumbnail") continue;
            if (key == "__thumbnail_codec" || key == "__thumbnail_qp" ||
                key == "__shared_instance" || key == "__shared_lease_seconds" ||
                key == "__shared_work") continue;
            if (key == "__ui" || key == "__auto_delete" || key == "__keep_originals" ||
                key == "__cleanup_dry_run" || key == "__allow_missing_metadata") continue;
            throw std::invalid_argument("Unknown option: " + key);
        }
    }
    check_cleanup_policy(values);
    const int workers = values.count("__procs") ? integer_option(values.at("__procs"), "--procs") : 1;
    if (workers < 1) throw std::invalid_argument("--procs must be positive");
    uf::Dashboard dashboard(required(values, "__output_root"), workers,
        values.count("__ui") ? values.at("__ui") : "auto");
    return dashboard_backend([&]() -> int {
        if (target_selection) uf::event("target_psnr_selected", *target_selection);
        if (values.count("__input_file") && image_source(values.at("__input_file"))) {
            if (!values.count("__thumbnail_codec") ||
                values.at("__thumbnail_codec") != "dcvc-intra") {
                throw std::invalid_argument("Image inputs require --thumbnail_codec dcvc-intra");
            }
            uf::event("run_discovered", {{"total", 1}});
            const auto report = run_image_encode_values(values);
            const std::set<std::filesystem::path> sources = {
                uf::python_path_absolute(values.at("__input_file"))};
            finish_lifecycle(values, dashboard, report.failed_channels, sources);
            return report.failed || report.busy || report.worker_failed ? 1 : 0;
        }
        std::optional<std::string> remote_binary;
        if (remote_input) {
            if (values.count("__max_frames")) {
                if (!uf::cli::unsigned_integer(values.at("__max_frames"), "--max_frames")) {
                    throw std::invalid_argument("--max_frames must be positive");
                }
                throw std::invalid_argument("Streaming requires full videos; omit --max_frames");
            }
            remote_binary = uf::resolve_ytdlp_binary(values.count("__ytdlp_bin")
                ? values.at("__ytdlp_bin") : "yt-dlp");
        }
        if (values.count("__thumbnail_codec") &&
            values.at("__thumbnail_codec") != "keep" &&
            values.at("__thumbnail_codec") != "dcvc-intra") {
            throw std::invalid_argument("--thumbnail_codec must be keep or dcvc-intra");
        }
        if (values.count("__thumbnail_qp") &&
            (integer_option(values.at("__thumbnail_qp"), "--thumbnail_qp") < 0 ||
             integer_option(values.at("__thumbnail_qp"), "--thumbnail_qp") >= 64)) {
            throw std::invalid_argument("--thumbnail_qp must be between 0 and 63");
        }
        uf::EncodeOptions config;
        config.runtime = encode_runtime(values);
        config.device = codec_device(values, config.runtime);
        config.workers = workers;
        config.devices = worker_devices(values, config.device);
        config.prefetch_frames = config.runtime == "int16" ? 8 : 0;
        cpu_set_t allowed_cpus;
        if (config.runtime == "int16") {
            const long available = sched_getaffinity(0, sizeof(allowed_cpus), &allowed_cpus) == 0
                ? CPU_COUNT(&allowed_cpus) : sysconf(_SC_NPROCESSORS_ONLN);
            const int count = static_cast<int>(std::max<long>(1, available));
            config.input_threads = std::max(1, std::min(4, count / config.workers - 1));
        }
        const bool folder = values.count("__base_root") != 0;
        const bool single = values.count("__input_file") != 0;
        const bool remote = remote_input;
        if (static_cast<int>(folder) + static_cast<int>(single) +
            static_cast<int>(remote) != 1) {
            throw std::invalid_argument("Select one of local input, folder, or remote URLs/channels");
        }
        if (single) config.input_file = values.at("__input_file");
        const auto channels = list_option(values, "__channel_ids");
        const bool recursive = !values.count("__recursive") || values.at("__recursive") == "true";
        config.output_root = required(values, "__output_root");
        if (values.count("__model_structure")) config.variant = values.at("__model_structure");
        if (values.count("__qp_i")) {
            config.qp_i = integer_option(values.at("__qp_i"), "--qp_i");
        }
        if (values.count("__qp_p")) {
            config.qp_p = integer_option(values.at("__qp_p"), "--qp_p");
        }
        if (values.count("__fps")) config.fps = uf::cli::number(values.at("__fps"), "--fps");
        if (values.count("__resolution")) config.resolution = integer_option(values.at("__resolution"), "--resolution");
        if (values.count("__reset_interval")) config.reset_interval = integer_option(values.at("__reset_interval"), "--reset_interval");
        if (values.count("__intra_period")) {
            config.intra_period = integer_option(values.at("__intra_period"), "--intra_period");
        }
        if (values.count("__max_frames")) {
            const auto& text = values.at("__max_frames");
            const auto count = uf::cli::unsigned_integer(text, "--max_frames");
            if (!count) throw std::invalid_argument("--max_frames must be positive");
            config.max_frames = count;
        }
        if (values.count("__skip_thres")) {
            config.skip_threshold = uf::cli::number(values.at("__skip_thres"), "--skip_thres");
            if (!std::isfinite(config.skip_threshold) || config.skip_threshold < 0 ||
                config.skip_threshold > 16) {
                throw std::invalid_argument("--skip_thres must be between 0 and 16");
            }
        }
        if (values.count("__audio")) config.audio = values.at("__audio");
        if (values.count("__opus_channels")) config.opus_channels = values.at("__opus_channels");
        if (values.count("__opus_bitrate")) config.opus_bitrate = values.at("__opus_bitrate");
        if (values.count("__opus_frame_ms")) config.opus_frame_ms = uf::cli::number(values.at("__opus_frame_ms"), "--opus_frame_ms");
        if (values.count("__opus_complexity")) config.opus_complexity = integer_option(values.at("__opus_complexity"), "--opus_complexity");
        if (values.count("__opus_vbr")) config.opus_vbr = values.at("__opus_vbr");
        if (values.count("__input_threads")) {
            config.input_threads = integer_option(values.at("__input_threads"), "--input_threads");
        }
        if (config.input_threads < 1) throw std::invalid_argument("--input_threads must be positive");
        if (values.count("__prefetch_frames")) {
            config.prefetch_frames = integer_option(values.at("__prefetch_frames"), "--prefetch_frames");
        }
        if (config.prefetch_frames < 0) throw std::invalid_argument("--prefetch_frames must be nonnegative");
        if (values.count("__ff_hwaccel")) config.hwaccel = values.at("__ff_hwaccel");
        if (config.hwaccel != "none" && config.hwaccel != "auto" && config.hwaccel != "jetson") {
            throw std::invalid_argument("--ff_hwaccel must be none, auto or jetson");
        }
        if (values.count("__shared_instance")) config.shared_instance = values.at("__shared_instance");
        if (values.count("__shared_lease_seconds")) {
            config.shared_lease_seconds = uf::cli::number(values.at("__shared_lease_seconds"), "--shared-lease-seconds");
        }
        const auto image_checkpoint = values.count("__model_path_i")
            ? std::filesystem::path(values.at("__model_path_i"))
            : checkpoint_root() / "cvpr2026_image.pth.tar";
        const auto video_checkpoint = values.count("__model_path_p")
            ? std::filesystem::path(values.at("__model_path_p"))
            : checkpoint_root() / ("cvpr2026_video_" + config.variant + ".pth.tar");
        config.checkpoint_i = image_checkpoint;
        config.checkpoint_p = video_checkpoint;
        const bool video_jobs = !folder ||
            !folder_sources(values.at("__base_root"), channels, recursive, false).empty();
        if (config.runtime == "int16") {
            config.prepared_i = uf::prepared_artifact_path(image_checkpoint, "image", values.count("__prepared_i")
                ? std::optional<std::filesystem::path>(values.at("__prepared_i")) : std::nullopt);
            config.prepared_p = video_jobs
                ? uf::prepared_artifact_path(video_checkpoint, config.variant, values.count("__prepared_p")
                    ? std::optional<std::filesystem::path>(values.at("__prepared_p")) : std::nullopt)
                : video_checkpoint;
        }
        if (remote) {
            uf::RemoteOptions source_options;
            source_options.pipeline = uf::encode_pipeline(config);
            source_options.source_urls = list_option(values, "__source_urls");
            source_options.youtube_channels = list_option(values, "__youtube_channels");
            source_options.twitch_channels = list_option(values, "__twitch_channels");
            source_options.output_root = config.output_root;
            if (values.count("__cookies")) source_options.cookies = values.at("__cookies");
            source_options.ytdlp_binary = *remote_binary;
            if (values.count("__source_max_height")) {
                source_options.max_height = integer_option(values.at("__source_max_height"), "--source-max-height");
            }
            if (values.count("__max_videos")) {
                source_options.max_videos = integer_option(values.at("__max_videos"), "--max-videos");
            }
            source_options.write_thumbnail = !values.count("__no_thumbnail");
            if (values.count("__thumbnail_codec")) {
                source_options.thumbnail_codec = values.at("__thumbnail_codec");
            }
            if (values.count("__thumbnail_qp")) {
                source_options.thumbnail_qp = integer_option(values.at("__thumbnail_qp"), "--thumbnail_qp");
            }
            uf::RemoteAheadJobs producer(std::move(source_options));
            uf::EncodeScheduler scheduler(config.workers, config.devices, true);
            std::optional<uf::RemoteJob> pending;
            std::shared_ptr<uf::PreparedDownload> warm;
            bool exhausted = false;
            while (!exhausted || pending || scheduler.running()) {
                scheduler.poll();
                if (!pending && !exhausted) {
                    auto next = producer.poll();
                    exhausted = next.finished;
                    pending = std::move(next.job);
                }
                if (pending && !warm && (scheduler.has_capacity() || scheduler.near_end()) &&
                    !std::filesystem::exists(pending->manifest) &&
                    !std::filesystem::exists(pending->manifest.parent_path() / (pending->video_id + ".uf"))) {
                    warm = std::make_shared<uf::PreparedDownload>(*pending);
                    uf::event(scheduler.has_capacity() ? "stream_download_started" : "stream_next_prefetched",
                        {{"source", pending->url}});
                }
                if (pending && scheduler.has_capacity()) {
                    auto job = config;
                    job.remote = std::move(pending);
                    pending.reset();
                    job.relative_path = job.remote->channel_id + "/" + job.remote->video_id;
                    scheduler.submit(job, std::move(warm));
                }
                if (!exhausted || pending || scheduler.running()) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(20));
                }
            }
            const auto discovery = producer.finish();
            const auto report = scheduler.finish();
            const auto failed = static_cast<std::size_t>(discovery.failed) + report.failed;
            uf::event("stream_finished", {{"encoded", report.encoded}, {"resumed", discovery.resumed + report.resumed},
                {"busy", report.busy}, {"failed", failed}, {"skipped_live", discovery.skipped_live}});
            finish_lifecycle(values, dashboard, report.failed_channels,
                std::set<std::filesystem::path>{});
            return failed ? 1 : 0;
        } else if (folder) {
            const auto jobs = uf::discover_local_jobs(values.at("__base_root"), config.output_root,
                channels, recursive, config);
            const auto images = values.count("__thumbnail_codec") &&
                values.at("__thumbnail_codec") == "dcvc-intra"
                ? folder_sources(values.at("__base_root"), channels, recursive, true)
                : std::vector<std::filesystem::path>{};
            std::set<std::filesystem::path> sources;
            std::vector<ValidationJob> validation_jobs;
            for (const auto& job : jobs) {
                sources.insert(uf::python_path_absolute(job.input_file));
                const auto root = uf::python_path_absolute(job.output_root);
                const auto stem = job.input_file.stem().string();
                validation_jobs.push_back({root / (stem + ".uf.json"),
                    root / (job.input_file.filename().string() + ".uf")});
            }
            for (const auto& source : images)
                sources.insert(uf::python_path_absolute(source));
            uf::event("run_discovered", {{"total", jobs.size() + images.size()}});
            uf::EncodeScheduler scheduler(config.workers, config.devices);
            for (const auto& job : jobs) scheduler.submit(job);
            auto report = scheduler.finish();
            uf::ImageRun image_report;
            if (!images.empty()) {
                std::vector<uf::ImageEncodeOptions> image_jobs;
                image_jobs.reserve(images.size());
                uf::ImageEncodeOptions image;
                image.source_base = uf::python_path_resolve(values.at("__base_root"));
                image.prepared_image = config.prepared_i;
                image.checkpoint_image = config.checkpoint_i;
                image.runtime = config.runtime;
                image.device = config.devices.empty() ? config.device : config.devices.front();
                image.qp = values.count("__thumbnail_qp")
                    ? integer_option(values.at("__thumbnail_qp"), "--thumbnail_qp") : 45;
                image.skip_threshold = config.skip_threshold;
                image.shared_instance = config.shared_instance;
                image.shared_lease_seconds = config.shared_lease_seconds;
                for (const auto& source : images) {
                    uf::ProcessInterrupt::check();
                    image.source = source;
                    const auto relative = uf::python_path_relative_to(source, *image.source_base);
                    image.target = uf::python_path_resolve(config.output_root) /
                        relative.parent_path() /
                        (source.filename().string() + "_qI" + std::to_string(image.qp) + ".dcvci");
                    image_jobs.push_back(image);
                }
                image_report = uf::encode_image_pass(image_jobs);
                report.failed_channels.insert(image_report.failed_channels.begin(),
                                              image_report.failed_channels.end());
            }
            uf::event("archive_run_completed", {{"encoded", report.encoded}, {"resumed", report.resumed},
                {"busy", report.busy}, {"failed", report.failed}, {"images", images.size()},
                {"image_encoded", image_report.encoded}, {"image_resumed", image_report.resumed},
                {"image_busy", image_report.busy}, {"image_failed", image_report.failed},
                {"image_worker_failed", image_report.worker_failed}});
            finish_lifecycle(values, dashboard, report.failed_channels, sources, validation_jobs);
            return report.failed || image_report.failed || image_report.busy ||
                image_report.worker_failed ? 1 : 0;
        } else {
            config.input_file = uf::python_path_absolute(config.input_file);
            if (uf::python_path_is_symlink(config.input_file) ||
                !uf::python_path_is_file(config.input_file)) {
                throw std::invalid_argument(
                    "input_file must be an existing regular file, not a symbolic link");
            }
            config.output_root = uf::python_path_resolve(config.output_root);
            uf::event("run_discovered", {{"total", 1}});
            const auto root = uf::python_path_absolute(config.output_root);
            const auto stem = config.input_file.stem().string();
            const std::vector<ValidationJob> validation_jobs = {
                {root / (stem + ".uf.json"),
                 root / (config.input_file.filename().string() + ".uf")}};
            const std::set<std::filesystem::path> sources = {
                uf::python_path_absolute(config.input_file)};
            uf::EncodeScheduler scheduler(config.workers, config.devices);
            scheduler.submit(config);
            const auto report = scheduler.finish();
            uf::event("archive_run_completed", {{"encoded", report.encoded}, {"resumed", report.resumed},
                {"busy", report.busy}, {"failed", report.failed}});
            finish_lifecycle(values, dashboard, report.failed_channels, sources, validation_jobs);
            return report.failed ? 1 : 0;
        }
        return 0;
    });
}

int run_prepare(int argc, char** argv) {
    uf::ProcessInterrupt interruption;
    const auto values = options(argc, argv);
    for (const auto& [key, _] : values) {
        if (key != "__model_structure" && key != "__model_path_i" &&
            key != "__model_path_p" && key != "__prepared_i" && key != "__prepared_p") {
            throw std::invalid_argument("Unknown option: " + key);
        }
    }
    const auto variant = values.count("__model_structure")
        ? values.at("__model_structure") : std::string("hts");
    const auto root = checkpoint_root();
    const auto source_i = values.count("__model_path_i")
        ? std::filesystem::path(values.at("__model_path_i")) : root / "cvpr2026_image.pth.tar";
    const auto source_p = values.count("__model_path_p")
        ? std::filesystem::path(values.at("__model_path_p"))
        : root / ("cvpr2026_video_" + variant + ".pth.tar");
    const auto prepare_one = [&](const std::filesystem::path& source,
                                 const std::string& kind, const char* option) {
        const auto resolved = uf::resolve_prepared(source, kind, values.count(option)
            ? std::optional<std::filesystem::path>(values.at(option)) : std::nullopt);
        uf::event("integer_model_prepared", {{"path", uf::python_path_text(resolved.path)},
            {"variant", kind}, {"identity", resolved.data.identity}});
    };
    prepare_one(source_i, "image", "__prepared_i");
    prepare_one(source_p, variant, "__prepared_p");
    return 0;
}

int run_pareto(int argc, char** argv) {
    uf::ProcessInterrupt interruption;
    const auto values = options(argc, argv);
    for (const auto& [key, _] : values) {
        if (key != "__input_file" && key != "__output_json" &&
            key != "__resolution" && key != "__model_structure" &&
            key != "__model_path_i" && key != "__model_path_p" &&
            key != "__prepared_i" && key != "__prepared_p" &&
            key != "__cuda_idx" && key != "__device" &&
            key != "__trials" && key != "__initial_samples" &&
            key != "__seed" && key != "__input_threads" &&
            key != "__skip_thresholds" && key != "__resume") {
            throw std::invalid_argument("Unknown Pareto option: " + key);
        }
    }
    uf::ParetoOptions config;
    config.input_file = required(values, "__input_file");
    config.output_json = required(values, "__output_json");
    if (values.count("__resolution")) config.resolution = integer_option(values.at("__resolution"), "--resolution");
    if (values.count("__model_structure")) config.variant = values.at("__model_structure");
    if (values.count("__trials")) config.trials = integer_option(values.at("__trials"), "--trials");
    if (values.count("__initial_samples")) {
        config.initial_samples = integer_option(values.at("__initial_samples"), "--initial_samples");
    }
    if (values.count("__seed")) {
        config.seed = uf::PythonInteger(values.at("__seed"));
    }
    if (values.count("__resume")) config.resume = values.at("__resume") == "true";
    cpu_set_t allowed_cpus;
    const long available = sched_getaffinity(0, sizeof(allowed_cpus), &allowed_cpus) == 0
        ? CPU_COUNT(&allowed_cpus) : sysconf(_SC_NPROCESSORS_ONLN);
    config.input_threads = static_cast<int>(std::max<long>(1, std::min<long>(4, available - 1)));
    if (values.count("__input_threads")) {
        config.input_threads = integer_option(values.at("__input_threads"), "--input_threads");
    }
    if (values.count("__skip_thresholds")) {
        config.skip_thresholds.clear();
        for (const auto& value : list_option(values, "__skip_thresholds")) {
            config.skip_thresholds.push_back(uf::cli::number(value, "--skip_thresholds"));
        }
    }
    if (values.count("__cuda_idx")) {
        const auto indexes = cuda_indices(values);
        if (indexes.size() != 1) {
            throw std::invalid_argument("Pareto --cuda_idx requires one nonnegative device index");
        }
    }
    config.device = codec_device(values, "int16");
    const auto root = checkpoint_root();
    const auto checkpoint_i = values.count("__model_path_i")
        ? std::filesystem::path(values.at("__model_path_i"))
        : root / "cvpr2026_image.pth.tar";
    const auto checkpoint_p = values.count("__model_path_p")
        ? std::filesystem::path(values.at("__model_path_p"))
        : root / ("cvpr2026_video_" + config.variant + ".pth.tar");
    config.prepared_i = uf::prepared_artifact_path(checkpoint_i, "image", values.count("__prepared_i")
        ? std::optional<std::filesystem::path>(values.at("__prepared_i")) : std::nullopt);
    config.prepared_p = uf::prepared_artifact_path(checkpoint_p, config.variant, values.count("__prepared_p")
        ? std::optional<std::filesystem::path>(values.at("__prepared_p")) : std::nullopt);
    uf::run_pareto_search(config);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    std::signal(SIGPIPE, SIG_IGN);
    const std::string command = argc >= 2 ? argv[1] : "";
    try {
        (void)uf::python_integer_digit_limit();
        if (argc < 2) {
            usage();
            return 2;
        }
        if (command == "--help" || command == "-h") {
            usage(std::cout);
            return 0;
        }
        if (command == "check-setup") {
            const auto values = options(argc, argv);
            return uf::report_setup(std::cout, checkpoint_root(),
                                    values.count("__require_native") != 0,
                                    values.count("__require_int16_native") != 0);
        }
        if (command == "check-int16") {
            return uf::run_int16_validation_command(argc, argv, checkpoint_root(), std::cout);
        }
        if (command == "dcvc2021-download-checkpoints") {
#ifdef UF_HAS_DCVC2021
            return uf::dcvc2021::run_checkpoint_download_command(argc, argv);
#else
            throw std::runtime_error("Original DCVC checkpoint helper requires UF_BUILD_DCVC2021");
#endif
        }
        if (command == "dcvc2021-test-video") {
#if defined(UF_HAS_DCVC2021) && defined(UF_HAS_EVALUATION)
            return uf::dcvc2021::run_video_evaluation_command(argc, argv);
#else
            throw std::runtime_error("Original DCVC evaluation requires UF_BUILD_DCVC2021 and UF_BUILD_EVALUATION");
#endif
        }
        if (command == "dcvc-tcm-test-video") {
#if defined(UF_HAS_DCVC_TCM) && defined(UF_HAS_EVALUATION)
            return uf::tcm::run_video_evaluation_command(argc, argv);
#else
            throw std::runtime_error("TCM evaluation requires UF_BUILD_DCVC_TCM and UF_BUILD_EVALUATION");
#endif
        }
        if (command == "dcvc-hem-download-checkpoints") {
#ifdef UF_HAS_DCVC_HEM
            return uf::hem::run_checkpoint_download_command(argc, argv);
#else
            throw std::runtime_error("HEM checkpoint helper requires UF_BUILD_DCVC_HEM");
#endif
        }
        if (command == "dcvc-hem-test-video") {
#if defined(UF_HAS_DCVC_HEM) && defined(UF_HAS_EVALUATION)
            return uf::hem::run_video_evaluation_command(argc, argv);
#else
            throw std::runtime_error("HEM evaluation requires UF_BUILD_DCVC_HEM and UF_BUILD_EVALUATION");
#endif
        }
        if (command == "dcvc-dc-download-checkpoints") {
#ifdef UF_HAS_DCVC_DC
            return uf::dc::run_checkpoint_download_command(argc, argv);
#else
            throw std::runtime_error("DC checkpoint helper requires UF_BUILD_DCVC_DC");
#endif
        }
        if (command == "dcvc-dc-data-to-png") {
#ifdef UF_HAS_DCVC_DC
            return uf::dc::run_data_to_png_command(argc, argv);
#else
            throw std::runtime_error("DC dataset conversion requires UF_BUILD_DCVC_DC");
#endif
        }
        if (command == "dcvc-dc-test-video") {
#if defined(UF_HAS_DCVC_DC) && defined(UF_HAS_EVALUATION)
            return uf::dc::run_video_evaluation_command(argc, argv);
#else
            throw std::runtime_error("DC evaluation requires UF_BUILD_DCVC_DC and UF_BUILD_EVALUATION");
#endif
        }
        if (command == "test-video" || command == "test-compress-time" || command == "_evaluation_worker") {
#ifdef UF_HAS_EVALUATION
            if (command == "_evaluation_worker") {
                if (argc != 2) throw std::invalid_argument("Invalid private evaluation worker invocation");
                return uf::evaluation::run_evaluation_worker();
            }
            return command == "test-video" ? uf::evaluation::run_video_evaluation_command(argc, argv)
                                          : uf::evaluation::run_compress_time_command(argc, argv);
#else
            throw std::runtime_error("Evaluation commands require UF_BUILD_EVALUATION");
#endif
        }
        if (command == "compare-bd-rate") {
#ifdef UF_HAS_EVALUATION
            return uf::evaluation::run_bd_rate_command(argc, argv);
#else
            throw std::runtime_error("BD-rate comparison requires UF_BUILD_EVALUATION");
#endif
        }
        if (command == "_train_loader_worker") {
#ifdef UF_HAS_TRAINING
            if (argc != 2) throw std::invalid_argument("Invalid private training loader invocation");
            return uf::training::run_training_loader_worker();
#else
            throw std::runtime_error("Training loader workers require UF_BUILD_TRAINING");
#endif
        }
        if (command == "train-image" || command == "train-video" ||
            command == "_train_image_worker" || command == "_train_video_worker") {
#ifdef UF_HAS_TRAINING
            return uf::training::run_training_command(argc, argv,
                command == "train-video" || command == "_train_video_worker",
                command == "_train_image_worker" || command == "_train_video_worker");
#else
            throw std::runtime_error("Training commands require UF_BUILD_TRAINING");
#endif
        }
        if (command == "_encode_worker") {
            if (argc != 2) throw std::invalid_argument("Invalid private UF worker invocation");
            return uf::run_encode_worker();
        }
        if (command == "_image_worker") {
            if (argc != 2) throw std::invalid_argument("Invalid private UF image worker invocation");
            return uf::run_image_worker();
        }
        if (command == "_decode_worker") {
            if (argc != 2) throw std::invalid_argument("Invalid private UF decode worker invocation");
            return uf::run_decode_worker([](const uf::cli::Options& values) {
                return run_archive_decode_values(values, true);
            });
        }
        if (command == "decode") {
            uf::ProcessInterrupt interruption;
            auto values = options(argc, argv);
            if (nonempty_option(values, "__input_folder") || nonempty_option(values, "__output_folder")) {
                return run_batch_decode(values);
            }
            // Python accepts these switches for either decode form; only the
            // folder worker path uses them. A single archive uses the first GPU.
            for (const auto* name : {"__worker", "__recursive", "__input_kind",
                                    "__input_folder", "__output_folder"}) values.erase(name);
            if (values.count("__cuda_idx")) {
                values["__cuda_idx"] = std::to_string(cuda_indices(values).front());
            }
            return run_archive_decode_values(values, true);
        }
        if (command == "verify") {
            return run_archive_decode(argc, argv, command == "decode");
        }
        if (command == "view") return run_view(argc, argv);
        if (command == "encode-raw") return run_raw_encode(argc, argv);
        if (command == "encode-image") return run_image_encode(argc, argv);
        if (command == "encode" || command == "stream") return run_local_encode(argc, argv);
        if (command == "cleanup") return run_cleanup(argc, argv);
        if (command == "manage") return run_manage(argc, argv);
        if (command == "prepare-int16") return run_prepare(argc, argv);
        if (command == "pareto-search") return run_pareto(argc, argv);
        if (argc < 3) { usage(); return 2; }
        const std::filesystem::path path(argv[2]);
        if (command == "inspect") {
            if (argc != 3) { usage(); return 2; }
            if (lower_extension(path) == ".dcvci") {
                std::cout << uf::python_pretty_json(uf::read_intra_image(path, false).metadata) << '\n';
            } else {
                std::cout << uf::python_pretty_json(uf::load_archive(path, false).metadata) << '\n';
            }
        } else if (command == "check-archive") {
            if (argc != 3) { usage(); return 2; }
            if (lower_extension(path) == ".dcvci") {
                const auto archive = uf::read_intra_image(path, true, false);
                const auto scan = scan_intra_payload(archive);
                std::cout << "archive=" << archive.path << " packets=" << scan.packets
                          << " frames=1 storage_and_structure=ok\n";
            } else {
                const auto archive = uf::load_archive(path);
                const auto scan = uf::scan_archive_bitstream(archive);
                std::cout << "archive=" << archive.manifest << " packets=" << scan.packets
                          << " frames=" << archive.metadata["video"]["frames"]
                          << " storage_and_structure=ok\n";
            }
        } else if (command == "inspect-prepared") {
            if (argc != 3) { usage(); return 2; }
            const auto model = uf::load_prepared(path);
            std::cout << "variant=" << model.variant << " identity=" << model.identity
                      << " tensors=" << model.tensors.size() << '\n';
        } else if (command == "scan-bitstream") {
            if (argc != 3 && argc != 5) { usage(); return 2; }
            std::string variant = "hts";
            if (argc == 5) {
                if (std::string(argv[3]) != "--variant") { usage(); return 2; }
                variant = argv[4];
            }
            if (lower_extension(path) == ".dcvci") {
                const auto scan = scan_intra_payload(uf::read_intra_image(path, true, false));
                std::cout << "width=" << scan.dimensions.width
                          << " height=" << scan.dimensions.height
                          << " packets=" << scan.packets << " intra=1"
                          << " integer=" << scan.preamble.integer << '\n';
            } else if (path.extension() == ".json" || std::filesystem::is_directory(path)) {
                const auto archive = uf::load_archive(path);
                const auto scan = uf::scan_archive_bitstream(archive);
                std::cout << "packets=" << scan.packets << " intra=" << scan.intra_packets
                          << " predicted=" << scan.predicted_packets
                          << " integer=" << scan.preamble.integer << '\n';
            } else {
                const auto scan = uf::scan_bitstream(path, variant);
                std::cout << "width=" << scan.dimensions.width << " height=" << scan.dimensions.height
                          << " packets=" << scan.packets << " intra=" << scan.intra_packets
                          << " predicted=" << scan.predicted_packets
                          << " integer=" << scan.preamble.integer << '\n';
            }
        } else {
            usage();
            return 2;
        }
        if (!std::cout) throw std::runtime_error("Cannot write UF command output");
    } catch (const uf::cli::HelpRequested&) {
        try {
            uf::cli::help(std::cout, command);
            return 0;
        } catch (const std::exception& error) {
            std::cerr << "dcvc-uf: " << error.what() << '\n';
            return 1;
        }
    } catch (const uf::cli::ParseError& error) {
        std::cerr << "dcvc-uf: " << error.what() << '\n';
        return 2;
    } catch (const std::exception& error) {
        std::cerr << "dcvc-uf: " << error.what() << '\n';
        return uf::ProcessInterrupt::requested() ? 128 + uf::ProcessInterrupt::requested() : 1;
    }
    return 0;
}
