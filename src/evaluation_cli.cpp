// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#include "evaluation_workflow.hpp"
#include "evaluation_cuda_policy.hpp"
#include "uf/evaluation.hpp"
#include "uf/fp16_model.hpp"
#include "uf/process.hpp"
#include "uf/cli.hpp"
#include "uf/reference_paths.hpp"
#include "uf/python_json.hpp"
#include "uf/python_numeric.hpp"
#include "uf/json_compat.hpp"

#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace uf::evaluation {
namespace {

void video_help(std::ostream& output) {
    cli::help(output, "test-video");
}
std::int64_t config_integer(const Json& value, const char* name) {
    // Python JSON Booleans are integers when used for dimensions/counts.
    if (!value.is_number_integer() && !value.is_boolean() && !python_json_extended_integer(value)) {
        throw std::invalid_argument(std::string("Evaluation configuration requires an integer ") + name);
    }
    return python_json_index_int64(nlohmann::json(value));
}
bool enabled(const Json& value) {
    // The reference skips `test == 0`; None and empty containers are enabled.
    if (value.is_boolean()) return value.get<bool>();
    if (value.is_number()) return value.get<double>() != 0;
    if (python_json_extended_integer(value)) return python_json_extended_integer_decimal(value) != "0";
    return true;
}

std::string replace_all(std::string text, const std::string& old, const std::string& replacement) {
    for (std::size_t position = 0; (position = text.find(old, position)) != std::string::npos; position += replacement.size()) {
        text.replace(position, old.size(), replacement);
    }
    return text;
}

VideoEvaluationOptions video_options(const cli::Options& values) {
    VideoEvaluationOptions options;
    options.config = values.at("__test_config");
    options.output = values.at("__output_path");
    for (const auto* name : {"__qp_i", "__qp_p", "__cuda_idx"}) {
        if (!values.count(name)) continue;
        std::vector<int> integers;
        for (const auto& value : cli::list(values, name)) integers.push_back(cli::integer(value, name));
        if (std::string_view(name) == "__qp_i") options.qp_i = std::move(integers);
        else if (std::string_view(name) == "__qp_p") options.qp_p = std::move(integers);
        else options.cuda_indices = std::move(integers);
    }
    if (values.count("__skip_thres")) options.skip_threshold = cli::number(values.at("__skip_thres"), "--skip_thres");
    if (values.count("__model_path_i")) options.model_i = values.at("__model_path_i");
    if (values.count("__model_path_p")) options.model_p = values.at("__model_path_p");
    if (values.count("__stream_path")) options.stream_path = values.at("__stream_path");
    if (values.count("__force_root_path")) options.root_path = values.at("__force_root_path");
    if (values.count("__model_structure")) options.variant = values.at("__model_structure");
    if (values.count("__rate_num")) options.rate_count = cli::integer(values.at("__rate_num"), "--rate_num");
    if (values.count("__worker")) options.workers = cli::integer(values.at("__worker"), "--worker");
    if (values.count("__reset_interval")) options.reset_interval = cli::integer(values.at("__reset_interval"), "--reset_interval");
    if (values.count("__verbose")) options.verbose = cli::integer(values.at("__verbose"), "--verbose");
    if (values.count("__force_frame_num")) options.force_frame_count = cli::signed_integer(values.at("__force_frame_num"), "--force_frame_num");
    if (values.count("__force_intra_period")) options.force_intra_period = cli::signed_integer(values.at("__force_intra_period"), "--force_intra_period");
    if (values.count("__force_intra")) options.force_intra = values.at("__force_intra") == "true";
    if (values.count("__calc_ssim")) options.calc_ssim = values.at("__calc_ssim") == "true";
    if (values.count("__check_existing")) options.check_existing = values.at("__check_existing") == "true";
    if (values.count("__save_decoded_frame")) options.save_decoded = values.at("__save_decoded_frame") == "true";
    if (values.count("__verbose_json")) options.verbose_json = values.at("__verbose_json") == "true";
    return options;
}

std::vector<int> qps(const std::optional<std::vector<int>>& supplied, int rate_count) {
    if (supplied) {
        if (supplied->size() != static_cast<std::size_t>(rate_count)) throw std::invalid_argument("QP list length must equal --rate_num");
        return *supplied;
    }
    if (rate_count < 2 || rate_count > 64) throw std::invalid_argument("Default QP spacing requires 2 to 64 rates");
    const auto step = 63.0 / (rate_count - 1);
    std::vector<int> result;
    for (int index = 0; index < rate_count; ++index) result.push_back(static_cast<int>((index + 1 == rate_count ? 63.0 : index * step) + 0.5));
    return result;
}

std::string command_text(const std::vector<std::string>& arguments) {
    std::string result;
    for (const auto& argument : arguments) {
        if (!result.empty()) result += ' ';
        result += '\'' + replace_all(argument, "'", "'\"'\"'") + '\'';
    }
    return result;
}
void wait_benchmark(ChildProcess& process) {
    try {
        // test_compress_time ignores os.system's return value and proceeds
        // to read the requested report, including an existing report.
        (void)process.wait();
    } catch (...) {
        process.signal(SIGTERM);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        for (;;) {
            try { if (process.poll_for_cleanup()) break; } catch (...) { break; }
            if (std::chrono::steady_clock::now() >= deadline) { process.signal(SIGKILL); break; }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        throw;
    }
}

double timing_average(const std::vector<Json>& values) {
    PythonInteger integer;
    std::optional<double> floating;
    for (const auto& value : values) {
        if (value.is_number_float()) {
            const auto first = floating ? *floating : integer.as_double();
            floating = first + value.get<double>();
        } else if (value.is_boolean() || value.is_number_integer() || python_json_extended_integer(value)) {
            const auto next = python_json_int(nlohmann::json(value));
            if (floating) *floating += next.as_double();
            else integer += next;
        } else {
            // sum() adds scalars; it does not apply float(str) coercion.
            throw PythonJsonTypeError("Timing benchmark statistic must be numeric");
        }
    }
    if (values.empty()) throw std::runtime_error("Timing benchmark HEVC_B contains no rate points");
    if (floating) return *floating / static_cast<double>(values.size());
    // Retain exact integer accumulation and round the ratio only at division.
    return python_fraction_double(integer.decimal() + "/" + std::to_string(values.size()));
}

void require_evaluation_cuda() {
    const auto* availability_policy = std::getenv("PYTORCH_NVML_BASED_CUDA_CHECK");
    const auto nvml_available = evaluation_cuda_compiled() && availability_policy && std::string_view(availability_policy) == "1"
        ? evaluation_nvml_devices() : std::optional<int>{};
    // A successful NVML check must not initialize the parent's CUDA runtime
    // before an explicit mask change. Keep the fallback lazy.
    const auto available = nvml_available ? *nvml_available : evaluation_cuda_devices();
    if (available < 1 || !fp16_compiled())
        throw std::runtime_error("test-video requires CUDA and the native FP16 backend");
}

}  // namespace

Json evaluation_settings_json(const VideoEvaluationOptions& options) {
    // Only immutable worker settings; rates, paths and dimensions travel with
    // each point. Nonfinite threshold values retain JSON's IEEE extensions.
    return Json{{"model_i", options.model_i.string()}, {"model_p", options.model_p.string()},
        {"variant", options.variant}, {"skip_threshold", options.skip_threshold},
        {"force_intra", options.force_intra}, {"reset_interval", options.reset_interval},
        {"calc_ssim", options.calc_ssim}, {"check_existing", options.check_existing},
        {"save_decoded", options.save_decoded}, {"verbose_json", options.verbose_json}, {"verbose", options.verbose}};
}
VideoEvaluationOptions evaluation_settings(const Json& value) {
    VideoEvaluationOptions options;
    options.model_i = value.at("model_i").get<std::string>(); options.model_p = value.at("model_p").get<std::string>();
    options.variant = value.at("variant").get<std::string>(); options.skip_threshold = value.at("skip_threshold").get<double>();
    options.force_intra = value.at("force_intra").get<bool>(); options.reset_interval = value.at("reset_interval").get<int>();
    options.calc_ssim = value.at("calc_ssim").get<bool>(); options.check_existing = value.at("check_existing").get<bool>();
    options.save_decoded = value.at("save_decoded").get<bool>(); options.verbose_json = value.at("verbose_json").get<bool>();
    options.verbose = value.at("verbose").get<int>();
    return options;
}
Json evaluation_point_json(const EvaluationPoint& point) {
    return Json{{"source", point.source.string()}, {"bin_folder", point.bin_folder.string()}, {"bin_path", point.bin_path.string()},
        {"reconstruction_path", point.reconstruction_path.string()}, {"json_path", point.json_path.string()},
        {"dataset", point.dataset}, {"sequence", point.sequence}, {"source_type", point.source_type},
        {"width", point.width}, {"height", point.height}, {"frame_count", point.frame_count},
        {"intra_period", point.intra_period}, {"rate_index", point.rate_index}, {"qp_i", point.qp_i}, {"qp_p", point.qp_p}};
}
EvaluationPoint evaluation_point(const Json& value) {
    EvaluationPoint point;
    point.source = value.at("source").get<std::string>(); point.bin_folder = value.at("bin_folder").get<std::string>();
    point.bin_path = value.at("bin_path").get<std::string>(); point.reconstruction_path = value.at("reconstruction_path").get<std::string>();
    point.json_path = value.at("json_path").get<std::string>(); point.dataset = value.at("dataset").get<std::string>();
    point.sequence = value.at("sequence").get<std::string>(); point.source_type = value.at("source_type").get<std::string>();
    point.width = value.at("width").get<std::uint32_t>(); point.height = value.at("height").get<std::uint32_t>();
    point.frame_count = value.at("frame_count").get<std::uint64_t>(); point.intra_period = value.at("intra_period").get<std::int64_t>();
    point.rate_index = value.at("rate_index").get<int>(); point.qp_i = value.at("qp_i").get<int>(); point.qp_p = value.at("qp_p").get<int>();
    if ((point.source_type != "png" && point.source_type != "yuv420") || !point.width || !point.height || !point.frame_count ||
        (point.source_type == "yuv420" && (point.width % 2 || point.height % 2)) ||
        point.width > static_cast<std::uint32_t>(std::numeric_limits<int>::max() - 64) ||
        point.height > static_cast<std::uint32_t>(std::numeric_limits<int>::max() - 64) ||
        static_cast<std::uint64_t>(point.width) * point.height > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) / 3) {
        throw std::invalid_argument("Invalid evaluation source dimensions or frame count");
    }
    return point;
}

namespace {

int evaluate_video_impl(VideoEvaluationOptions options,
                        const std::chrono::system_clock::time_point& started) {
    options.skip_threshold = std::max(0.0, options.skip_threshold);
    if (options.cuda_indices && (options.cuda_indices->empty() || options.cuda_indices->size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))) {
        throw std::invalid_argument("Invalid evaluation CUDA index list");
    }
    if (options.cuda_indices) {
        std::string mask;
        for (const int index : *options.cuda_indices) {
            if (!mask.empty()) mask += ',';
            mask += std::to_string(index);
        }
        if (setenv("CUDA_VISIBLE_DEVICES", mask.c_str(), 1) != 0) {
            throw std::runtime_error("Cannot configure evaluation CUDA visibility");
        }
    }
    if (options.workers < 1) throw std::invalid_argument("--worker must be at least 1");
    const auto config = read_json_file(options.config);
    const auto nvml_count = evaluation_nvml_devices();
    const auto gpu_count = nvml_count ? *nvml_count : evaluation_cuda_devices();
    // ProcessPoolExecutor is created before QP selection in test_video.main;
    // actual workers still start lazily when rate points are submitted.
    EvaluationExecutor executor(options, gpu_count);
    const auto qp_i = qps(options.qp_i, options.rate_count);
    const auto qp_p = options.force_intra ? qp_i : options.qp_p ? qps(options.qp_p, options.rate_count) : qp_i;
    std::cout << "testing " << options.rate_count << " rates, using qp: ";
    for (std::size_t index = 0; index < qp_i.size(); ++index) std::cout << (index ? ", " : "") << qp_i[index];
    std::cout << '\n' << std::flush;
    const auto root = options.root_path ? *options.root_path : std::filesystem::path(config.at("root_path").get<std::string>());
    const auto& classes = config.at("test_classes");
    if (!classes.is_object()) throw std::invalid_argument("Evaluation test_classes must be an object");
    Json log = Json::object();
    std::uint64_t count_frames = 0, count_sequences = 0;
    for (auto dataset = classes.begin(); dataset != classes.end(); ++dataset) {
        ProcessInterrupt::check();
        if (!enabled(dataset.value().at("test"))) continue;
        auto& dataset_log = log[dataset.key()] = Json::object();
        const auto& sequences = dataset.value().at("sequences");
        if (!sequences.is_object()) throw std::invalid_argument("Evaluation sequences must be an object");
        for (auto sequence = sequences.begin(); sequence != sequences.end(); ++sequence) {
            ++count_sequences;
            dataset_log[sequence.key()] = Json::object();
            for (int index = 0; index < options.rate_count; ++index) {
                EvaluationPoint point;
                point.dataset = dataset.key(); point.sequence = sequence.key(); point.rate_index = index;
                point.qp_i = qp_i[index]; point.qp_p = qp_p[index];
                point.source_type = dataset.value().at("src_type").get<std::string>();
                const auto height = config_integer(sequence.value().at("height"), "height");
                const auto width = config_integer(sequence.value().at("width"), "width");
                if (width <= 0 || height <= 0 || width > std::numeric_limits<int>::max() - 64 || height > std::numeric_limits<int>::max() - 64) {
                    throw std::invalid_argument("Invalid evaluation sequence dimensions");
                }
                point.width = static_cast<std::uint32_t>(width); point.height = static_cast<std::uint32_t>(height);
                // The source always looks up these fields, then replaces them
                // with forced values before the worker consumes their types.
                const auto& configured_intra = sequence.value().at("intra_period");
                point.intra_period = options.force_intra_period > 0 ? options.force_intra_period
                    : options.force_intra ? 1 : config_integer(configured_intra, "intra_period");
                const auto& configured_frames = sequence.value().at("frames");
                const auto frames = options.force_frame_count > 0 ? options.force_frame_count
                    : config_integer(configured_frames, "frames");
                if (frames <= 0) throw std::invalid_argument("Evaluation frame count must be positive");
                point.frame_count = static_cast<std::uint64_t>(frames);
                point.source = root / dataset.value().at("base_path").get<std::string>() / point.sequence;
                point.bin_folder = options.stream_path / point.dataset;
                point.bin_path = point.bin_folder / (point.sequence + "_q" + std::to_string(point.qp_i) + ".bin");
                point.reconstruction_path = replace_all(point.bin_path.string(), ".bin", ".mp4");
                point.json_path = replace_all(point.bin_path.string(), ".bin", ".json");
                point = evaluation_point(evaluation_point_json(point));
                if (count_frames > std::numeric_limits<std::uint64_t>::max() - point.frame_count) throw std::overflow_error("Evaluation total frame count overflow");
                count_frames += point.frame_count;
                executor.submit(point);
            }
        }
    }
    const auto results = executor.finish();
    for (const auto& result : results) {
        std::ostringstream rate;
        rate << std::setfill('0') << std::setw(3) << result.at("rate_idx").get<int>();
        log[result.at("ds_name").get<std::string>()][result.at("seq").get<std::string>()][rate.str()] = result;
    }
    if (!options.output.parent_path().empty()) std::filesystem::create_directories(options.output.parent_path());
    write_json_file(log, options.output, 6, 2);
    const auto minutes = std::chrono::duration<double>(std::chrono::system_clock::now() - started).count() / 60;
    std::cout << "Test finished\nTested " << count_frames << " frames from " << count_sequences
              << " sequences\nTotal elapsed time: " << std::fixed << std::setprecision(1) << minutes << " min\n";
    return 0;
}

}  // namespace

int evaluate_video(VideoEvaluationOptions options) {
    const auto started = std::chrono::system_clock::now();
    const ProcessInterrupt interruption;
    require_evaluation_cuda();
    return evaluate_video_impl(std::move(options), started);
}

int run_video_evaluation_command(int argc, char** argv) {
    const auto started = std::chrono::system_clock::now();
    cli::Options values;
    try { values = cli::parse(argc, argv); }
    catch (const cli::HelpRequested&) { video_help(std::cout); return 0; }
    catch (const cli::ParseError& error) {
        video_help(std::cerr);
        std::cerr << "error: " << error.what() << '\n';
        return 2;
    }
    const ProcessInterrupt interruption;
    require_evaluation_cuda();
    // Python parses arbitrary-width integers, then checks CUDA availability.
    // Native width limits belong to the workflow, after this same preflight,
    // rather than the argument parser's usage/exit-2 handler.
    auto options = video_options(values);
    values.clear();
    return evaluate_video_impl(std::move(options), started);
}

int run_compress_time_command(int argc, char** argv) {
    cli::Options values;
    try {
        values = cli::parse(argc, argv);
    } catch (const cli::HelpRequested&) {
        cli::help(std::cout, "test-compress-time");
        return 0;
    } catch (const cli::ParseError& error) {
        cli::help(std::cerr, "test-compress-time");
        std::cerr << "error: " << error.what() << '\n';
        return 2;
    }
    const bool image = values.count("__img") && values.at("__img") == "true";
    const auto variant = values.count("__model_structure") ? values.at("__model_structure") : std::string("ld");
    const auto output_path = values.count("__output_path") ? values.at("__output_path") : std::string("temp.json");
    values.clear();
    const ProcessInterrupt interruption;
    const std::vector<std::string> command{std::filesystem::read_symlink("/proc/self/exe").string(), "test-video",
        "--verbose", "2", "--rate_num", "4", "--force_intra", image ? "1" : "0",
        "--test_config", reference_path("test_cfg/runtime_avg.json").string(), "--force_frame_num", "-1",
        "--cuda_idx", "0", "-w", "1", "--skip_thres", "0.15", "--output_path", output_path,
        "--model_path_i", reference_path("checkpoints/cvpr2026_image.pth.tar").string(),
        "--model_path_p", reference_path("checkpoints/cvpr2026_video_" + variant + ".pth.tar").string(),
        "--model_structure", variant};
    std::cout << command_text(command) << '\n' << std::flush;
    ChildProcess benchmark(command, {}, true);
    wait_benchmark(benchmark);
    const auto report = read_json_file(output_path);
    const auto& dataset = report.at("HEVC_B");
    std::vector<Json> encoding_times, decoding_times;
    for (const auto& sequence : dataset) for (const auto& point : sequence) {
        encoding_times.push_back(point.at("avg_frame_encoding_time"));
        decoding_times.push_back(point.at("avg_frame_decoding_time"));
    }
    const auto encoding = timing_average(encoding_times);
    const auto decoding = timing_average(decoding_times);
    const auto delay = variant == "ld" ? 1 : 8;
    if (encoding == 0) throw std::runtime_error("Timing benchmark average encoding time is zero");
    std::cout << std::fixed << std::setprecision(4)
              << "Average encoding time on HEVC_B = " << encoding * 1000 << " ms / " << delay / encoding << " fps\n";
    if (decoding == 0) throw std::runtime_error("Timing benchmark average decoding time is zero");
    std::cout << "Average decoding time on HEVC_B = " << decoding * 1000 << " ms / " << delay / decoding << " fps\n";
    return 0;
}

}  // namespace uf::evaluation
