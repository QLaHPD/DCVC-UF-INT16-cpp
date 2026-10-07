// Copyright (c) Microsoft Corporation. Licensed under the MIT License.
// Native counterpart of DCVC-family/DCVC-TCM/test_video.py.
#include "uf/dcvc_tcm_evaluation.hpp"
#include "uf/dcvc_tcm_image.hpp"
#include "uf/dcvc_tcm.hpp"
#include "uf/cli.hpp"
#include "uf/image_pixels.hpp"
#include "uf/process.hpp"
#include "dcvc_tcm_evaluation_internal.hpp"
#include "evaluation_workflow.hpp"
#include <ATen/Context.h>
#include <torch/cuda.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace uf::tcm {
namespace {
using Json = uf::evaluation::Json;
using Clock = std::chrono::system_clock;
double seconds(Clock::time_point start) { return std::chrono::duration<double>(Clock::now() - start).count(); }
std::int64_t frame_count(const Json& value) {
    if (value.is_boolean()) return value.get<bool>() ? 1 : 0;
    if (!value.is_number_integer() || (value.is_number_unsigned() && value.get<std::uint64_t>() >
        static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())))
        throw std::invalid_argument("TCM sequence frames must fit a native integer");
    return value.get<std::int64_t>();
}
bool intra_frame(std::int64_t index, const Json& gop) {
    if (gop.is_boolean()) {
        if (!gop.get<bool>()) throw std::invalid_argument("TCM GOP cannot be zero");
        return true;
    }
    if (gop.is_number_unsigned()) {
        const auto period = gop.get<std::uint64_t>();
        if (!period) throw std::invalid_argument("TCM GOP cannot be zero");
        return static_cast<std::uint64_t>(index) % period == 0;
    }
    if (gop.is_number_integer()) {
        const auto period = gop.get<std::int64_t>();
        if (!period) throw std::invalid_argument("TCM GOP cannot be zero");
        return index % period == 0;
    }
    if (gop.is_number_float()) {
        const auto period = gop.get<double>();
        if (period == 0) throw std::invalid_argument("TCM GOP cannot be zero");
        return std::fmod(static_cast<double>(index), period) == 0;
    }
    throw std::invalid_argument("TCM GOP must be numeric");
}
bool disabled_dataset(const Json& dataset) {
    const auto& flag = dataset.at("test");
    return (flag.is_boolean() && !flag.get<bool>()) || (flag.is_number() && flag.get<double>() == 0);
}
std::uint64_t product(std::uint64_t left, std::uint64_t right) {
    if (left && right > std::numeric_limits<std::uint64_t>::max() / left)
        throw std::overflow_error("TCM evaluation pixel count exceeds the native integer range");
    return left * right;
}
std::string basename(const std::string& path) { return std::filesystem::path(path).filename().string(); }
std::string frame_name(std::int64_t index, std::size_t padding) {
    auto number = std::to_string(index + 1);
    if (number.size() < padding) number.insert(0, padding - number.size(), '0');
    return "im" + number + ".png";
}
torch::Tensor read_frame(const std::filesystem::path& path) {
    const auto pixels = read_image_rgb(path, false);
    auto tensor = torch::from_blob(const_cast<std::uint8_t*>(pixels.rgb.data()),
        {pixels.height, pixels.width, 3}, torch::TensorOptions().dtype(torch::kUInt8))
        .to(torch::kFloat64).permute({2, 0, 1}).to(torch::kFloat32).unsqueeze(0);
    return tensor / 255;
}
void write_frame(const torch::Tensor& frame, const std::filesystem::path& path) {
    if (frame.dim() != 4 || frame.size(0) != 1 || frame.size(1) != 3 || frame.size(2) < 1 || frame.size(3) < 1 ||
        frame.size(2) > std::numeric_limits<std::uint32_t>::max() ||
        frame.size(3) > std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument("Invalid TCM RGB reconstruction shape");
    const auto values = (frame.squeeze(0).permute({1, 2, 0}).detach().to(torch::kCPU) * 255)
        .round().clamp(0, 255).to(torch::kUInt8).contiguous();
    ImagePixels pixels;
    pixels.width = static_cast<std::uint32_t>(frame.size(3));
    pixels.height = static_cast<std::uint32_t>(frame.size(2));
    const auto* bytes = values.data_ptr<std::uint8_t>();
    pixels.rgb.assign(bytes, bytes + values.numel());
    write_image_png(path, pixels);
}
struct FrameLog { int type; double bit, mv_y, mv_z, y, z, psnr, msssim; };
Json filter_result(const Json& result) {
    static const std::vector<std::string> keys{"i_frame_num", "p_frame_num",
        "ave_i_frame_bpp", "ave_i_frame_psnr", "ave_i_frame_msssim", "ave_p_frame_bpp",
        "ave_p_frame_bpp_mv_y", "ave_p_frame_bpp_mv_z", "ave_p_frame_bpp_y", "ave_p_frame_bpp_z",
        "ave_p_frame_psnr", "ave_p_frame_msssim", "frame_bpp", "frame_bpp_mv_y", "frame_bpp_mv_z",
        "frame_bpp_y", "frame_bpp_z", "frame_psnr", "frame_msssim", "frame_type", "test_time"};
    Json filtered = Json::object();
    for (auto field = result.begin(); field != result.end(); ++field)
        if (std::find(keys.begin(), keys.end(), field.key()) != keys.end()) filtered[field.key()] = field.value();
    const auto i_count = filtered.at("i_frame_num").get<std::int64_t>();
    const auto p_count = filtered.at("p_frame_num").get<std::int64_t>();
    for (const auto* metric : {"bpp", "psnr", "msssim"}) {
        const auto suffix = std::string("_frame_") + metric;
        const auto sum = i_count * filtered.at("ave_i" + suffix).get<double>() +
                         p_count * filtered.at("ave_p" + suffix).get<double>();
        filtered["ave_all" + suffix] = sum / (i_count + p_count);
    }
    return filtered;
}
}  // namespace

Json evaluate_document(const Json& point, bool use_cuda, int worker_index) {
    at::globalContext().setUserEnabledCuDNN(true);
    at::globalContext().setBenchmarkCuDNN(false);
    at::globalContext().setDeterministicAlgorithms(true, false);
    torch::manual_seed(0);
    torch::set_num_threads(1);
    // The source seeds NumPy but does not consume its RNG here.
    const auto gpu_count = use_cuda ? torch::cuda::device_count() : 0;
    const auto device = gpu_count ? torch::Device(torch::kCUDA,
        static_cast<c10::DeviceIndex>(static_cast<std::size_t>(worker_index) % gpu_count)) : torch::Device(torch::kCPU);
    if (point.at("i_frame_model_name") != "IntraNoAR") throw std::invalid_argument("Unknown TCM I-frame architecture");
    auto intra = image::from_checkpoint(point.at("i_frame_model_path").get<std::string>());
    intra->eval(); intra->to(device); intra->eval();
    DMC video{nullptr};
    const auto model_path = point.at("model_path").get<std::string>();
    if (!point.at("force_intra").get<bool>()) {
        video = DMC(); video->load_checkpoint(model_path); video->to(device); video->eval();
    }
    const auto write_stream = point.at("write_stream").get<bool>();
    if (write_stream) { if (video) video->update(true); intra->update(true); }

    const auto sequence = point.at("video_path").get<std::string>();
    const auto model_index = std::to_string(point.at("model_idx").get<std::size_t>());
    const auto source = std::filesystem::path(point.at("dataset_path").get<std::string>()) / sequence;
    const auto bin_folder = std::filesystem::path(point.at("stream_path").get<std::string>()) / sequence / model_index;
    if (write_stream && !std::filesystem::exists(bin_folder)) std::filesystem::create_directories(bin_folder);
    const auto save_decoded = point.at("save_decoded_frame").get<bool>();
    const auto decoded_folder = std::filesystem::path(point.at("decoded_frame_path").get<std::string>()) / sequence / model_index;
    if (save_decoded) std::filesystem::create_directories(decoded_folder);
    bool short_names = false, long_names = false;
    for (const auto& entry : std::filesystem::directory_iterator(source)) {
        short_names |= entry.path().filename() == "im1.png";
        long_names |= entry.path().filename() == "im00001.png";
    }
    if (!short_names && !long_names) throw std::invalid_argument("unknown image naming convention; please specify");
    const auto padding = short_names ? 1u : 5u;
    const auto frames = frame_count(point.at("frame_num"));
    if (frames <= 0) throw std::invalid_argument("TCM evaluation requires at least one frame");
    const auto& gop = point.at("gop");
    const auto verbose = point.at("verbose").get<int>();
    std::vector<FrameLog> history;
    std::uint64_t pixels = 0;
    std::int64_t p_frames = 0;
    double decoding_time = 0;
    torch::Tensor reference, reference_feature;
    MSSSIM metric;
    const auto start = Clock::now();
    {
        torch::NoGradGuard no_grad;
        for (std::int64_t index = 0; index < frames; ++index) {
            ProcessInterrupt::check();
            const auto frame_start = Clock::now();
            const auto original = read_frame(source / frame_name(index, padding)).to(device);
            const auto height = original.size(2), width = original.size(3);
            const auto area = product(static_cast<std::uint64_t>(height), static_cast<std::uint64_t>(width));
            if (!pixels) pixels = area;
            else if (pixels != area) throw std::invalid_argument("TCM frame pixel counts differ within a sequence");
            const auto padded = tcm::pad(original);
            const auto output = write_stream ? std::optional<std::filesystem::path>(bin_folder / (std::to_string(index) + ".bin"))
                                             : std::nullopt;
            FrameLog log{};
            if (intra_frame(index, gop)) {
                const auto result = intra->encode_decode(padded, output, width, height);
                reference = result.x_hat; reference_feature = torch::Tensor{}; log.bit = result.bit;
            } else {
                if (!video) throw std::runtime_error("TCM force_intra with this GOP reaches a P-frame without a video model");
                const auto result = video->encode_decode(padded, reference, reference_feature, output, width, height);
                reference = result.x_hat; reference_feature = result.feature;
                log.type = 1; log.bit = result.bit; log.mv_y = result.bit_mv_y;
                log.mv_z = result.bit_mv_z; log.y = result.bit_y; log.z = result.bit_z;
                ++p_frames; decoding_time += result.decoding_time;
            }
            reference.clamp_(0, 1);
            const auto reconstruction = tcm::crop(reference, height, width);
            const auto mse = (reconstruction - original).pow(2).mean();
            log.psnr = (20 * torch::log10(1 / torch::sqrt(mse))).item<double>();
            log.msssim = metric(reconstruction, original).item<double>();
            history.push_back(log);
            const auto elapsed = seconds(frame_start);
            if (verbose >= 2) {
                std::ostringstream line; line.imbue(std::locale::classic()); line << std::fixed;
                line << "frame " << index << ", " << std::setprecision(3) << elapsed << " seconds, bits: " << log.bit
                     << ", PSNR: " << std::setprecision(4) << log.psnr << ", MS-SSIM: " << log.msssim << " \n";
                std::cout << line.str();
            }
            if (save_decoded) write_frame(reconstruction, decoded_folder / (std::to_string(index) + ".png"));
        }
    }
    const auto test_time = seconds(start);
    if (verbose >= 1 && p_frames > 0) {
        std::ostringstream line; line.imbue(std::locale::classic());
        line << "decoding " << p_frames << " P frames, average " << std::fixed << std::setprecision(0)
             << decoding_time / p_frames * 1000 << " ms.\n";
        std::cout << line.str();
    }
    std::int64_t i_count = 0, p_count = 0;
    double i_bits = 0, i_psnr = 0, i_msssim = 0, p_bits = 0, p_psnr = 0, p_msssim = 0;
    double p_mv_y = 0, p_mv_z = 0, p_y = 0, p_z = 0;
    for (const auto& frame : history) {
        if (!frame.type) { ++i_count; i_bits += frame.bit; i_psnr += frame.psnr; i_msssim += frame.msssim; }
        else {
            ++p_count; p_bits += frame.bit; p_psnr += frame.psnr; p_msssim += frame.msssim;
            p_mv_y += frame.mv_y; p_mv_z += frame.mv_z; p_y += frame.y; p_z += frame.z;
        }
    }
    if (!i_count || !pixels) throw std::invalid_argument("TCM evaluation has no I-frame pixels");
    Json log{{"frame_pixel_num", pixels}, {"i_frame_num", i_count}, {"p_frame_num", p_count},
        {"ave_i_frame_bpp", i_bits / i_count / pixels}, {"ave_i_frame_psnr", i_psnr / i_count},
        {"ave_i_frame_msssim", i_msssim / i_count}};
    for (const auto* name : {"frame_bpp", "frame_bpp_mv_y", "frame_bpp_mv_z", "frame_bpp_y", "frame_bpp_z",
                             "frame_psnr", "frame_msssim", "frame_type"}) log[name] = Json::array();
    for (const auto& frame : history) {
        log["frame_bpp"].push_back(frame.bit / pixels); log["frame_bpp_mv_y"].push_back(frame.mv_y / pixels);
        log["frame_bpp_mv_z"].push_back(frame.mv_z / pixels); log["frame_bpp_y"].push_back(frame.y / pixels);
        log["frame_bpp_z"].push_back(frame.z / pixels); log["frame_psnr"].push_back(frame.psnr);
        log["frame_msssim"].push_back(frame.msssim); log["frame_type"].push_back(frame.type);
    }
    log["test_time"] = test_time;
    if (p_count) {
        const auto p_pixels = product(static_cast<std::uint64_t>(p_count), pixels);
        log["ave_p_frame_bpp"] = p_bits / p_pixels; log["ave_p_frame_bpp_mv_y"] = p_mv_y / p_pixels;
        log["ave_p_frame_bpp_mv_z"] = p_mv_z / p_pixels; log["ave_p_frame_bpp_y"] = p_y / p_pixels;
        log["ave_p_frame_bpp_z"] = p_z / p_pixels; log["ave_p_frame_psnr"] = p_psnr / p_count;
        log["ave_p_frame_msssim"] = p_msssim / p_count;
    } else {
        for (const auto* name : {"ave_p_frame_bpp", "ave_p_frame_psnr", "ave_p_frame_msssim", "ave_p_frame_bpp_mv_y",
                                 "ave_p_frame_bpp_mv_z", "ave_p_frame_bpp_y", "ave_p_frame_bpp_z"}) log[name] = 0;
    }
    log["ave_all_frame_bpp"] = (i_bits + p_bits) / product(static_cast<std::uint64_t>(frames), pixels);
    log["ave_all_frame_psnr"] = (i_psnr + p_psnr) / frames;
    log["ave_all_frame_msssim"] = (i_msssim + p_msssim) / frames;
    log["name"] = basename(model_path) + "_" + sequence;
    log["ds_name"] = point.at("ds_name"); log["video_path"] = sequence; log["model_idx"] = point.at("model_idx");
    return log;
}

int run_video_evaluation_command(int argc, char** argv) {
    const auto start = Clock::now();
    at::globalContext().setUserEnabledCuDNN(true);
    const auto options = cli::parse(argc, argv);
    const ProcessInterrupt interruption;
    const auto get = [&](const char* name, const char* fallback) {
        const auto found = options.find(name); return found == options.end() ? std::string(fallback) : found->second;
    };
    const auto enabled = [&](const char* name) { return get(name, "false") == "true"; };
    const auto workers = cli::integer(get("__worker", "1"), "--worker");
    if (workers < 1) throw std::invalid_argument("--worker must be at least 1");
    if (options.count("__cuda_device") && !options.at("__cuda_device").empty() &&
        setenv("CUDA_VISIBLE_DEVICES", options.at("__cuda_device").c_str(), 1) != 0)
        throw std::runtime_error("Cannot set CUDA_VISIBLE_DEVICES");
    if (setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8", 1) != 0) throw std::runtime_error("Cannot set CUBLAS_WORKSPACE_CONFIG");
    const auto intra_models = cli::list(options, "__i_frame_model_path");
    const auto force_intra = enabled("__force_intra");
    const auto models = force_intra ? intra_models : cli::list(options, "__model_path");
    if (models.empty()) throw std::invalid_argument("TCM evaluation requires model checkpoint paths");
    if (intra_models.size() < models.size()) throw std::invalid_argument("TCM evaluation requires an I-frame checkpoint for each model");
    const auto forced_frames = cli::signed_integer(get("__force_frame_num", "-1"), "--force_frame_num");
    const auto forced_period = cli::signed_integer(get("__force_intra_period", "-1"), "--force_intra_period");
    const auto verbose = cli::integer(get("__verbose", "0"), "--verbose");
    std::ifstream input(options.at("__test_config"));
    if (!input) throw std::runtime_error("Cannot open TCM evaluation configuration");
    const auto config = uf::evaluation::read_json(input);
    if (!config.is_object()) throw std::invalid_argument("TCM evaluation configuration must be an object");
    std::vector<Json> points;
    std::uint64_t count_frames = 0, count_sequences = 0;
    for (auto dataset = config.begin(); dataset != config.end(); ++dataset) {
        if (disabled_dataset(dataset.value())) continue;
        const auto& sequences = dataset.value().at("sequences");
        if (!sequences.is_object()) throw std::invalid_argument("TCM evaluation sequences must be an object");
        for (auto sequence = sequences.begin(); sequence != sequences.end(); ++sequence) {
            ++count_sequences;
            const auto configured_frames = frame_count(sequence.value().at("frames"));
            const auto frames = forced_frames > 0 ? forced_frames : configured_frames;
            if (frames <= 0) throw std::invalid_argument("TCM evaluation requires at least one frame");
            auto gop = sequence.value().at("gop");
            if (force_intra) gop = 1;
            if (forced_period > 0) gop = forced_period;
            for (std::size_t model = 0; model < models.size(); ++model) {
                if (count_frames > std::numeric_limits<std::uint64_t>::max() - static_cast<std::uint64_t>(frames))
                    throw std::overflow_error("TCM evaluation total frame count exceeds the native integer range");
                count_frames += static_cast<std::uint64_t>(frames);
                points.push_back(Json{{"model_idx", model}, {"i_frame_model_path", intra_models[model]},
                    {"i_frame_model_name", get("__i_frame_model_name", "IntraNoAR")}, {"force_intra", force_intra},
                    {"model_path", models[model]}, {"video_path", sequence.key()}, {"gop", gop}, {"frame_num", frames},
                    {"dataset_path", dataset.value().at("base_path")}, {"write_stream", enabled("__write_stream")},
                    {"stream_path", get("__stream_path", "out_bin") + "/bit_stream"},
                    {"save_decoded_frame", enabled("__save_decoded_frame")},
                    {"decoded_frame_path", get("__decoded_frame_path", "decoded_frames") + "/decoded_video"},
                    {"ds_name", dataset.key()}, {"verbose", verbose}});
            }
        }
    }
    const Json initialization{{"backend", "dcvc_tcm"}, {"settings", Json{{"cuda", enabled("__cuda")}}}};
    const auto results = uf::evaluation::evaluate_documents(std::vector<Json>(static_cast<std::size_t>(workers), initialization), points);
    Json log = Json::object();
    for (auto dataset = config.begin(); dataset != config.end(); ++dataset) {
        if (disabled_dataset(dataset.value())) continue;
        auto& dataset_log = log[dataset.key()] = Json::object();
        const auto& sequences = dataset.value().at("sequences");
        for (auto sequence = sequences.begin(); sequence != sequences.end(); ++sequence) {
            auto& sequence_log = dataset_log[sequence.key()] = Json::object();
            for (const auto& model : models) {
                const auto checkpoint = basename(model);
                // Preserve startswith, basename collisions and the last match.
                for (const auto& result : results) {
                    if (result.at("name").get<std::string>().rfind(checkpoint, 0) == 0 &&
                        result.at("ds_name") == dataset.key() && result.at("video_path") == sequence.key())
                        sequence_log[checkpoint] = filter_result(result);
                }
            }
        }
    }
    const std::filesystem::path output_path = options.at("__output_path");
    if (!output_path.parent_path().empty() && !std::filesystem::exists(output_path.parent_path()))
        std::filesystem::create_directories(output_path.parent_path());
    std::ofstream output(output_path, std::ios::trunc);
    if (!output) throw std::runtime_error("Cannot create TCM evaluation result JSON");
    uf::evaluation::write_json(log, output, -1, 2); output.close();
    if (!output) throw std::runtime_error("Cannot write TCM evaluation result JSON");
    std::ostringstream elapsed; elapsed.imbue(std::locale::classic());
    elapsed << std::fixed << std::setprecision(1) << seconds(start) / 60;
    std::cout << "Test finished\nTested " << models.size() << " models on " << count_frames / models.size()
        << " frames from " << count_sequences << " sequences\nTotal elapsed time: " << elapsed.str() << " min\n";
    return 0;
}
}  // namespace uf::tcm
