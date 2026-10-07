// Copyright (c) Microsoft Corporation. Licensed under the MIT License.
// Native counterpart of DCVC-family/DCVC/test_video.py.
#include "uf/dcvc2021_evaluation.hpp"
#include "uf/dcvc2021_image.hpp"
#include "uf/cli.hpp"
#include "uf/image_pixels.hpp"
#include "uf/process.hpp"
#include "dcvc2021_evaluation_internal.hpp"
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
#include <utility>
#include <vector>

namespace uf::dcvc2021 {
namespace {
using Json = uf::evaluation::Json;

std::int64_t frame_count(const Json& value) {
    if (value.is_boolean()) return value.get<bool>() ? 1 : 0;
    if (!value.is_number_integer() || (value.is_number_unsigned() && value.get<std::uint64_t>() >
        static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))) {
        throw std::invalid_argument("DCVC sequence frames must fit a native integer");
    }
    return value.get<std::int64_t>();
}

bool intra_frame(std::int64_t index, const Json& gop) {
    if (gop.is_boolean()) {
        if (!gop.get<bool>()) throw std::invalid_argument("DCVC GOP cannot be zero");
        return true;
    }
    if (gop.is_number_unsigned()) {
        const auto period = gop.get<std::uint64_t>();
        if (!period) throw std::invalid_argument("DCVC GOP cannot be zero");
        return static_cast<std::uint64_t>(index) % period == 0;
    }
    if (gop.is_number_integer()) {
        const auto period = gop.get<std::int64_t>();
        if (!period) throw std::invalid_argument("DCVC GOP cannot be zero");
        return index % period == 0;
    }
    if (gop.is_number_float()) {
        const auto period = gop.get<double>();
        if (period == 0) throw std::invalid_argument("DCVC GOP cannot be zero");
        return std::fmod(static_cast<double>(index), period) == 0;
    }
    throw std::invalid_argument("DCVC GOP must be numeric");
}

std::uint64_t product(std::uint64_t left, std::uint64_t right) {
    if (left && right > std::numeric_limits<std::uint64_t>::max() / left) {
        throw std::overflow_error("DCVC evaluation pixel count exceeds the native integer range");
    }
    return left * right;
}

std::string model_basename(const std::string& path) { return std::filesystem::path(path).filename().string(); }
std::string remove_last_four_characters(std::string value) {
    for (int count = 0; count < 4 && !value.empty(); ++count) {
        auto offset = value.size() - 1;
        while (offset && (static_cast<unsigned char>(value[offset]) & 0xc0) == 0x80) --offset;
        value.resize(offset);
    }
    return value;
}
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
    auto values = (frame.clone().to(torch::kCPU).detach().permute({1, 2, 0}) * 255)
        .round().clamp(0, 255).to(torch::kUInt8).contiguous();
    if (frame.dim() != 3 || frame.size(0) != 3 || frame.size(1) < 1 || frame.size(2) < 1 ||
        frame.size(1) > std::numeric_limits<std::uint32_t>::max() ||
        frame.size(2) > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("Invalid DCVC RGB reconstruction shape");
    }
    ImagePixels pixels;
    pixels.width = static_cast<std::uint32_t>(frame.size(2));
    pixels.height = static_cast<std::uint32_t>(frame.size(1));
    const auto* bytes = values.data_ptr<std::uint8_t>();
    pixels.rgb.assign(bytes, bytes + values.numel());
    write_image_png(path, pixels);
}

Json filter_result(const Json& result) {
    static const std::vector<std::string> keys{"i_frame_num", "p_frame_num", "ave_i_frame_bpp",
        "ave_i_frame_quality", "ave_p_frame_bpp", "ave_p_frame_bpp_mv_y", "ave_p_frame_bpp_mv_z",
        "ave_p_frame_bpp_y", "ave_p_frame_bpp_z", "ave_p_frame_quality", "ave_all_frame_bpp",
        "ave_all_frame_quality"};
    Json filtered = Json::object();
    for (auto field = result.begin(); field != result.end(); ++field) {
        if (std::find(keys.begin(), keys.end(), field.key()) != keys.end()) filtered[field.key()] = field.value();
    }
    return filtered;
}
}  // namespace

Json evaluate_document(const Json& point, bool use_cuda, int worker_index) {
    const auto write_stream = point.at("write_stream").get<bool>();
    at::globalContext().setUserEnabledCuDNN(true);
    if (write_stream) {
        at::globalContext().setBenchmarkCuDNN(false);
        at::globalContext().setDeterministicAlgorithms(true, false);
    }
    torch::manual_seed(0);
    torch::set_num_threads(1);
    // NumPy's seeded RNG is not consumed by this evaluation path.
    const auto gpu_count = use_cuda ? torch::cuda::device_count() : 0;
    const auto device = gpu_count ? torch::Device(torch::kCUDA,
        static_cast<c10::DeviceIndex>(static_cast<std::size_t>(worker_index) % gpu_count)) : torch::Device(torch::kCPU);
    auto intra = image::from_checkpoint(point.at("i_frame_model_path").get<std::string>(),
        point.at("i_frame_model_name").get<std::string>());
    intra->eval();
    VideoModel video;
    const auto model_path = point.at("model_path").get<std::string>();
    video->load_checkpoint(model_path);
    video->to(device); video->eval();
    intra->to(device); intra->eval();
    if (write_stream) { video->update(true); intra->update({}, true); }

    const auto sequence = point.at("video_path").get<std::string>();
    const auto source = std::filesystem::path(point.at("dataset_path").get<std::string>()) / sequence;
    const auto folder = std::filesystem::path(point.at("recon_bin_path").get<std::string>()) /
        sequence / remove_last_four_characters(model_basename(model_path));
    // The reference creates this directory even without stream/reconstruction output.
    if (!std::filesystem::exists(folder)) std::filesystem::create_directories(folder);
    bool short_names = false, long_names = false;
    for (const auto& entry : std::filesystem::directory_iterator(source)) {
        short_names |= entry.path().filename() == "im1.png";
        long_names |= entry.path().filename() == "im00001.png";
    }
    if (!short_names && !long_names) throw std::invalid_argument("unknown image naming convention; please specify");
    const auto padding = short_names ? 1u : 5u;
    const auto frames = frame_count(point.at("frame_num"));
    if (frames <= 0) throw std::invalid_argument("DCVC evaluation requires at least one frame");
    const auto& gop = point.at("gop");
    const auto save_reconstruction = point.at("write_recon_frame").get<bool>();
    const auto psnr = point.at("model_type").get<std::string>() == "psnr";
    std::uint64_t pixels = 0;
    std::int64_t i_count = 0, p_count = 0;
    double i_bits = 0, i_quality = 0, p_bits = 0, p_quality = 0;
    double p_mv_y = 0, p_mv_z = 0, p_y = 0, p_z = 0;
    torch::Tensor reference;
    torch::NoGradGuard no_grad;
    for (std::int64_t index = 0; index < frames; ++index) {
        ProcessInterrupt::check();
        const auto original = read_frame(source / frame_name(index, padding)).to(device);
        const auto area = product(static_cast<std::uint64_t>(original.size(2)),
            static_cast<std::uint64_t>(original.size(3)));
        if (!pixels) pixels = area;
        else if (pixels != area) throw std::invalid_argument("DCVC frame pixel counts differ within a sequence");
        const auto is_intra = intra_frame(index, gop);
        double bits = 0, mv_y = 0, mv_z = 0, y = 0, z = 0;
        if (write_stream) {
            const auto stream = folder / (std::to_string(index) + ".bin");
            if (is_intra) {
                const auto result = intra->encode_decode(original, stream);
                reference = result.x_hat; bits = result.bpp * pixels;
            } else {
                const auto result = video->encode_decode(reference, original, stream);
                reference = result.recon_image;
                bits = result.bpp * pixels; mv_y = result.bpp_mv_y * pixels;
                mv_z = result.bpp_mv_z * pixels; y = result.bpp_y * pixels; z = result.bpp_z * pixels;
            }
        } else if (is_intra) {
            const auto result = intra->forward(original);
            reference = result.x_hat;
            torch::Tensor total;
            for (const auto& likelihood : result.likelihoods) {
                const auto term = torch::log(likelihood.second).sum() / (-std::log(2.0));
                total = total.defined() ? total + term : 0 + term;
            }
            bits = total.item<double>();
        } else {
            const auto result = video->forward(reference, original);
            reference = result.recon_image;
            bits = result.bpp.item<double>() * pixels;
            mv_y = result.bpp_mv_y.item<double>() * pixels; mv_z = result.bpp_mv_z.item<double>() * pixels;
            y = result.bpp_y.item<double>() * pixels; z = result.bpp_z.item<double>() * pixels;
        }
        reference.clamp_(0, 1);
        if (save_reconstruction) write_frame(reference.squeeze(), folder / ("recon_frame_" + std::to_string(index) + ".png"));
        const auto quality = psnr ? evaluation_psnr(reference, original) : evaluation_msssim(reference, original);
        if (is_intra) { ++i_count; i_bits += bits; i_quality += quality; }
        else {
            ++p_count; p_bits += bits; p_quality += quality;
            p_mv_y += mv_y; p_mv_z += mv_z; p_y += y; p_z += z;
        }
    }
    if (!i_count || !pixels) throw std::invalid_argument("DCVC evaluation has no I-frame pixels");
    Json log{{"name", model_basename(model_path) + "_" + sequence},
        {"ds_name", point.at("ds_name")}, {"video_path", sequence}, {"frame_pixel_num", pixels},
        {"i_frame_num", i_count}, {"p_frame_num", p_count},
        {"ave_i_frame_bpp", i_bits / i_count / pixels}, {"ave_i_frame_quality", i_quality / i_count}};
    if (p_count) {
        const auto p_pixels = product(static_cast<std::uint64_t>(p_count), pixels);
        log["ave_p_frame_bpp"] = p_bits / p_pixels;
        log["ave_p_frame_bpp_mv_y"] = p_mv_y / p_pixels; log["ave_p_frame_bpp_mv_z"] = p_mv_z / p_pixels;
        log["ave_p_frame_bpp_y"] = p_y / p_pixels; log["ave_p_frame_bpp_z"] = p_z / p_pixels;
        log["ave_p_frame_quality"] = p_quality / p_count;
    } else {
        log["ave_p_frame_bpp"] = 0; log["ave_p_frame_quality"] = 0;
        log["ave_p_frame_bpp_mv_y"] = 0; log["ave_p_frame_bpp_mv_z"] = 0;
        log["ave_p_frame_bpp_y"] = 0; log["ave_p_frame_bpp_z"] = 0;
    }
    log["ave_all_frame_bpp"] = (i_bits + p_bits) / product(static_cast<std::uint64_t>(frames), pixels);
    log["ave_all_frame_quality"] = (i_quality + p_quality) / frames;
    log["model_idx"] = point.at("model_idx");
    return log;
}

int run_video_evaluation_command(int argc, char** argv) {
    const auto options = cli::parse(argc, argv);
    const ProcessInterrupt interruption;
    const auto workers = options.count("__worker") ? cli::integer(options.at("__worker"), "--worker") : 1;
    if (workers < 1) throw std::invalid_argument("--worker must be at least 1");
    if (options.count("__cuda_device") && !options.at("__cuda_device").empty() &&
        setenv("CUDA_VISIBLE_DEVICES", options.at("__cuda_device").c_str(), 1) != 0) {
        throw std::runtime_error("Cannot set CUDA_VISIBLE_DEVICES");
    }
    if (setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8", 1) != 0) throw std::runtime_error("Cannot set CUBLAS_WORKSPACE_CONFIG");
    const auto models = cli::list(options, "__model_path");
    const auto intra_models = cli::list(options, "__i_frame_model_path");
    if (models.empty()) throw std::invalid_argument("DCVC evaluation requires --model_path");
    if (intra_models.size() < models.size()) throw std::invalid_argument("DCVC evaluation requires an I-frame checkpoint for each P-frame checkpoint");
    std::ifstream input(options.at("__test_config"));
    if (!input) throw std::runtime_error("Cannot open DCVC evaluation configuration");
    const auto config = uf::evaluation::read_json(input);
    if (!config.is_object()) throw std::invalid_argument("DCVC evaluation configuration must be an object");
    const auto get = [&](const char* name, const char* fallback) {
        const auto found = options.find(name); return found == options.end() ? std::string(fallback) : found->second;
    };
    const auto enabled = [&](const char* name) { const auto found = options.find(name); return found != options.end() && found->second == "true"; };
    const auto start = std::chrono::steady_clock::now();
    std::vector<Json> points;
    std::uint64_t count_frames = 0, count_sequences = 0;
    for (auto dataset = config.begin(); dataset != config.end(); ++dataset) {
        const auto& sequences = dataset.value().at("sequences");
        if (!sequences.is_object()) throw std::invalid_argument("DCVC evaluation sequences must be an object");
        for (auto sequence = sequences.begin(); sequence != sequences.end(); ++sequence) {
            ++count_sequences;
            const auto frames = frame_count(sequence.value().at("frames"));
            if (frames <= 0) throw std::invalid_argument("DCVC evaluation requires at least one frame");
            for (std::size_t model = 0; model < models.size(); ++model) {
                if (count_frames > std::numeric_limits<std::uint64_t>::max() - static_cast<std::uint64_t>(frames)) {
                    throw std::overflow_error("DCVC evaluation total frame count exceeds the native integer range");
                }
                count_frames += static_cast<std::uint64_t>(frames);
                points.push_back(Json{{"model_idx", model}, {"i_frame_model_path", intra_models[model]},
                    {"i_frame_model_name", get("__i_frame_model_name", "cheng2020-anchor")}, {"model_path", models[model]},
                    {"video_path", sequence.key()}, {"gop", sequence.value().at("gop")}, {"frame_num", frames},
                    {"dataset_path", dataset.value().at("base_path")}, {"write_stream", enabled("__write_stream")},
                    {"write_recon_frame", enabled("__write_recon_frame")}, {"recon_bin_path", get("__recon_bin_path", "recon_bin_path")},
                    {"model_type", get("__model_type", "psnr")}, {"ds_name", dataset.key()}});
            }
        }
    }
    const Json initialization{{"backend", "dcvc2021"}, {"settings", Json{{"cuda", enabled("__cuda")}}}};
    const auto results = uf::evaluation::evaluate_documents(std::vector<Json>(static_cast<std::size_t>(workers), initialization), points);
    Json log = Json::object();
    for (auto dataset = config.begin(); dataset != config.end(); ++dataset) {
        auto& dataset_log = log[dataset.key()] = Json::object();
        const auto& sequences = dataset.value().at("sequences");
        for (auto sequence = sequences.begin(); sequence != sequences.end(); ++sequence) {
            auto& sequence_log = dataset_log[sequence.key()] = Json::object();
            for (const auto& model : models) {
                const auto checkpoint = model_basename(model);
                // Preserve the original startswith matching, including basename
                // collisions and last matching result behavior.
                for (const auto& result : results) {
                    if (result.at("name").get<std::string>().rfind(checkpoint, 0) == 0 &&
                        result.at("ds_name") == dataset.key() && result.at("video_path") == sequence.key()) {
                        sequence_log[checkpoint] = filter_result(result);
                    }
                }
            }
        }
    }
    std::ofstream output(options.at("__output_json_result_path"), std::ios::trunc);
    if (!output) throw std::runtime_error("Cannot create DCVC evaluation result JSON");
    uf::evaluation::write_json(log, output, -1, 2);
    output.close();
    if (!output) throw std::runtime_error("Cannot write DCVC evaluation result JSON");
    const auto minutes = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() / 60;
    std::ostringstream elapsed;
    elapsed.imbue(std::locale::classic()); elapsed << std::fixed << std::setprecision(1) << minutes;
    std::cout << "Test finished\nTested " << models.size() << " models on " << count_frames / models.size()
        << " frames from " << count_sequences << " sequences\nTotal elapsed time: " << elapsed.str() << " min\n";
    return 0;
}
}  // namespace uf::dcvc2021
