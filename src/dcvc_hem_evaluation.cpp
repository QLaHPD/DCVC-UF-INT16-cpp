// Copyright (c) Microsoft Corporation. Licensed under the MIT License.
// Native counterpart of DCVC-family/DCVC-HEM/test_video.py and PNGReader.
// Log interpolation also adapts NumPy's BSD-3-Clause linspace implementation;
// its notice is retained in licenses/NumPy.txt.
#include "uf/dcvc_hem_evaluation.hpp"
#include "uf/dcvc_hem.hpp"
#include "uf/cli.hpp"
#include "uf/image_pixels.hpp"
#include "uf/process.hpp"
#include "dcvc_hem_evaluation_internal.hpp"
#include "evaluation_workflow.hpp"
#include <ATen/Context.h>
#include <torch/cuda.h>
#include <nlohmann/json.hpp>
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
#include <utility>
#include <vector>

namespace uf::hem {
namespace {
using Json = uf::evaluation::Json;
using Clock = std::chrono::system_clock;
double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
std::int64_t frame_count(const Json& value) {
    if (value.is_boolean()) return value.get<bool>() ? 1 : 0;
    if (!value.is_number_integer() || (value.is_number_unsigned() && value.get<std::uint64_t>() >
        static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())))
        throw std::invalid_argument("HEM sequence dimensions and frame counts must fit native integers");
    return value.get<std::int64_t>();
}
bool intra_frame(std::int64_t index, const Json& gop) {
    if (gop.is_boolean()) {
        if (!gop.get<bool>()) throw std::invalid_argument("HEM GOP cannot be zero");
        return true;
    }
    if (gop.is_number_unsigned()) {
        const auto period = gop.get<std::uint64_t>();
        if (!period) throw std::invalid_argument("HEM GOP cannot be zero");
        return static_cast<std::uint64_t>(index) % period == 0;
    }
    if (gop.is_number_integer()) {
        const auto period = gop.get<std::int64_t>();
        if (!period) throw std::invalid_argument("HEM GOP cannot be zero");
        return index % period == 0;
    }
    if (gop.is_number_float()) {
        const auto period = gop.get<double>();
        if (period == 0) throw std::invalid_argument("HEM GOP cannot be zero");
        return std::fmod(static_cast<double>(index), period) == 0;
    }
    throw std::invalid_argument("HEM GOP must be numeric");
}
bool disabled_dataset(const Json& dataset) {
    const auto& flag = dataset.at("test");
    return (flag.is_boolean() && !flag.get<bool>()) || (flag.is_number() && flag.get<double>() == 0);
}
std::uint64_t product(std::uint64_t left, std::uint64_t right) {
    if (left && right > std::numeric_limits<std::uint64_t>::max() / left)
        throw std::overflow_error("HEM evaluation pixel count exceeds the native integer range");
    return left * right;
}
void create_folder(const std::filesystem::path& path, bool print = false) {
    if (!std::filesystem::exists(path)) {
        std::filesystem::create_directories(path);
        if (print) std::cout << "created folder: " << path.string() << '\n';
    }
}
torch::Tensor rgb_tensor(const ImagePixels& pixels, torch::ScalarType dtype) {
    return torch::from_blob(const_cast<std::uint8_t*>(pixels.rgb.data()),
        {pixels.height, pixels.width, 3}, torch::TensorOptions().dtype(torch::kUInt8))
        .to(dtype).permute({2, 0, 1});
}
struct Quality { double value; torch::ScalarType dtype; };
std::vector<Quality> checkpoint_qualities(const torch::Tensor& tensor) {
    const auto values = tensor.reshape({-1}).to(torch::kCPU);
    const auto type = values.scalar_type();
    if (type != torch::kFloat32 && type != torch::kFloat64 && type != torch::kFloat16)
        throw std::invalid_argument("HEM evaluation checkpoint q_scales must be Float16, Float32 or Float64");
    std::vector<Quality> result;
    for (std::int64_t index = 0; index < values.numel(); ++index)
        result.push_back({values[index].item<double>(), type});
    return result;
}
std::vector<Quality> explicit_qualities(const cli::Options& options, const std::string& name, int count) {
    const auto items = cli::list(options, name);
    if (items.size() != static_cast<std::size_t>(count))
        throw std::invalid_argument(name + " must contain rate_num values");
    std::vector<Quality> result;
    for (const auto& item : items) result.push_back({cli::number(item, name), torch::kFloat64});
    return result;
}
Json quality_document(const Quality& scale) {
    return Json{{"value", scale.value}, {"dtype", static_cast<int>(scale.dtype)}};
}
torch::Tensor quality_tensor(const Json& document) {
    const auto type = static_cast<torch::ScalarType>(document.at("dtype").get<int>());
    if (type != torch::kFloat32 && type != torch::kFloat64 && type != torch::kFloat16)
        throw std::invalid_argument("Unsupported HEM evaluation quality dtype");
    return torch::tensor(document.at("value").get<double>(), torch::TensorOptions().dtype(type));
}
double endpoint_log(const Quality& scale) {
    // np.log preserves the checkpoint scalar dtype before linspace promotes
    // its zero-dimensional endpoints to Float64 in the installed NumPy source.
    if (scale.dtype == torch::kFloat32) return std::log(static_cast<float>(scale.value));
    if (scale.dtype == torch::kFloat16)
        return static_cast<double>(static_cast<c10::Half>(std::log(static_cast<float>(scale.value))));
    return std::log(scale.value);
}
std::vector<Quality> interpolate_qualities(const std::vector<Quality>& source, int count) {
    if (source.empty() || !(source.front().value > source.back().value) || !(source.back().value > 0))
        throw std::invalid_argument("HEM log interpolation requires max_q > min_q > 0");
    const auto first = endpoint_log(source.front()), last = endpoint_log(source.back());
    const auto delta = last - first;
    const auto divisor = count - 1;
    const auto step = divisor > 0 ? delta / divisor : std::numeric_limits<double>::quiet_NaN();
    std::vector<Quality> result;
    for (int index = 0; index < count; ++index) {
        double value;
        if (divisor <= 0) value = static_cast<double>(index) * delta;
        else if (step == 0) value = (static_cast<double>(index) / divisor) * delta;
        else value = static_cast<double>(index) * step;
        value += first;
        if (divisor > 0 && index == count - 1) value = last;
        result.push_back({std::exp(value), torch::kFloat64});
    }
    return result;
}
void print_qualities(const std::string& label, const std::vector<Quality>& scales) {
    std::ostringstream line; line.imbue(std::locale::classic());
    line << label << std::fixed << std::setprecision(3);
    for (const auto& scale : scales) line << scale.value << ", ";
    std::cout << line.str() << '\n';
}
std::string rate_key(int rate) {
    auto result = std::to_string(rate);
    if (result.size() < 3) result.insert(0, 3 - result.size(), '0');
    return result;
}
struct FrameLog { int type; double bit, psnr, msssim; };
Json generate_log(const std::vector<FrameLog>& frames, std::uint64_t pixels, double time) {
    std::int64_t i_count = 0, p_count = 0;
    double i_bits = 0, i_psnr = 0, i_ssim = 0, p_bits = 0, p_psnr = 0, p_ssim = 0;
    for (const auto& frame : frames) {
        if (!frame.type) { ++i_count; i_bits += frame.bit; i_psnr += frame.psnr; i_ssim += frame.msssim; }
        else { ++p_count; p_bits += frame.bit; p_psnr += frame.psnr; p_ssim += frame.msssim; }
    }
    if (!i_count || !pixels) throw std::invalid_argument("HEM evaluation has no I-frame pixels");
    Json result{{"frame_pixel_num", pixels}, {"i_frame_num", i_count}, {"p_frame_num", p_count},
        {"ave_i_frame_bpp", i_bits / i_count / pixels}, {"ave_i_frame_psnr", i_psnr / i_count},
        {"ave_i_frame_msssim", i_ssim / i_count}};
    for (const auto* field : {"frame_bpp", "frame_psnr", "frame_msssim", "frame_type"})
        result[field] = Json::array();
    for (const auto& frame : frames) {
        result["frame_bpp"].push_back(frame.bit / pixels); result["frame_psnr"].push_back(frame.psnr);
        result["frame_msssim"].push_back(frame.msssim); result["frame_type"].push_back(frame.type);
    }
    result["test_time"] = time;
    if (p_count) {
        result["ave_p_frame_bpp"] = p_bits / product(static_cast<std::uint64_t>(p_count), pixels);
        result["ave_p_frame_psnr"] = p_psnr / p_count; result["ave_p_frame_msssim"] = p_ssim / p_count;
    } else {
        for (const auto* field : {"ave_p_frame_bpp", "ave_p_frame_psnr", "ave_p_frame_msssim"}) result[field] = 0;
    }
    result["ave_all_frame_bpp"] = (i_bits + p_bits) / product(frames.size(), pixels);
    result["ave_all_frame_psnr"] = (i_psnr + p_psnr) / frames.size();
    result["ave_all_frame_msssim"] = (i_ssim + p_ssim) / frames.size();
    return result;
}
}  // namespace

torch::Tensor read_image_to_torch(const std::filesystem::path& path) {
    return rgb_tensor(read_image_rgb(path, false), torch::kFloat64).to(torch::kFloat32).unsqueeze(0) / 255;
}
void save_torch_image(const torch::Tensor& image, const std::filesystem::path& path) {
    if (image.dim() != 4 || image.size(0) != 1 || image.size(1) != 3 || image.size(2) < 1 || image.size(3) < 1 ||
        image.size(2) > std::numeric_limits<std::uint32_t>::max() ||
        image.size(3) > std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument("Invalid HEM RGB reconstruction shape");
    const auto values = (image.squeeze(0).permute({1, 2, 0}).detach().to(torch::kCPU) * 255)
        .round().clamp(0, 255).to(torch::kUInt8).contiguous();
    ImagePixels pixels;
    pixels.width = static_cast<std::uint32_t>(image.size(3)); pixels.height = static_cast<std::uint32_t>(image.size(2));
    const auto* bytes = values.data_ptr<std::uint8_t>();
    pixels.rgb.assign(bytes, bytes + values.numel());
    write_image_png(path, pixels);
}
double psnr(const torch::Tensor& first, const torch::Tensor& second) {
    const auto mse = (first - second).pow(2).mean();
    return (20 * torch::log10(1 / torch::sqrt(mse))).item<double>();
}
PNGReader::PNGReader(std::filesystem::path folder, std::int64_t width, std::int64_t height)
    : folder_(std::move(folder)), width_(width), height_(height) {
    bool short_names = false, long_names = false;
    for (const auto& entry : std::filesystem::directory_iterator(folder_)) {
        short_names |= entry.path().filename() == "im1.png";
        long_names |= entry.path().filename() == "im00001.png";
    }
    if (!short_names && !long_names)
        throw std::invalid_argument("unknown image naming convention; please specify");
    padding_ = short_names ? 1u : 5u;
}
torch::Tensor PNGReader::read_one_frame(const std::string& format) {
    if (format != "rgb") throw std::invalid_argument("Native HEM PNGReader supports RGB output");
    if (eof_) return {};
    auto number = std::to_string(index_);
    if (number.size() < padding_) number.insert(0, padding_ - number.size(), '0');
    const auto path = folder_ / ("im" + number + ".png");
    if (!std::filesystem::exists(path)) { eof_ = true; return {}; }
    const auto pixels = read_image_rgb(path, false);
    auto rgb = rgb_tensor(pixels, torch::kFloat32) / 255;
    if (pixels.height != height_ || pixels.width != width_)
        throw std::invalid_argument("HEM PNG dimensions differ from the sequence configuration");
    if (index_ == std::numeric_limits<std::int64_t>::max())
        throw std::overflow_error("HEM PNG frame index exceeds the native integer range");
    ++index_;
    return rgb;
}
void PNGReader::close() { index_ = 1; }

Json evaluate_document(const Json& point, bool use_cuda, int worker_index) {
    at::globalContext().setUserEnabledCuDNN(true);
    at::globalContext().setBenchmarkCuDNN(false);
    at::globalContext().setDeterministicAlgorithms(true, false);
    torch::manual_seed(0);
    torch::set_num_threads(1);
    // NumPy's source seed has no subsequent RNG consumer in this worker.
    const auto gpu_count = use_cuda ? torch::cuda::device_count() : 0;
    const auto device = gpu_count ? torch::Device(torch::kCUDA,
        static_cast<c10::DeviceIndex>(static_cast<std::size_t>(worker_index) % gpu_count)) : torch::Device(torch::kCPU);
    const auto intra_state = get_state_dict(point.at("i_frame_model_path").get<std::string>());
    IntraNoAR intra;
    intra->load_state_dict(intra_state); intra->to(device); intra->eval();
    DMC video{nullptr};
    if (!point.at("force_intra").get<bool>()) {
        const auto video_state = get_state_dict(point.at("model_path").get<std::string>());
        video = DMC(); video->load_state_dict(video_state); video->to(device); video->eval();
    }
    const auto write_stream = point.at("write_stream").get<bool>();
    if (write_stream) { if (video) video->update(true); intra->update(true); }
    const auto sequence = point.at("video_path").get<std::string>();
    const auto rate = std::to_string(point.at("rate_idx").get<int>());
    const auto bin_folder = std::filesystem::path(point.at("stream_path").get<std::string>()) / sequence / rate;
    if (write_stream) create_folder(bin_folder, true);
    const auto save_decoded = point.at("save_decoded_frame").get<bool>();
    const auto decoded_folder = std::filesystem::path(point.at("decoded_frame_path").get<std::string>()) / sequence / rate;
    if (save_decoded) create_folder(decoded_folder);
    if (point.at("src_type") != "png") throw std::invalid_argument("HEM evaluation supports PNG sequences");
    PNGReader reader(std::filesystem::path(point.at("dataset_path").get<std::string>()) / sequence,
                     frame_count(point.at("src_width")), frame_count(point.at("src_height")));
    const auto frames = frame_count(point.at("frame_num"));
    if (frames <= 0) throw std::invalid_argument("HEM evaluation requires at least one frame");
    const auto& gop = point.at("gop");
    const auto verbose = point.at("verbose").get<int>();
    const auto intra_q = quality_tensor(point.at("i_frame_q_scale"));
    const auto motion_q = video ? quality_tensor(point.at("p_frame_mv_y_q_scale")) : torch::Tensor{};
    const auto image_q = video ? quality_tensor(point.at("p_frame_y_q_scale")) : torch::Tensor{};
    std::vector<FrameLog> history;
    std::uint64_t pixels = 0;
    std::int64_t p_frames = 0;
    double decoding_time = 0;
    Dpb reference;
    MSSSIM metric;
    const auto start = Clock::now();
    {
        torch::NoGradGuard no_grad;
        for (std::int64_t index = 0; index < frames; ++index) {
            ProcessInterrupt::check();
            const auto frame_start = Clock::now();
            const auto rgb = reader.read_one_frame();
            if (!rgb.defined()) throw std::runtime_error("HEM PNG sequence ended before frame_num");
            const auto original = rgb.to(torch::kFloat32).unsqueeze(0).to(device);
            const auto height = original.size(2), width = original.size(3);
            const auto area = product(static_cast<std::uint64_t>(height), static_cast<std::uint64_t>(width));
            if (!pixels) pixels = area;
            else if (pixels != area) throw std::invalid_argument("HEM frame pixel counts differ within a sequence");
            const auto padded = hem::pad(original);
            const auto output = write_stream ? std::optional<std::filesystem::path>(bin_folder / (std::to_string(index) + ".bin"))
                                             : std::nullopt;
            FrameLog log{};
            if (intra_frame(index, gop)) {
                const auto result = intra->encode_decode(padded, intra_q, output, width, height);
                reference = Dpb{result.x_hat, {}, {}, {}}; log.bit = result.bit;
            } else {
                if (!video) throw std::runtime_error("HEM force_intra with this GOP reaches a P-frame without a video model");
                const auto result = video->encode_decode(padded, reference, output, width, height, motion_q, image_q);
                reference = result.dpb; log.type = 1; log.bit = result.bit;
                ++p_frames; decoding_time += result.decoding_time;
            }
            reference.ref_frame.clamp_(0, 1);
            const auto reconstruction = hem::crop(reference.ref_frame, height, width);
            log.psnr = hem::psnr(reconstruction, original);
            log.msssim = metric(reconstruction, original).item<double>();
            history.push_back(log);
            const auto elapsed = seconds(frame_start);
            if (verbose >= 2) {
                std::ostringstream line; line.imbue(std::locale::classic()); line << std::fixed;
                line << "frame " << index << ", " << std::setprecision(3) << elapsed << " seconds, bits: " << log.bit
                     << ", PSNR: " << std::setprecision(4) << log.psnr << ", MS-SSIM: " << log.msssim << " \n";
                std::cout << line.str();
            }
            if (save_decoded) save_torch_image(reconstruction, decoded_folder / (std::to_string(index) + ".png"));
        }
    }
    const auto test_time = seconds(start);
    if (verbose >= 1 && p_frames > 0) {
        std::ostringstream line; line.imbue(std::locale::classic());
        line << "decoding " << p_frames << " P frames, average " << std::fixed << std::setprecision(0)
             << decoding_time / p_frames * 1000 << " ms.\n";
        std::cout << line.str();
    }
    auto log = generate_log(history, pixels, test_time);
    log["ds_name"] = point.at("ds_name"); log["video_path"] = sequence; log["rate_idx"] = point.at("rate_idx");
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
    if (setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8", 1) != 0)
        throw std::runtime_error("Cannot set CUBLAS_WORKSPACE_CONFIG");
    std::ifstream input(options.at("__test_config"));
    if (!input) throw std::runtime_error("Cannot open HEM evaluation configuration");
    const auto config = uf::evaluation::read_json(input);
    if (!config.is_object()) throw std::invalid_argument("HEM evaluation configuration must be an object");
    const auto rates = cli::integer(get("__rate_num", "4"), "--rate_num");
    if (rates < 1) throw std::invalid_argument("--rate_num must be at least 1");
    const auto intra_path = get("__i_frame_model_path", "");
    auto intra_scales = checkpoint_qualities(IntraNoARImpl::get_q_scales_from_ckpt(intra_path));
    print_qualities("q_scales in intra ckpt: ", intra_scales);
    std::string label;
    if (options.count("__i_frame_q_scales")) {
        intra_scales = explicit_qualities(options, "__i_frame_q_scales", rates);
        label = "testing " + std::to_string(rates) + " rate points with pre-defined intra y q_scales: ";
    } else if (intra_scales.size() == static_cast<std::size_t>(rates)) {
        label = "testing " + std::to_string(rates) + " rate points with intra y q_scales in ckpt: ";
    } else {
        intra_scales = interpolate_qualities(intra_scales, rates);
        label = "testing " + std::to_string(rates) + " rates, using intra y q_scales: ";
    }
    print_qualities(label, intra_scales);
    const auto force_intra = enabled("__force_intra");
    const auto video_path = get("__model_path", "");
    std::vector<Quality> image_scales, motion_scales;
    if (!force_intra) {
        const auto checkpoint = DMCImpl::get_q_scales_from_ckpt(video_path);
        image_scales = checkpoint_qualities(checkpoint.first); motion_scales = checkpoint_qualities(checkpoint.second);
        print_qualities("y_q_scales in inter ckpt: ", image_scales);
        print_qualities("mv_y_q_scales in inter ckpt: ", motion_scales);
        if (options.count("__p_frame_y_q_scales")) {
            image_scales = explicit_qualities(options, "__p_frame_y_q_scales", rates);
            motion_scales = explicit_qualities(options, "__p_frame_mv_y_q_scales", rates);
            std::cout << "testing " << rates << " rate points with pre-defined inter q_scales\n";
        } else if (image_scales.size() == static_cast<std::size_t>(rates)) {
            std::cout << "testing " << rates << " rate points with inter q_scales in ckpt\n";
        } else {
            image_scales = interpolate_qualities(image_scales, rates);
            motion_scales = interpolate_qualities(motion_scales, rates);
        }
        print_qualities("y_q_scales for testing: ", image_scales);
        print_qualities("mv_y_q_scales for testing: ", motion_scales);
    }
    const auto root_path = options.count("__force_root_path") ? options.at("__force_root_path")
        : config.at("root_path").get<std::string>();
    const auto& datasets = config.at("test_classes");
    if (!datasets.is_object()) throw std::invalid_argument("HEM test_classes must be an object");
    const auto forced_frames = cli::signed_integer(get("__force_frame_num", "-1"), "--force_frame_num");
    const auto forced_period = cli::signed_integer(get("__force_intra_period", "-1"), "--force_intra_period");
    const auto verbose = cli::integer(get("__verbose", "0"), "--verbose");
    std::vector<Json> points;
    std::uint64_t count_frames = 0, count_sequences = 0;
    for (auto dataset = datasets.begin(); dataset != datasets.end(); ++dataset) {
        if (disabled_dataset(dataset.value())) continue;
        const auto& sequences = dataset.value().at("sequences");
        if (!sequences.is_object()) throw std::invalid_argument("HEM sequences must be an object");
        for (auto sequence = sequences.begin(); sequence != sequences.end(); ++sequence) {
            ++count_sequences;
            const auto configured_frames = frame_count(sequence.value().at("frames"));
            const auto frames = forced_frames > 0 ? forced_frames : configured_frames;
            if (frames <= 0) throw std::invalid_argument("HEM evaluation requires at least one frame");
            auto gop = sequence.value().at("gop");
            if (force_intra) gop = 1;
            if (forced_period > 0) gop = forced_period;
            for (int rate = 0; rate < rates; ++rate) {
                if (count_frames > std::numeric_limits<std::uint64_t>::max() - static_cast<std::uint64_t>(frames))
                    throw std::overflow_error("HEM evaluation total frame count exceeds the native integer range");
                count_frames += static_cast<std::uint64_t>(frames);
                Json point{{"rate_idx", rate}, {"i_frame_model_path", intra_path},
                    {"i_frame_q_scale", quality_document(intra_scales.at(static_cast<std::size_t>(rate)))},
                    {"force_intra", force_intra}, {"video_path", sequence.key()}, {"src_type", dataset.value().at("src_type")},
                    {"src_height", sequence.value().at("height")}, {"src_width", sequence.value().at("width")},
                    {"gop", gop}, {"frame_num", frames},
                    {"dataset_path", (std::filesystem::path(root_path) / dataset.value().at("base_path").get<std::string>()).string()},
                    {"write_stream", enabled("__write_stream")}, {"stream_path", get("__stream_path", "out_bin")},
                    {"save_decoded_frame", enabled("__save_decoded_frame")},
                    {"decoded_frame_path", get("__decoded_frame_path", "decoded_frames") + "_DMC_" + std::to_string(rate)},
                    {"ds_name", dataset.key()}, {"verbose", verbose}};
                if (!force_intra) {
                    point["model_path"] = video_path;
                    point["p_frame_y_q_scale"] = quality_document(image_scales.at(static_cast<std::size_t>(rate)));
                    point["p_frame_mv_y_q_scale"] = quality_document(motion_scales.at(static_cast<std::size_t>(rate)));
                }
                points.push_back(std::move(point));
            }
        }
    }
    const Json initialization{{"backend", "dcvc_hem"}, {"settings", Json{{"cuda", enabled("__cuda")}}}};
    const auto results = uf::evaluation::evaluate_documents(std::vector<Json>(static_cast<std::size_t>(workers), initialization), points);
    Json log = Json::object();
    for (auto dataset = datasets.begin(); dataset != datasets.end(); ++dataset) {
        if (disabled_dataset(dataset.value())) continue;
        auto& dataset_log = log[dataset.key()] = Json::object();
        const auto& sequences = dataset.value().at("sequences");
        for (auto sequence = sequences.begin(); sequence != sequences.end(); ++sequence) {
            auto& sequence_log = dataset_log[sequence.key()] = Json::object();
            for (int rate = 0; rate < rates; ++rate)
                for (const auto& result : results)
                    if (result.at("rate_idx") == rate && result.at("ds_name") == dataset.key() &&
                        result.at("video_path") == sequence.key()) sequence_log[rate_key(rate)] = result;
        }
    }
    const std::filesystem::path output_path = options.at("__output_path");
    if (!output_path.parent_path().empty()) create_folder(output_path.parent_path(), true);
    std::ofstream output(output_path, std::ios::trunc);
    if (!output) throw std::runtime_error("Cannot create HEM evaluation result JSON");
    uf::evaluation::write_json(log, output, 6, 2); output.close();
    if (!output) throw std::runtime_error("Cannot write HEM evaluation result JSON");
    std::ostringstream elapsed; elapsed.imbue(std::locale::classic());
    elapsed << std::fixed << std::setprecision(1) << seconds(start) / 60;
    std::cout << "Test finished\nTested " << count_frames << " frames from " << count_sequences
              << " sequences\nTotal elapsed time: " << elapsed.str() << " min\n";
    return 0;
}
}  // namespace uf::hem
