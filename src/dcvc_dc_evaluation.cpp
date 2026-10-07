// Copyright (c) Microsoft Corporation. Licensed under the MIT License.
// Native counterpart of DCVC-family/DCVC-DC/test_video.py.
#include "uf/dcvc_dc_evaluation.hpp"
#include "uf/dcvc_dc.hpp"
#include "uf/dcvc_tcm.hpp"
#include "uf/cli.hpp"
#include "uf/process.hpp"
#include "evaluation_workflow.hpp"
#include <ATen/Context.h>
#include <torch/cuda.h>
#include <torch/nn/functional.h>
#include <nlohmann/json.hpp>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <optional>
#include <sstream>
#include <stdexcept>

namespace uf::dc {
namespace {
using Clock = std::chrono::system_clock;
double seconds(Clock::time_point start) { return std::chrono::duration<double>(Clock::now() - start).count(); }
std::int64_t integer(const Json& value) {
    if (value.is_boolean()) return value.get<bool>() ? 1 : 0;
    if (!value.is_number_integer() || (value.is_number_unsigned() && value.get<std::uint64_t>() >
        static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())))
        throw std::invalid_argument("DC dimensions and frame counts must fit native integers");
    return value.get<std::int64_t>();
}
bool intra_frame(std::int64_t index, const Json& gop) {
    if (gop.is_number_float()) {
        const auto period = gop.get<double>();
        if (period == 0) throw std::invalid_argument("DC GOP cannot be zero");
        return std::fmod(static_cast<double>(index), period) == 0;
    }
    if (gop.is_number_unsigned()) {
        const auto period = gop.get<std::uint64_t>();
        if (!period) throw std::invalid_argument("DC GOP cannot be zero");
        return static_cast<std::uint64_t>(index) % period == 0;
    }
    const auto period = integer(gop);
    if (!period) throw std::invalid_argument("DC GOP cannot be zero");
    return index % period == 0;
}
bool disabled(const Json& dataset) {
    const auto& test = dataset.at("test");
    return (test.is_boolean() && !test.get<bool>()) || (test.is_number() && test.get<double>() == 0);
}
std::string rate_key(int rate) {
    auto key = std::to_string(rate); if (key.size() < 3) key.insert(0, 3 - key.size(), '0'); return key;
}
void print_scales(const char* label, const torch::Tensor& scales) {
    std::ostringstream line; line.imbue(std::locale::classic()); line << label << std::fixed << std::setprecision(3);
    const auto values = scales.reshape({-1}).to(torch::kCPU);
    for (std::int64_t index = 0; index < values.numel(); ++index) line << values[index].item<double>() << ", ";
    std::cout << line.str() << '\n';
}
std::uint64_t pixels(std::int64_t height, std::int64_t width) {
    if (height <= 0 || width <= 0 || static_cast<std::uint64_t>(height) >
        std::numeric_limits<std::uint64_t>::max() / static_cast<std::uint64_t>(width))
        throw std::invalid_argument("DC evaluation requires bounded, positive frame dimensions");
    return static_cast<std::uint64_t>(height) * static_cast<std::uint64_t>(width);
}
}  // namespace

struct EvaluationModels::Impl {
    IntraNoAR intra{nullptr}; DMC inter{nullptr};
    tcm::MSSSIM rgb_ssim{nullptr};
    torch::Device device{torch::kCPU};
    Impl(const Json& settings, int worker_index) {
        at::globalContext().setUserEnabledCuDNN(true);
        at::globalContext().setBenchmarkCuDNN(false);
        at::globalContext().setDeterministicAlgorithms(true, false);
        torch::manual_seed(0); torch::set_num_threads(1);
        const auto gpu_count = settings.at("cuda").get<bool>() ? torch::cuda::device_count() : 0;
        if (gpu_count) device = torch::Device(torch::kCUDA,
            static_cast<c10::DeviceIndex>(static_cast<std::size_t>(worker_index) % gpu_count));
        // Read state before module construction to preserve initializer order.
        const auto intra_state = get_state_dict(settings.at("i_frame_model_path").get<std::string>());
        intra = IntraNoAR(256, 4, settings.at("ec_thread").get<bool>(), settings.at("stream_part_i").get<int>(), true);
        intra->load_state_dict(intra_state); intra->to(device); intra->eval();
        if (!settings.at("force_intra").get<bool>()) {
            const auto inter_state = get_state_dict(settings.at("p_frame_model_path").get<std::string>());
            inter = DMC(4, settings.at("ec_thread").get<bool>(), settings.at("stream_part_p").get<int>(), true);
            inter->load_state_dict(inter_state); inter->to(device); inter->eval();
        }
        if (settings.at("write_stream").get<bool>()) {
            if (inter) inter->update(true);
            intra->update(true);
        }
    }
    Json run(const Json& point) {
        const auto frames = integer(point.at("frame_num"));
        if (frames <= 0) throw std::invalid_argument("DC evaluation requires at least one frame");
        const auto sequence = point.at("video_path").get<std::string>();
        const auto rate = point.at("rate_idx").get<int>();
        const auto bin_folder = std::filesystem::path(point.at("stream_path").get<std::string>()) / sequence / std::to_string(rate);
        const auto write_stream = point.at("write_stream").get<bool>();
        if (write_stream) create_folder(bin_folder, true);
        const auto save = point.at("save_decoded_frame").get<bool>();
        std::filesystem::path recon_path;
        if (save) {
            recon_path = std::filesystem::path(point.at("decoded_frame_path").get<std::string>()) / sequence / std::to_string(rate);
            create_folder(recon_path);
        }
        const auto src_path = std::filesystem::path(point.at("dataset_path").get<std::string>()) / sequence;
        const auto width = integer(point.at("src_width")), height = integer(point.at("src_height"));
        const auto source_type = point.at("src_type").get<std::string>();
        std::unique_ptr<VideoReader> reader; std::unique_ptr<VideoWriter> writer;
        if (source_type == "png") {
            reader = std::make_unique<PNGReader>(src_path, width, height);
            if (save) writer = std::make_unique<PNGWriter>(recon_path, width, height);
        } else if (source_type == "yuv420") {
            reader = std::make_unique<YUVReader>(src_path, width, height);
            if (save) writer = std::make_unique<YUVWriter>(recon_path, width, height);
        } else throw std::invalid_argument("DC src_type must be png or yuv420");
        const auto yuv = point.at("dist_in_yuv420").get<bool>(), calc_ssim = point.at("calc_ssim").get<bool>();
        const auto verbose = point.value("verbose", 0);
        std::vector<int> types; std::vector<double> bits; std::vector<FrameQuality> quality;
        std::uint64_t frame_pixels = 0; std::int64_t pic_width = 0, pic_height = 0, p_count = 0;
        double encoding_time = 0, decoding_time = 0;
        Dpb reference; const auto start = Clock::now();
        {
            const torch::NoGradGuard no_grad;
            for (std::int64_t frame_index = 0; frame_index < frames; ++frame_index) {
                ProcessInterrupt::check(); const auto frame_start = Clock::now();
                const auto source = reader->read_one_frame(yuv ? "420" : "rgb");
                if (!source.defined()) throw std::runtime_error("DC sequence ended before configured frame count");
                const auto x = np_image_to_tensor(yuv ? ycbcr420_to_444(source.y, source.uv, 0) : source.rgb).to(device);
                pic_height = x.size(2); pic_width = x.size(3);
                const auto area = pixels(pic_height, pic_width);
                if (frame_pixels && frame_pixels != area) throw std::invalid_argument("DC frame pixel counts differ");
                frame_pixels = area;
                const auto right = (16 - pic_width % 16) % 16, bottom = (16 - pic_height % 16) % 16;
                const auto padded = torch::nn::functional::pad(x,
                    torch::nn::functional::PadFuncOptions({0, right, 0, bottom}).mode(torch::kReplicate));
                const auto output = write_stream ? std::optional<std::filesystem::path>(bin_folder / (std::to_string(frame_index) + ".bin")) : std::nullopt;
                const auto in_checkpoint = point.at("q_in_ckpt").get<bool>();
                const auto q_index = integer(point.at("i_frame_q_index"));
                if (intra_frame(frame_index, point.at("gop_size"))) {
                    const auto result = intra->encode_decode(padded, in_checkpoint, q_index, output, pic_width, pic_height);
                    reference = {result.x_hat, {}, {}, {}, {}};
                    types.push_back(0); bits.push_back(result.bit);
                } else {
                    if (!inter) throw std::invalid_argument("DC GOP requested a P frame with force_intra enabled");
                    // The Python evaluation uses its I index for P frames too.
                    const auto result = inter->encode_decode(padded, reference, in_checkpoint, q_index, output,
                        pic_width, pic_height, frame_index % 4);
                    reference = result.dpb; types.push_back(1); bits.push_back(result.bit);
                    ++p_count; encoding_time += result.encoding_time; decoding_time += result.decoding_time;
                }
                reference.ref_frame.clamp_(0, 1);
                const auto x_hat = torch::nn::functional::pad(reference.ref_frame,
                    torch::nn::functional::PadFuncOptions({0, -right, 0, -bottom}));
                FrameQuality measurement; MediaFrame decoded;
                if (yuv) {
                    const auto planes = ycbcr444_to_420(x_hat.squeeze(0).to(torch::kCPU));
                    const auto y_rec = planes.first[0], u_rec = planes.second[0], v_rec = planes.second[1];
                    const auto y_src = source.y[0], u_src = source.uv[0], v_src = source.uv[1];
                    const auto py = uf::dc::calc_psnr(y_src, y_rec, 1), pu = uf::dc::calc_psnr(u_src, u_rec, 1), pv = uf::dc::calc_psnr(v_src, v_rec, 1);
                    const auto sy = calc_ssim ? uf::dc::calc_msssim(y_src, y_rec, 1) : 0;
                    const auto su = calc_ssim ? uf::dc::calc_msssim(u_src, u_rec, 1) : 0;
                    const auto sv = calc_ssim ? uf::dc::calc_msssim(v_src, v_rec, 1) : 0;
                    measurement.psnr = {(6 * py + pu + pv) / 8, py, pu, pv};
                    measurement.msssim = {(6 * sy + su + sv) / 8, sy, su, sv};
                    decoded = {{}, y_rec, planes.second};
                } else {
                    measurement.psnr = {PSNR(x_hat, x)};
                    if (calc_ssim && !rgb_ssim) rgb_ssim = tcm::MSSSIM(11, true);
                    measurement.msssim = {calc_ssim ? rgb_ssim->forward(x_hat, x).item<double>() : 0};
                }
                quality.push_back(measurement);
                const auto elapsed = seconds(frame_start);
                if (verbose >= 2) {
                    std::ostringstream line; line.imbue(std::locale::classic());
                    line << "frame " << frame_index << ", " << std::fixed << std::setprecision(3) << elapsed
                         << " seconds, bits: " << bits.back() << ", PSNR: " << std::setprecision(4) << measurement.psnr[0]
                         << ", MS-SSIM: " << measurement.msssim[0] << " \n"; std::cout << line.str();
                }
                if (save) {
                    if (!yuv) decoded.rgb = x_hat.squeeze(0).to(torch::kCPU);
                    writer->write_one_frame(decoded, yuv ? "420" : "rgb");
                }
            }
        }
        if (save) {
            double bit_sum = 0, psnr_sum = 0;
            for (std::size_t index = 0; index < bits.size(); ++index) { bit_sum += bits[index]; psnr_sum += quality[index].psnr[0]; }
            const auto bpp = bit_sum / bits.size() / pic_width / pic_height;
            std::ostringstream name; name.imbue(std::locale::classic());
            name << rate << '_' << std::fixed << std::setprecision(4) << bpp << '_' << psnr_sum / quality.size();
            // Keep the YUV writer open across the POSIX rename, as Python does.
            std::filesystem::rename(recon_path, std::filesystem::path(recon_path.string() + "/../" + name.str()));
        }
        const auto time = seconds(start);
        if (verbose >= 1 && p_count > 0) {
            std::ostringstream line; line.imbue(std::locale::classic());
            line << "encoding/decoding " << p_count << " P frames, average encoding time " << std::fixed << std::setprecision(0)
                 << encoding_time / p_count * 1000 << " ms, average decoding time " << decoding_time / p_count * 1000 << " ms.\n";
            std::cout << line.str();
        }
        auto result = uf::dc::generate_log_json(frames, frame_pixels, time, types, bits, quality);
        result["ds_name"] = point.at("ds_name"); result["video_path"] = sequence; result["rate_idx"] = rate;
        return result;
    }
};
EvaluationModels::EvaluationModels(const Json& settings, int worker_index) : impl_(std::make_unique<Impl>(settings, worker_index)) {}
EvaluationModels::~EvaluationModels() = default;
Json EvaluationModels::run(const Json& point) { return impl_->run(point); }

int run_video_evaluation_command(int argc, char** argv) {
    const auto start = Clock::now(); at::globalContext().setUserEnabledCuDNN(true);
    const auto options = cli::parse(argc, argv); const ProcessInterrupt interruption;
    const auto get = [&](const char* key, const char* fallback) {
        const auto found = options.find(key); return found == options.end() ? std::string(fallback) : found->second;
    };
    const auto enabled = [&](const char* key) { return get(key, "false") == "true"; };
    if (options.count("__cuda_device") && !options.at("__cuda_device").empty() &&
        setenv("CUDA_VISIBLE_DEVICES", options.at("__cuda_device").c_str(), 1) != 0)
        throw std::runtime_error("Cannot set CUDA_VISIBLE_DEVICES");
    if (setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8", 1) != 0) throw std::runtime_error("Cannot set CUBLAS_WORKSPACE_CONFIG");
    const auto workers = cli::integer(get("__worker", "1"), "--worker");
    if (workers < 1) throw std::invalid_argument("--worker must be at least one");
    std::ifstream input(options.at("__test_config"));
    if (!input) throw std::runtime_error("Cannot open DC evaluation configuration");
    const auto config = uf::evaluation::read_json(input);
    const auto rates = cli::integer(get("__rate_num", "4"), "--rate_num");
    const auto intra_path = get("__i_frame_model_path", ""), inter_path = get("__p_frame_model_path", "");
    const auto scales = IntraNoARImpl::get_q_scales_from_ckpt(intra_path);
    print_scales("q_scale_enc in intra ckpt: ", scales.first); print_scales("q_scale_dec in intra ckpt: ", scales.second);
    std::vector<std::int64_t> indexes; bool in_checkpoint = false;
    if (options.count("__i_frame_q_indexes")) {
        const auto values = cli::list(options, "__i_frame_q_indexes");
        if (rates < 0 || values.size() != static_cast<std::size_t>(rates)) throw std::invalid_argument("i_frame_q_indexes must contain rate_num indexes");
        for (const auto& value : values) indexes.push_back(cli::signed_integer(value, "--i_frame_q_indexes"));
    } else if (scales.first.numel() == rates) {
        if (rates != 4) throw std::invalid_argument("DC checkpoint rate count must be four");
        in_checkpoint = true; indexes = {0, 1, 2, 3};
    } else {
        if (rates < 2 || rates > 64) throw std::invalid_argument("Interpolated DC rate_num must be between two and 64");
        const auto step = 63.0 / (rates - 1);
        for (int rate = 0; rate < rates; ++rate) {
            const auto value = rate == rates - 1 ? 63.0 : static_cast<double>(rate) * step;
            indexes.push_back(static_cast<std::int64_t>(value + 0.5));
        }
    }
    const auto force_intra = enabled("__force_intra");
    if (!force_intra) {
        const auto inter_scales = DMCImpl::get_q_scales_from_ckpt(inter_path);
        print_scales("y_q_scale_enc in inter ckpt: ", inter_scales[0]); print_scales("y_q_scale_dec in inter ckpt: ", inter_scales[1]);
        print_scales("mv_y_q_scale_enc in inter ckpt: ", inter_scales[2]); print_scales("mv_y_q_scale_dec in inter ckpt: ", inter_scales[3]);
    }
    // p_frame_q_indexes is parsed and ignored by the original main function.
    std::cout << "testing " << rates << " rates, using q_indexes: ";
    for (const auto index : indexes) std::cout << index << ", "; std::cout << '\n';
    const auto root = config.at("root_path").get<std::string>(); const auto& datasets = config.at("test_classes");
    if (!datasets.is_object()) throw std::invalid_argument("DC test_classes must be an object");
    const auto forced_frames = cli::signed_integer(get("__force_frame_num", "-1"), "--force_frame_num");
    const auto forced_period = cli::signed_integer(get("__force_intra_period", "-1"), "--force_intra_period");
    const auto verbose = cli::integer(get("__verbose", "0"), "--verbose");
    std::uint64_t count_frames = 0, count_sequences = 0; std::vector<Json> points;
    for (auto dataset = datasets.begin(); dataset != datasets.end(); ++dataset) {
        if (disabled(dataset.value())) continue;
        const auto& sequences = dataset.value().at("sequences");
        if (!sequences.is_object()) throw std::invalid_argument("DC sequences must be an object");
        for (auto sequence = sequences.begin(); sequence != sequences.end(); ++sequence) {
            if (count_sequences == std::numeric_limits<std::uint64_t>::max()) throw std::overflow_error("DC sequence count exceeds native integers");
            ++count_sequences; auto gop = sequence.value().at("gop");
            if (force_intra) gop = 1; if (forced_period > 0) gop = forced_period;
            const auto frame_count = forced_frames > 0 ? forced_frames : integer(sequence.value().at("frames"));
            if (frame_count <= 0) throw std::invalid_argument("DC evaluation requires at least one frame");
            for (int rate = 0; rate < rates; ++rate) {
                if (count_frames > std::numeric_limits<std::uint64_t>::max() - static_cast<std::uint64_t>(frame_count))
                    throw std::overflow_error("DC frame count exceeds native integers");
                count_frames += static_cast<std::uint64_t>(frame_count);
                Json point{{"rate_idx", rate}, {"q_in_ckpt", in_checkpoint}, {"i_frame_q_index", indexes.at(rate)},
                    {"force_intra", force_intra}, {"video_path", sequence.key()}, {"dist_in_yuv420", enabled("__yuv420")},
                    {"src_type", dataset.value().at("src_type")}, {"src_height", sequence.value().at("height")},
                    {"src_width", sequence.value().at("width")}, {"gop_size", gop}, {"frame_num", frame_count},
                    {"calc_ssim", enabled("__calc_ssim")}, {"dataset_path", (std::filesystem::path(root) / dataset.value().at("base_path").get<std::string>()).string()},
                    {"write_stream", enabled("__write_stream")}, {"stream_path", get("__stream_path", "out_bin")},
                    {"save_decoded_frame", enabled("__save_decoded_frame")}, {"decoded_frame_path", get("__decoded_frame_path", "decoded_frames")},
                    {"ds_name", dataset.key()}, {"verbose", verbose}};
                if (!force_intra) point["p_frame_q_index"] = indexes.at(rate);
                points.push_back(std::move(point));
            }
        }
    }
    const Json settings{{"cuda", enabled("__cuda")}, {"i_frame_model_path", intra_path}, {"p_frame_model_path", inter_path},
        {"force_intra", force_intra}, {"ec_thread", enabled("__ec_thread")}, {"stream_part_i", cli::integer(get("__stream_part_i", "1"), "--stream_part_i")},
        {"stream_part_p", cli::integer(get("__stream_part_p", "1"), "--stream_part_p")}, {"write_stream", enabled("__write_stream")}};
    const Json init{{"backend", "dcvc_dc"}, {"settings", settings}};
    const auto results = uf::evaluation::evaluate_documents(std::vector<Json>(static_cast<std::size_t>(workers), init), points);
    Json log = Json::object();
    for (auto dataset = datasets.begin(); dataset != datasets.end(); ++dataset) {
        if (disabled(dataset.value())) continue;
        auto& dataset_log = log[dataset.key()] = Json::object(); const auto& sequences = dataset.value().at("sequences");
        for (auto sequence = sequences.begin(); sequence != sequences.end(); ++sequence) {
            auto& sequence_log = dataset_log[sequence.key()] = Json::object();
            for (int rate = 0; rate < rates; ++rate) for (const auto& result : results)
                if (result.at("rate_idx") == rate && result.at("ds_name") == dataset.key() && result.at("video_path") == sequence.key())
                    sequence_log[rate_key(rate)] = result;
        }
    }
    const std::filesystem::path output_path = options.at("__output_path");
    if (!output_path.parent_path().empty()) create_folder(output_path.parent_path(), true);
    std::ofstream output(output_path, std::ios::trunc);
    if (!output) throw std::runtime_error("Cannot create DC evaluation result JSON");
    dump_json(log, output, 6, 2); output.close();
    if (!output) throw std::runtime_error("Cannot write DC evaluation result JSON");
    std::ostringstream elapsed; elapsed.imbue(std::locale::classic()); elapsed << std::fixed << std::setprecision(1) << seconds(start) / 60;
    std::cout << "Test finished\nTested " << count_frames << " frames from " << count_sequences << " sequences\nTotal elapsed time: " << elapsed.str() << " min\n";
    return 0;
}
}  // namespace uf::dc
