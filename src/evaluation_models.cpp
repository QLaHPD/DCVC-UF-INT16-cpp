// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#include "evaluation_workflow.hpp"
#include "uf/bitstream.hpp"
#include "uf/evaluation_metrics.hpp"
#include "uf/fp16_model.hpp"
#include "uf/video_io.hpp"
#include "uf/transforms.hpp"
#include "uf/process.hpp"
#include "uf/python_json.hpp"

#include <nlohmann/json.hpp>
#include <c10/core/InferenceMode.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <utility>

#ifdef UF_HAS_EVALUATION_CUDA
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#endif

namespace uf::evaluation {
namespace {

std::string replace_all(std::string value, const std::string& old, const std::string& replacement) {
    for (std::size_t position = 0; (position = value.find(old, position)) != std::string::npos; position += replacement.size()) {
        value.replace(position, old.size(), replacement);
    }
    return value;
}

template<class Stream, class Function>
void with_checked_close(Stream& stream, const std::string& message, Function&& body) {
    bool attempted = false;
    const auto close = [&]() {
        attempted = true;
        stream.clear();
        stream.close();
        if (!stream) throw std::runtime_error(message);
    };
    try {
        body();
        close();
    } catch (...) {
        if (!attempted) close();
        throw;
    }
}

#ifdef UF_HAS_EVALUATION_CUDA
void cuda_check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

bool existing_frame_count_matches(const Json& first, const Json& second, std::uint64_t expected) {
    // Python adds the stored values before comparing with the requested integer.
    // Do not truncate fractional fields or wrap negative ones through UInt64.
    const auto numeric = [](const Json& value) {
        return value.is_number() || value.is_boolean() || python_json_extended_integer(value);
    };
    if (!numeric(first) || !numeric(second)) {
        // Python can concatenate two strings/lists, but neither result equals
        // an integer. Other combinations fail at the addition operation.
        if ((first.is_string() && second.is_string()) || (first.is_array() && second.is_array())) return false;
        throw std::invalid_argument("Existing evaluation frame counts cannot be added");
    }
    if (first.is_number_float() || second.is_number_float()) {
        const auto floating = [](const Json& value) {
            return python_json_as_double(nlohmann::json(value));
        };
        const double sum = floating(first) + floating(second);
        return std::isfinite(sum) && sum >= 0 && std::trunc(sum) == sum &&
            sum < std::ldexp(1.0, 64) && static_cast<std::uint64_t>(sum) == expected;
    }
    auto sum = python_json_int(nlohmann::json(first));
    sum += python_json_int(nlohmann::json(second));
    return sum.decimal() == PythonInteger::from_unsigned(expected).decimal();
}

struct TimingEvents {
    cudaEvent_t start{}, end{};
    TimingEvents() {
        cuda_check(cudaEventCreate(&start));
        const auto status = cudaEventCreate(&end);
        if (status != cudaSuccess) { cudaEventDestroy(start); start = nullptr; cuda_check(status); }
    }
    ~TimingEvents() { if (end) cudaEventDestroy(end); if (start) cudaEventDestroy(start); }
    void begin() {
        ProcessInterrupt::check();
        cuda_check(cudaDeviceSynchronize());
        cuda_check(cudaEventRecord(start, c10::cuda::getCurrentCUDAStream().stream()));
    }
    double finish() {
        cuda_check(cudaEventRecord(end, c10::cuda::getCurrentCUDAStream().stream()));
        cuda_check(cudaDeviceSynchronize());
        ProcessInterrupt::check();
        float milliseconds = 0;
        cuda_check(cudaEventElapsedTime(&milliseconds, start, end));
        return static_cast<double>(milliseconds) / 1000.0;
    }
};

struct SourceFrame {
    torch::Tensor input, y, uv, rgb;
};

class SourceReader {
public:
    explicit SourceReader(const EvaluationPoint& point) {
        if (point.source_type == "png")
            png_ = std::make_unique<video_io::PNGReader>(point.source, point.width, point.height);
        else yuv_ = std::make_unique<video_io::YUV420Reader>(point.source, point.width, point.height);
    }
    SourceFrame read() {
        ProcessInterrupt::check();
        SourceFrame result;
        if (png_) {
            result.rgb = png_->read_one_frame();
            if (!result.rgb.defined()) throw std::runtime_error("Evaluation PNG source ended before its declared frame count");
        } else {
            auto planes = yuv_->read_one_frame();
            result.y = std::move(planes.first); result.uv = std::move(planes.second);
            if (!result.y.defined()) throw std::runtime_error("Evaluation YUV source ended before its declared frame count");
            result.input = transforms::ycbcr420_to_444_np(result.y, result.uv).unsqueeze(0);
        }
        return result;
    }
    void close() {
        if (png_) png_->close();
        if (yuv_) yuv_->close();
    }
private:
    std::unique_ptr<video_io::PNGReader> png_;
    std::unique_ptr<video_io::YUV420Reader> yuv_;
};

struct SourceGroup {
    torch::Tensor input;
    std::vector<SourceFrame> frames;
};
SourceGroup read_group(SourceReader& reader, const EvaluationPoint& point, const torch::Device& device,
                       int real_count, int delay, bool intra, bool build_input) {
    SourceGroup group;
    for (int index = 0; index < real_count; ++index) {
        auto frame = reader.read();
        if (point.source_type == "png") {
            const auto rgb = frame.rgb.unsqueeze(0).to(device, true).to(torch::kFloat32) / 255.0;
            frame.input = transforms::rgb2ycbcr(rgb);
        }
        group.frames.push_back(std::move(frame));
    }
    if (!intra) while (group.frames.size() < static_cast<std::size_t>(delay)) group.frames.push_back(group.frames.back());
    // The Python np_only branch also prepares per-frame inputs, but returns
    // before cat/half/normalization. Keep that work outside decode event timing.
    if (!build_input) return group;
    std::vector<torch::Tensor> inputs;
    for (const auto& frame : group.frames) inputs.push_back(frame.input);
    auto input = torch::cat(inputs, 1);
    if (point.source_type == "yuv420") input = input.to(device, true);
    input = input.to(torch::kFloat16);
    if (point.source_type == "yuv420") input = input / 255.0;
    group.input = (input - 0.5).contiguous(torch::MemoryFormat::ChannelsLast);
    return group;
}

std::pair<torch::Tensor, torch::Tensor> reconstruction_yuv(const torch::Tensor& frame) {
    const auto planes = transforms::yuv_444_to_420(frame + 0.5);
    return {(planes.first * 255).clamp(0, 255), (planes.second * 255).clamp(0, 255)};
}

FrameQuality distortion(const EvaluationPoint& point, const VideoEvaluationOptions& options,
                        const SourceFrame& source, const torch::Tensor& frame) {
    if (point.source_type == "png") {
        const auto rgb = (transforms::ycbcr2rgb(frame + 0.5) * 255).clamp(0, 255).squeeze(0).to(torch::kCPU);
        return {{calc_psnr(source.rgb, rgb)}, {options.calc_ssim ? calc_msssim_rgb(source.rgb, rgb) : 0.0}};
    }
    const auto rec = reconstruction_yuv(frame);
    const auto y = rec.first.squeeze(0).select(0, 0).to(torch::kCPU), uv = rec.second.squeeze(0).to(torch::kCPU);
    const auto psnr_y = calc_psnr(source.y.select(0, 0), y);
    const auto psnr_u = calc_psnr(source.uv.select(0, 0), uv.select(0, 0));
    const auto psnr_v = calc_psnr(source.uv.select(0, 1), uv.select(0, 1));
    const auto ssim_y = options.calc_ssim ? calc_msssim(source.y.select(0, 0), y) : 0.0;
    const auto ssim_u = options.calc_ssim ? calc_msssim(source.uv.select(0, 0), uv.select(0, 0)) : 0.0;
    const auto ssim_v = options.calc_ssim ? calc_msssim(source.uv.select(0, 1), uv.select(0, 1)) : 0.0;
    return {{(6 * psnr_y + psnr_u + psnr_v) / 8, psnr_y, psnr_u, psnr_v},
            {(6 * ssim_y + ssim_u + ssim_v) / 8, ssim_y, ssim_u, ssim_v}};
}

class ReconstructionWriter {
public:
    ReconstructionWriter(const EvaluationPoint& point, std::int64_t bitrate) : point_(point) {
        if (point.source_type == "png")
            png_ = std::make_unique<video_io::PNGWriter>(point.bin_folder, point.width, point.height);
        else {
            // Preserve YUV420Writer's path rules. The reference worker passes
            // a .mp4 path, so it attempts to open that path as a directory.
            const auto name = replace_all(point.reconstruction_path.string(), ".yuv", "_" + std::to_string(bitrate) + "kbps.yuv");
            yuv_ = std::make_unique<video_io::YUV420Writer>(name, point.width, point.height);
        }
    }
    void write(const torch::Tensor& frame) {
        ProcessInterrupt::check();
        if (point_.source_type == "png") {
            const auto rgb = (transforms::ycbcr2rgb(frame + 0.5) * 255).clamp(0, 255).round().to(torch::kUInt8)
                .squeeze(0).to(torch::kCPU);
            png_->write_one_frame(rgb);
        } else {
            const auto rec = reconstruction_yuv(frame);
            const auto y = rec.first.round().to(torch::kUInt8).squeeze(0).to(torch::kCPU);
            const auto uv = rec.second.to(torch::kUInt8).squeeze(0).to(torch::kCPU);
            yuv_->write_one_frame(y, uv);
        }
    }
    void close() {
        if (png_) png_->close();
        if (yuv_) yuv_->close();
    }
private:
    const EvaluationPoint& point_;
    std::unique_ptr<video_io::PNGWriter> png_;
    std::unique_ptr<video_io::YUV420Writer> yuv_;
};
#endif

}  // namespace

bool evaluation_cuda_compiled() {
#ifdef UF_HAS_EVALUATION_CUDA
    return true;
#else
    return false;
#endif
}

int evaluation_cuda_devices() {
#ifdef UF_HAS_EVALUATION_CUDA
    return static_cast<int>(torch::cuda::device_count());
#else
    return 0;
#endif
}

struct EvaluationModels::Impl {
    VideoEvaluationOptions options;
#ifdef UF_HAS_EVALUATION_CUDA
    torch::Device device{torch::kCUDA, 0};
    std::unique_ptr<c10::cuda::CUDAGuard> device_guard;
    std::optional<c10::cuda::CUDAStream> stream;
    std::unique_ptr<c10::cuda::CUDAStreamGuard> stream_guard;
    std::unique_ptr<Fp16Model> image, video;
#endif
    explicit Impl(VideoEvaluationOptions selected, const std::string& selector) : options(std::move(selected)) {
#ifdef UF_HAS_EVALUATION_CUDA
        if (setenv("CUDA_VISIBLE_DEVICES", selector.c_str(), 1) != 0 ||
            setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8", 1) != 0) {
            throw std::runtime_error("Cannot configure evaluation CUDA environment");
        }
        at::globalContext().setUserEnabledCuDNN(true);
        at::globalContext().setBenchmarkCuDNN(false);
        at::globalContext().setDeterministicAlgorithms(true, false);
        at::globalContext().setDeterministicFillUninitializedMemory(false);
        torch::manual_seed(0);
        torch::set_num_threads(1);
        if (!fp16_available()) throw std::runtime_error("Video evaluation requires the native FP16 CUDA backend");
        device_guard = std::make_unique<c10::cuda::CUDAGuard>(device);
        stream = c10::cuda::getStreamFromPool(false, 0);
        stream_guard = std::make_unique<c10::cuda::CUDAStreamGuard>(*stream);
        image = std::make_unique<Fp16Model>(options.model_i, "image", device, options.skip_threshold);
        if (!options.force_intra) video = std::make_unique<Fp16Model>(options.model_p, options.variant, device, options.skip_threshold);
#else
        (void)selector;
        throw std::runtime_error("Video evaluation requires the native FP16 CUDA backend");
#endif
    }
};

EvaluationModels::EvaluationModels(VideoEvaluationOptions options, const std::string& selector)
    : impl_(std::make_unique<Impl>(std::move(options), selector)) {}
EvaluationModels::~EvaluationModels() = default;

Json EvaluationModels::run(const EvaluationPoint& point) {
#ifdef UF_HAS_EVALUATION_CUDA
    const c10::InferenceMode inference;
    const auto& options = impl_->options;
    if (options.check_existing && std::filesystem::exists(point.json_path) && std::filesystem::exists(point.bin_path)) {
        auto log = read_json_file(point.json_path);
        if (existing_frame_count_matches(log.at("i_frame_num"), log.at("p_frame_num"), point.frame_count)) return log;
        std::cout << "incorrect log for " << point.json_path.string() << ", try to rerun.\n";
    }
    const auto delay = options.variant == "ld" ? 1 : 8;
    std::unique_ptr<SourceReader> reader = std::make_unique<SourceReader>(point);
    const auto padding_right = static_cast<int>((16 - point.width % 16) % 16);
    const auto padding_bottom = static_cast<int>((16 - point.height % 16) % 16);
    std::vector<int> frame_types;
    std::vector<std::uint64_t> bits;
    std::vector<FrameQuality> qualities;
    std::vector<double> encoding_times, decoding_times;
    TimingEvents events;
    const auto started = std::chrono::system_clock::now();
    std::stringstream output(std::ios::binary | std::ios::in | std::ios::out);
    SpsHelper encode_sps;
    for (std::uint64_t frame = 0; frame < point.frame_count;) {
        ProcessInterrupt::check();
        const bool intra = frame == 0 || point.intra_period == 1 ||
            (point.intra_period > 1 && frame != 1 && frame % point.intra_period == 1);
        if (point.intra_period > 1 && frame != 1 && point.intra_period % delay) {
            throw std::invalid_argument("Evaluation intra period must be divisible by the model frame delay");
        }
        const int real_count = intra ? 1 : static_cast<int>(std::min<std::uint64_t>(delay, point.frame_count - frame));
        const auto group = read_group(*reader, point, impl_->device, real_count, delay, intra, true);
        events.begin();
        const bool reset = !intra && options.reset_interval > 0 && (frame + delay) % options.reset_interval == 1;
        const auto qp = intra ? point.qp_i : point.qp_p;
        if (!intra && !impl_->video) throw std::runtime_error("Predicted evaluation packet requires a video model");
        auto encoded = intra ? impl_->image->compress(group.input, qp, padding_bottom, padding_right)
                             : impl_->video->compress(group.input, qp, padding_bottom, padding_right, reset);
        if (intra && impl_->video) impl_->video->add_reference(encoded.reconstruction);
        frame_types.insert(frame_types.end(), real_count, intra ? 0 : 1);
        Sps sps{0, point.width, point.height};
        const auto [sps_id, sps_new] = encode_sps.get_sps_id(sps);
        sps.id = sps_id;
        std::uint64_t count = 0;
        if (sps_new) {
            count = write_sps(output, sps);
            if (options.verbose >= 2) std::cout << "new sps {sps_id: " << static_cast<unsigned>(sps.id)
                << ", height: " << sps.height << ", width: " << sps.width << "}\n";
        }
        count += write_packet(output, Packet{intra ? NalType::intra : NalType::predicted, sps_id,
            static_cast<std::uint8_t>(qp), encoded.entropy_parallel, reset, std::move(encoded.payload)});
        if (count > std::numeric_limits<std::uint64_t>::max() / 8) {
            throw std::overflow_error("Evaluation packet bit count overflow");
        }
        bits.push_back(count * 8);
        bits.insert(bits.end(), real_count - 1, 0);
        const auto seconds = events.finish();
        encoding_times.push_back(seconds);
        if (options.verbose >= 2) std::cout << "frame " << frame << " encoded, " << std::fixed << std::setprecision(3)
            << seconds * 1000 << " ms, bits: " << bits[bits.size() - real_count] << '\n';
        frame += real_count;
    }
    reader->close();
    reader.reset();
    std::uint64_t total_bytes = 0;
    {
        const auto end = output.tellp();
        if (end == std::streampos(-1)) {
            throw std::runtime_error("Cannot measure evaluation bitstream");
        }
        const auto size = static_cast<std::streamoff>(end);
        if (size < 0 || static_cast<std::uintmax_t>(size) > std::numeric_limits<std::uint64_t>::max()) {
            throw std::overflow_error("Evaluation bitstream file size overflow");
        }
        total_bytes = static_cast<std::uint64_t>(size);
        output.seekg(0);
        if (!output) throw std::runtime_error("Cannot rewind evaluation bitstream");
        std::ofstream file(point.bin_path, std::ios::binary | std::ios::trunc);
        if (!file) throw std::runtime_error("Cannot create evaluation bitstream: " + point.bin_path.string());
        // Transfer the retained stream in bounded blocks. str() would copy the
        // entire encoded video while the original buffer is still allocated.
        with_checked_close(file, "Cannot close evaluation bitstream output", [&]() {
            std::array<char, 64 * 1024> block{};
            for (std::uint64_t remaining = total_bytes; remaining;) {
                ProcessInterrupt::check();
                const auto count = static_cast<std::streamsize>(std::min<std::uint64_t>(remaining, block.size()));
                output.read(block.data(), count);
                if (!output || output.gcount() != count) throw std::runtime_error("Cannot read evaluation output buffer");
                file.write(block.data(), count);
                if (!file) throw std::runtime_error("Cannot write evaluation bitstream");
                remaining -= static_cast<std::uint64_t>(count);
            }
        });
    }
    const auto bitrate = (static_cast<double>(total_bytes) * 8) / (static_cast<double>(point.frame_count) / 30) / 1000;
    if (!std::isfinite(bitrate) || bitrate >= static_cast<double>(std::numeric_limits<std::int64_t>::max())) {
        throw std::overflow_error("Evaluation reconstructed bitrate overflow");
    }
    const auto total_kbps = static_cast<std::int64_t>(bitrate);
    // Swap into a scoped stream so its destructor frees the old capacity.
    // Assigning an empty string or stream can retain that allocation.
    {
        std::stringstream empty(std::ios::binary | std::ios::in | std::ios::out);
        output.swap(empty);
    }
    // Release encoder records before starting the decode phase.
    encode_sps = SpsHelper{};
    std::ifstream encoded_file(point.bin_path, std::ios::binary);
    if (!encoded_file) throw std::runtime_error("Cannot reopen evaluation bitstream");
    std::string input_bytes;
    with_checked_close(encoded_file, "Cannot close evaluation bitstream input", [&]() {
        input_bytes.assign(std::istreambuf_iterator<char>(encoded_file), std::istreambuf_iterator<char>());
        if (encoded_file.bad()) throw std::runtime_error("Cannot read evaluation bitstream");
    });
    std::istringstream input(input_bytes, std::ios::binary | std::ios::in);
    std::string{}.swap(input_bytes);
    reader = std::make_unique<SourceReader>(point);
    std::unique_ptr<ReconstructionWriter> writer;
    if (options.save_decoded) writer = std::make_unique<ReconstructionWriter>(point, total_kbps);
    SpsHelper decode_sps;
    for (std::uint64_t frame = 0; frame < point.frame_count;) {
        events.begin();
        Packet packet;
        Sps sps;
        for (;;) {
            const auto header = read_optional_header(input);
            if (!header) throw std::runtime_error("Invalid or incomplete evaluation bitstream");
            const auto type = header->type;
            const auto id = header->sps_id;
            if (type == NalType::sps) {
                const auto registered = read_sps(input, id);
                decode_sps.add_sps_by_id(registered);
                if (options.verbose >= 2) std::cout << "new sps {sps_id: " << static_cast<unsigned>(id)
                    << ", height: " << registered.height << ", width: " << registered.width << "}\n";
                continue;
            }
            const auto registered = decode_sps.get_sps_by_id(id);
            if (!registered) throw std::runtime_error("Evaluation packet uses an unknown SPS");
            sps = *registered;
            packet = read_packet(input, type, id);
            break;
        }
        const bool intra = packet.type == NalType::intra;
        if (!intra && !impl_->video) throw std::runtime_error("Predicted evaluation packet requires a video model");
        auto reconstruction = intra ? impl_->image->decompress(packet.payload, sps.width, sps.height, packet.qp, packet.entropy_parallel)
            : impl_->video->decompress(packet.payload, sps.width, sps.height, packet.qp, packet.entropy_parallel, packet.reset_feature_memory);
        if (intra && impl_->video) impl_->video->add_reference(reconstruction.at(0), false);
        const auto seconds = events.finish();
        decoding_times.push_back(seconds);
        const int real_count = intra ? 1 : static_cast<int>(std::min<std::uint64_t>(delay, point.frame_count - frame));
        if (reconstruction.size() != static_cast<std::size_t>(intra ? 1 : delay)) throw std::runtime_error("Unexpected evaluation decoder frame count");
        const auto source = read_group(*reader, point, impl_->device, real_count, delay, intra, false);
        for (int index = 0; index < real_count; ++index) {
            const auto cropped = reconstruction[index].slice(2, 0, point.height).slice(3, 0, point.width);
            qualities.push_back(distortion(point, options, source.frames[index], cropped));
        }
        if (options.verbose >= 2) {
            std::cout << "frame " << frame << " decoded, " << std::fixed << std::setprecision(3)
                      << seconds * 1000 << " ms, bits: " << packet.payload.size() * 8 << ", ";
            for (int index = 0; index < real_count; ++index) std::cout << "PSNR: " << std::setprecision(4)
                << qualities[qualities.size() - real_count + index].psnr[0] << ", ";
            std::cout << '\n';
        }
        if (writer) for (int index = 0; index < real_count; ++index) {
            writer->write(reconstruction[index].slice(2, 0, point.height).slice(3, 0, point.width));
        }
        frame += real_count;
    }
    {
        std::istringstream empty(std::ios::binary | std::ios::in);
        input.swap(empty);
    }
    reader->close();
    reader.reset();
    if (writer) writer->close();
    const auto test_seconds = std::chrono::duration<double>(std::chrono::system_clock::now() - started).count();
    std::optional<double> average_encoding, average_decoding;
    if (options.verbose >= 1 && encoding_times.size() > 4) {
        const auto count = encoding_times.size() - 4;
        average_encoding = std::accumulate(encoding_times.begin() + 4, encoding_times.end(), 0.0) / count;
        average_decoding = std::accumulate(decoding_times.begin() + 4, decoding_times.end(), 0.0) / count;
        std::cout << "encoding/decoding " << count << " * " << delay << " frames, average encoding time "
                  << std::fixed << std::setprecision(3) << *average_encoding * 1000 << " ms, average decoding time "
                  << *average_decoding * 1000 << " ms.\n";
    }
    auto log = generate_log_json(point.frame_count,
        static_cast<std::uint64_t>(point.width) * point.height, test_seconds,
        frame_types, bits, qualities, options.verbose_json, average_encoding, average_decoding);
    write_json_file(log, point.json_path);
    return log;
#else
    (void)point;
    throw std::runtime_error("Video evaluation requires the native FP16 CUDA backend");
#endif
}

}  // namespace uf::evaluation
