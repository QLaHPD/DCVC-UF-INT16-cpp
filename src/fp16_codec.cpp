#include "uf/fp16_codec.hpp"
#include "uf/process.hpp"
#include "uf/transforms.hpp"
#include "bitstream_output.hpp"
#include "inference_environment.hpp"

#include <ATen/TensorIndexing.h>
#include <torch/cuda.h>

#include <openssl/evp.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>

#ifdef UF_HAS_FP16
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#endif

namespace uf {

struct Fp16Codec::Impl {
    std::string variant;
    torch::Device device;
    int chunk_size;
    std::unique_ptr<Fp16Model> image;
    std::unique_ptr<Fp16Model> video;
#ifdef UF_HAS_FP16
    c10::cuda::CUDAStream stream;
    Impl(std::string selected, torch::Device target)
        : variant(std::move(selected)), device(std::move(target)),
          chunk_size(variant == "ld" ? 1 : 8),
          stream(c10::cuda::getStreamFromPool(false, device.index())) {}
#else
    Impl(std::string selected, torch::Device target)
        : variant(std::move(selected)), device(std::move(target)),
          chunk_size(variant == "ld" ? 1 : 8) {}
#endif
};

Fp16Codec::Fp16Codec(const std::filesystem::path& image,
                     const std::optional<std::filesystem::path>& video,
                     std::string variant, torch::Device device,
                     double skip_threshold, const std::optional<std::string>& image_sha256,
                     const std::optional<std::string>& video_sha256) {
    if (!device.is_cuda() || (variant != "hts" && variant != "htl" && variant != "ld")) {
        throw std::invalid_argument("UF FP16 requires CUDA and an hts/htl/ld variant");
    }
#ifdef UF_HAS_FP16
    detail::set_inference_environment();
    if (!device.has_index()) device = torch::Device(torch::kCUDA, c10::cuda::current_device());
    impl_ = std::make_unique<Impl>(std::move(variant), device);
    c10::cuda::CUDAStreamGuard stream_guard(impl_->stream);
#else
    impl_ = std::make_unique<Impl>(std::move(variant), device);
#endif
    impl_->image = std::make_unique<Fp16Model>(image, "image", device, skip_threshold, image_sha256);
    if (video) impl_->video = std::make_unique<Fp16Model>(
        *video, impl_->variant, device, skip_threshold, video_sha256);
}

Fp16Codec::~Fp16Codec() = default;

torch::Tensor Fp16Codec::frame_tensor(
    const std::vector<std::vector<std::uint8_t>>& frames,
    std::uint32_t width, std::uint32_t height) const {
    std::vector<std::uint8_t> bytes;
    const auto limit = std::min(static_cast<std::uintmax_t>(bytes.max_size()),
        static_cast<std::uintmax_t>(std::numeric_limits<std::int64_t>::max()));
    if (!width || !height || frames.empty() ||
        static_cast<std::uintmax_t>(width) > limit / height) {
        throw std::length_error("UF frame dimensions exceed tensor buffer size limit");
    }
    const auto plane = static_cast<std::size_t>(width) * height;
    if (plane > limit / 3) {
        throw std::length_error("UF frame exceeds tensor buffer size limit");
    }
    const auto frame_bytes = plane * 3;
    if (frames.size() > limit / frame_bytes) {
        throw std::length_error("UF chunk exceeds tensor buffer size limit");
    }
    bytes.reserve(frames.size() * frame_bytes);
    for (const auto& frame : frames) {
        if (frame.size() != frame_bytes) throw std::invalid_argument("UF input frame has the wrong size");
        bytes.insert(bytes.end(), frame.begin(), frame.end());
    }
    // The byte vector survives device transfer and the allocating FP16 cast;
    // a separate CPU clone would duplicate the concatenated input chunk.
    auto tensor = torch::from_blob(bytes.data(),
        {1, static_cast<std::int64_t>(frames.size() * 3), height, width},
        torch::TensorOptions().dtype(torch::kUInt8)).to(impl_->device);
    return (tensor.to(torch::kFloat16) / 255.0 - 0.5)
        .contiguous(torch::MemoryFormat::ChannelsLast);
}

CodecStatistics Fp16Codec::encode(FrameSource source, const std::filesystem::path& output,
                                  std::uint32_t width, std::uint32_t height,
                                  int qp_i, int qp_p, int reset_interval, int intra_period,
                                  std::optional<std::uint64_t> max_frames,
                                  EncodeProgress progress) {
    ProcessInterrupt::check();
    if (!source || !width || !height || width % 2 || height % 2 ||
        width > static_cast<std::uint32_t>(std::numeric_limits<int>::max() - 16) ||
        height > static_cast<std::uint32_t>(std::numeric_limits<int>::max() - 16) ||
        qp_i < 0 || qp_i >= 64 || qp_p < 0 || qp_p >= 64 ||
        (intra_period > 1 && intra_period % impl_->chunk_size)) {
        throw std::invalid_argument("Invalid UF FP16 encode settings");
    }
#ifdef UF_HAS_FP16
    c10::cuda::CUDAStreamGuard stream_guard(impl_->stream);
#endif
    torch::NoGradGuard no_grad;
    const int padding_bottom = static_cast<int>((16 - height % 16) % 16);
    const int padding_right = static_cast<int>((16 - width % 16) % 16);
    CodecStatistics stats;
    const auto started = std::chrono::steady_clock::now();
    BitstreamOutput file(output);
    try {
        auto& stream = file.stream();
        write_sps(stream, Sps{0, width, height});
        while (!max_frames || stats.frames < *max_frames) {
            ProcessInterrupt::check();
            const auto count = stats.frames;
            const bool intra = count == 0 || intra_period == 1 ||
                (intra_period > 1 && count != 1 && count % intra_period == 1);
            std::uint64_t maximum = intra ? 1 : impl_->chunk_size;
            if (max_frames) maximum = std::min(maximum, *max_frames - count);
            std::vector<std::vector<std::uint8_t>> frames;
            for (std::uint64_t i = 0; i < maximum; ++i) {
                ProcessInterrupt::check();
                auto frame = source();
                if (!frame) break;
                frames.push_back(std::move(*frame));
            }
            if (frames.empty()) break;
            const auto valid = frames.size();
            if (!intra) {
                if (!impl_->video) throw std::runtime_error("UF P packet requires a video model");
                while (frames.size() < static_cast<std::size_t>(impl_->chunk_size)) frames.push_back(frames.back());
            }
            const auto input = frame_tensor(frames, width, height);
            const bool reset = !intra && reset_interval > 0 &&
                (count + impl_->chunk_size) % reset_interval == 1;
            auto result = intra ? impl_->image->compress(input, qp_i, padding_bottom, padding_right)
                : impl_->video->compress(input, qp_p, padding_bottom, padding_right, reset);
            ProcessInterrupt::check();
            if (intra && impl_->video) impl_->video->add_reference(result.reconstruction, true);
            write_packet(stream, Packet{intra ? NalType::intra : NalType::predicted, 0,
                static_cast<std::uint8_t>(intra ? qp_i : qp_p), result.entropy_parallel,
                reset, std::move(result.payload)});
            stats.frames += valid;
            ++stats.packets;
            if (progress) {
                const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
                progress(stats.frames, stats.frames / std::max(seconds, 1e-9));
            }
        }
        ProcessInterrupt::check();
        file.finish();
    } catch (...) {
        file.close_checked();
        throw;
    }
    if (!stats.frames) throw std::runtime_error("UF input yielded no frames");
#ifdef UF_HAS_FP16
    torch::cuda::synchronize(impl_->device.index());
#endif
    stats.encode_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    return stats;
}

std::vector<std::uint8_t> Fp16Codec::raw_frame(const torch::Tensor& frame,
                                              std::int64_t width, std::int64_t height) {
    ProcessInterrupt::check();
    using at::indexing::None;
    using at::indexing::Slice;
    const auto cropped = frame.index({Slice(), Slice(), Slice(None, height), Slice(None, width)}) + 0.5;
    const auto planes = transforms::yuv_444_to_420(cropped);
    const auto y = (planes.first * 255).clamp(0, 255).round()
        .to(torch::kUInt8).squeeze(0).to(torch::kCPU).contiguous();
    const auto uv = (planes.second * 255).clamp(0, 255)
        .to(torch::kUInt8).squeeze(0).to(torch::kCPU).contiguous();
    ProcessInterrupt::check();
    std::vector<std::uint8_t> result;
    const auto y_size = static_cast<std::size_t>(y.numel());
    const auto uv_size = static_cast<std::size_t>(uv.numel());
    if (y_size > result.max_size() || uv_size > result.max_size() - y_size)
        throw std::length_error("UF raw frame exceeds byte buffer size limit");
    result.reserve(y_size + uv_size);
    if (y_size) result.insert(result.end(), y.data_ptr<std::uint8_t>(), y.data_ptr<std::uint8_t>() + y_size);
    if (uv_size) result.insert(result.end(), uv.data_ptr<std::uint8_t>(), uv.data_ptr<std::uint8_t>() + uv_size);
    return result;
}

CodecStatistics Fp16Codec::decode(const std::filesystem::path& input,
                                  std::uint64_t total_frames, std::uint64_t total_packets,
                                  FrameSink sink, std::optional<Sps> dimensions,
                                  DecodeProgress progress) {
    const auto started = std::chrono::steady_clock::now();
    std::ifstream stream(input, std::ios::binary);
    if (!stream) throw std::runtime_error("Cannot open UF FP16 bitstream");
    bool close_attempted = false;
    const auto close_input = [&] {
        if (std::exchange(close_attempted, true)) return;
        stream.clear();
        stream.close();
        if (stream.fail()) throw std::runtime_error("Cannot close UF FP16 bitstream");
    };
    try {
        auto stats = decode_stream_impl(stream, total_frames, total_packets,
            std::move(sink), dimensions, std::move(progress), close_input);
        stats.decode_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        return stats;
    } catch (...) {
        close_input();
        throw;
    }
}

std::vector<std::vector<std::uint8_t>> Fp16Codec::decode_packet(const Packet& packet, const Sps& sps,
    std::optional<std::uint64_t> valid_frames) {
    ProcessInterrupt::check();
#ifdef UF_HAS_FP16
    c10::cuda::CUDAStreamGuard stream_guard(impl_->stream);
#endif
    torch::NoGradGuard no_grad;
    if (!sps.width || !sps.height || sps.width % 2 || sps.height % 2 ||
        sps.id != packet.sps_id || packet.qp >= 64 ||
        (packet.type != NalType::intra && packet.type != NalType::predicted)) {
        throw std::invalid_argument("Invalid UF packet or dimensions for frame decoding");
    }
    const bool intra = packet.type == NalType::intra;
    std::vector<torch::Tensor> frames;
    if (intra) {
        frames = impl_->image->decompress(packet.payload, sps.width, sps.height,
                                          packet.qp, packet.entropy_parallel);
        // Native add_reference(false) replaces the I reference and invalidates
        // temporal feature memory, including after a backward seek.
        if (impl_->video) impl_->video->add_reference(frames.at(0), false);
    } else {
        if (!impl_->video) throw std::runtime_error("UF P packet requires a video model");
        frames = impl_->video->decompress(packet.payload, sps.width, sps.height,
            packet.qp, packet.entropy_parallel, packet.reset_feature_memory);
    }
    if (frames.size() != static_cast<std::size_t>(intra ? 1 : impl_->chunk_size))
        throw std::runtime_error("UF FP16 decoder returned an unexpected chunk length");
    // FrameSource checks every reconstruction before evaluating the sliced
    // raw_frame list, even when the last packet has a padded tail.
    for (const auto& frame : frames) {
        ProcessInterrupt::check();
        if (!torch::isfinite(frame).all().item<bool>())
            throw std::runtime_error("UF FP16 decoder produced non-finite pixels");
    }
    const auto count = valid_frames ? static_cast<std::size_t>(
        std::min(*valid_frames, static_cast<std::uint64_t>(frames.size()))) : frames.size();
    std::vector<std::vector<std::uint8_t>> result;
    result.reserve(count);
    for (std::size_t index = 0; index < count; ++index)
        result.push_back(raw_frame(frames[index], sps.width, sps.height));
    return result;
}

CodecStatistics Fp16Codec::decode_stream(std::istream& stream,
                                         std::uint64_t total_frames, std::uint64_t total_packets,
                                         FrameSink sink, std::optional<Sps> dimensions,
                                         DecodeProgress progress) {
    return decode_stream_impl(stream, total_frames, total_packets, std::move(sink),
        dimensions, std::move(progress), {});
}

CodecStatistics Fp16Codec::decode_stream_impl(std::istream& stream,
    std::uint64_t total_frames, std::uint64_t total_packets, FrameSink sink,
    std::optional<Sps> dimensions, DecodeProgress progress,
    const std::function<void()>& close_input) {
    ProcessInterrupt::check();
#ifdef UF_HAS_FP16
    c10::cuda::CUDAStreamGuard stream_guard(impl_->stream);
#endif
    torch::NoGradGuard no_grad;
    const auto started = std::chrono::steady_clock::now();
    if (read_preamble(stream).integer) throw std::runtime_error("INT16 stream cannot use the FP16 runtime");
    std::array<std::optional<Sps>, 16> sps_table{};
    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> digest(EVP_MD_CTX_new(), EVP_MD_CTX_free);
    if (!digest || EVP_DigestInit_ex(digest.get(), EVP_sha256(), nullptr) != 1) {
        throw std::runtime_error("Cannot initialize UF decoded-pixel digest");
    }
    CodecStatistics stats;
    bool first = true;
    while (const auto header = read_optional_header(stream)) {
        ProcessInterrupt::check();
        const auto type = header->type;
        const auto sps_id = header->sps_id;
        if (type == NalType::sps) {
            const auto sps = read_sps(stream, sps_id);
            if (!sps.width || !sps.height || sps.width % 2 || sps.height % 2) {
                throw std::runtime_error("Invalid UF YUV420 bitstream dimensions");
            }
            if (dimensions && (sps.width != dimensions->width || sps.height != dimensions->height)) {
                throw std::runtime_error("UF bitstream dimensions differ from archive metadata");
            }
            if (!dimensions) dimensions = sps;
            sps_table[sps_id] = sps;
            continue;
        }
        if (!sps_table[sps_id] || stats.frames >= total_frames) {
            throw std::runtime_error("Unexpected UF packet or unknown SPS");
        }
        const auto& sps = *sps_table[sps_id];
        const auto packet = read_packet(stream, type, sps_id);
        const bool intra = type == NalType::intra;
        if (first && !intra) throw std::runtime_error("UF stream must start with an I packet");
        first = false;
        std::vector<torch::Tensor> frames;
        if (intra) {
            frames = impl_->image->decompress(packet.payload, sps.width, sps.height,
                                              packet.qp, packet.entropy_parallel);
            if (impl_->video) impl_->video->add_reference(frames.at(0), false);
        } else {
            if (!impl_->video) throw std::runtime_error("UF P packet requires a video model");
            frames = impl_->video->decompress(packet.payload, sps.width, sps.height,
                packet.qp, packet.entropy_parallel, packet.reset_feature_memory);
        }
        if (frames.size() != static_cast<std::size_t>(intra ? 1 : impl_->chunk_size)) {
            throw std::runtime_error("UF FP16 decoder returned an unexpected chunk length");
        }
        for (const auto& frame : frames) {
            ProcessInterrupt::check();
            if (stats.frames == total_frames) break;
            if (!torch::isfinite(frame).all().item<bool>()) {
                throw std::runtime_error("UF FP16 decoder produced non-finite pixels");
            }
            const auto raw = raw_frame(frame, sps.width, sps.height);
            if (EVP_DigestUpdate(digest.get(), raw.data(), raw.size()) != 1) {
                throw std::runtime_error("Cannot update UF decoded-pixel digest");
            }
            if (sink) sink(raw);
            ++stats.frames;
        }
        ++stats.packets;
        if (progress) progress(stats.frames, total_frames);
    }
    if (close_input) close_input();
    ProcessInterrupt::check();
    if (stats.frames != total_frames || stats.packets != total_packets) {
        throw std::runtime_error("Incomplete UF FP16 stream");
    }
    std::array<unsigned char, EVP_MAX_MD_SIZE> hash{};
    unsigned int length = 0;
    if (EVP_DigestFinal_ex(digest.get(), hash.data(), &length) != 1 || length != 32) {
        throw std::runtime_error("Cannot finish UF decoded-pixel digest");
    }
    std::ostringstream encoded;
    encoded << std::hex << std::setfill('0');
    for (unsigned int i = 0; i < length; ++i) encoded << std::setw(2) << static_cast<unsigned>(hash[i]);
    stats.decoded_yuv_sha256 = encoded.str();
    stats.decode_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    return stats;
}

}  // namespace uf
