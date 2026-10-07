#include "uf/integer_codec.hpp"
#include "uf/int16_ops.hpp"
#include "uf/process.hpp"
#include "bitstream_output.hpp"
#include "inference_environment.hpp"

#include <ATen/TensorIndexing.h>
#include <torch/cuda.h>
#ifdef UF_HAS_CUDA
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#endif

#include <openssl/evp.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace uf {

struct IntegerCodec::RuntimeState {
#ifdef UF_HAS_CUDA
    std::optional<c10::cuda::CUDAStream> stream;
#endif

    explicit RuntimeState(torch::Device& device) {
        if (device.is_cpu()) return;
        if (!device.is_cuda()) {
            throw std::invalid_argument("UF INT16 requires a CPU or CUDA device");
        }
#ifdef UF_HAS_CUDA
        if (!device.has_index()) {
            device = torch::Device(torch::kCUDA, c10::cuda::current_device());
        }
        stream = c10::cuda::getStreamFromPool(false, device.index());
#else
        throw std::runtime_error("UF INT16 CUDA inference was not compiled; select the CPU reference device");
#endif
    }
};

namespace {

std::array<std::uint8_t, 32> hex_identity(const std::string& value) {
    if (value.size() != 64) throw std::invalid_argument("Invalid UF prepared identity");
    std::array<std::uint8_t, 32> result{};
    const auto digit = [](char c) {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        throw std::invalid_argument("Invalid UF prepared identity hex");
    };
    for (std::size_t i = 0; i < result.size(); ++i) {
        result[i] = static_cast<std::uint8_t>(digit(value[2 * i]) * 16 + digit(value[2 * i + 1]));
    }
    return result;
}

std::string finish_sha256(EVP_MD_CTX* context) {
    std::array<unsigned char, EVP_MAX_MD_SIZE> hash{};
    unsigned int length = 0;
    if (EVP_DigestFinal_ex(context, hash.data(), &length) != 1 || length != 32) {
        throw std::runtime_error("OpenSSL SHA-256 finalization failed");
    }
    std::ostringstream text;
    text << std::hex << std::setfill('0');
    for (unsigned int i = 0; i < length; ++i) text << std::setw(2) << static_cast<unsigned>(hash[i]);
    return text.str();
}

}  // namespace

RawYuv420Reader::RawYuv420Reader(const std::filesystem::path& path,
                                 std::uint32_t width, std::uint32_t height)
    : input_(path, std::ios::binary), width_(width), height_(height) {
    try {
        if (!input_ || width < 2 || height < 2 || width % 2 || height % 2) {
            throw std::invalid_argument("Raw YUV420 input or dimensions are invalid");
        }
        const auto buffer_limit = std::vector<std::uint8_t>{}.max_size();
        if (static_cast<std::uintmax_t>(width) >
            static_cast<std::uintmax_t>(buffer_limit) / height) {
            throw std::overflow_error("Raw YUV420 pixel count exceeds buffer size limit");
        }
        pixels_ = static_cast<std::size_t>(width) * height;
        if (pixels_ > buffer_limit / 3) {
            throw std::overflow_error("Expanded raw YUV444 frame exceeds buffer size limit");
        }
        // Even dimensions make these integer plane sizes exact. The expanded
        // bound above also bounds this addition without multiplying by 3 first.
        planar_bytes_ = pixels_ + pixels_ / 2;
        expanded_bytes_ = pixels_ * 3;
        if (static_cast<std::uintmax_t>(planar_bytes_) >
            static_cast<std::uintmax_t>(std::numeric_limits<std::streamsize>::max())) {
            throw std::overflow_error("Raw YUV420 frame exceeds stream read size limit");
        }
    } catch (...) {
        close_checked();
        throw;
    }
}

RawYuv420Reader::~RawYuv420Reader() {
    try { close_checked(); } catch (...) {}
}

void RawYuv420Reader::close_checked() {
    if (close_attempted_) return;
    close_attempted_ = true;
    if (!input_.is_open()) return;
    input_.clear();
    input_.close();
    if (input_.fail()) throw std::runtime_error("Cannot close raw YUV420 input");
}

std::optional<std::vector<std::uint8_t>> RawYuv420Reader::read() {
    if (close_attempted_) throw std::runtime_error("Read from closed raw YUV420 input");
    const auto pixels = pixels_;
    std::vector<std::uint8_t> planar(planar_bytes_);
    input_.read(reinterpret_cast<char*>(planar.data()),
                static_cast<std::streamsize>(planar.size()));
    if (input_.bad()) throw std::runtime_error("Cannot read raw YUV420 input");
    if (input_.gcount() == 0 && input_.eof()) return std::nullopt;
    if (input_.gcount() != static_cast<std::streamsize>(planar.size())) {
        throw std::runtime_error("Truncated raw YUV420 frame");
    }
    std::vector<std::uint8_t> full(expanded_bytes_);
    std::copy_n(planar.data(), pixels, full.data());
    for (int component = 0; component < 2; ++component) {
        const auto* source = planar.data() + pixels + component * pixels / 4;
        auto* target = full.data() + pixels * (component + 1);
        for (std::uint32_t y = 0; y < height_; ++y) {
            for (std::uint32_t x = 0; x < width_; ++x) {
                target[static_cast<std::size_t>(y) * width_ + x] =
                    source[static_cast<std::size_t>(y / 2) * (width_ / 2) + x / 2];
            }
        }
    }
    return full;
}

IntegerCodec::IntegerCodec(const PreparedModel& image, const PreparedModel* video,
                           std::string variant, torch::Device device,
                           double skip_threshold)
    : variant_(std::move(variant)), device_(std::move(device)),
      chunk_size_(variant_ == "ld" ? 1 : 8) {
    detail::set_inference_environment();
    if (image.variant != "image" || (video && video->variant != variant_) ||
        (variant_ != "hts" && variant_ != "htl" && variant_ != "ld")) {
        throw std::invalid_argument("UF prepared models do not match the codec variant");
    }
    runtime_ = std::make_unique<RuntimeState>(device_);
#ifdef UF_HAS_CUDA
    c10::cuda::OptionalCUDAStreamGuard stream_guard;
    if (runtime_->stream) stream_guard.reset_stream(*runtime_->stream);
#endif
    image_ = std::make_unique<IntegerModel>(image, device_);
    if (video) video_ = std::make_unique<IntegerModel>(*video, device_, false);
    set_skip_threshold(skip_threshold);
    preamble_.integer = true;
    preamble_.image_identity = hex_identity(image.identity);
    preamble_.video_identity = video ? hex_identity(video->identity) : std::array<std::uint8_t, 32>{};
}

IntegerCodec::~IntegerCodec() = default;
IntegerCodec::IntegerCodec(IntegerCodec&&) noexcept = default;
IntegerCodec& IntegerCodec::operator=(IntegerCodec&&) noexcept = default;

void IntegerCodec::set_skip_threshold(double threshold) {
    image_->set_skip_threshold(threshold);
    if (video_) video_->set_skip_threshold(threshold);
}

torch::Tensor IntegerCodec::frame_tensor(
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
    // Keep the byte vector alive through conversion. CPU from_bytes produces
    // owned Int16 storage; device transfer is synchronous for this CPU view.
    auto tensor = torch::from_blob(bytes.data(),
                    {1, static_cast<std::int64_t>(frames.size() * 3),
                     static_cast<std::int64_t>(height), static_cast<std::int64_t>(width)},
                     torch::TensorOptions().dtype(torch::kUInt8)).to(device_);
    return int16::from_bytes(tensor);
}

CodecStatistics IntegerCodec::encode(FrameSource source,
                                     const std::filesystem::path& output,
                                     std::uint32_t width, std::uint32_t height,
                                     int qp_i, int qp_p, int reset_interval,
                                     int intra_period,
                                     std::optional<std::uint64_t> max_frames,
                                     EncodeProgress progress) {
    ProcessInterrupt::check();
#ifdef UF_HAS_CUDA
    c10::cuda::OptionalCUDAStreamGuard stream_guard;
    if (runtime_->stream) stream_guard.reset_stream(*runtime_->stream);
#endif
    torch::NoGradGuard no_grad;
    if (!source || !width || !height || width % 2 || height % 2 ||
        qp_i < 0 || qp_i >= 64 || qp_p < 0 || qp_p >= 64 ||
        (intra_period > 1 && intra_period % chunk_size_)) {
        throw std::invalid_argument("Invalid UF encode settings");
    }
    CodecStatistics stats;
    const auto started = std::chrono::steady_clock::now();
    BitstreamOutput file(output);
    try {
        auto& stream = file.stream();
        write_preamble(stream, preamble_);
        write_sps(stream, Sps{0, width, height});
        while (!max_frames || stats.frames < *max_frames) {
            ProcessInterrupt::check();
            const auto count = stats.frames;
            const bool intra = count == 0 || intra_period == 1 ||
                (intra_period > 1 && count != 1 && count % intra_period == 1);
            std::uint64_t maximum = intra ? 1 : static_cast<std::uint64_t>(chunk_size_);
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
                if (!video_) throw std::runtime_error("UF video model is required for P packets");
                while (frames.size() < static_cast<std::size_t>(chunk_size_)) {
                    frames.push_back(frames.back());
                }
            }
            const auto x = frame_tensor(frames, width, height);
            const bool reset = !intra && reset_interval > 0 &&
                (count + chunk_size_) % reset_interval == 1;
            CompressedFrame result;
            if (intra) {
                result = image_->compress(x, qp_i);
                if (video_) {
                    video_->clear_dpb();
                    video_->add_ref_feature_from_frame(result.frames.at(0));
                }
            } else result = video_->compress(x, qp_p, reset);
            write_packet(stream, Packet{intra ? NalType::intra : NalType::predicted,
                                        0, static_cast<std::uint8_t>(intra ? qp_i : qp_p),
                                        1, reset, std::move(result.payload)});
            stats.frames += valid;
            ++stats.packets;
            if (progress) {
                const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
                progress(stats.frames, stats.frames / std::max(seconds, 1e-9));
            }
        }
        file.finish();
    } catch (...) {
        file.close_checked();
        throw;
    }
    if (!stats.frames) throw std::runtime_error("UF input yielded no frames");
    if (device_.is_cuda()) torch::cuda::synchronize(device_.index());
    stats.encode_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    return stats;
}

std::vector<std::uint8_t> IntegerCodec::raw_frame(const torch::Tensor& frame,
                                                   std::int64_t width,
                                                   std::int64_t height) {
    ProcessInterrupt::check();
    using at::indexing::None;
    using at::indexing::Slice;
    const auto values = frame.index({0, Slice(), Slice(None, height), Slice(None, width)})
        .to(torch::kInt64);
    const auto luminance = int16::divide_round((values.index({0}) + int16::feature_scale / 2) * 255,
                                              int16::feature_scale);
    const auto chroma = values.index({Slice(1, None)});
    const auto summed = chroma.index({Slice(), Slice(0, None, 2), Slice(0, None, 2)}) +
                        chroma.index({Slice(), Slice(1, None, 2), Slice(0, None, 2)}) +
                        chroma.index({Slice(), Slice(0, None, 2), Slice(1, None, 2)}) +
                        chroma.index({Slice(), Slice(1, None, 2), Slice(1, None, 2)});
    const auto downsampled = torch::div((summed + 2 * int16::feature_scale) * 255,
                                       4 * int16::feature_scale, "floor");
    const auto y = luminance.clamp(0, 255).to(torch::kUInt8).to(torch::kCPU).contiguous();
    const auto uv = downsampled.clamp(0, 255).to(torch::kUInt8).to(torch::kCPU).contiguous();
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

CodecStatistics IntegerCodec::decode(const std::filesystem::path& input,
                                      std::uint64_t total_frames,
                                      std::uint64_t total_packets,
                                      FrameSink sink, std::optional<Sps> dimensions,
                                      DecodeProgress progress) {
    const auto started = std::chrono::steady_clock::now();
    std::ifstream stream(input, std::ios::binary);
    if (!stream) throw std::runtime_error("Cannot open UF bitstream");
    bool close_attempted = false;
    const auto close_input = [&] {
        if (std::exchange(close_attempted, true)) return;
        // EOF/fail from probing belongs to reading, not this close result.
        stream.clear();
        stream.close();
        if (stream.fail()) throw std::runtime_error("Cannot close UF bitstream");
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

std::vector<std::vector<std::uint8_t>> IntegerCodec::decode_packet(const Packet& packet, const Sps& sps,
    std::optional<std::uint64_t> valid_frames) {
    ProcessInterrupt::check();
#ifdef UF_HAS_CUDA
    c10::cuda::OptionalCUDAStreamGuard stream_guard;
    if (runtime_->stream) stream_guard.reset_stream(*runtime_->stream);
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
        frames = image_->decompress(packet.payload, sps.width, sps.height,
                                   packet.qp, packet.entropy_parallel);
        if (video_) {
            video_->clear_dpb();
            video_->add_ref_feature_from_frame(frames.at(0), false);
        }
    } else {
        if (!video_) throw std::runtime_error("UF P packet requires a video model");
        frames = video_->decompress(packet.payload, sps.width, sps.height,
            packet.qp, packet.entropy_parallel, packet.reset_feature_memory);
    }
    if (frames.size() != static_cast<std::size_t>(intra ? 1 : chunk_size_))
        throw std::runtime_error("UF decoder returned an unexpected frame count");
    for (const auto& frame : frames) {
        ProcessInterrupt::check();
        if (!torch::isfinite(frame).all().item<bool>())
            throw std::runtime_error("UF integer decoder produced non-finite pixels");
    }
    const auto count = valid_frames ? static_cast<std::size_t>(
        std::min(*valid_frames, static_cast<std::uint64_t>(frames.size()))) : frames.size();
    std::vector<std::vector<std::uint8_t>> result;
    result.reserve(count);
    for (std::size_t index = 0; index < count; ++index)
        result.push_back(raw_frame(frames[index], sps.width, sps.height));
    return result;
}

CodecStatistics IntegerCodec::decode_stream(std::istream& stream,
                                             std::uint64_t total_frames,
                                             std::uint64_t total_packets,
                                             FrameSink sink, std::optional<Sps> dimensions,
                                             DecodeProgress progress) {
    return decode_stream_impl(stream, total_frames, total_packets, std::move(sink),
        dimensions, std::move(progress), {});
}

CodecStatistics IntegerCodec::decode_stream_impl(std::istream& stream,
    std::uint64_t total_frames, std::uint64_t total_packets, FrameSink sink,
    std::optional<Sps> dimensions, DecodeProgress progress,
    const std::function<void()>& close_input) {
    ProcessInterrupt::check();
#ifdef UF_HAS_CUDA
    c10::cuda::OptionalCUDAStreamGuard stream_guard;
    if (runtime_->stream) stream_guard.reset_stream(*runtime_->stream);
#endif
    torch::NoGradGuard no_grad;
    const auto started = std::chrono::steady_clock::now();
    const auto preamble = read_preamble(stream);
    if (!preamble.integer || preamble.image_identity != preamble_.image_identity ||
        preamble.video_identity != preamble_.video_identity) {
        throw std::runtime_error("UF integer runtime/prepared-model identity mismatch");
    }
    std::array<std::optional<Sps>, 16> sps_table{};
    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> digest(EVP_MD_CTX_new(), EVP_MD_CTX_free);
    if (!digest || EVP_DigestInit_ex(digest.get(), EVP_sha256(), nullptr) != 1) {
        throw std::runtime_error("OpenSSL SHA-256 initialization failed");
    }
    CodecStatistics stats;
    bool first = true;
    while (const auto header = read_optional_header(stream)) {
        ProcessInterrupt::check();
        const auto type = header->type;
        const auto sps_id = header->sps_id;
        if (type == NalType::sps) {
            const auto sps = read_sps(stream, sps_id);
            if (dimensions && (sps.width != dimensions->width || sps.height != dimensions->height)) {
                throw std::runtime_error("Bitstream dimensions differ from archive metadata");
            }
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
            frames = image_->decompress(packet.payload, sps.width, sps.height,
                                       packet.qp, packet.entropy_parallel);
            if (video_) {
                video_->clear_dpb();
                video_->add_ref_feature_from_frame(frames.at(0), false);
            }
        } else {
            if (!video_) throw std::runtime_error("UF P packet requires a video model");
            frames = video_->decompress(packet.payload, sps.width, sps.height,
                                        packet.qp, packet.entropy_parallel,
                                        packet.reset_feature_memory);
        }
        if (frames.size() != static_cast<std::size_t>(intra ? 1 : chunk_size_)) {
            throw std::runtime_error("UF decoder returned an unexpected frame count");
        }
        for (const auto& frame : frames) {
            ProcessInterrupt::check();
            if (stats.frames == total_frames) break;
            // IntegerCodec inherits the same valid-frame finite check as the
            // FP16 archive decoder, before raw conversion, digest and sink.
            if (!torch::isfinite(frame).all().item<bool>()) {
                throw std::runtime_error("UF integer decoder produced non-finite pixels");
            }
            const auto raw = raw_frame(frame, sps.width, sps.height);
            if (EVP_DigestUpdate(digest.get(), raw.data(), raw.size()) != 1) {
                throw std::runtime_error("OpenSSL SHA-256 update failed");
            }
            if (sink) sink(raw);
            ++stats.frames;
        }
        ++stats.packets;
        if (progress) progress(stats.frames, total_frames);
    }
    // Python leaves its with-open block before validating final counts and
    // computing the returned digest/statistics. Borrowed streams stay open.
    if (close_input) close_input();
    ProcessInterrupt::check();
    if (stats.frames != total_frames || stats.packets != total_packets) {
        throw std::runtime_error("Incomplete UF integer bitstream");
    }
    stats.decoded_yuv_sha256 = finish_sha256(digest.get());
    stats.decode_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    return stats;
}

}  // namespace uf
