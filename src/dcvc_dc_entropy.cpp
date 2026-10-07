// Neural estimators and stream partitioning: Microsoft Corporation, MIT.
// CDF/rANS adaptation: InterDigital Communications, Apache-2.0.
#include "uf/dcvc_dc.hpp"
#include "uf/process.hpp"
#include <algorithm>
#include <cmath>
#include <future>
#include <limits>
#include <stdexcept>

namespace uf::dc {
namespace {
std::vector<std::int32_t> integers(const torch::Tensor& input) {
    const auto cpu = input.to(torch::kCPU, torch::kInt32).contiguous().view({-1});
    if (!cpu.numel()) return {};
    return {cpu.data_ptr<std::int32_t>(), cpu.data_ptr<std::int32_t>() + cpu.numel()};
}
std::vector<std::int32_t> shorts(const torch::Tensor& input) {
    const auto cpu = input.to(torch::kInt16).to(torch::kCPU).contiguous().view({-1});
    std::vector<std::int32_t> result;
    if (cpu.numel()) result.assign(cpu.data_ptr<std::int16_t>(), cpu.data_ptr<std::int16_t>() + cpu.numel());
    return result;
}
CdfTables tables(const torch::Tensor& cdf, const torch::Tensor& lengths, const torch::Tensor& offsets) {
    if (!cdf.defined() || !lengths.defined() || !offsets.defined() || cdf.dim() != 2 ||
        lengths.numel() != cdf.size(0) || offsets.numel() != cdf.size(0))
        throw std::runtime_error("Invalid or uninitialized DC CDF helper");
    const auto values = integers(cdf), sizes = integers(lengths);
    CdfTables result; result.offsets = integers(offsets);
    for (std::size_t row = 0; row < sizes.size(); ++row) {
        if (sizes[row] < 3 || sizes[row] > cdf.size(1)) throw std::runtime_error("Invalid DC CDF length");
        const auto first = values.begin() + row * cdf.size(1);
        result.rows.emplace_back(first, first + sizes[row]);
    }
    result.validate();
    return result;
}
torch::Tensor floating(const std::vector<std::int32_t>& values) {
    auto integers = torch::empty({static_cast<std::int64_t>(values.size())},
        torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt16));
    for (std::size_t index = 0; index < values.size(); ++index) {
        const auto word = static_cast<std::uint16_t>(values[index]);
        integers.data_ptr<std::int16_t>()[index] = word < 32768 ? static_cast<std::int16_t>(word)
            : static_cast<std::int16_t>(static_cast<std::int32_t>(word) - 65536);
    }
    auto result = torch::empty(integers.sizes(), torch::TensorOptions().device(torch::kCPU));
    result.copy_(integers);
    return result;
}
void ready(const AEHelper& helper) {
    if (!helper.entropy_coder || !helper.offset.defined()) throw std::runtime_error("DC entropy model must be updated before coding");
}
void append_little(std::string& output, std::uint32_t value, int size) {
    for (int byte = 0; byte < size; ++byte) output.push_back(static_cast<char>(value >> (byte * 8)));
}
std::uint32_t read_little(const std::string& input, std::size_t& position, int size) {
    if (position > input.size() || static_cast<std::size_t>(size) > input.size() - position)
        throw std::runtime_error("Truncated DC entropy stream header");
    std::uint32_t result = 0;
    for (int byte = 0; byte < size; ++byte) result |= static_cast<std::uint32_t>(static_cast<unsigned char>(input[position++])) << (byte * 8);
    return result;
}
}  // namespace

struct EntropyCoder::Impl {
    struct Lane {
        std::vector<std::int32_t> symbols, indexes;
        CdfTables tables;
        std::future<std::string> pending;
        std::string stream;
        dcvc2021::RansDecoder decoder;
        bool flushed = false;
    };
    Impl(bool use_thread, int parts) : threaded(use_thread || parts > 1) {
        if (parts < 1 || parts > 16) throw std::invalid_argument("DC entropy stream_part must be between 1 and 16");
        lanes.resize(static_cast<std::size_t>(parts));
    }
    void finish(Lane& lane) {
        if (lane.pending.valid()) lane.stream = lane.pending.get();
    }
    bool threaded;
    std::vector<Lane> lanes;
};
EntropyCoder::EntropyCoder(bool threaded, int parts) : impl_(std::make_unique<Impl>(threaded, parts)) {}
EntropyCoder::~EntropyCoder() = default;
torch::Tensor normal_cdf(const torch::Tensor& input, const torch::Tensor& scales) {
    if (!(scales > 0).all().item<bool>() || !(input == input).all().item<bool>())
        throw std::invalid_argument("Invalid DC Normal distribution parameter/sample");
    const auto mu = torch::zeros_like(scales);
    return 0.5 * (1 + torch::erf((input - mu) * scales.reciprocal() / std::sqrt(2.0)));
}
torch::Tensor EntropyCoder::pmf_to_quantized_cdf(const torch::Tensor& pmf, int precision) {
    if (precision != 16 || pmf.dim() != 1) throw std::invalid_argument("DC native CDF requires a vector and precision 16");
    const auto probabilities = pmf.to(torch::kCPU, torch::kFloat32).contiguous();
    std::vector<float> values;
    if (probabilities.numel()) values.assign(probabilities.data_ptr<float>(), probabilities.data_ptr<float>() + probabilities.numel());
    const auto cdf = dcvc2021::quantized_cdf(values);
    auto result = torch::empty({static_cast<std::int64_t>(cdf.size())},
        torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt32));
    std::copy(cdf.begin(), cdf.end(), result.data_ptr<std::int32_t>());
    return result;
}
torch::Tensor EntropyCoder::pmf_to_cdf(const torch::Tensor& pmf, const torch::Tensor& tails,
                                     const torch::Tensor& lengths, std::int64_t maximum) {
    const auto probabilities = pmf.to(torch::kCPU, torch::kFloat32).contiguous();
    const auto tail = tails.to(torch::kCPU, torch::kFloat32).contiguous().view({-1});
    const auto sizes = integers(lengths);
    if (probabilities.dim() != 2 || probabilities.size(0) != static_cast<std::int64_t>(sizes.size()) ||
        tail.numel() != probabilities.size(0) || maximum < probabilities.size(1) ||
        maximum > std::numeric_limits<std::int64_t>::max() - 2)
        throw std::invalid_argument("Invalid DC PMF dimensions");
    auto result = torch::zeros({probabilities.size(0), maximum + 2},
        torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt32));
    for (std::size_t row = 0; row < sizes.size(); ++row) {
        ProcessInterrupt::check();
        if (sizes[row] <= 0 || sizes[row] > probabilities.size(1)) throw std::invalid_argument("Invalid DC PMF length");
        const auto* first = probabilities.data_ptr<float>() + row * probabilities.size(1);
        std::vector<float> values(first, first + sizes[row]); values.push_back(tail.data_ptr<float>()[row]);
        const auto cdf = dcvc2021::quantized_cdf(values);
        std::copy(cdf.begin(), cdf.end(), result.data_ptr<std::int32_t>() + row * result.size(1));
    }
    return result;
}
void EntropyCoder::reset() {
    for (auto& lane : impl_->lanes) {
        impl_->finish(lane);
        lane.symbols.clear(); lane.indexes.clear(); lane.tables.rows.clear(); lane.tables.offsets.clear();
        lane.flushed = false;  // Source reset retains the previously encoded bytes.
    }
}
void EntropyCoder::encode_with_indexes(const torch::Tensor& symbols, const torch::Tensor& indexes,
    const torch::Tensor& cdf, const torch::Tensor& lengths, const torch::Tensor& offsets) {
    const auto source_symbols = shorts(symbols.clamp(-30000, 30000)), source_indexes = shorts(indexes);
    if (source_symbols.size() != source_indexes.size()) throw std::invalid_argument("DC entropy symbol/index counts differ");
    const auto source_tables = tables(cdf, lengths, offsets);
    const auto each = source_symbols.size() / impl_->lanes.size();
    for (std::size_t part = 0; part < impl_->lanes.size(); ++part) {
        auto& lane = impl_->lanes[part];
        impl_->finish(lane);
        const auto limit = static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max());
        if (lane.tables.rows.size() > limit || source_tables.rows.size() > limit - lane.tables.rows.size())
            throw std::overflow_error("DC combined CDF table exceeds int32");
        const auto base = static_cast<std::int32_t>(lane.tables.rows.size());
        lane.tables.rows.insert(lane.tables.rows.end(), source_tables.rows.begin(), source_tables.rows.end());
        lane.tables.offsets.insert(lane.tables.offsets.end(), source_tables.offsets.begin(), source_tables.offsets.end());
        const auto begin = part * each;
        const auto end = part + 1 == impl_->lanes.size() ? source_symbols.size() : begin + each;
        for (auto index = begin; index < end; ++index) {
            const auto table = source_indexes[index];
            if (table < 0) continue;  // Encoder skips negative signed int16 indexes.
            if (static_cast<std::size_t>(table) >= source_tables.rows.size()) throw std::out_of_range("DC entropy CDF index");
            lane.symbols.push_back(source_symbols[index]); lane.indexes.push_back(base + table);
        }
        lane.flushed = false;
    }
}
void EntropyCoder::flush() {
    for (auto& lane : impl_->lanes) {
        impl_->finish(lane);
        if (lane.tables.rows.empty()) { lane.tables.rows = {{0, 1, 65536}}; lane.tables.offsets = {0}; }
        auto symbols = std::move(lane.symbols), indexes = std::move(lane.indexes);
        auto cdf = std::move(lane.tables);
        lane.symbols.clear(); lane.indexes.clear(); lane.tables.rows.clear(); lane.tables.offsets.clear();
        const auto encode = [symbols = std::move(symbols), indexes = std::move(indexes), cdf = std::move(cdf)] {
            return dcvc2021::rans_encode(symbols, indexes, cdf);
        };
        if (impl_->threaded) lane.pending = std::async(std::launch::async, encode);
        else lane.stream = encode();
        lane.flushed = true;
    }
}
std::string EntropyCoder::get_encoded_stream() {
    std::size_t maximum = 0;
    for (std::size_t part = 0; part < impl_->lanes.size(); ++part) {
        auto& lane = impl_->lanes[part];
        if (impl_->threaded && !lane.flushed) throw std::logic_error("DC threaded entropy stream requested before flush");
        impl_->finish(lane);
        if (part + 1 < impl_->lanes.size()) maximum = std::max(maximum, lane.stream.size());
    }
    const auto width = maximum > 65535 ? 4 : 2;
    std::string result(1, static_cast<char>(((impl_->lanes.size() - 1) << 4) + (width == 2 ? 1 : 0)));
    for (std::size_t part = 0; part + 1 < impl_->lanes.size(); ++part) {
        if (impl_->lanes[part].stream.size() > std::numeric_limits<std::uint32_t>::max())
            throw std::overflow_error("DC entropy partition exceeds uint32");
        append_little(result, static_cast<std::uint32_t>(impl_->lanes[part].stream.size()), width);
    }
    for (const auto& lane : impl_->lanes) result += lane.stream;
    return result;
}
void EntropyCoder::set_stream(const std::string& stream) {
    if (stream.empty()) throw std::runtime_error("Truncated DC entropy stream");
    const auto flag = static_cast<unsigned char>(stream.front());
    const auto parts = static_cast<std::size_t>((flag >> 4) + 1);
    if (parts != impl_->lanes.size()) throw std::invalid_argument("DC entropy stream_part differs from decoder");
    const auto width = (flag & 0x0f) == 1 ? 2 : 4;
    std::size_t position = 1, total = 0;
    std::vector<std::size_t> sizes;
    for (std::size_t part = 0; part + 1 < parts; ++part) {
        const auto size = read_little(stream, position, width);
        if (size > stream.size() - total) throw std::runtime_error("Invalid DC entropy partition length");
        sizes.push_back(size); total += size;
    }
    if (position > stream.size() || total > stream.size() - position) throw std::runtime_error("Truncated DC entropy partitions");
    sizes.push_back(stream.size() - position - total);
    for (std::size_t part = 0; part < parts; ++part) {
        impl_->lanes[part].decoder.set_stream(stream.substr(position, sizes[part]));
        position += sizes[part];
    }
}
torch::Tensor EntropyCoder::decode_stream(const torch::Tensor& indexes, const torch::Tensor& cdf,
    const torch::Tensor& lengths, const torch::Tensor& offsets) {
    const auto source = shorts(indexes);
    const auto source_tables = tables(cdf, lengths, offsets);
    for (const auto index : source) if (index < 0)
        throw std::out_of_range("DC decoder negative indexes access offsets out of bounds in the reference");
    const auto each = source.size() / impl_->lanes.size();
    std::vector<std::future<std::vector<std::int32_t>>> results;
    for (std::size_t part = 0; part < impl_->lanes.size(); ++part) {
        const auto begin = source.begin() + part * each;
        const auto end = part + 1 == impl_->lanes.size() ? source.end() : begin + each;
        std::vector<std::int32_t> selected(begin, end);
        results.push_back(std::async(std::launch::async, [this, part, selected = std::move(selected), source_tables] {
            return impl_->lanes[part].decoder.decode(selected, source_tables);
        }));
    }
    std::vector<std::int32_t> output;
    for (auto& result : results) { const auto values = result.get(); output.insert(output.end(), values.begin(), values.end()); }
    return floating(output);
}
void AEHelper::set_entropy_coder(std::shared_ptr<EntropyCoder> coder) { entropy_coder = std::move(coder); }
void AEHelper::set_cdf_info(const torch::Tensor& cdf, const torch::Tensor& lengths, const torch::Tensor& offsets) {
    quantized_cdf = cdf.to(torch::kCPU);
    cdf_length = lengths.reshape({-1}).to(torch::kInt32).to(torch::kCPU);
    offset = offsets.reshape({-1}).to(torch::kInt32).to(torch::kCPU);
}
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> AEHelper::get_cdf_info() const {
    return {quantized_cdf, cdf_length, offset};
}
BitparmImpl::BitparmImpl(std::int64_t channels, bool final) : final_(final) {
    h = register_parameter("h", torch::empty({channels}).view({1, -1, 1, 1}).normal_(0, 0.01));
    b = register_parameter("b", torch::empty({channels}).view({1, -1, 1, 1}).normal_(0, 0.01));
    if (!final_) a = register_parameter("a", torch::empty({channels}).view({1, -1, 1, 1}).normal_(0, 0.01));
}
torch::Tensor BitparmImpl::forward(const torch::Tensor& input) {
    const auto x = input * torch::softplus(h) + b;
    return final_ ? x : x + x.tanh() * a.tanh();
}
BitEstimatorImpl::BitEstimatorImpl(std::int64_t channels) : channel(channels) {
    f1 = register_module("f1", Bitparm(channel)); f2 = register_module("f2", Bitparm(channel));
    f3 = register_module("f3", Bitparm(channel)); f4 = register_module("f4", Bitparm(channel, true));
}
torch::Tensor BitEstimatorImpl::get_logits_cdf(const torch::Tensor& input) { return f4(f3(f2(f1(input)))); }
torch::Tensor BitEstimatorImpl::get_cdf(const torch::Tensor& input) { return get_logits_cdf(input).sigmoid(); }
torch::Tensor BitEstimatorImpl::forward(const torch::Tensor& input) { return get_cdf(input); }
void BitEstimatorImpl::update(bool force, std::shared_ptr<EntropyCoder> coder) {
    if (coder) entropy_coder = std::move(coder);
    if (!force && offset.defined()) return;
    torch::NoGradGuard no_grad;
    const auto medians = torch::zeros({channel}, torch::TensorOptions().device(f1->h.device()));
    auto minima = medians + 50, maxima = medians + 50;
    for (int index = 50; index > 1; --index) {
        ProcessInterrupt::check();
        const auto probability = forward((torch::zeros_like(medians) - index).view({1, channel, 1, 1})).squeeze();
        minima = torch::where(probability < torch::zeros_like(medians) + 0.0001, torch::zeros_like(medians) + index, minima);
    }
    for (int index = 50; index > 1; --index) {
        const auto probability = forward((torch::zeros_like(medians) + index).view({1, channel, 1, 1})).squeeze();
        maxima = torch::where(probability > torch::zeros_like(medians) + 0.9999, torch::zeros_like(medians) + index, maxima);
    }
    minima = minima.to(torch::kInt32); maxima = maxima.to(torch::kInt32);
    const auto sizes = maxima + minima + 1;
    const auto maximum = sizes.max().item<std::int64_t>();
    const auto samples = torch::arange(maximum, torch::TensorOptions().device(medians.device()))
        .view({1, 1, 1, maximum}) + (medians - minima).view({1, channel, 1, 1});
    const auto lower = forward(samples - 0.5).squeeze(0), upper = forward(samples + 0.5).squeeze(0);
    const auto pmf = upper.select(1, 0) - lower.select(1, 0);
    const auto tails = lower.select(1, 0).slice(1, 0, 1) + (1.0 - upper.select(1, 0).slice(1, maximum - 1));
    set_cdf_info(EntropyCoder::pmf_to_cdf(pmf, tails, sizes, maximum), sizes + 2, -minima);
}
torch::Tensor BitEstimatorImpl::build_indexes(const std::vector<std::int64_t>& size) {
    if (size.size() != 4) throw std::invalid_argument("DC entropy indexes require NCHW dimensions");
    return torch::arange(size[1], torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt32))
        .view({1, -1, 1, 1}).repeat({size[0], 1, size[2], size[3]});
}
void BitEstimatorImpl::encode(const torch::Tensor& input) {
    ready(*this);
    entropy_coder->encode_with_indexes(input.reshape({-1}), build_indexes(input.sizes().vec()).reshape({-1}),
        quantized_cdf, cdf_length, offset);
}
torch::Tensor BitEstimatorImpl::decode_stream(std::pair<std::int64_t, std::int64_t> size,
                                             torch::ScalarType dtype, torch::Device device) {
    ready(*this);
    const std::vector<std::int64_t> shape{1, channel, size.first, size.second};
    const auto indexes = build_indexes(shape);
    return entropy_coder->decode_stream(indexes.reshape({-1}), quantized_cdf, cdf_length, offset).reshape(shape).to(dtype).to(device);
}
GaussianEncoder::GaussianEncoder(std::string selected) : distribution(std::move(selected)) {
    if (distribution != "laplace" && distribution != "gaussian") throw std::invalid_argument("Invalid DC latent distribution");
    log_min_ = std::log(distribution == "laplace" ? 0.01 : 0.11);
    log_step_ = (std::log(64.0) - log_min_) / 255;
    scale_table = get_scale_table(distribution == "laplace" ? 0.01 : 0.11, 64, 256);
}
torch::Tensor GaussianEncoder::get_scale_table(double minimum, double maximum, std::int64_t levels) {
    return torch::linspace(std::log(minimum), std::log(maximum), levels).exp();
}
void GaussianEncoder::update(bool force, std::shared_ptr<EntropyCoder> coder) {
    if (coder) entropy_coder = std::move(coder);
    if (!force && offset.defined()) return;
    auto center = torch::zeros_like(scale_table) + 50;
    const auto scales = torch::zeros_like(center) + scale_table;
    const auto cdf = [&](const torch::Tensor& samples, const torch::Tensor& sigma) {
        return distribution == "laplace" ? dcvc2021::laplace_cdf(samples, sigma) : normal_cdf(samples, sigma);
    };
    for (int index = 50; index > 1; --index) {
        ProcessInterrupt::check();
        center = torch::where(cdf(torch::zeros_like(center) + index, scales).squeeze() > torch::zeros_like(center) + 0.9999,
            torch::zeros_like(center) + index, center);
    }
    center = center.to(torch::kInt32);
    const auto sizes = 2 * center + 1;
    const auto maximum = sizes.max().item<std::int64_t>();
    const auto samples = (torch::arange(maximum, torch::TensorOptions().device(center.device())).unsqueeze(0) - center.unsqueeze(1))
        .to(torch::kFloat32);
    const auto sigmas = torch::zeros_like(samples) + scale_table.unsqueeze(1);
    const auto upper = cdf(samples + 0.5, sigmas), lower = cdf(samples - 0.5, sigmas);
    set_cdf_info(EntropyCoder::pmf_to_cdf(upper - lower, 2 * lower.slice(1, 0, 1), sizes, maximum), sizes + 2, -center);
}
torch::Tensor GaussianEncoder::build_indexes(const torch::Tensor& scales) const {
    const auto bounded = torch::maximum(scales, torch::zeros_like(scales) + 1e-5);
    return ((bounded.log() - log_min_) / log_step_).clamp_(0, 255).to(torch::kInt32);
}
void GaussianEncoder::encode(const torch::Tensor& input, const torch::Tensor& scales) {
    ready(*this);
    entropy_coder->encode_with_indexes(input.reshape({-1}), build_indexes(scales).reshape({-1}), quantized_cdf, cdf_length, offset);
}
torch::Tensor GaussianEncoder::decode_stream(const torch::Tensor& scales, torch::ScalarType dtype, torch::Device device) {
    ready(*this);
    return entropy_coder->decode_stream(build_indexes(scales).reshape({-1}), quantized_cdf, cdf_length, offset)
        .reshape(scales.sizes()).to(device).to(dtype);
}
}  // namespace uf::dc
