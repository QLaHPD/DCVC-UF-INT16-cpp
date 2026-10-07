// Neural estimators and stream partitioning: Microsoft Corporation, MIT.
// CDF/rANS adaptation: InterDigital Communications, Apache-2.0.
#include "uf/dcvc_fm.hpp"
#include "uf/process.hpp"
#include "dcvc_fm_rans.hpp"
#include <cstring>
#include <algorithm>
#include <cmath>
#include <future>
#include <limits>
#include <stdexcept>

namespace uf::fm {
namespace {
std::vector<std::int32_t> integers(const torch::Tensor& input) {
    const auto cpu = input.to(torch::kCPU, torch::kInt32).contiguous().view({-1});
    if (!cpu.numel()) return {};
    return {cpu.data_ptr<std::int32_t>(), cpu.data_ptr<std::int32_t>() + cpu.numel()};
}
std::vector<std::int16_t> shorts(const torch::Tensor& input) {
    const auto cpu = input.to(torch::kInt16).to(torch::kCPU).contiguous().view({-1});
    if (!cpu.numel()) return {};
    return {cpu.data_ptr<std::int16_t>(), cpu.data_ptr<std::int16_t>() + cpu.numel()};
}
CdfTables tables(const torch::Tensor& cdf, const torch::Tensor& lengths, const torch::Tensor& offsets) {
    if (!cdf.defined() || !lengths.defined() || !offsets.defined() || cdf.dim() != 2 ||
        lengths.numel() != cdf.size(0) || offsets.numel() != cdf.size(0))
        throw std::runtime_error("Invalid or uninitialized FM CDF helper");
    const auto values = integers(cdf), sizes = integers(lengths);
    CdfTables result; result.offsets = integers(offsets);
    for (std::size_t row = 0; row < sizes.size(); ++row) {
        if (sizes[row] < 3 || sizes[row] > cdf.size(1)) throw std::runtime_error("Invalid FM CDF length");
        const auto first = values.begin() + row * cdf.size(1);
        result.rows.emplace_back(first, first + sizes[row]);
    }
    result.validate();
    return result;
}
torch::Tensor floating(const std::vector<std::int16_t>& values) {
    auto ints = torch::empty({static_cast<std::int64_t>(values.size())},
        torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt16));
    std::copy(values.begin(), values.end(), ints.data_ptr<std::int16_t>());
    auto result = torch::empty(ints.sizes(), torch::TensorOptions().device(torch::kCPU));
    result.copy_(ints);
    return result;
}
void ready(const AEHelper& helper) {
    if (!helper.entropy_coder || !helper.offset.defined() || !helper.cdf_group_index)
        throw std::runtime_error("FM entropy model must be updated before coding");
}
void append_size(std::string& output, std::uint32_t value, int size) {
    // The Python binding copies native-endian partition sizes, unlike the LE rANS state.
    if (size == 2) { const auto word = static_cast<std::uint16_t>(value); output.append(reinterpret_cast<const char*>(&word), 2); }
    else output.append(reinterpret_cast<const char*>(&value), 4);
}
std::uint32_t read_size(const std::string& input, std::size_t& position, int size) {
    if (position > input.size() || static_cast<std::size_t>(size) > input.size() - position)
        throw std::runtime_error("Truncated FM entropy partition header");
    std::uint32_t result = 0;
    if (size == 2) { std::uint16_t word; std::memcpy(&word, input.data() + position, 2); result = word; }
    else std::memcpy(&result, input.data() + position, 4);
    position += size; return result;
}
}  // namespace

struct EntropyCoder::Impl {
    Impl(bool threaded, int parts) {
        if (parts < 1 || parts > 16) throw std::invalid_argument("FM entropy stream_part must be between 1 and 16");
        for (int i = 0; i < parts; ++i) {
            encoders.push_back(std::make_unique<detail::EncoderLane>(threaded || parts > 1));
            decoders.emplace_back();
        }
    }
    std::shared_ptr<const CdfTables> group(int index) const {
        if (index < 0 || static_cast<std::size_t>(index) >= groups.size())
            throw std::out_of_range("FM CDF group is not registered in this entropy coder");
        return groups[index];
    }
    std::vector<std::unique_ptr<detail::EncoderLane>> encoders;
    std::vector<detail::DecoderLane> decoders;
    std::vector<std::shared_ptr<const CdfTables>> groups;
};
EntropyCoder::EntropyCoder(bool threaded, int parts) : impl_(std::make_unique<Impl>(threaded, parts)) {}
EntropyCoder::~EntropyCoder() = default;
torch::Tensor normal_cdf(const torch::Tensor& input, const torch::Tensor& scales) {
    if (!(scales > 0).all().item<bool>() || !(input == input).all().item<bool>())
        throw std::invalid_argument("Invalid FM Normal distribution parameter/sample");
    const auto mu = torch::zeros_like(scales);
    return 0.5 * (1 + torch::erf((input - mu) * scales.reciprocal() / std::sqrt(2.0)));
}
torch::Tensor EntropyCoder::pmf_to_quantized_cdf(const torch::Tensor& pmf, int precision) {
    if (pmf.dim() != 1) throw std::invalid_argument("FM native CDF requires a vector");
    const auto probabilities = pmf.to(torch::kCPU, torch::kFloat32).contiguous();
    std::vector<float> values;
    if (probabilities.numel()) values.assign(probabilities.data_ptr<float>(), probabilities.data_ptr<float>() + probabilities.numel());
    const auto cdf = detail::quantized_cdf(values, precision);
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
        throw std::invalid_argument("Invalid FM PMF dimensions");
    auto result = torch::zeros({probabilities.size(0), maximum + 2},
        torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt32));
    for (std::size_t row = 0; row < sizes.size(); ++row) {
        ProcessInterrupt::check();
        if (sizes[row] <= 0 || sizes[row] > probabilities.size(1)) throw std::invalid_argument("Invalid FM PMF length");
        const auto* first = probabilities.data_ptr<float>() + row * probabilities.size(1);
        std::vector<float> values(first, first + sizes[row]); values.push_back(tail.data_ptr<float>()[row]);
        const auto cdf = detail::quantized_cdf(values, 16);
        std::copy(cdf.begin(), cdf.end(), result.data_ptr<std::int32_t>() + row * result.size(1));
    }
    return result;
}
int EntropyCoder::add_cdf(const torch::Tensor& cdf, const torch::Tensor& lengths, const torch::Tensor& offsets) {
    if (impl_->groups.size() >= static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::overflow_error("Too many FM CDF groups");
    impl_->groups.push_back(std::make_shared<const CdfTables>(tables(cdf, lengths, offsets)));
    return static_cast<int>(impl_->groups.size()) - 1;
}
void EntropyCoder::empty_cdf_buffer() {
    for (auto& lane : impl_->encoders) lane->wait_idle();
    impl_->groups.clear();
}
void EntropyCoder::reset() { for (auto& lane : impl_->encoders) lane->reset(); }
void EntropyCoder::encode_with_indexes(const torch::Tensor& symbols, const torch::Tensor& indexes, int group) {
    encode_with_indexes_np(shorts(symbols.clamp(-30000, 30000)), shorts(indexes), group);
}
void EntropyCoder::encode_with_indexes_np(const std::vector<std::int16_t>& symbols,
                                          const std::vector<std::int16_t>& indexes, int group) {
    if (symbols.size() != indexes.size()) throw std::invalid_argument("FM symbol/index counts differ");
    const auto cdf = impl_->group(group);
    const auto each = symbols.size() / impl_->encoders.size();
    for (std::size_t part = 0; part < impl_->encoders.size(); ++part) {
        const auto begin = part * each;
        const auto end = part + 1 == impl_->encoders.size() ? symbols.size() : begin + each;
        std::vector<std::int16_t> selected_symbols(symbols.begin() + begin, symbols.begin() + end);
        for (auto& symbol : selected_symbols) symbol = std::clamp<std::int16_t>(symbol, -30000, 30000);
        impl_->encoders[part]->encode(std::move(selected_symbols), {indexes.begin() + begin, indexes.begin() + end}, cdf);
    }
}
void EntropyCoder::flush() { for (auto& lane : impl_->encoders) lane->flush(); }
std::string EntropyCoder::get_encoded_stream() {
    std::vector<std::string> streams;
    std::size_t maximum = 0;
    for (std::size_t part = 0; part < impl_->encoders.size(); ++part) {
        streams.push_back(impl_->encoders[part]->get_stream());
        if (part + 1 < impl_->encoders.size()) maximum = std::max(maximum, streams.back().size());
    }
    const auto width = maximum > 65535 ? 4 : 2;
    std::string result(1, static_cast<char>(((streams.size() - 1) << 4) + (width == 2 ? 1 : 0)));
    for (std::size_t part = 0; part + 1 < streams.size(); ++part) {
        if (streams[part].size() > std::numeric_limits<std::uint32_t>::max()) throw std::overflow_error("FM entropy partition exceeds uint32");
        append_size(result, static_cast<std::uint32_t>(streams[part].size()), width);
    }
    for (const auto& stream : streams) result += stream;
    return result;
}
void EntropyCoder::set_stream(const std::string& stream) {
    if (stream.empty()) throw std::runtime_error("Truncated FM entropy stream");
    const auto flag = static_cast<unsigned char>(stream.front());
    const auto parts = static_cast<std::size_t>((flag >> 4) + 1);
    if (parts != impl_->decoders.size()) throw std::invalid_argument("FM entropy stream_part differs from decoder");
    const auto width = (flag & 0x0f) == 1 ? 2 : 4;
    std::size_t position = 1, total = 0;
    std::vector<std::size_t> sizes;
    for (std::size_t part = 0; part + 1 < parts; ++part) {
        const auto size = read_size(stream, position, width);
        if (size > stream.size() - total) throw std::runtime_error("Invalid FM entropy partition length");
        sizes.push_back(size); total += size;
    }
    if (total > stream.size() - position) throw std::runtime_error("Truncated FM entropy partitions");
    sizes.push_back(stream.size() - position - total);
    for (std::size_t part = 0; part < parts; ++part) {
        impl_->decoders[part].set_stream(stream.substr(position, sizes[part])); position += sizes[part];
    }
}
std::vector<std::int16_t> EntropyCoder::decode_stream_np(const std::vector<std::int16_t>& indexes, int group) {
    const auto cdf = impl_->group(group);
    const auto each = indexes.size() / impl_->decoders.size();
    std::vector<std::future<std::vector<std::int16_t>>> results;
    for (std::size_t part = 0; part < impl_->decoders.size(); ++part) {
        const auto begin = indexes.begin() + part * each;
        const auto end = part + 1 == impl_->decoders.size() ? indexes.end() : begin + each;
        std::vector<std::int16_t> selected(begin, end);
        results.push_back(std::async(std::launch::async, [this, part, selected = std::move(selected), cdf] {
            return impl_->decoders[part].decode(selected, *cdf);
        }));
    }
    std::vector<std::int16_t> output;
    for (auto& result : results) { const auto values = result.get(); output.insert(output.end(), values.begin(), values.end()); }
    return output;
}
torch::Tensor EntropyCoder::decode_stream(const torch::Tensor& indexes, int group) {
    return floating(decode_stream_np(shorts(indexes), group));
}
void AEHelper::set_cdf_info(const torch::Tensor& cdf, const torch::Tensor& lengths, const torch::Tensor& offsets) {
    quantized_cdf = cdf.to(torch::kCPU);
    cdf_length = lengths.reshape({-1}).to(torch::kInt32).to(torch::kCPU);
    offset = offsets.reshape({-1}).to(torch::kInt32).to(torch::kCPU);
}
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> AEHelper::get_cdf_info() const {
    return {quantized_cdf, cdf_length, offset};
}
BitparmImpl::BitparmImpl(std::int64_t qp_num, std::int64_t channels, bool final) : final_(final) {
    h = register_parameter("h", torch::empty({qp_num, channels, 1, 1}).normal_(0, 0.01));
    b = register_parameter("b", torch::empty({qp_num, channels, 1, 1}).normal_(0, 0.01));
    if (!final_) a = register_parameter("a", torch::empty({qp_num, channels, 1, 1}).normal_(0, 0.01));
}
torch::Tensor BitparmImpl::forward(const torch::Tensor& input, const torch::Tensor& index) {
    const auto selected_h = torch::index_select(h, 0, index), selected_b = torch::index_select(b, 0, index);
    const auto x = input * torch::softplus(selected_h) + selected_b;
    return final_ ? x : x + x.tanh() * torch::index_select(a, 0, index).tanh();
}
BitEstimatorImpl::BitEstimatorImpl(std::int64_t qp, std::int64_t channels) : qp_num(qp), channel(channels) {
    f1 = register_module("f1", Bitparm(qp_num, channel)); f2 = register_module("f2", Bitparm(qp_num, channel));
    f3 = register_module("f3", Bitparm(qp_num, channel)); f4 = register_module("f4", Bitparm(qp_num, channel, true));
}
torch::Tensor BitEstimatorImpl::get_logits_cdf(const torch::Tensor& input, const torch::Tensor& index) {
    return f4(f3(f2(f1(input, index), index), index), index);
}
torch::Tensor BitEstimatorImpl::get_cdf(const torch::Tensor& input, const torch::Tensor& index) { return get_logits_cdf(input, index).sigmoid(); }
torch::Tensor BitEstimatorImpl::forward(const torch::Tensor& input, const torch::Tensor& index) { return get_cdf(input, index); }
void BitEstimatorImpl::update(bool force, std::shared_ptr<EntropyCoder> coder) {
    if (!coder) throw std::invalid_argument("FM BitEstimator update requires an entropy coder");
    entropy_coder = std::move(coder);
    // Source changes coder before this return, without registering cached CDFs again.
    if (!force && offset.defined()) return;
    torch::NoGradGuard no_grad;
    const auto medians = torch::zeros({qp_num, channel, 1, 1}, torch::TensorOptions().device(f1->h.device()));
    const auto index = torch::arange(qp_num, torch::TensorOptions().device(f1->h.device()).dtype(torch::kInt32));
    auto minima = medians + 50, maxima = medians + 50;
    for (int i = 50; i > 1; --i) {
        ProcessInterrupt::check();
        minima = torch::where(forward(torch::zeros_like(medians) - i, index) < torch::zeros_like(medians) + 0.0001,
                              torch::zeros_like(medians) + i, minima);
    }
    for (int i = 50; i > 1; --i)
        maxima = torch::where(forward(torch::zeros_like(medians) + i, index) > torch::zeros_like(medians) + 0.9999,
                              torch::zeros_like(medians) + i, maxima);
    minima = minima.to(torch::kInt32); maxima = maxima.to(torch::kInt32);
    const auto sizes = maxima + minima + 1;
    const auto maximum = sizes.max().item<std::int64_t>();
    const auto samples = torch::arange(maximum, torch::TensorOptions().device(medians.device()).dtype(torch::kInt64))
        .view({1, 1, 1, maximum}) + medians - minima;
    const auto lower = forward(samples - 0.5, index).select(2, 0), upper = forward(samples + 0.5, index).select(2, 0);
    const auto pmf = (upper - lower).reshape({-1, maximum});
    const auto tails = (lower.slice(2, 0, 1) + (1.0 - upper.slice(2, maximum - 1))).reshape({-1, 1});
    set_cdf_info(EntropyCoder::pmf_to_cdf(pmf, tails, sizes.reshape({-1}), maximum), sizes + 2, -minima);
    cdf_group_index = entropy_coder->add_cdf(quantized_cdf, cdf_length, offset);
}
torch::Tensor BitEstimatorImpl::build_indexes(const std::vector<std::int64_t>& size, std::int64_t qp) const {
    if (size.size() != 4) throw std::invalid_argument("FM entropy indexes require NCHW dimensions");
    if (qp < std::numeric_limits<std::int32_t>::min() / channel || qp > std::numeric_limits<std::int32_t>::max() / channel)
        throw std::overflow_error("FM QP index offset exceeds int32");
    return (torch::arange(size[1], torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt32))
        .view({1, -1, 1, 1}) + qp * channel).repeat({size[0], 1, size[2], size[3]});
}
std::vector<std::int16_t> BitEstimatorImpl::build_indexes_np(const std::vector<std::int64_t>& size, std::int64_t qp) const {
    return shorts(build_indexes(size, qp));
}
void BitEstimatorImpl::encode(const torch::Tensor& input, std::int64_t qp) {
    ready(*this);
    entropy_coder->encode_with_indexes(input.reshape({-1}), build_indexes(input.sizes().vec(), qp).reshape({-1}), *cdf_group_index);
}
torch::Tensor BitEstimatorImpl::decode_stream(std::pair<std::int64_t, std::int64_t> size,
                                             torch::ScalarType dtype, torch::Device device, std::int64_t qp) {
    ready(*this);
    const std::vector<std::int64_t> shape{1, channel, size.first, size.second};
    return entropy_coder->decode_stream(build_indexes(shape, qp).reshape({-1}), *cdf_group_index).reshape(shape).to(dtype).to(device);
}
GaussianEncoder::GaussianEncoder(std::string selected) : distribution(std::move(selected)) {
    if (distribution != "laplace" && distribution != "gaussian") throw std::invalid_argument("Invalid FM latent distribution");
    log_min_ = std::log(distribution == "laplace" ? 0.01 : 0.11);
    log_step_ = (std::log(64.0) - log_min_) / 255;
    scale_table = get_scale_table(distribution == "laplace" ? 0.01 : 0.11, 64, 256);
}
torch::Tensor GaussianEncoder::get_scale_table(double minimum, double maximum, std::int64_t levels) {
    return torch::linspace(std::log(minimum), std::log(maximum), levels).exp();
}
void GaussianEncoder::update(bool force, std::shared_ptr<EntropyCoder> coder) {
    if (!coder) throw std::invalid_argument("FM GaussianEncoder update requires an entropy coder");
    entropy_coder = std::move(coder);
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
    cdf_group_index = entropy_coder->add_cdf(quantized_cdf, cdf_length, offset);
}
torch::Tensor GaussianEncoder::build_indexes(const torch::Tensor& scales) const {
    const auto bounded = torch::maximum(scales, torch::zeros_like(scales) + 1e-5);
    return ((bounded.log() - log_min_) / log_step_).clamp_(0, 255).to(torch::kInt32);
}
void GaussianEncoder::encode(const torch::Tensor& input, const torch::Tensor& scales) {
    ready(*this);
    entropy_coder->encode_with_indexes(input.reshape({-1}), build_indexes(scales).reshape({-1}), *cdf_group_index);
}
torch::Tensor GaussianEncoder::decode_stream(const torch::Tensor& scales, torch::ScalarType dtype, torch::Device device) {
    ready(*this);
    return entropy_coder->decode_stream(build_indexes(scales).reshape({-1}), *cdf_group_index)
        .reshape(scales.sizes()).to(device).to(dtype);
}
}  // namespace uf::fm
