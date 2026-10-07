// Neural estimators: Microsoft Corporation, MIT.
// Buffered rANS/CDF adaptation: InterDigital Communications, Apache-2.0.
#include "dcvc_hem_internal.hpp"
#include "uf/process.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace uf::hem::detail {
std::vector<std::int32_t> integers(const torch::Tensor& input) {
    const auto cpu = input.to(torch::kCPU, torch::kInt32).contiguous().view({-1});
    if (!cpu.numel()) return {};
    return {cpu.data_ptr<std::int32_t>(), cpu.data_ptr<std::int32_t>() + cpu.numel()};
}
torch::Tensor floating(const std::vector<std::int32_t>& values) {
    auto integers = torch::empty({static_cast<std::int64_t>(values.size())},
        torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt32));
    if (!values.empty()) std::copy(values.begin(), values.end(), integers.data_ptr<std::int32_t>());
    auto result = torch::empty(integers.sizes(), torch::TensorOptions().device(torch::kCPU));
    result.copy_(integers);
    return result;
}
}  // namespace uf::hem::detail

namespace uf::hem {
namespace {
const CdfHelper& ready(const std::shared_ptr<EntropyCoder>& coder, const std::shared_ptr<CdfHelper>& helper) {
    if (!coder || !helper) throw std::runtime_error("HEM entropy model must be updated before coding");
    return *helper;
}
}  // namespace
torch::Tensor normal_cdf(const torch::Tensor& input, const torch::Tensor& scales) {
    if (!(scales > 0).all().item<bool>() || !(input == input).all().item<bool>())
        throw std::invalid_argument("Invalid HEM Normal distribution parameter/sample");
    const auto mu = torch::zeros_like(scales);
    return 0.5 * (1 + torch::erf((input - mu) * scales.reciprocal() / std::sqrt(2.0)));
}
void CdfHelper::set_cdf(const torch::Tensor& offsets, const torch::Tensor& cdf, const torch::Tensor& sizes) {
    offset = offsets.reshape({-1}).to(torch::kInt32).to(torch::kCPU);
    quantized = cdf.to(torch::kCPU);
    lengths = sizes.reshape({-1}).to(torch::kInt32).to(torch::kCPU);
}
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> CdfHelper::get_cdf_info() const {
    return {quantized, lengths, offset};
}
CdfTables CdfHelper::tables() const {
    if (!quantized.defined() || !lengths.defined() || !offset.defined() || quantized.dim() != 2 ||
        lengths.numel() != quantized.size(0) || offset.numel() != quantized.size(0))
        throw std::runtime_error("Invalid or uninitialized HEM CDF helper");
    const auto values = detail::integers(quantized), sizes = detail::integers(lengths);
    CdfTables result; result.offsets = detail::integers(offset);
    for (std::size_t i = 0; i < sizes.size(); ++i) {
        if (sizes[i] < 3 || sizes[i] > quantized.size(1)) throw std::runtime_error("Invalid HEM CDF length");
        const auto first = values.begin() + i * quantized.size(1);
        result.rows.emplace_back(first, first + sizes[i]);
    }
    result.validate(); return result;
}
torch::Tensor EntropyCoder::pmf_to_quantized_cdf(const torch::Tensor& pmf, int precision) {
    if (precision != 16 || pmf.dim() != 1) throw std::invalid_argument("HEM native CDF requires a vector and precision 16");
    const auto probabilities = pmf.to(torch::kCPU, torch::kFloat32).contiguous();
    std::vector<float> values;
    if (probabilities.numel()) values.assign(probabilities.data_ptr<float>(), probabilities.data_ptr<float>() + probabilities.numel());
    const auto cdf = dcvc2021::quantized_cdf(values);
    auto result = torch::empty({static_cast<std::int64_t>(cdf.size())},
        torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt32));
    std::copy(cdf.begin(), cdf.end(), result.data_ptr<std::int32_t>()); return result;
}
torch::Tensor EntropyCoder::pmf_to_cdf(const torch::Tensor& pmf, const torch::Tensor& tails,
                                     const torch::Tensor& lengths, std::int64_t maximum) {
    const auto probabilities = pmf.to(torch::kCPU, torch::kFloat32).contiguous();
    const auto tail = tails.to(torch::kCPU, torch::kFloat32).contiguous().view({-1});
    const auto sizes = detail::integers(lengths);
    if (probabilities.dim() != 2 || probabilities.size(0) != static_cast<std::int64_t>(sizes.size()) ||
        tail.numel() != probabilities.size(0) || maximum < probabilities.size(1) ||
        maximum > std::numeric_limits<std::int64_t>::max() - 2)
        throw std::invalid_argument("Invalid HEM PMF dimensions");
    auto result = torch::zeros({probabilities.size(0), maximum + 2},
        torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt32));
    for (std::size_t i = 0; i < sizes.size(); ++i) {
        ProcessInterrupt::check();
        if (sizes[i] <= 0 || sizes[i] > probabilities.size(1)) throw std::invalid_argument("Invalid HEM PMF length");
        const auto* first = probabilities.data_ptr<float>() + i * probabilities.size(1);
        std::vector<float> row(first, first + sizes[i]); row.push_back(tail.data_ptr<float>()[i]);
        const auto cdf = dcvc2021::quantized_cdf(row);
        std::copy(cdf.begin(), cdf.end(), result.data_ptr<std::int32_t>() + i * result.size(1));
    }
    return result;
}
void EntropyCoder::reset_encoder() { symbols_.clear(); indexes_.clear(); tables_.rows.clear(); tables_.offsets.clear(); }
void EntropyCoder::encode_with_indexes(const std::vector<std::int32_t>& symbols,
    const std::vector<std::int32_t>& indexes, const CdfTables& tables) {
    tables.validate();
    if (symbols.size() != indexes.size()) throw std::invalid_argument("HEM entropy symbol/index counts differ");
    const auto limit = static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max());
    if (tables_.rows.size() > limit || tables.rows.size() > limit - tables_.rows.size())
        throw std::overflow_error("HEM combined CDF table exceeds int32");
    for (const auto index : indexes) if (index < 0 || static_cast<std::size_t>(index) >= tables.rows.size())
        throw std::out_of_range("HEM entropy CDF index");
    const auto base = static_cast<std::int32_t>(tables_.rows.size());
    tables_.rows.insert(tables_.rows.end(), tables.rows.begin(), tables.rows.end());
    tables_.offsets.insert(tables_.offsets.end(), tables.offsets.begin(), tables.offsets.end());
    symbols_.insert(symbols_.end(), symbols.begin(), symbols.end());
    for (const auto index : indexes) indexes_.push_back(base + index);
}
std::string EntropyCoder::flush_encoder() {
    if (tables_.rows.empty()) { tables_.rows = {{0, 1, 65536}}; tables_.offsets = {0}; }
    const auto stream = dcvc2021::rans_encode(symbols_, indexes_, tables_); reset_encoder(); return stream;
}
void EntropyCoder::set_stream(const std::string& stream) { decoder_.set_stream(stream); }
torch::Tensor EntropyCoder::decode_stream(const std::vector<std::int32_t>& indexes, const CdfTables& tables) {
    return detail::floating(decoder_.decode(indexes, tables)).view({1, -1, 1, 1});
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
    if (entropy_coder && !force) return;
    entropy_coder = std::move(coder); cdf_helper = std::make_shared<CdfHelper>();
    torch::NoGradGuard no_grad;
    const auto medians = torch::zeros({channel}, torch::TensorOptions().device(f1->h.device()));
    auto minima = medians + 50, maxima = medians + 50;
    for (int i = 50; i > 1; --i) {
        ProcessInterrupt::check();
        const auto probability = forward((torch::zeros_like(medians) - i).view({1, channel, 1, 1})).squeeze();
        minima = torch::where(probability < torch::zeros_like(medians) + 0.0001, torch::zeros_like(medians) + i, minima);
    }
    for (int i = 50; i > 1; --i) {
        const auto probability = forward((torch::zeros_like(medians) + i).view({1, channel, 1, 1})).squeeze();
        maxima = torch::where(probability > torch::zeros_like(medians) + 0.9999, torch::zeros_like(medians) + i, maxima);
    }
    minima = minima.to(torch::kInt32); maxima = maxima.to(torch::kInt32);
    const auto sizes = maxima + minima + 1;
    const auto maximum = sizes.max().item<std::int64_t>();
    const auto samples = torch::arange(maximum, torch::TensorOptions().device(medians.device()))
        .view({1, 1, 1, maximum}) + (medians - minima).view({1, channel, 1, 1});
    const auto lower = forward(samples - 0.5).squeeze(0), upper = forward(samples + 0.5).squeeze(0);
    const auto pmf = upper.select(1, 0) - lower.select(1, 0);
    const auto tails = lower.select(1, 0).slice(1, 0, 1) + (1.0 - upper.select(1, 0).slice(1, maximum - 1));
    cdf_helper->set_cdf(-minima, EntropyCoder::pmf_to_cdf(pmf, tails, sizes, maximum), sizes + 2);
}
torch::Tensor BitEstimatorImpl::build_indexes(const std::vector<std::int64_t>& size) {
    if (size.size() != 4) throw std::invalid_argument("HEM entropy indexes require NCHW dimensions");
    return torch::arange(size[1], torch::TensorOptions().device(torch::kCPU)).view({1, -1, 1, 1})
        .to(torch::kInt32).repeat({size[0], 1, size[2], size[3]});
}
void BitEstimatorImpl::encode(const torch::Tensor& input) {
    const auto& helper = ready(entropy_coder, cdf_helper);
    if (input.dim() != 4 || input.size(0) != 1) throw std::invalid_argument("HEM hyperlatent coding requires batch one");
    const auto indexes = build_indexes(input.sizes().vec());
    entropy_coder->encode_with_indexes(detail::integers(input), detail::integers(indexes.select(0, 0)), helper.tables());
}
torch::Tensor BitEstimatorImpl::decode_stream(std::pair<std::int64_t, std::int64_t> size) {
    const auto& helper = ready(entropy_coder, cdf_helper);
    const std::vector<std::int64_t> shape{1, channel, size.first, size.second};
    const auto indexes = build_indexes(shape);
    return entropy_coder->decode_stream(detail::integers(indexes), helper.tables()).reshape(shape).to(torch::kFloat32);
}
GaussianEncoder::GaussianEncoder(std::string selected) : distribution(std::move(selected)) {
    if (distribution != "laplace" && distribution != "gaussian") throw std::invalid_argument("Invalid HEM latent distribution");
    log_min_ = std::log(distribution == "laplace" ? 0.01 : 0.11);
    log_step_ = (std::log(64.0) - log_min_) / 255;
    scale_table = get_scale_table(distribution == "laplace" ? 0.01 : 0.11, 64.0, 256);
}
torch::Tensor GaussianEncoder::get_scale_table(double minimum, double maximum, std::int64_t levels) {
    return torch::linspace(std::log(minimum), std::log(maximum), levels).exp();
}
void GaussianEncoder::update(bool force, std::shared_ptr<EntropyCoder> coder) {
    if (entropy_coder && !force) return;
    entropy_coder = std::move(coder); cdf_helper = std::make_shared<CdfHelper>();
    auto center = torch::zeros_like(scale_table) + 50;
    const auto scales = torch::zeros_like(center) + scale_table;
    const auto cdf = [&](const torch::Tensor& samples, const torch::Tensor& sigma) {
        return distribution == "laplace" ? dcvc2021::laplace_cdf(samples, sigma) : normal_cdf(samples, sigma);
    };
    for (int i = 50; i > 1; --i) {
        ProcessInterrupt::check();
        const auto samples = torch::zeros_like(center) + i;
        center = torch::where(cdf(samples, scales).squeeze() > torch::zeros_like(center) + 0.9999,
            torch::zeros_like(center) + i, center);
    }
    center = center.to(torch::kInt32);
    const auto sizes = 2 * center + 1;
    const auto maximum = sizes.max().item<std::int64_t>();
    const auto samples = (torch::arange(maximum, torch::TensorOptions().device(center.device())).unsqueeze(0)
        - center.unsqueeze(1)).to(torch::kFloat32);
    const auto sigmas = torch::zeros_like(samples) + scale_table.unsqueeze(1);
    const auto upper = cdf(samples + 0.5, sigmas), lower = cdf(samples - 0.5, sigmas);
    cdf_helper->set_cdf(-center, EntropyCoder::pmf_to_cdf(upper - lower, 2 * lower.slice(1, 0, 1), sizes, maximum), sizes + 2);
}
torch::Tensor GaussianEncoder::build_indexes(const torch::Tensor& scales) const {
    const auto bounded = torch::maximum(scales, torch::zeros_like(scales) + 1e-5);
    return ((bounded.log() - log_min_) / log_step_).clamp_(0, 255).to(torch::kInt32);
}
void GaussianEncoder::encode(const torch::Tensor& input, const torch::Tensor& scales) {
    encode_with_indexes(detail::integers(input), detail::integers(build_indexes(scales)));
}
void GaussianEncoder::encode_with_indexes(const std::vector<std::int32_t>& symbols, const std::vector<std::int32_t>& indexes) {
    const auto& helper = ready(entropy_coder, cdf_helper);
    entropy_coder->encode_with_indexes(symbols, indexes, helper.tables());
}
torch::Tensor GaussianEncoder::decode_stream(const torch::Tensor& scales) {
    const auto& helper = ready(entropy_coder, cdf_helper);
    return entropy_coder->decode_stream(detail::integers(build_indexes(scales)), helper.tables()).reshape(scales.sizes()).to(torch::kFloat32);
}
void GaussianEncoder::set_decoder_cdf() {
    throw std::logic_error("The HEM reference EntropyCoder does not implement set_decoder_cdf()");
}
}  // namespace uf::hem
