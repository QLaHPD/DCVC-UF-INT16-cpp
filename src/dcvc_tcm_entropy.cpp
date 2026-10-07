// Neural estimators: Microsoft Corporation, MIT.
// Buffered rANS adaptation: InterDigital Communications, Apache-2.0.
// See licenses/Apache-2.0.txt and licenses/RygRans.txt.
#include "dcvc_tcm_internal.hpp"
#include "uf/process.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace uf::tcm::detail {
std::vector<std::int32_t> integers(const torch::Tensor& input) {
    const auto cpu = input.to(torch::kCPU, torch::kInt32).contiguous().view({-1});
    if (!cpu.numel()) return {};
    return {cpu.data_ptr<std::int32_t>(), cpu.data_ptr<std::int32_t>() + cpu.numel()};
}
torch::Tensor floating(const std::vector<std::int32_t>& input) {
    auto integers = torch::empty({static_cast<std::int64_t>(input.size())},
        torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt32));
    if (!input.empty()) std::copy(input.begin(), input.end(), integers.data_ptr<std::int32_t>());
    auto result = torch::empty(integers.sizes(), torch::TensorOptions().device(torch::kCPU));
    result.copy_(integers);
    return result;
}
CdfTables make_tables(const torch::Tensor& pmf, const torch::Tensor& tails,
                      const torch::Tensor& lengths, const torch::Tensor& offsets) {
    const auto probabilities = pmf.to(torch::kCPU, torch::kFloat32).contiguous();
    const auto tail = tails.to(torch::kCPU, torch::kFloat32).contiguous().view({-1});
    const auto counts = integers(lengths);
    CdfTables result;
    result.offsets = integers(offsets);
    if (probabilities.dim() != 2 || probabilities.size(0) != static_cast<std::int64_t>(counts.size()) ||
        tail.numel() != static_cast<std::int64_t>(counts.size())) throw std::invalid_argument("Invalid TCM PMF dimensions");
    for (std::size_t i = 0; i < counts.size(); ++i) {
        ProcessInterrupt::check();
        if (counts[i] <= 0 || counts[i] > probabilities.size(1)) throw std::invalid_argument("Invalid TCM PMF length");
        const auto* begin = probabilities.data_ptr<float>() + i * probabilities.size(1);
        std::vector<float> row(begin, begin + counts[i]);
        row.push_back(tail.data_ptr<float>()[i]);
        result.rows.push_back(dcvc2021::quantized_cdf(row));
    }
    result.validate();
    return result;
}
}  // namespace uf::tcm::detail

namespace uf::tcm {
namespace {
const CdfTables& ready(const std::shared_ptr<EntropyCoder>& coder, const std::optional<CdfTables>& tables) {
    if (!coder || !tables) throw std::runtime_error("TCM entropy model must be updated before coding");
    return *tables;
}
}  // namespace
EntropyCoder::EntropyCoder(int precision) {
    if (precision != 16) throw std::invalid_argument("TCM's rANS backend requires precision 16");
}
void EntropyCoder::reset_encoder() { symbols_.clear(); indexes_.clear(); tables_.rows.clear(); tables_.offsets.clear(); }
void EntropyCoder::encode_with_indexes(const std::vector<std::int32_t>& symbols,
    const std::vector<std::int32_t>& indexes, const CdfTables& tables) {
    tables.validate();
    if (symbols.size() != indexes.size()) throw std::invalid_argument("TCM entropy symbol/index counts differ");
    const auto limit = static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max());
    if (tables_.rows.size() > limit || tables.rows.size() > limit - tables_.rows.size())
        throw std::overflow_error("TCM combined CDF table exceeds int32");
    for (const auto index : indexes)
        if (index < 0 || static_cast<std::size_t>(index) >= tables.rows.size()) throw std::out_of_range("TCM CDF index");
    const auto base = static_cast<std::int32_t>(tables_.rows.size());
    tables_.rows.insert(tables_.rows.end(), tables.rows.begin(), tables.rows.end());
    tables_.offsets.insert(tables_.offsets.end(), tables.offsets.begin(), tables.offsets.end());
    symbols_.insert(symbols_.end(), symbols.begin(), symbols.end());
    for (const auto index : indexes) indexes_.push_back(base + index);
}
std::string EntropyCoder::flush_encoder() {
    // A zero-symbol flush still has an rANS state; the dummy CDF is unused.
    if (tables_.rows.empty()) { tables_.rows = {{0, 1, 65536}}; tables_.offsets = {0}; }
    const auto stream = dcvc2021::rans_encode(symbols_, indexes_, tables_);
    reset_encoder();
    return stream;
}
void EntropyCoder::set_stream(const std::string& stream) { decoder_.set_stream(stream); }
void EntropyCoder::set_decoder_cdf(CdfTables tables) { tables.validate(); decoder_tables_ = std::move(tables); }
std::vector<std::int32_t> EntropyCoder::decode_stream(const std::vector<std::int32_t>& indexes, const CdfTables& tables) {
    return decoder_.decode(indexes, tables);
}
std::vector<std::int32_t> EntropyCoder::decode_stream_only_indexes(const std::vector<std::int32_t>& indexes) {
    if (!decoder_tables_) throw std::runtime_error("TCM decoder CDF has not been set");
    return decoder_.decode(indexes, *decoder_tables_);
}
BitEstimatorImpl::BitEstimatorImpl(std::int64_t channels) : channels_(channels) {
    f1 = register_module("f1", Bitparm(channels));
    f2 = register_module("f2", Bitparm(channels));
    f3 = register_module("f3", Bitparm(channels));
    f4 = register_module("f4", Bitparm(channels, true));
}
torch::Tensor BitEstimatorImpl::forward(const torch::Tensor& input) { return f4(f3(f2(f1(input)))); }
void BitEstimatorImpl::update(bool force, std::shared_ptr<EntropyCoder> coder) {
    if (coder_ && !force) return;
    coder_ = std::move(coder);
    if (!coder_) throw std::invalid_argument("TCM BitEstimator update needs an entropy coder");
    torch::NoGradGuard no_grad;
    const auto medians = torch::zeros({channels_}, torch::TensorOptions().device(f1->h.device()));
    auto minima = medians + 50, maxima = medians + 50;
    for (int i = 50; i > 1; --i) {
        ProcessInterrupt::check();
        minima = torch::where(forward((torch::zeros_like(medians) - i).view({1, channels_, 1, 1})).squeeze() <
            torch::zeros_like(medians) + 0.0001, torch::zeros_like(medians) + i, minima);
    }
    for (int i = 50; i > 1; --i) {
        ProcessInterrupt::check();
        maxima = torch::where(forward((torch::zeros_like(medians) + i).view({1, channels_, 1, 1})).squeeze() >
            torch::zeros_like(medians) + 0.9999, torch::zeros_like(medians) + i, maxima);
    }
    minima = minima.to(torch::kInt32); maxima = maxima.to(torch::kInt32);
    const auto lengths = maxima + minima + 1;
    const auto maximum = lengths.max().item<std::int64_t>();
    const auto samples = torch::arange(maximum, torch::TensorOptions().device(medians.device()))
        .view({1, 1, 1, maximum}) + (medians - minima).view({1, channels_, 1, 1});
    const auto lower = forward(samples - 0.5).squeeze(0), upper = forward(samples + 0.5).squeeze(0);
    tables_ = detail::make_tables((upper - lower).select(1, 0),
        lower.select(1, 0).slice(1, 0, 1) + (1.0 - upper.select(1, 0).slice(1, maximum - 1)),
        lengths, -minima);
}
torch::Tensor BitEstimatorImpl::build_indexes(const std::vector<std::int64_t>& size) {
    if (size.size() != 4) throw std::invalid_argument("TCM hyperlatent must be NCHW");
    return torch::arange(size[1], torch::TensorOptions().device(torch::kCPU)).to(torch::kInt32)
        .view({1, size[1], 1, 1}).repeat({size[0], 1, size[2], size[3]});
}
void BitEstimatorImpl::encode(const torch::Tensor& input) {
    const auto& tables = ready(coder_, tables_);
    if (input.dim() != 4 || input.size(0) != 1 || input.size(1) != channels_)
        throw std::invalid_argument("TCM hyperlatent coding requires one NCHW sample");
    const auto indexes = build_indexes(input.sizes().vec()).select(0, 0);
    coder_->encode_with_indexes(detail::integers(input), detail::integers(indexes), tables);
}
torch::Tensor BitEstimatorImpl::decode_stream(std::pair<std::int64_t, std::int64_t> size) {
    const auto& tables = ready(coder_, tables_);
    const std::vector<std::int64_t> shape{1, channels_, size.first, size.second};
    const auto indexes = build_indexes(shape);
    return detail::floating(coder_->decode_stream(detail::integers(indexes), tables)).view(shape).to(torch::kFloat32);
}
GaussianEncoder::GaussianEncoder(std::string distribution)
    : distribution_(std::move(distribution)), scale_table_(torch::linspace(std::log(0.01), std::log(64.0), 256).exp()),
      log_min_(std::log(0.01)), log_step_((std::log(64.0) - log_min_) / 255) {
    if (distribution_ != "laplace" && distribution_ != "gaussian") throw std::invalid_argument("Invalid TCM distribution");
}
void GaussianEncoder::update(bool force, std::shared_ptr<EntropyCoder> coder) {
    if (coder_ && !force) return;
    coder_ = std::move(coder);
    if (!coder_) throw std::invalid_argument("TCM GaussianEncoder update needs an entropy coder");
    torch::NoGradGuard no_grad;
    auto center = torch::zeros_like(scale_table_) + 50;
    const auto scales = torch::zeros_like(center) + scale_table_, mu = torch::zeros_like(scales);
    for (int i = 50; i > 1; --i) {
        const auto samples = torch::zeros_like(center) + i;
        const auto cdf = distribution_ == "laplace" ? dcvc2021::laplace_cdf(samples, scales) :
            0.5 * (1 + torch::erf((samples - mu) * scales.reciprocal() / std::sqrt(2.0)));
        center = torch::where(cdf.squeeze() > torch::zeros_like(center) + 0.9999, torch::zeros_like(center) + i, center);
    }
    center = center.to(torch::kInt32);
    const auto lengths = center * 2 + 1;
    const auto maximum = lengths.max().item<std::int64_t>();
    const auto samples = (torch::arange(maximum).unsqueeze(0) - center.unsqueeze(1)).to(torch::kFloat32);
    const auto pmf_scales = torch::zeros_like(samples) + scale_table_.unsqueeze(1);
    const auto lower = dcvc2021::laplace_cdf(samples - 0.5, pmf_scales);
    const auto upper = dcvc2021::laplace_cdf(samples + 0.5, pmf_scales);
    tables_ = detail::make_tables(upper - lower, 2 * lower.slice(1, 0, 1), lengths, -center);
}
torch::Tensor GaussianEncoder::build_indexes(const torch::Tensor& scales) const {
    const auto bounded = torch::maximum(scales, torch::zeros_like(scales) + 1e-5);
    auto indexes = (torch::log(bounded) - log_min_) / log_step_;
    return indexes.clamp_(0, 255).to(torch::kInt32);
}
void GaussianEncoder::encode(const torch::Tensor& input, const torch::Tensor& scales) {
    encode_with_indexes(detail::integers(input), detail::integers(build_indexes(scales)));
}
torch::Tensor GaussianEncoder::decode_stream(const torch::Tensor& scales) {
    const auto& tables = ready(coder_, tables_);
    return detail::floating(coder_->decode_stream(detail::integers(build_indexes(scales)), tables)).view(scales.sizes());
}
void GaussianEncoder::set_decoder_cdf() {
    const auto& tables = ready(coder_, tables_); coder_->set_decoder_cdf(tables);
}
void GaussianEncoder::encode_with_indexes(const std::vector<std::int32_t>& symbols, const std::vector<std::int32_t>& indexes) {
    const auto& tables = ready(coder_, tables_); coder_->encode_with_indexes(symbols, indexes, tables);
}
std::vector<std::int32_t> GaussianEncoder::decode_stream_only_indexes(const std::vector<std::int32_t>& indexes) {
    ready(coder_, tables_); return coder_->decode_stream_only_indexes(indexes);
}
}  // namespace uf::tcm
