// Neural estimator: Copyright (c) Microsoft Corporation, MIT License.
// CDF/rANS interface adaptation: Copyright 2020 InterDigital Communications,
// Inc., Apache-2.0. See licenses/Apache-2.0.txt.
// 64-bit rANS arithmetic: Fabian "ryg" Giesen 2014, public domain/CC0.
// Reference: ryg-rans-legacy, revision c9d162d996fd600315af9ae8eb89d832576cb32d.
#include "uf/dcvc2021.hpp"
#include "uf/process.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace uf::dcvc2021 {
namespace {
constexpr std::uint64_t lower_bound = std::uint64_t(1) << 31;
constexpr std::uint32_t precision = 16, total_mass = 1u << precision;

struct Symbol {
    std::uint32_t start, frequency;
    bool bypass;
};

std::vector<std::int32_t> integer_values(const torch::Tensor& input) {
    const auto cpu = input.to(torch::kCPU, torch::kInt32).contiguous().view({-1});
    if (!cpu.numel()) return {};
    return {cpu.data_ptr<std::int32_t>(), cpu.data_ptr<std::int32_t>() + cpu.numel()};
}

torch::Tensor channel_indexes(std::int64_t channels, std::int64_t height, std::int64_t width) {
    return torch::arange(channels, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU))
        .view({1, channels, 1, 1}).repeat({1, 1, height, width});
}

const CdfTables& ready(const std::optional<CdfTables>& tables) {
    if (!tables) throw std::runtime_error("Uninitialized DCVC CDFs; call update() first");
    return *tables;
}

std::string compress_tensor(const torch::Tensor& input, const torch::Tensor& indexes,
                            const CdfTables& tables) {
    if (input.dim() != 4 || input.size(0) != 1 || input.sizes() != indexes.sizes()) {
        throw std::invalid_argument("DCVC entropy coding requires equal 4-D tensors with batch size one");
    }
    return rans_encode(integer_values(input.select(0, 0)), integer_values(indexes.select(0, 0)), tables);
}

torch::Tensor decompress_tensor(const std::string& encoded, const torch::Tensor& indexes,
                                const CdfTables& tables) {
    if (indexes.dim() != 4 || indexes.size(0) != 1) {
        throw std::invalid_argument("DCVC entropy decoding requires a 4-D tensor with batch size one");
    }
    RansDecoder decoder;
    decoder.set_stream(encoded);
    const auto values = decoder.decode(integer_values(indexes.select(0, 0)), tables);
    auto output = torch::empty(indexes.sizes(), torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt32));
    std::copy(values.begin(), values.end(), output.data_ptr<std::int32_t>());
    return output.to(torch::kFloat32);
}

CdfTables make_tables(const torch::Tensor& pmf, const torch::Tensor& tails,
                      const torch::Tensor& lengths, const torch::Tensor& offsets) {
    const auto probabilities = pmf.to(torch::kCPU, torch::kFloat32).contiguous();
    const auto tail = tails.to(torch::kCPU, torch::kFloat32).contiguous();
    const auto sizes = integer_values(lengths);
    CdfTables result;
    result.offsets = integer_values(offsets);
    for (std::size_t i = 0; i < sizes.size(); ++i) {
        ProcessInterrupt::check();
        const auto* start = probabilities.data_ptr<float>() + i * probabilities.size(1);
        std::vector<float> row(start, start + sizes[i]);
        row.push_back(tail.data_ptr<float>()[i]);
        result.rows.push_back(quantized_cdf(row));
    }
    result.validate();
    return result;
}
}  // namespace

void CdfTables::validate() const {
    if (rows.empty() || rows.size() != offsets.size()) throw std::invalid_argument("Invalid DCVC CDF table shape");
    for (const auto& row : rows) {
        if (row.size() < 3 || row.size() > total_mass + 1 || row.front() != 0 || row.back() != total_mass) {
            throw std::invalid_argument("Invalid DCVC CDF endpoints");
        }
        for (std::size_t i = 1; i < row.size(); ++i) {
            if (row[i] <= row[i - 1]) throw std::invalid_argument("DCVC CDF must increase strictly");
        }
    }
}

std::vector<std::int32_t> quantized_cdf(const std::vector<float>& pmf) {
    if (pmf.empty() || pmf.size() > total_mass) throw std::invalid_argument("Invalid DCVC PMF length");
    std::vector<std::uint32_t> cdf(pmf.size() + 1);
    for (std::size_t i = 0; i < pmf.size(); ++i) {
        if (!std::isfinite(pmf[i]) || pmf[i] < 0 || pmf[i] > 1) throw std::invalid_argument("Invalid DCVC PMF");
        // Keep the original ops.cpp round(), including its tie behavior.
        cdf[i + 1] = static_cast<std::uint32_t>(std::round(pmf[i] * total_mass) + 0.5);
    }
    const auto sum = std::accumulate(cdf.begin(), cdf.end(), std::uint32_t{});
    if (!sum) throw std::invalid_argument("DCVC PMF has zero mass");
    for (auto& value : cdf) value = static_cast<std::uint32_t>(std::uint64_t(total_mass) * value / sum);
    std::partial_sum(cdf.begin(), cdf.end(), cdf.begin());
    cdf.back() = total_mass;
    for (std::size_t i = 0; i + 1 < cdf.size(); ++i) {
        if (cdf[i] != cdf[i + 1]) continue;
        std::uint32_t best_frequency = std::numeric_limits<std::uint32_t>::max();
        std::size_t donor = cdf.size();
        for (std::size_t j = 0; j + 1 < cdf.size(); ++j) {
            const auto frequency = cdf[j + 1] - cdf[j];
            if (frequency > 1 && frequency < best_frequency) { best_frequency = frequency; donor = j; }
        }
        if (donor == cdf.size()) throw std::invalid_argument("DCVC PMF cannot assign nonzero frequencies");
        if (donor < i) for (std::size_t j = donor + 1; j <= i; ++j) --cdf[j];
        else for (std::size_t j = i + 1; j <= donor; ++j) ++cdf[j];
    }
    return {cdf.begin(), cdf.end()};
}

std::string rans_encode(const std::vector<std::int32_t>& symbols,
                        const std::vector<std::int32_t>& indexes, const CdfTables& tables) {
    tables.validate();
    if (symbols.size() != indexes.size()) throw std::invalid_argument("DCVC symbol/index size mismatch");
    std::vector<Symbol> operations;
    operations.reserve(symbols.size());
    for (std::size_t i = 0; i < symbols.size(); ++i) {
        if ((i & 65535) == 0) ProcessInterrupt::check();
        const auto index = indexes[i];
        if (index < 0 || static_cast<std::size_t>(index) >= tables.rows.size()) throw std::out_of_range("DCVC CDF index");
        const auto& cdf = tables.rows[index];
        const auto maximum = static_cast<std::int64_t>(cdf.size() - 2);
        auto value = std::int64_t(symbols[i]) - tables.offsets[index];
        std::uint64_t raw = 0;
        if (value < 0) { raw = std::uint64_t(-2 * value - 1); value = maximum; }
        else if (value >= maximum) { raw = std::uint64_t(2 * (value - maximum)); value = maximum; }
        if (raw > std::numeric_limits<std::uint32_t>::max()) throw std::out_of_range("DCVC bypass value exceeds 32 bits");
        operations.push_back({static_cast<std::uint32_t>(cdf[value]),
                               static_cast<std::uint32_t>(cdf[value + 1] - cdf[value]), false});
        if (value == maximum) {
            std::uint32_t count = 0;
            for (auto remaining = raw; remaining; remaining >>= 4) ++count;
            auto remainder = count;
            while (remainder >= 15) { operations.push_back({15, 0, true}); remainder -= 15; }
            operations.push_back({remainder, 0, true});
            for (std::uint32_t j = 0; j < count; ++j) operations.push_back({
                static_cast<std::uint32_t>((raw >> (j * 4)) & 15), 0, true});
        }
    }
    // Append emitted words, then reverse them after flushing the state.
    // This avoids the original binding's undersized reverse-write buffer.
    std::vector<std::uint32_t> words;
    std::uint64_t state = lower_bound;
    std::size_t consumed = 0;
    for (auto it = operations.rbegin(); it != operations.rend(); ++it) {
        if ((consumed++ & 65535) == 0) ProcessInterrupt::check();
        const auto frequency = it->bypass ? (1u << 12) : it->frequency;
        const auto threshold = ((lower_bound >> precision) << 32) * frequency;
        if (state >= threshold) { words.push_back(static_cast<std::uint32_t>(state)); state >>= 32; }
        state = it->bypass ? (state << 4) | it->start
            : ((state / frequency) << precision) + state % frequency + it->start;
    }
    words.push_back(static_cast<std::uint32_t>(state >> 32));
    words.push_back(static_cast<std::uint32_t>(state));
    std::string encoded;
    encoded.reserve(words.size() * 4);
    for (auto it = words.rbegin(); it != words.rend(); ++it) {
        for (unsigned shift = 0; shift < 32; shift += 8) encoded.push_back(static_cast<char>(*it >> shift));
    }
    return encoded;
}

std::uint32_t RansDecoder::word() {
    if (position_ > stream_.size() || stream_.size() - position_ < 4) throw std::runtime_error("Truncated DCVC rANS word");
    std::uint32_t result = 0;
    for (unsigned shift = 0; shift < 32; shift += 8) result |=
        std::uint32_t(static_cast<unsigned char>(stream_[position_++])) << shift;
    return result;
}

void RansDecoder::set_stream(std::string encoded) {
    initialized_ = false;
    stream_ = std::move(encoded);
    position_ = 0;
    const auto low = word(), high = word();
    state_ = std::uint64_t(low) | (std::uint64_t(high) << 32);
    if (state_ < lower_bound || state_ >= (std::uint64_t(1) << 63)) throw std::runtime_error("Invalid DCVC rANS state");
    initialized_ = true;
}

void RansDecoder::renormalize() {
    if (state_ < lower_bound) {
        state_ = (state_ << 32) | word();
        if (state_ < lower_bound) throw std::runtime_error("Invalid DCVC rANS normalization");
    }
}

std::uint32_t RansDecoder::bypass() {
    const auto value = static_cast<std::uint32_t>(state_ & 15);
    state_ >>= 4;
    renormalize();
    return value;
}

std::vector<std::int32_t> RansDecoder::decode(const std::vector<std::int32_t>& indexes,
                                             const CdfTables& tables) {
    if (!initialized_) throw std::runtime_error("DCVC rANS stream has not been set");
    tables.validate();
    std::vector<std::int32_t> result;
    result.reserve(indexes.size());
    for (std::size_t i = 0; i < indexes.size(); ++i) {
        if ((i & 65535) == 0) ProcessInterrupt::check();
        const auto index = indexes[i];
        if (index < 0 || static_cast<std::size_t>(index) >= tables.rows.size()) throw std::out_of_range("DCVC CDF index");
        const auto& cdf = tables.rows[index];
        const auto cumulative = static_cast<std::uint32_t>(state_ & (total_mass - 1));
        const auto upper = std::upper_bound(cdf.begin(), cdf.end(), cumulative);
        const auto symbol = static_cast<std::size_t>(upper - cdf.begin() - 1);
        const auto maximum = cdf.size() - 2;
        state_ = std::uint64_t(cdf[symbol + 1] - cdf[symbol]) * (state_ >> precision) +
                 cumulative - cdf[symbol];
        renormalize();
        std::int64_t value = symbol;
        if (symbol == maximum) {
            auto part = bypass(), count = part;
            while (part == 15) {
                part = bypass(); count += part;
                if (count > 8) throw std::runtime_error("DCVC bypass count exceeds 32 bits");
            }
            if (count > 8) throw std::runtime_error("DCVC bypass count exceeds 32 bits");
            std::uint32_t raw = 0;
            for (std::uint32_t j = 0; j < count; ++j) raw |= bypass() << (j * 4);
            value = (raw & 1) ? -std::int64_t(raw >> 1) - 1
                              : std::int64_t(raw >> 1) + static_cast<std::int64_t>(maximum);
        }
        value += tables.offsets[index];
        if (value < std::numeric_limits<std::int32_t>::min() || value > std::numeric_limits<std::int32_t>::max()) {
            throw std::runtime_error("Decoded DCVC symbol exceeds int32");
        }
        result.push_back(static_cast<std::int32_t>(value));
    }
    return result;
}

torch::Tensor laplace_cdf(const torch::Tensor& values, const torch::Tensor& scales) {
    if (torch::isnan(values).any().item<bool>() || !(scales > 0).all().item<bool>()) {
        throw std::invalid_argument("Invalid DCVC Laplace values or scales");
    }
    // Match the installed torch.distributions.Laplace.cdf expression.
    return 0.5 - 0.5 * values.sign() * torch::expm1(-values.abs() / scales);
}

BitparmImpl::BitparmImpl(std::int64_t channels, bool final) : final_(final) {
    h = register_parameter("h", torch::empty({1, channels, 1, 1}).normal_(0, 0.01));
    b = register_parameter("b", torch::empty({1, channels, 1, 1}).normal_(0, 0.01));
    if (!final_) a = register_parameter("a", torch::empty({1, channels, 1, 1}).normal_(0, 0.01));
}

torch::Tensor BitparmImpl::forward(const torch::Tensor& input) {
    const auto value = input * torch::softplus(h) + b;
    return final_ ? value.sigmoid() : value + value.tanh() * a.tanh();
}

BitEstimatorImpl::BitEstimatorImpl(std::int64_t channels) : channels_(channels) {
    f1 = register_module("f1", Bitparm(channels));
    f2 = register_module("f2", Bitparm(channels));
    f3 = register_module("f3", Bitparm(channels));
    f4 = register_module("f4", Bitparm(channels, true));
}

torch::Tensor BitEstimatorImpl::forward(const torch::Tensor& input) { return f4(f3(f2(f1(input)))); }
void BitEstimatorImpl::invalidate() { tables_.reset(); }

void BitEstimatorImpl::update(bool force) {
    if (tables_ && !force) return;
    torch::NoGradGuard no_grad;
    const auto medians = torch::zeros({channels_}, torch::TensorOptions().device(f1->h.device()));
    auto minima = medians + 50, maxima = medians + 50;
    for (int i = 50; i > 1; --i) {
        ProcessInterrupt::check();
        const auto samples = (torch::zeros_like(medians) - i).view({1, channels_, 1, 1});
        minima = torch::where(forward(samples).squeeze() < torch::zeros_like(medians) + 0.0001,
                             torch::zeros_like(medians) + i, minima);
    }
    for (int i = 50; i > 1; --i) {
        ProcessInterrupt::check();
        const auto samples = (torch::zeros_like(medians) + i).view({1, channels_, 1, 1});
        maxima = torch::where(forward(samples).squeeze() > torch::zeros_like(medians) + 0.9999,
                             torch::zeros_like(medians) + i, maxima);
    }
    minima = minima.to(torch::kInt32); maxima = maxima.to(torch::kInt32);
    const auto lengths = maxima + minima + 1;
    const auto maximum = lengths.max().item<std::int64_t>();
    const auto samples = torch::arange(maximum, torch::TensorOptions().device(medians.device()))
        .view({1, 1, 1, maximum}) + (medians - minima).view({1, channels_, 1, 1});
    const auto lower = forward(samples - 0.5).squeeze(0);
    const auto upper = forward(samples + 0.5).squeeze(0);
    // The Python tail uses the final column of the shared maximum-length
    // grid, even for a channel with a shorter PMF.
    tables_ = make_tables((upper - lower).select(1, 0),
        lower.select(1, 0).slice(1, 0, 1) + (1.0 - upper.select(1, 0).slice(1, maximum - 1)),
        lengths, -minima);
}

std::string BitEstimatorImpl::compress(const torch::Tensor& input) const {
    if (input.dim() != 4 || input.size(1) != channels_) throw std::invalid_argument("Invalid DCVC hyperlatent shape");
    return compress_tensor(input, channel_indexes(channels_, input.size(2), input.size(3)), ready(tables_));
}

torch::Tensor BitEstimatorImpl::decompress(const std::string& encoded, std::int64_t height, std::int64_t width) const {
    if (height <= 0 || width <= 0) throw std::invalid_argument("Invalid DCVC hyperlatent dimensions");
    return decompress_tensor(encoded, channel_indexes(channels_, height, width), ready(tables_));
}

LaplaceEncoder::LaplaceEncoder() : scale_table_(torch::linspace(std::log(0.01), std::log(16.0), 64).exp()) {}

void LaplaceEncoder::update(bool force) {
    if (tables_ && !force) return;
    torch::NoGradGuard no_grad;
    auto center = torch::zeros_like(scale_table_) + 50;
    for (int i = 50; i > 1; --i) {
        const auto samples = torch::zeros_like(center) + i;
        center = torch::where(laplace_cdf(samples, scale_table_).squeeze() > torch::zeros_like(center) + 0.9999,
                              torch::zeros_like(center) + i, center);
    }
    center = center.to(torch::kInt32);
    const auto lengths = 2 * center + 1;
    const auto maximum = lengths.max().item<std::int64_t>();
    const auto samples = (torch::arange(maximum).unsqueeze(0) - center.unsqueeze(1)).to(torch::kFloat32);
    const auto scales = torch::zeros_like(samples) + scale_table_.unsqueeze(1);
    const auto upper = laplace_cdf(samples + 0.5, scales);
    const auto lower = laplace_cdf(samples - 0.5, scales);
    tables_ = make_tables(upper - lower, 2 * lower.slice(1, 0, 1), lengths, -center);
}

torch::Tensor LaplaceEncoder::build_indexes(const torch::Tensor& scales) const {
    const auto bounded = torch::maximum(scales, torch::zeros_like(scales) + 1e-5);
    auto indexes = torch::full_like(scales, scale_table_.numel() - 1, torch::TensorOptions().dtype(torch::kInt32));
    for (std::int64_t i = 0; i + 1 < scale_table_.numel(); ++i) {
        indexes -= (bounded <= scale_table_.select(0, i)).to(torch::kInt32);
    }
    return indexes;
}

std::string LaplaceEncoder::compress(const torch::Tensor& symbols, const torch::Tensor& scales) const {
    return compress_tensor(symbols, build_indexes(scales), ready(tables_));
}
torch::Tensor LaplaceEncoder::decompress(const std::string& encoded, const torch::Tensor& scales) const {
    return decompress_tensor(encoded, build_indexes(scales), ready(tables_));
}
void LaplaceEncoder::set_stream(const std::string& encoded) { ready(tables_); decoder_.set_stream(encoded); }

torch::Tensor LaplaceEncoder::decode_stream(const torch::Tensor& scales) {
    const auto values = decoder_.decode(integer_values(build_indexes(scales)), ready(tables_));
    const std::vector<std::int64_t> shape{1, static_cast<std::int64_t>(values.size()), 1, 1};
    auto integers = torch::empty(shape, torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt32));
    std::copy(values.begin(), values.end(), integers.data_ptr<std::int32_t>());
    // Python Tensor(list) uses the default floating dtype here, unlike
    // full hyperlatent decompression, which explicitly finishes with .float().
    auto result = torch::empty(shape, torch::TensorOptions().device(torch::kCPU));
    result.copy_(integers);
    return result;
}
}  // namespace uf::dcvc2021
