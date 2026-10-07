// Derived from Microsoft's DCVC-UF py_rans/rans.cpp (MIT license).
#include "uf/entropy.hpp"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>

namespace uf {
namespace {

constexpr std::uint32_t scale_bits = 16;
constexpr std::uint32_t state_lower_bound = 1u << 23u;
constexpr std::uint32_t probability_mask = (1u << scale_bits) - 1u;
constexpr std::uint32_t bypass_bits = 2;
constexpr std::uint32_t max_bypass = (1u << bypass_bits) - 1u;

struct Cdf {
    std::vector<std::vector<std::int32_t>> rows;
    std::vector<int> maximum;
};

Cdf unpack_cdf(const std::vector<std::int32_t>& flat,
               const std::vector<std::int32_t>& lengths) {
    if (lengths.empty() || flat.size() % lengths.size()) {
        throw std::invalid_argument("UF CDF table shape is invalid");
    }
    const auto width = flat.size() / lengths.size();
    Cdf result;
    result.rows.reserve(lengths.size());
    result.maximum.reserve(lengths.size());
    for (std::size_t i = 0; i < lengths.size(); ++i) {
        const auto length = lengths[i];
        if (length < 2 || static_cast<std::size_t>(length) > width || length - 2 > 127) {
            throw std::invalid_argument("UF CDF row length is invalid");
        }
        std::vector<std::int32_t> row(flat.begin() + i * width,
                                      flat.begin() + i * width + length);
        if (row.front() != 0 || row.back() != (1 << scale_bits)) {
            throw std::invalid_argument("UF CDF does not span 16-bit probability space");
        }
        for (std::size_t j = 1; j < row.size(); ++j) {
            if (row[j] <= row[j - 1]) throw std::invalid_argument("UF CDF is not increasing");
        }
        result.rows.push_back(std::move(row));
        result.maximum.push_back(length - 2);
    }
    return result;
}

std::size_t z_cdf_base(std::size_t rows, int offset, int channels) {
    if (offset < 0 || channels <= 0) {
        throw std::invalid_argument("UF Z CDF range is invalid");
    }
    const auto base = static_cast<std::size_t>(offset);
    const auto count = static_cast<std::size_t>(channels);
    // Validate by subtraction: offset + channels can overflow a signed int.
    if (base > rows || count > rows - base) {
        throw std::invalid_argument("UF Z CDF range is invalid");
    }
    return base;
}

void put(std::uint32_t& state, std::vector<std::uint8_t>& emitted,
         std::uint32_t start, std::uint32_t frequency) {
    if (!frequency) throw std::runtime_error("Zero UF rANS frequency");
    const auto limit = frequency << (23u - scale_bits + 8u);
    while (state >= limit) {
        emitted.push_back(static_cast<std::uint8_t>(state));
        state >>= 8u;
    }
    state = ((state / frequency) << scale_bits) + (state % frequency) + start;
}

void put_bypass(std::uint32_t& state, std::vector<std::uint8_t>& emitted,
                std::uint32_t value) {
    if (value > max_bypass) throw std::invalid_argument("UF rANS bypass value exceeds two bits");
    constexpr std::uint32_t limit = (1u << (scale_bits - bypass_bits))
                                    << (23u - scale_bits + 8u);
    while (state >= limit) {
        emitted.push_back(static_cast<std::uint8_t>(state));
        state >>= 8u;
    }
    state = (state << bypass_bits) | value;
}

void encode_one(std::uint32_t& state, std::vector<std::uint8_t>& emitted,
                int symbol, const std::vector<std::int32_t>& cdf, int maximum) {
    int value = std::abs(symbol) * 2 - (symbol > 0);
    if (value >= maximum) {
        const std::uint32_t raw = value - maximum;
        value = maximum;
        std::vector<std::uint32_t> bins;
        int count = 0;
        while ((raw >> (count * bypass_bits)) != 0) ++count;
        int remainder = count;
        while (remainder >= static_cast<int>(max_bypass)) {
            bins.push_back(max_bypass);
            remainder -= max_bypass;
        }
        bins.push_back(remainder);
        for (int i = 0; i < count; ++i) bins.push_back((raw >> (i * bypass_bits)) & max_bypass);
        for (auto it = bins.rbegin(); it != bins.rend(); ++it) put_bypass(state, emitted, *it);
    }
    put(state, emitted, cdf[value], cdf[value + 1] - cdf[value]);
}

std::uint8_t next_byte(const std::vector<std::uint8_t>& stream, std::size_t& offset) {
    if (offset >= stream.size()) throw std::runtime_error("Truncated UF rANS stream");
    return stream[offset++];
}

std::uint32_t get_bypass(std::uint32_t& state,
                         const std::vector<std::uint8_t>& stream, std::size_t& offset) {
    const auto value = state & max_bypass;
    state >>= bypass_bits;
    if (state < state_lower_bound) state = (state << 8u) | next_byte(stream, offset);
    return value;
}

std::int8_t decode_one(std::uint32_t& state,
                       const std::vector<std::uint8_t>& stream, std::size_t& offset,
                       const std::vector<std::int32_t>& cdf, int maximum) {
    const auto cumulative = state & probability_mask;
    const auto upper = std::upper_bound(cdf.begin(), cdf.end(), cumulative);
    if (upper == cdf.begin() || upper == cdf.end()) throw std::runtime_error("Invalid UF rANS state");
    const auto index = static_cast<std::size_t>(upper - cdf.begin() - 1);
    const auto frequency = static_cast<std::uint32_t>(cdf[index + 1] - cdf[index]);
    state = frequency * (state >> scale_bits) + cumulative - cdf[index];
    while (state < state_lower_bound) state = (state << 8u) | next_byte(stream, offset);
    std::uint32_t value = index;
    if (value == static_cast<std::uint32_t>(maximum)) {
        std::uint32_t part = get_bypass(state, stream, offset);
        std::uint32_t count = part;
        while (part == max_bypass) {
            part = get_bypass(state, stream, offset);
            count += part;
            if (count > 16) throw std::runtime_error("UF rANS bypass symbol is too large");
        }
        std::uint32_t raw = 0;
        for (std::uint32_t i = 0; i < count; ++i) {
            raw |= get_bypass(state, stream, offset) << (i * bypass_bits);
        }
        // INT8 zigzag symbols occupy [0, 256]. Validate before adding the
        // escape index so an oversized raw value cannot wrap into that range.
        if (raw > 256u - static_cast<std::uint32_t>(maximum)) {
            throw std::runtime_error("Decoded UF rANS symbol exceeds INT8 range");
        }
        value = raw + static_cast<std::uint32_t>(maximum);
    }
    const auto signed_value = (value & 1u) ? static_cast<int>((value + 1u) / 2u)
                                            : -static_cast<int>((value + 1u) / 2u);
    if (signed_value < -128 || signed_value > 127) {
        throw std::runtime_error("Decoded UF rANS symbol exceeds INT8 range");
    }
    return static_cast<std::int8_t>(signed_value);
}

}  // namespace

void EntropyEncoder::set_cdf(int table, const std::vector<std::int32_t>& flat,
                             const std::vector<std::int32_t>& lengths) {
    if (table < 0 || table > 1) throw std::invalid_argument("UF CDF table index is invalid");
    auto unpacked = unpack_cdf(flat, lengths);
    tables_[table].rows = std::move(unpacked.rows);
    tables_[table].maximum = std::move(unpacked.maximum);
}

void EntropyEncoder::reset() {
    state_ = state_lower_bound;
    emitted_.clear();
}

void EntropyEncoder::encode_y(const std::vector<std::int16_t>& packed) {
    const auto& table = tables_[1];
    for (auto it = packed.rbegin(); it != packed.rend(); ++it) {
        const auto index = static_cast<std::uint16_t>(*it) & 0xffu;
        if (index >= table.rows.size()) throw std::runtime_error("UF Y CDF index out of range");
        const auto high = static_cast<std::uint16_t>(*it) >> 8u;
        const auto signed_high = static_cast<int>(high) - (high >= 128u ? 256 : 0);
        const auto symbol = static_cast<std::int8_t>(signed_high);
        encode_one(state_, emitted_, symbol, table.rows[index], table.maximum[index]);
    }
}

void EntropyEncoder::encode_z(const std::vector<std::int8_t>& symbols,
                              int cdf_offset, int channels) {
    const auto& table = tables_[0];
    const auto base = z_cdf_base(table.rows.size(), cdf_offset, channels);
    for (std::size_t i = symbols.size(); i-- > 0;) {
        const auto index = base + i % static_cast<std::size_t>(channels);
        encode_one(state_, emitted_, symbols[i], table.rows[index], table.maximum[index]);
    }
}

std::vector<std::uint8_t> EntropyEncoder::flush() {
    std::vector<std::uint8_t> result;
    if (emitted_.size() > result.max_size() - 4) {
        throw std::length_error("UF rANS stream exceeds vector size limits");
    }
    result.reserve(emitted_.size() + 4);
    for (int shift = 0; shift < 32; shift += 8) {
        result.push_back(static_cast<std::uint8_t>(state_ >> shift));
    }
    result.insert(result.end(), emitted_.rbegin(), emitted_.rend());
    return result;
}

void EntropyDecoder::set_cdf(int table, const std::vector<std::int32_t>& flat,
                             const std::vector<std::int32_t>& lengths) {
    if (table < 0 || table > 1) throw std::invalid_argument("UF CDF table index is invalid");
    auto unpacked = unpack_cdf(flat, lengths);
    tables_[table].rows = std::move(unpacked.rows);
    tables_[table].maximum = std::move(unpacked.maximum);
}

void EntropyDecoder::set_stream(const std::vector<std::uint8_t>& stream) {
    if (stream.size() < 4) throw std::invalid_argument("UF rANS stream is too short");
    stream_ = stream;
    state_ = 0;
    offset_ = 0;
    for (int shift = 0; shift < 32; shift += 8) {
        state_ |= static_cast<std::uint32_t>(next_byte(stream_, offset_)) << shift;
    }
}

std::vector<std::int8_t> EntropyDecoder::decode_z(std::size_t count,
                                                   int cdf_offset, int channels) {
    const auto& table = tables_[0];
    const auto base = z_cdf_base(table.rows.size(), cdf_offset, channels);
    std::vector<std::int8_t> result(count);
    for (std::size_t i = 0; i < count; ++i) {
        const auto index = base + i % static_cast<std::size_t>(channels);
        result[i] = decode_one(state_, stream_, offset_, table.rows[index], table.maximum[index]);
    }
    return result;
}

std::vector<std::int8_t> EntropyDecoder::decode_y(const std::vector<std::uint8_t>& indexes) {
    const auto& table = tables_[1];
    std::vector<std::int8_t> result(indexes.size());
    for (std::size_t i = 0; i < indexes.size(); ++i) {
        if (indexes[i] >= table.rows.size()) throw std::runtime_error("UF Y CDF index out of range");
        result[i] = decode_one(state_, stream_, offset_, table.rows[indexes[i]],
                               table.maximum[indexes[i]]);
    }
    return result;
}

}  // namespace uf
