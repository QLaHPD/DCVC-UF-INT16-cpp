#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace uf {

// UF's single-lane rANS stream, using the exact integer CDFs in prepared
// models. The class order matches src/int16/model.py: encode Y partitions in
// reverse order, then Z; decode Z first, then Y partitions in forward order.
class EntropyEncoder {
public:
    void set_cdf(int table, const std::vector<std::int32_t>& flat_cdf,
                 const std::vector<std::int32_t>& lengths);
    void reset();
    void encode_y(const std::vector<std::int16_t>& packed_symbols);
    void encode_z(const std::vector<std::int8_t>& symbols, int cdf_offset, int channels);
    std::vector<std::uint8_t> flush();

private:
    struct Table {
        std::vector<std::vector<std::int32_t>> rows;
        std::vector<int> maximum;
    } tables_[2];
    std::uint32_t state_{1u << 23u};
    std::vector<std::uint8_t> emitted_;
};

class EntropyDecoder {
public:
    void set_cdf(int table, const std::vector<std::int32_t>& flat_cdf,
                 const std::vector<std::int32_t>& lengths);
    void set_stream(const std::vector<std::uint8_t>& stream);
    std::vector<std::int8_t> decode_z(std::size_t count, int cdf_offset, int channels);
    std::vector<std::int8_t> decode_y(const std::vector<std::uint8_t>& indexes);

private:
    struct Table {
        std::vector<std::vector<std::int32_t>> rows;
        std::vector<int> maximum;
    } tables_[2];
    std::vector<std::uint8_t> stream_;
    std::size_t offset_{};
    std::uint32_t state_{};
};

}  // namespace uf
