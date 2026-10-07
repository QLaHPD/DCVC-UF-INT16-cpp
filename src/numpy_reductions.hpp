// Pairwise summation adapted from NumPy 1.26.1 loops_utils.h.src (BSD-3-Clause).
// Retained notice: licenses/NumPy.txt. Reference: numpy-reference/dcvc-dc.
#pragma once
#include <cstdint>

namespace uf::numpy_detail {
template<class Number> Number pairwise_sum(const Number* values, std::int64_t count) {
    if (count < 8) {
        Number result = -0.0;
        for (std::int64_t index = 0; index < count; ++index) result += values[index];
        return result;
    }
    if (count <= 128) {
        Number partial[8];
        for (int index = 0; index < 8; ++index) partial[index] = values[index];
        std::int64_t index = 8;
        for (; index < count - count % 8; index += 8)
            for (int lane = 0; lane < 8; ++lane) partial[lane] += values[index + lane];
        Number result = ((partial[0] + partial[1]) + (partial[2] + partial[3])) +
                        ((partial[4] + partial[5]) + (partial[6] + partial[7]));
        for (; index < count; ++index) result += values[index];
        return result;
    }
    auto split = count / 2;
    split -= split % 8;
    return pairwise_sum(values, split) + pairwise_sum(values + split, count - split);
}
}  // namespace uf::numpy_detail
