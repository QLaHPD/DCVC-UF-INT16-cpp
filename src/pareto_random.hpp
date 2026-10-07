#pragma once

#include "uf/python_integer.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace uf::pareto_detail {

// CPython 3.10 set(iterable_of_strings) table order, including duplicate
// removal and incremental growth. Inputs are valid UTF-8; hashing uses
// Python's 1/2/4-byte Unicode representation and the shared process secret.
std::vector<std::string> string_set_order(const std::vector<std::string>& values);

// The Python reference uses CPython 3.10 Random with integer seeds. Standard
// C++ engines/distributions have different seeding and draw consumption.
class PythonRandom {
public:
    // Random(None): initialize from entropy, with time/PID fallback.
    PythonRandom();
    explicit PythonRandom(std::int64_t seed, std::uint64_t increment = 0);
    explicit PythonRandom(const PythonInteger& seed, std::uint64_t increment = 0);
    double random();
    std::uint64_t below(std::uint64_t bound);
    double gauss(double mean, double sigma);

    template <typename T>
    void shuffle(std::vector<T>& values) {
        for (auto remaining = values.size(); remaining > 1; --remaining) {
            const auto selected = static_cast<std::size_t>(below(remaining));
            std::swap(values[remaining - 1], values[selected]);
        }
    }

private:
    std::array<std::uint32_t, 624> state_{};
    std::size_t index_{624};
    std::optional<double> gaussian_;
    std::uint32_t word();
    std::uint64_t bits(unsigned count);
    void initialize(const std::vector<std::uint32_t>& key);
};

// CPython iterates the mutation-name set in hash-table order. Its hash secret
// is controlled by PYTHONHASHSEED, independently of the Pareto random seed.
class MutationOrder {
public:
    MutationOrder();
    std::vector<std::size_t> indices(const std::array<bool, 5>& selected) const;
private:
    std::array<std::uint64_t, 5> hashes_{};
};

}  // namespace uf::pareto_detail
