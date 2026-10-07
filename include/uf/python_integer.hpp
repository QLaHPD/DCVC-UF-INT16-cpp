#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace uf {

// An unbounded signed decimal integer for Python's integer-seeded Random and
// comparison arithmetic. This supplies the operations needed by the port,
// rather than exposing a general-purpose arithmetic library. UTF-8 input accepts
// Unicode decimal digits/spaces, a sign and underscores between digits, and
// follows the reference interpreter's decimal digit-limit policy.
class PythonInteger {
public:
    PythonInteger() = default;
    PythonInteger(std::int64_t value);
    explicit PythonInteger(std::string_view value);

    static PythonInteger from_unsigned(std::uint64_t value);
    PythonInteger& operator+=(const PythonInteger& other);
    PythonInteger operator*(const PythonInteger& other) const;
    // Like PyLong_AsDouble: round to nearest, ties to even, and fail on overflow.
    double as_double() const;

    const std::string& decimal() const noexcept { return decimal_; }

    // CPython seeds with little-endian 32-bit words of abs(seed + increment).
    // Add before taking the absolute value, including for negative seeds that
    // cross zero when a resumed search adds its completed-trial offset.
    std::vector<std::uint32_t> seed_words(std::uint64_t increment = 0) const;

private:
    std::string decimal_{"0"};
};

}  // namespace uf
