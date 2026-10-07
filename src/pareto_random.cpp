// Native adaptation of CPython 3.10.20 Random, hash-secret initialization and
// string sets used by UF Pareto/comparison and temporary names. Python headers are not used.
// Licenses/attribution: licenses/CPython-3.10.20.txt, licenses/MT19937.txt and
// licenses/SipHash.txt. See licenses/README.md for the adaptation summary.
#include "pareto_random.hpp"

#include <openssl/rand.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <time.h>
#include <unistd.h>

namespace uf::pareto_detail {
namespace {

std::uint64_t little_word(const unsigned char* bytes) {
    std::uint64_t result = 0;
    for (unsigned index = 0; index < 8; ++index) {
        result |= static_cast<std::uint64_t>(bytes[index]) << (index * 8);
    }
    return result;
}

std::array<unsigned char, 24> hash_secret() {
    std::array<unsigned char, 24> secret{};
    const char* environment = std::getenv("PYTHONHASHSEED");
    if (!environment || !*environment || std::string_view(environment) == "random") {
        if (RAND_bytes(secret.data(), static_cast<int>(secret.size())) != 1) {
            throw std::runtime_error("Cannot initialize Python string hash secret");
        }
        return secret;
    }
    errno = 0;
    char* end = nullptr;
    const auto value = std::strtoull(environment, &end, 10);
    if (errno == ERANGE || end == environment || *end || value > 0xffffffffull) {
        throw std::invalid_argument("PYTHONHASHSEED must be random or an integer from 0 to 4294967295");
    }
    if (value) {
        auto state = static_cast<std::uint32_t>(value);
        for (auto& byte : secret) {
            state = state * 214013u + 2531011u;
            byte = static_cast<unsigned char>((state >> 16) & 0xffu);
        }
    }
    // Seed zero disables randomization and keeps every secret byte zero.
    return secret;
}

const std::array<unsigned char, 24>& process_hash_secret() {
    static const auto secret = hash_secret();
    return secret;
}

std::uint64_t rotate(std::uint64_t value, unsigned count) {
    return (value << count) | (value >> (64 - count));
}

std::uint64_t string_hash(std::string_view text, const std::array<unsigned char, 24>& secret) {
    if (text.empty()) return 0;
    const auto k0 = little_word(secret.data());
    const auto k1 = little_word(secret.data() + 8);
    auto v0 = k0 ^ 0x736f6d6570736575ull;
    auto v1 = k1 ^ 0x646f72616e646f6dull;
    auto v2 = k0 ^ 0x6c7967656e657261ull;
    auto v3 = k1 ^ 0x7465646279746573ull;
    const auto round = [&]() {
        v0 += v1;
        v1 = rotate(v1, 13) ^ v0;
        v0 = rotate(v0, 32);
        v2 += v3;
        v3 = rotate(v3, 16) ^ v2;
        v0 += v3;
        v3 = rotate(v3, 21) ^ v0;
        v2 += v1;
        v1 = rotate(v1, 17) ^ v2;
        v2 = rotate(v2, 32);
    };
    const auto* bytes = reinterpret_cast<const unsigned char*>(text.data());
    std::size_t offset = 0;
    while (text.size() - offset >= 8) {
        const auto block = little_word(bytes + offset);
        v3 ^= block;
        round();
        round();
        v0 ^= block;
        offset += 8;
    }
    auto tail = static_cast<std::uint64_t>(text.size()) << 56;
    for (unsigned shift = 0; offset < text.size(); ++offset, shift += 8) {
        tail |= static_cast<std::uint64_t>(bytes[offset]) << shift;
    }
    v3 ^= tail;
    round();
    round();
    v0 ^= tail;
    v2 ^= 0xff;
    for (int index = 0; index < 4; ++index) round();
    const auto result = v0 ^ v1 ^ v2 ^ v3;
    // Python reserves signed hash -1 for errors; sets cast the result to
    // an unsigned word when probing, which preserves all other bit patterns.
    return result == std::numeric_limits<std::uint64_t>::max() ? result - 1 : result;
}

std::string unicode_hash_bytes(std::string_view text) {
    if (std::all_of(text.begin(), text.end(), [](unsigned char byte) { return byte < 128; })) return std::string(text);
    std::vector<std::uint32_t> characters;
    characters.reserve(text.size());
    std::uint32_t maximum = 0;
    for (std::size_t index = 0; index < text.size();) {
        const auto first = static_cast<unsigned char>(text[index++]);
        std::uint32_t character = first, minimum = 0;
        unsigned following = 0;
        if (first >= 0xc2 && first <= 0xdf) {
            character = first & 0x1f; minimum = 0x80; following = 1;
        } else if (first >= 0xe0 && first <= 0xef) {
            character = first & 0x0f; minimum = 0x800; following = 2;
        } else if (first >= 0xf0 && first <= 0xf4) {
            character = first & 0x07; minimum = 0x10000; following = 3;
        } else if (first >= 128) throw std::invalid_argument("Invalid UTF-8 string for Python set hashing");
        if (following > text.size() - index) throw std::invalid_argument("Truncated UTF-8 string for Python set hashing");
        for (unsigned part = 0; part < following; ++part) {
            const auto byte = static_cast<unsigned char>(text[index++]);
            if ((byte & 0xc0) != 0x80) throw std::invalid_argument("Invalid UTF-8 continuation for Python set hashing");
            character = (character << 6) | (byte & 0x3f);
        }
        if (character < minimum || character > 0x10ffff || (character >= 0xd800 && character <= 0xdfff))
            throw std::invalid_argument("Invalid Unicode character for Python set hashing");
        characters.push_back(character);
        maximum = std::max(maximum, character);
    }
    const unsigned width = maximum <= 0xff ? 1 : maximum <= 0xffff ? 2 : 4;
    const std::uint16_t endian_marker = 1;
    const bool little = *reinterpret_cast<const unsigned char*>(&endian_marker) == 1;
    std::string bytes;
    if (characters.size() > bytes.max_size() / width) throw std::length_error("Python string hash byte count overflow");
    bytes.reserve(characters.size() * width);
    // unicode_hash passes native-endian PyUnicode_DATA, without its terminator.
    for (const auto character : characters) for (unsigned part = 0; part < width; ++part) {
        const auto shift = (little ? part : width - part - 1) * 8;
        bytes.push_back(static_cast<char>((character >> shift) & 0xff));
    }
    return bytes;
}

struct StringEntry {
    std::size_t index = std::numeric_limits<std::size_t>::max();
    std::uint64_t hash = 0;
    bool empty() const { return index == std::numeric_limits<std::size_t>::max(); }
};

bool insert_string(std::vector<StringEntry>& table, std::size_t index, std::uint64_t hash,
                   const std::vector<std::string>& values) {
    const auto mask = table.size() - 1;
    auto perturb = hash;
    auto slot = static_cast<std::size_t>(hash) & mask;
    for (;;) {
        const auto probes = slot + 9 <= mask ? 9u : 0u;
        for (unsigned offset = 0; offset <= probes; ++offset) {
            auto& entry = table[slot + offset];
            if (entry.empty()) { entry = {index, hash}; return true; }
            if (entry.hash == hash && values[entry.index] == values[index]) return false;
        }
        perturb >>= 5;
        slot = (slot * 5 + 1 + perturb) & mask;
    }
}

struct Entry {
    int field{-1};
    std::uint64_t hash{};
};

void insert(std::array<Entry, 32>& table, std::size_t mask, int field, std::uint64_t hash) {
    auto perturb = hash;
    auto slot = static_cast<std::size_t>(hash) & mask;
    for (;;) {
        // CPython's set_add_entry/set_insert_clean check the initial slot
        // plus nine consecutive entries only if they all fit in the table.
        const auto probes = slot + 9 <= mask ? 9u : 0u;
        for (unsigned offset = 0; offset <= probes; ++offset) {
            auto& entry = table[slot + offset];
            if (entry.field == field) return;
            if (entry.field < 0) {
                entry = {field, hash};
                return;
            }
        }
        perturb >>= 5;
        slot = (slot * 5 + 1 + perturb) & mask;
    }
}

}  // namespace

std::vector<std::string> string_set_order(const std::vector<std::string>& values) {
    static_assert(sizeof(std::size_t) == 8, "Python string set order follows 64-bit CPython 3.10");
    const auto& secret = process_hash_secret();
    std::vector<StringEntry> table(8);
    std::size_t used = 0;
    for (std::size_t index = 0; index < values.size(); ++index) {
        const auto bytes = unicode_hash_bytes(values[index]);
        if (!insert_string(table, index, string_hash(bytes, secret), values)) continue;
        ++used;
        // There are no deletions/dummy entries, so fill equals used. A dict
        // keys view takes CPython's iterable branch and grows incrementally.
        if (used * 5 < (table.size() - 1) * 3) continue;
        const auto multiplier = used > 50000 ? 2u : 4u;
        if (used > table.max_size() / multiplier) throw std::length_error("Python string set table size overflow");
        const auto minimum = used * multiplier;
        std::size_t size = 8;
        while (size <= minimum) {
            if (size > table.max_size() / 2) throw std::length_error("Python string set table size overflow");
            size *= 2;
        }
        std::vector<StringEntry> larger(size);
        // set_insert_clean visits old slots in ascending order and uses the
        // same probes without equality callbacks, since all keys are distinct.
        for (const auto& entry : table) if (!entry.empty()) insert_string(larger, entry.index, entry.hash, values);
        table = std::move(larger);
    }
    std::vector<std::string> result;
    result.reserve(used);
    for (const auto& entry : table) if (!entry.empty()) result.push_back(values[entry.index]);
    return result;
}

PythonRandom::PythonRandom() {
    // CPython seeds Random(None) with 624 native UInt32 entropy words, then
    // init_by_array. Reuse the installed entropy provider used by this module.
    std::array<std::uint32_t, 624> entropy{};
    if (RAND_bytes(reinterpret_cast<unsigned char*>(entropy.data()),
                   static_cast<int>(sizeof(entropy))) == 1) {
        initialize(std::vector<std::uint32_t>(entropy.begin(), entropy.end()));
        return;
    }
    const auto ticks = [](clockid_t clock) {
        timespec now{};
        if (clock_gettime(clock, &now) != 0) {
            const int error = errno;
            throw std::system_error(error, std::generic_category(), "Cannot seed native Python Random");
        }
        // Unsigned arithmetic retains the low 64 clock bits without signed
        // overflow. Python splits the same nanosecond clock value into words.
        return static_cast<std::uint64_t>(now.tv_sec) * 1000000000ull +
               static_cast<std::uint64_t>(now.tv_nsec);
    };
    const auto realtime = ticks(CLOCK_REALTIME);
    const auto monotonic = ticks(CLOCK_MONOTONIC);
    initialize({static_cast<std::uint32_t>(realtime), static_cast<std::uint32_t>(realtime >> 32),
                static_cast<std::uint32_t>(getpid()), static_cast<std::uint32_t>(monotonic),
                static_cast<std::uint32_t>(monotonic >> 32)});
}

PythonRandom::PythonRandom(std::int64_t seed, std::uint64_t increment)
    : PythonRandom(PythonInteger(seed), increment) {}

PythonRandom::PythonRandom(const PythonInteger& seed, std::uint64_t increment) {
    initialize(seed.seed_words(increment));
}

void PythonRandom::initialize(const std::vector<std::uint32_t>& key) {
    state_[0] = 19650218u;
    for (std::size_t index = 1; index < state_.size(); ++index) {
        const auto prior = state_[index - 1];
        state_[index] = 1812433253u * (prior ^ (prior >> 30)) + static_cast<std::uint32_t>(index);
    }
    std::size_t index = 1, key_index = 0;
    for (auto remaining = std::max(state_.size(), key.size()); remaining; --remaining) {
        const auto prior = state_[index - 1];
        state_[index] = (state_[index] ^ ((prior ^ (prior >> 30)) * 1664525u)) +
                        key[key_index] + static_cast<std::uint32_t>(key_index);
        if (++index == state_.size()) { state_[0] = state_.back(); index = 1; }
        if (++key_index == key.size()) key_index = 0;
    }
    for (std::size_t remaining = state_.size() - 1; remaining; --remaining) {
        const auto prior = state_[index - 1];
        state_[index] = (state_[index] ^ ((prior ^ (prior >> 30)) * 1566083941u)) -
                        static_cast<std::uint32_t>(index);
        if (++index == state_.size()) { state_[0] = state_.back(); index = 1; }
    }
    state_[0] = 0x80000000u;
    index_ = state_.size();
    gaussian_.reset();
}

std::uint32_t PythonRandom::word() {
    if (index_ == state_.size()) {
        for (std::size_t index = 0; index < state_.size(); ++index) {
            const auto mixed = (state_[index] & 0x80000000u) |
                               (state_[(index + 1) % state_.size()] & 0x7fffffffu);
            state_[index] = state_[(index + 397) % state_.size()] ^ (mixed >> 1) ^
                            ((mixed & 1u) ? 0x9908b0dfu : 0u);
        }
        index_ = 0;
    }
    auto value = state_[index_++];
    value ^= value >> 11;
    value ^= (value << 7) & 0x9d2c5680u;
    value ^= (value << 15) & 0xefc60000u;
    value ^= value >> 18;
    return value;
}

std::uint64_t PythonRandom::bits(unsigned count) {
    if (!count) return 0;
    if (count <= 32) return word() >> (32 - count);
    if (count > 64) throw std::invalid_argument("Pareto random bit count exceeds 64");
    const auto low = word();
    const auto high = word() >> (64 - count);
    return static_cast<std::uint64_t>(low) | (static_cast<std::uint64_t>(high) << 32);
}

double PythonRandom::random() {
    const auto high = word() >> 5;
    const auto low = word() >> 6;
    return (high * 67108864.0 + low) * (1.0 / 9007199254740992.0);
}

std::uint64_t PythonRandom::below(std::uint64_t bound) {
    if (!bound) return 0;
    unsigned count = 0;
    for (auto value = bound; value; value >>= 1) ++count;
    // n.bit_length(), including the extra bit for powers of two and n=1.
    auto result = bits(count);
    while (result >= bound) result = bits(count);
    return result;
}

double PythonRandom::gauss(double mean, double sigma) {
    double value;
    if (gaussian_) {
        value = *gaussian_;
        gaussian_.reset();
    } else {
        const auto angle = random() * 0x1.921fb54442d18p+2;  // math.tau
        const auto radius = std::sqrt(-2.0 * std::log(1.0 - random()));
        value = std::cos(angle) * radius;
        gaussian_ = std::sin(angle) * radius;
    }
    // CPython caches the unscaled second variate; a later call may use a
    // different sigma because the parameter domains have different sizes.
    const volatile double scaled = value * sigma;
    return mean + scaled;
}

MutationOrder::MutationOrder() {
    static_assert(sizeof(std::size_t) == 8, "Pareto mutation order follows 64-bit CPython 3.10");
    const auto& secret = process_hash_secret();
    constexpr std::array<std::string_view, 5> names{
        "qp_i", "qp_p", "intra_period", "reset_interval", "skip_threshold"
    };
    for (std::size_t index = 0; index < names.size(); ++index) {
        hashes_[index] = string_hash(names[index], secret);
    }
}

std::vector<std::size_t> MutationOrder::indices(const std::array<bool, 5>& selected) const {
    std::array<Entry, 32> table{};
    std::size_t mask = 7, used = 0;
    for (std::size_t field = 0; field < selected.size(); ++field) {
        if (!selected[field]) continue;
        insert(table, mask, static_cast<int>(field), hashes_[field]);
        ++used;
        if (used * 5 >= mask * 3) {
            // Five distinct entries grow CPython's initial eight-entry set
            // to the smallest power of two greater than used*4: 32.
            std::array<Entry, 32> larger{};
            for (std::size_t slot = 0; slot <= mask; ++slot) {
                if (table[slot].field >= 0) {
                    insert(larger, 31, table[slot].field, table[slot].hash);
                }
            }
            table = larger;
            mask = 31;
        }
    }
    std::vector<std::size_t> result;
    result.reserve(used);
    for (std::size_t slot = 0; slot <= mask; ++slot) {
        if (table[slot].field >= 0) result.push_back(static_cast<std::size_t>(table[slot].field));
    }
    return result;
}

}  // namespace uf::pareto_detail
