// FM byte-rANS adaptation. InterDigital, Apache-2.0; Fabian Giesen, public
// domain; Microsoft, MIT. See licenses/ for retained notices.
#include "dcvc_fm_rans.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace uf::fm::detail {
namespace {
constexpr std::uint32_t state_lower_bound = 1u << 23;
std::int16_t short_word(std::int64_t value) {
    const auto word = static_cast<std::uint16_t>(value);
    return word < 32768 ? static_cast<std::int16_t>(word)
                       : static_cast<std::int16_t>(static_cast<std::int32_t>(word) - 65536);
}
void buffer(std::vector<RansSymbol>& output, const std::vector<std::int16_t>& symbols,
            const std::vector<std::int16_t>& indexes, const CdfTables& tables) {
    for (std::size_t i = 0; i < symbols.size(); ++i) {
        const auto index = indexes[i];
        if (index < 0) continue;
        const auto& cdf = tables.rows.at(index);
        const auto maximum = static_cast<std::int64_t>(cdf.size()) - 2;
        auto value = static_cast<std::int64_t>(symbols[i]) - tables.offsets.at(index);
        std::uint64_t raw = 0;
        if (value < 0) { raw = static_cast<std::uint64_t>(-2 * value - 1); value = maximum; }
        else if (value >= maximum) { raw = static_cast<std::uint64_t>(2 * (value - maximum)); value = maximum; }
        // The original int32 mapping has undefined behavior outside this range.
        if (raw > std::numeric_limits<std::int32_t>::max())
            throw std::overflow_error("FM bypass symbol exceeds the reference int32 range");
        output.push_back({static_cast<std::uint16_t>(cdf.at(value)),
                          static_cast<std::uint16_t>(cdf.at(value + 1) - cdf.at(value))});
        if (value == maximum) {
            int count = 0;
            for (auto remainder = raw; remainder; remainder >>= 2) ++count;
            auto remaining = count;
            while (remaining >= 3) { output.push_back({3, 0}); remaining -= 3; }
            output.push_back({static_cast<std::uint16_t>(remaining), 0});
            for (int j = 0; j < count; ++j)
                output.push_back({static_cast<std::uint16_t>((raw >> (2 * j)) & 3), 0});
        }
    }
}
std::string encode(const std::vector<RansSymbol>& symbols) {
    std::uint32_t state = state_lower_bound;
    std::string emitted;
    for (auto it = symbols.rbegin(); it != symbols.rend(); ++it) {
        const auto frequency = it->range ? static_cast<std::uint32_t>(it->range) : (1u << 14);
        const auto maximum = frequency << 15;
        while (state >= maximum) { emitted.push_back(static_cast<char>(state)); state >>= 8; }
        if (it->range) state = ((state / frequency) << 16) + (state % frequency) + it->start;
        else state = (state << 2) | it->start;
    }
    std::string result;
    result.reserve(4 + emitted.size());
    for (int i = 0; i < 4; ++i) result.push_back(static_cast<char>(state >> (8 * i)));
    result.append(emitted.rbegin(), emitted.rend());
    return result;
}
}
EncoderLane::EncoderLane(bool threaded) : threaded_(threaded) {
    if (threaded_) thread_ = std::thread(&EncoderLane::worker, this);
}
EncoderLane::~EncoderLane() {
    if (!threaded_) return;
    { std::lock_guard<std::mutex> lock(mutex_); stopping_ = true; }
    changed_.notify_all();
    thread_.join();
}
void EncoderLane::submit(std::function<void()> work) {
    if (!threaded_) { work(); return; }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (error_) std::rethrow_exception(error_);
        pending_.push_back(std::move(work));
    }
    changed_.notify_all();
}
void EncoderLane::worker() {
    std::unique_lock<std::mutex> lock(mutex_);
    for (;;) {
        changed_.wait(lock, [this] { return stopping_ || !pending_.empty(); });
        if (pending_.empty() && stopping_) return;
        auto work = std::move(pending_.front()); pending_.pop_front(); busy_ = true;
        lock.unlock();
        std::exception_ptr error;
        try { work(); } catch (...) { error = std::current_exception(); }
        lock.lock();
        if (error) { error_ = error; pending_.clear(); }
        busy_ = false;
        changed_.notify_all();
    }
}
void EncoderLane::wait_idle() {
    if (!threaded_) return;
    std::unique_lock<std::mutex> lock(mutex_);
    changed_.wait(lock, [this] { return pending_.empty() && !busy_; });
    if (error_) std::rethrow_exception(error_);
}
void EncoderLane::encode(std::vector<std::int16_t> symbols, std::vector<std::int16_t> indexes,
                         std::shared_ptr<const CdfTables> tables) {
    submit([this, symbols = std::move(symbols), indexes = std::move(indexes), tables = std::move(tables)] {
        buffer(symbols_, symbols, indexes, *tables);
    });
}
void EncoderLane::flush() {
    submit([this] { stream_ = detail::encode(symbols_); });
    flushed_ = true;
}
void EncoderLane::reset() {
    wait_idle(); symbols_.clear(); flushed_ = false;  // Encoded bytes and CDF groups survive reset.
}
std::string EncoderLane::get_stream() {
    if (threaded_ && !flushed_) throw std::logic_error("FM threaded entropy stream requested before flush");
    wait_idle(); return stream_;
}
std::uint8_t DecoderLane::byte() {
    if (position_ >= stream_.size()) throw std::runtime_error("Truncated FM byte-rANS payload");
    return static_cast<std::uint8_t>(stream_[position_++]);
}
void DecoderLane::set_stream(std::string bytes) {
    stream_ = std::move(bytes); position_ = 0; initialized_ = false;
    state_ = 0;
    for (int i = 0; i < 4; ++i) state_ |= static_cast<std::uint32_t>(byte()) << (8 * i);
    if (state_ < state_lower_bound) throw std::runtime_error("Invalid FM byte-rANS state");
    initialized_ = true;
}
std::uint32_t DecoderLane::bits() {
    const auto result = state_ & 3; state_ >>= 2;
    if (state_ < state_lower_bound) {
        state_ = (state_ << 8) | byte();
        if (state_ < state_lower_bound) throw std::runtime_error("Invalid FM bypass state");
    }
    return result;
}
std::vector<std::int16_t> DecoderLane::decode(const std::vector<std::int16_t>& indexes, const CdfTables& tables) {
    if (!initialized_) throw std::logic_error("FM decoder has no stream");
    std::vector<std::int16_t> result(indexes.size());
    for (std::size_t i = 0; i < indexes.size(); ++i) {
        const auto index = indexes[i];
        if (index < 0) { result[i] = 0; continue; }
        const auto& cdf = tables.rows.at(index);
        const auto cumulative = state_ & 65535;
        const auto found = std::upper_bound(cdf.begin(), cdf.end(), static_cast<std::int32_t>(cumulative));
        const auto symbol = static_cast<std::size_t>(found - cdf.begin() - 1);
        state_ = (cdf.at(symbol + 1) - cdf.at(symbol)) * (state_ >> 16) + cumulative - cdf.at(symbol);
        while (state_ < state_lower_bound) state_ = (state_ << 8) | byte();
        const auto maximum = static_cast<std::int64_t>(cdf.size()) - 2;
        auto value = static_cast<std::int64_t>(symbol);
        if (value == maximum) {
            auto part = bits(); auto count = part;
            while (part == 3) { part = bits(); count += part; if (count > 16) throw std::runtime_error("Invalid FM bypass count"); }
            if (count > 16) throw std::runtime_error("Invalid FM bypass count");
            std::uint32_t raw = 0;
            for (std::uint32_t j = 0; j < count; ++j) raw |= bits() << (2 * j);
            if (raw > std::numeric_limits<std::int32_t>::max()) throw std::runtime_error("FM bypass value exceeds int32");
            value = raw & 1 ? -static_cast<std::int64_t>(raw >> 1) - 1 : static_cast<std::int64_t>(raw >> 1) + maximum;
        }
        result[i] = short_word(value + tables.offsets.at(index));
    }
    return result;
}
std::vector<std::int32_t> quantized_cdf(const std::vector<float>& pmf, int precision) {
    if (precision < 1 || precision > 30 || pmf.empty()) throw std::invalid_argument("Invalid FM PMF precision/size");
    const auto target = 1u << precision;
    if (pmf.size() > target) throw std::invalid_argument("FM PMF has more symbols than available frequencies");
    std::vector<std::uint32_t> cdf(pmf.size() + 1);
    std::uint64_t total = 0;
    for (std::size_t i = 0; i < pmf.size(); ++i) {
        if (!std::isfinite(pmf[i]) || pmf[i] < 0) throw std::invalid_argument("Invalid FM PMF probability");
        const auto frequency = std::round(pmf[i] * static_cast<float>(target)) + 0.5;
        if (frequency > std::numeric_limits<std::int32_t>::max()) throw std::overflow_error("FM PMF frequency exceeds int32");
        cdf[i + 1] = static_cast<std::uint32_t>(frequency); total += cdf[i + 1];
    }
    if (total == 0 || total > std::numeric_limits<std::int32_t>::max()) throw std::invalid_argument("Invalid FM PMF total");
    for (auto& frequency : cdf) frequency = static_cast<std::uint32_t>((static_cast<std::uint64_t>(target) * frequency) / total);
    std::partial_sum(cdf.begin(), cdf.end(), cdf.begin()); cdf.back() = target;
    for (std::size_t i = 0; i + 1 < cdf.size(); ++i) {
        if (cdf[i] != cdf[i + 1]) continue;
        std::uint32_t best = std::numeric_limits<std::uint32_t>::max();
        std::size_t donor = cdf.size();
        for (std::size_t j = 0; j + 1 < cdf.size(); ++j) {
            const auto frequency = cdf[j + 1] - cdf[j];
            if (frequency > 1 && frequency < best) { best = frequency; donor = j; }
        }
        if (donor == cdf.size()) throw std::runtime_error("FM PMF cannot supply a nonzero frequency");
        if (donor < i) for (auto j = donor + 1; j <= i; ++j) --cdf[j];
        else for (auto j = i + 1; j <= donor; ++j) ++cdf[j];
    }
    return {cdf.begin(), cdf.end()};
}
}  // namespace uf::fm::detail
