// FM byte-rANS adaptation. InterDigital, Apache-2.0; Fabian Giesen, public
// domain; Microsoft, MIT. See licenses/ for retained notices.
#pragma once
#include "uf/dcvc_fm.hpp"
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>

namespace uf::fm::detail {
struct RansSymbol { std::uint16_t start, range; };
class EncoderLane {
public:
    explicit EncoderLane(bool threaded);
    ~EncoderLane();
    void encode(std::vector<std::int16_t> symbols, std::vector<std::int16_t> indexes,
                std::shared_ptr<const CdfTables> tables);
    void flush();
    void reset();
    void wait_idle();
    std::string get_stream();
private:
    void submit(std::function<void()> work);
    void worker();
    std::vector<RansSymbol> symbols_;
    std::string stream_;
    bool threaded_, stopping_ = false, busy_ = false, flushed_ = false;
    std::exception_ptr error_;
    std::deque<std::function<void()>> pending_;
    std::mutex mutex_;
    std::condition_variable changed_;
    std::thread thread_;
};
class DecoderLane {
public:
    void set_stream(std::string bytes);
    std::vector<std::int16_t> decode(const std::vector<std::int16_t>& indexes, const CdfTables& tables);
private:
    std::uint32_t bits();
    std::uint8_t byte();
    std::string stream_;
    std::size_t position_ = 0;
    std::uint32_t state_ = 0;
    bool initialized_ = false;
};
std::vector<std::int32_t> quantized_cdf(const std::vector<float>& pmf, int precision);
}  // namespace uf::fm::detail
