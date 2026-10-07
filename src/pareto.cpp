#include "uf/pareto.hpp"

#include "uf/archive.hpp"
#include "uf/integer_codec.hpp"
#include "uf/media.hpp"
#include "uf/frame_buffer.hpp"
#include "uf/prepared.hpp"
#include "uf/process.hpp"
#include "uf/python_json.hpp"
#include "uf/json_compat.hpp"
#include "pareto_random.hpp"
#include "pareto_state.hpp"
#include "pareto_progress.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <tuple>
#include <vector>
#include <cerrno>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace uf {
namespace {

using Clock = std::chrono::steady_clock;
using Json = nlohmann::json;
using pareto_detail::MutationOrder;
using pareto_detail::PythonRandom;

double sampled_seconds() {
    const auto elapsed = Clock::now().time_since_epoch();
    const auto whole = std::chrono::duration_cast<std::chrono::seconds>(elapsed);
    if (elapsed == whole) return static_cast<double>(whole.count());
    return std::chrono::duration<double>(elapsed).count();
}

double seconds_since(double start) {
    return sampled_seconds() - start;
}

bool has_status(const Json& row, const char* expected) {
    const auto& status = python_json_dict_get(row, "status");
    return status.is_string() && status.get_ref<const std::string&>() == expected;
}

Json total_encode_seconds(const Json& results) {
    static const Json zero(0);
    PythonInteger integer;
    double real = 0;
    bool floating = false;
    for (const auto& row : results) {
        const auto& value = python_json_dict_get(row, "encode_seconds", zero);
        if (!value.is_number() && !value.is_boolean() && !python_json_extended_integer(value)) {
            throw PythonJsonTypeError("unsupported operand type(s) for +: '" +
                std::string(floating ? "float" : "int") + "' and '" +
                json_compat_detail::python_type_name(value) + "'");
        }
        if (!floating && !value.is_number_float()) integer += python_json_int(value);
        else {
            if (!floating) real = integer.as_double();
            real += python_json_as_double(value);
            floating = true;
        }
    }
    return floating ? Json(real) : python_json_integer(integer);
}

Json search_elapsed(const Json& previous, double started) {
    const auto now = sampled_seconds();
    if (!previous.is_number() && !previous.is_boolean() && !python_json_extended_integer(previous)) {
        throw PythonJsonTypeError("unsupported operand type(s) for +: '" +
            std::string(json_compat_detail::python_type_name(previous)) + "' and 'float'");
    }
    // Python evaluates previous + monotonic_now - started from left to right.
    volatile double advanced = python_json_as_double(previous) + now;
    return advanced - started;
}

std::string timestamp() {
    const auto now = std::chrono::system_clock::now();
    const auto whole = std::chrono::system_clock::to_time_t(now);
    std::tm utc{};
    gmtime_r(&whole, &utc);
    std::ostringstream result;
    result << std::put_time(&utc, "%Y-%m-%dT%H:%M:%S") << "+00:00";
    return result.str();
}

void atomic_json(const std::filesystem::path& output, const Json& state) {
    // The search keeps an unbounded seed as decimal text internally. Persist
    // it as a Python integer without narrowing, then use storage.atomic_json.
    auto persisted = state;
    auto& seed = persisted.at("settings").at("seed");
    if (seed.is_string()) seed = python_json_integer(PythonInteger(seed.get<std::string>()));
    write_python_json_file_atomic(output, persisted);
}

class TrialFile {
public:
    TrialFile() {
        auto temporary = python_temporary_file(".uf-pareto-", ".bin");
        path_ = std::move(temporary.path);
        temporary.descriptor.close_checked("Cannot close Pareto bitstream reservation");
        if (unlink(path_.c_str()) != 0) {
            const int error = errno;
            throw std::system_error(error, std::generic_category(), "Cannot remove Pareto bitstream reservation");
        }
    }
    ~TrialFile() {
        if (!cleanup_attempted_) {
            try { remove(); } catch (...) {}
        }
    }
    TrialFile(const TrialFile&) = delete;
    TrialFile& operator=(const TrialFile&) = delete;
    const std::filesystem::path& path() const { return path_; }
    void remove() {
        if (cleanup_attempted_) return;
        cleanup_attempted_ = true;
        if (unlink(path_.c_str()) != 0) {
            const int error = errno;
            if (error != ENOENT)
                throw std::system_error(error, std::generic_category(), "Cannot remove Pareto trial bitstream");
        }
    }
private:
    std::filesystem::path path_;
    bool cleanup_attempted_{};
};

struct Config {
    int qp_i{}, qp_p{}, intra_period{}, reset_interval{};
    double skip_threshold{};
    auto key() const {
        return std::tuple{qp_i, qp_p, intra_period, reset_interval, skip_threshold};
    }
    Json json() const {
        return {{"qp_i", qp_i}, {"qp_p", qp_p},
                {"intra_period", intra_period}, {"reset_interval", reset_interval},
                {"skip_threshold", skip_threshold}};
    }
};

Config config_from_json(const Json& item) {
    return {item.at("qp_i").get<int>(), item.at("qp_p").get<int>(),
            item.at("intra_period").get<int>(), item.at("reset_interval").get<int>(),
            item.at("skip_threshold").get<double>()};
}

std::vector<int> valid_intra_periods(const std::string& variant) {
    const int chunk = variant == "ld" ? 1 : 8;
    std::vector<int> result{-1};
    for (int period = 1; period <= 60; ++period) {
        if (period == 1 || period % chunk == 0) result.push_back(period);
    }
    return result;
}

struct Domains {
    std::vector<int> qp_i, qp_p, intra_period, reset_interval;
    std::vector<double> skip_threshold;
    Json json() const {
        return {{"qp_i", qp_i}, {"qp_p", qp_p},
                {"intra_period", intra_period}, {"reset_interval", reset_interval},
                {"skip_threshold", skip_threshold}};
    }
    std::uint64_t capacity() const {
        const std::array<std::size_t, 5> sizes{qp_i.size(), qp_p.size(),
            intra_period.size(), reset_interval.size(), skip_threshold.size()};
        if (std::any_of(sizes.begin(), sizes.end(), [](auto size) { return !size; })) return 0;
        std::uint64_t result = 1;
        for (const auto size : sizes) {
            // Capacity is only compared with bounded native sample counts.
            // A larger Python math.prod value must not wrap below that count.
            if (size > std::numeric_limits<std::uint64_t>::max() / result)
                return std::numeric_limits<std::uint64_t>::max();
            result *= size;
        }
        return result;
    }
};

Domains make_domains(const ParetoOptions& options) {
    Domains result;
    result.qp_i.resize(64);
    result.qp_p.resize(64);
    std::iota(result.qp_i.begin(), result.qp_i.end(), 0);
    std::iota(result.qp_p.begin(), result.qp_p.end(), 0);
    result.intra_period = valid_intra_periods(options.variant);
    result.reset_interval.resize(60);
    std::iota(result.reset_interval.begin(), result.reset_interval.end(), 1);
    result.skip_threshold = options.skip_thresholds;
    for (const auto value : result.skip_threshold) {
        if (!std::isfinite(value) || value < 0 || value > 16) {
            throw std::invalid_argument("Pareto skip thresholds must be finite values from 0 to 16");
        }
    }
    // Python set keeps the first equal value; stable sorting before unique
    // preserves that representative too, including -0.0 versus +0.0.
    std::stable_sort(result.skip_threshold.begin(), result.skip_threshold.end());
    result.skip_threshold.erase(std::unique(result.skip_threshold.begin(),
                                            result.skip_threshold.end()),
                                result.skip_threshold.end());
    if (result.skip_threshold.empty()) {
        throw std::invalid_argument("At least one Pareto skip threshold is required");
    }
    return result;
}

std::vector<Config> uniform_configurations(const Domains& domains, int count,
                                         const PythonInteger& seed) {
    if (static_cast<std::uint64_t>(count) > domains.capacity()) {
        throw std::invalid_argument("Initial Pareto samples exceed the configuration space");
    }
    PythonRandom rng(seed);
    for (int attempt = 0; attempt < 1000; ++attempt) {
        ProcessInterrupt::check();
        std::array<std::vector<std::size_t>, 5> columns;
        const std::array<std::size_t, 5> lengths{domains.qp_i.size(), domains.qp_p.size(),
            domains.intra_period.size(), domains.reset_interval.size(),
            domains.skip_threshold.size()};
        for (std::size_t column = 0; column < columns.size(); ++column) {
            auto& values = columns[column];
            values.reserve(count);
            for (int row = 0; row < count; ++row) values.push_back(row % lengths[column]);
            rng.shuffle(values);
        }
        std::vector<Config> samples;
        std::set<decltype(Config{}.key())> distinct;
        for (int row = 0; row < count; ++row) {
            Config candidate{domains.qp_i[columns[0][row]],
                domains.qp_p[columns[1][row]],
                domains.intra_period[columns[2][row]],
                domains.reset_interval[columns[3][row]],
                domains.skip_threshold[columns[4][row]]};
            distinct.insert(candidate.key());
            samples.push_back(candidate);
        }
        if (distinct.size() == samples.size()) return samples;
    }
    throw std::runtime_error("Cannot construct unique balanced Pareto samples");
}

std::vector<int> pareto_trials(const Json& results) {
    std::vector<const Json*> successful;
    for (const auto& item : results) {
        if (has_status(item, "ok")) successful.push_back(&item);
    }
    std::sort(successful.begin(), successful.end(), [](const Json* a, const Json* b) {
        return std::tuple{a->at("bitstream_bytes").get<std::uint64_t>(),
                          -a->at("psnr_db").get<double>(), a->at("trial").get<int>()} <
               std::tuple{b->at("bitstream_bytes").get<std::uint64_t>(),
                          -b->at("psnr_db").get<double>(), b->at("trial").get<int>()};
    });
    std::vector<int> frontier;
    double best = -std::numeric_limits<double>::infinity();
    for (const auto* item : successful) {
        const double quality = item->at("psnr_db").get<double>();
        if (quality > best) {
            frontier.push_back(item->at("trial").get<int>());
            best = quality;
        }
    }
    return frontier;
}

template <typename T>
const T& choice(const std::vector<T>& values, PythonRandom& rng) {
    if (values.empty()) throw std::invalid_argument("Cannot sample an empty Pareto domain");
    return values[static_cast<std::size_t>(rng.below(values.size()))];
}

bool chance(double probability, PythonRandom& rng) {
    return rng.random() < probability;
}

int rounded_step(double value) {
    // Python round(x) uses ties-to-even, independent of the floating-point
    // rounding mode. std::round instead rounds half values away from zero.
    const auto magnitude = std::abs(value);
    auto whole = std::floor(magnitude);
    const auto fraction = magnitude - whole;
    if (fraction > 0.5 || (fraction == 0.5 && std::fmod(whole, 2.0) != 0)) ++whole;
    return static_cast<int>(value < 0 ? -whole : whole);
}

template <typename T>
void mutate_value(T& value, const std::vector<T>& domain, PythonRandom& rng) {
    const auto it = std::find(domain.begin(), domain.end(), value);
    if (it == domain.end()) throw std::runtime_error("Pareto configuration is outside search space");
    const auto sigma = std::max(1.0, static_cast<double>(domain.size()) / 8.0);
    int step = 0;
    while (!step) step = rounded_step(rng.gauss(0.0, sigma));
    const auto index = static_cast<int>(it - domain.begin());
    value = domain[std::clamp(index + step, 0, static_cast<int>(domain.size()) - 1)];
}

Config adaptive_configuration(const Domains& domains, const Json& results,
                               const std::set<decltype(Config{}.key())>& seen,
                               PythonRandom& rng, const MutationOrder& mutation_order) {
    const auto frontier_ids = pareto_trials(results);
    std::vector<Config> parents;
    for (const auto& item : results) {
        if (has_status(item, "ok") &&
            std::find(frontier_ids.begin(), frontier_ids.end(),
                      item.at("trial").get<int>()) != frontier_ids.end()) {
            parents.push_back(config_from_json(item.at("config")));
        }
    }
    for (int attempt = 0; attempt < 10000; ++attempt) {
        ProcessInterrupt::check();
        Config candidate;
        if (parents.size() < 2 || chance(0.15, rng)) {
            candidate = {choice(domains.qp_i, rng), choice(domains.qp_p, rng),
                choice(domains.intra_period, rng), choice(domains.reset_interval, rng),
                choice(domains.skip_threshold, rng)};
        } else {
            const auto& first = choice(parents, rng);
            const auto& second = choice(parents, rng);
            candidate = {chance(0.5, rng) ? first.qp_i : second.qp_i,
                chance(0.5, rng) ? first.qp_p : second.qp_p,
                chance(0.5, rng) ? first.intra_period : second.intra_period,
                chance(0.5, rng) ? first.reset_interval : second.reset_interval,
                chance(0.5, rng) ? first.skip_threshold : second.skip_threshold};
            std::array<bool, 5> mutations{};
            for (auto& selected : mutations) selected = chance(0.35, rng);
            if (std::none_of(mutations.begin(), mutations.end(), [](bool value) { return value; })) {
                mutations[static_cast<std::size_t>(rng.below(mutations.size()))] = true;
            }
            for (const auto field : mutation_order.indices(mutations)) {
                switch (field) {
                case 0: mutate_value(candidate.qp_i, domains.qp_i, rng); break;
                case 1: mutate_value(candidate.qp_p, domains.qp_p, rng); break;
                case 2: mutate_value(candidate.intra_period, domains.intra_period, rng); break;
                case 3: mutate_value(candidate.reset_interval, domains.reset_interval, rng); break;
                case 4: mutate_value(candidate.skip_threshold, domains.skip_threshold, rng); break;
                default: throw std::logic_error("Unknown Pareto mutation field");
                }
            }
        }
        if (!seen.count(candidate.key())) return candidate;
    }
    throw std::runtime_error("Cannot find another unique Pareto configuration");
}

std::vector<std::uint8_t> raw_yuv420(const std::vector<std::uint8_t>& frame,
                                     std::uint32_t width, std::uint32_t height) {
    const auto area = static_cast<std::size_t>(width) * height;
    if (frame.size() != area * 3 || width % 2 || height % 2) {
        throw std::invalid_argument("Invalid Pareto reference frame");
    }
    std::vector<std::uint8_t> raw(area * 3 / 2);
    std::copy_n(frame.data(), area, raw.data());
    for (int channel = 0; channel < 2; ++channel) {
        const auto* full = frame.data() + (channel + 1) * area;
        auto* reduced = raw.data() + area + channel * area / 4;
        for (std::uint32_t y = 0; y < height / 2; ++y) {
            for (std::uint32_t x = 0; x < width / 2; ++x) {
                reduced[static_cast<std::size_t>(y) * width / 2 + x] =
                    full[static_cast<std::size_t>(y * 2) * width + x * 2];
            }
        }
    }
    return raw;
}

class PsnrSink {
public:
    PsnrSink(const std::vector<std::vector<std::uint8_t>>& references,
             std::uint32_t width, std::uint32_t height)
        : references_(references), luma_(static_cast<std::size_t>(width) * height) {}

    void write(const std::vector<std::uint8_t>& decoded) {
        if (index_ >= references_.size() || decoded.size() != references_[index_].size()) {
            throw std::runtime_error("Pareto decoded frame differs from source reference shape");
        }
        const auto& reference = references_[index_];
        const auto chroma = luma_ / 4;
        const std::array<double, 3> component{
            psnr(reference.data(), decoded.data(), luma_),
            psnr(reference.data() + luma_, decoded.data() + luma_, chroma),
            psnr(reference.data() + luma_ + chroma,
                 decoded.data() + luma_ + chroma, chroma)};
        const volatile double weighted_y = 6 * component[0];
        const volatile double weighted_yu = weighted_y + component[1];
        sums_[0] += (weighted_yu + component[2]) / 8;
        for (int i = 0; i < 3; ++i) sums_[i + 1] += component[i];
        ++index_;
    }

    Json result() const {
        if (!index_ || index_ != references_.size()) {
            throw std::runtime_error("Pareto decoder did not compare every source frame");
        }
        return {{"psnr_db", sums_[0] / index_}, {"psnr_y_db", sums_[1] / index_},
                {"psnr_u_db", sums_[2] / index_}, {"psnr_v_db", sums_[3] / index_}};
    }

private:
    static double psnr(const std::uint8_t* reference,
                       const std::uint8_t* decoded, std::size_t length) {
        double squared_error = 0;
        for (std::size_t i = 0; i < length; ++i) {
            if (!(i % 4096)) ProcessInterrupt::check();
            const double error = static_cast<double>(reference[i]) - decoded[i];
            squared_error += error * error;
        }
        const double mse = squared_error / length;
        return mse <= 1e-10 ? 99.9 :
            std::min(99.9, 10.0 * std::log10(255.0 * 255.0 / mse));
    }
    const std::vector<std::vector<std::uint8_t>>& references_;
    std::size_t luma_{};
    std::size_t index_{};
    std::array<double, 4> sums_{};
};

void save_state(const std::filesystem::path& path, Json& state, const Json& elapsed) {
    auto& results = state.at("results");
    state["completed"] = results.size();
    std::size_t successful = 0, failed = 0;
    for (const auto& row : results) {
        if (has_status(row, "ok")) ++successful;
        if (has_status(row, "failed")) ++failed;
    }
    state["successful"] = successful;
    state["failed"] = failed;
    state["total_encode_seconds"] = total_encode_seconds(results);
    state["total_search_seconds"] = elapsed;
    state["pareto_frontier_trials"] = pareto_trials(results);
    state["updated_at"] = timestamp();
    atomic_json(path, state);
}

}  // namespace

void run_pareto_search(const ParetoOptions& options) {
    ProcessInterrupt::check();
    if (options.variant != "hts" && options.variant != "htl" && options.variant != "ld") {
        throw std::invalid_argument("Unknown Pareto model structure");
    }
    if (options.trials < 1 || options.initial_samples < 1 ||
        options.initial_samples > options.trials || options.resolution < 1 ||
        options.input_threads < 1) {
        throw std::invalid_argument("Invalid Pareto search counts, resolution, or input threads");
    }
    const auto source = std::filesystem::canonical(options.input_file);
    if (!std::filesystem::is_regular_file(source)) {
        throw std::invalid_argument("Pareto source must be a regular video file");
    }
    const auto output = std::filesystem::weakly_canonical(std::filesystem::absolute(options.output_json));
    if (output == source) {
        throw std::invalid_argument("Pareto JSON cannot replace its source video");
    }
    const auto original = probe_media(source);
    const auto [width, height] = scaled_dimensions(original.width, original.height,
                                                   static_cast<std::uint32_t>(options.resolution));
    const auto domains = make_domains(options);
    auto image = load_prepared(options.prepared_i, std::nullopt, std::string("image"));
    auto video = load_prepared(options.prepared_p, std::nullopt, options.variant);
    const Json source_media = {{"width", original.width}, {"height", original.height},
                               {"fps", original.fps}, {"duration", original.duration},
                               {"audio", original.audio}};
    const Json settings = {
        {"input_file", source.string()}, {"input_sha256", sha256_file(source)},
        {"source", source_media},
        {"resolution", std::to_string(original.width) + "x" +
                       std::to_string(original.height) + " -> " +
                       std::to_string(width) + "x" + std::to_string(height)},
        {"runtime", "int16"}, {"model_structure", options.variant},
        {"integer", {{"arithmetic", image.arithmetic}, {"image", image.identity},
                      {"video", video.identity}}},
        {"trials", options.trials}, {"initial_samples", options.initial_samples},
        {"seed", options.seed.decimal()}, {"input_threads", options.input_threads},
        {"search_space", domains.json()},
        {"objective", {{"minimize", "bitstream_bytes"}, {"maximize", "psnr_db"}}},
        {"psnr", "mean per-frame YUV420 PSNR, 6:1:1 Y:U:V weighting"},
        {"optimizer", "balanced categorical Latin sampling, then Pareto-front genetic crossover/mutation"}
    };
    Json state;
    if (std::filesystem::exists(output)) {
        if (!options.resume) throw std::runtime_error("Pareto JSON already exists");
        state = pareto_detail::read_state_file(output);
        if (state.value("format", std::string{}) !=
                                  "dcvc-uf-pareto-search-v1" ||
            !pareto_detail::same_settings(state.at("settings"), settings) || !state.at("results").is_array()) {
            throw std::runtime_error("Existing Pareto JSON has different source or settings");
        }
    } else {
        const auto created = timestamp();
        state = {{"format", "dcvc-uf-pareto-search-v1"}, {"status", "running"},
                 {"created_at", created}, {"updated_at", created},
                 {"settings", settings}, {"completed", 0},
                 {"total_encode_seconds", 0.0}, {"total_search_seconds", 0.0},
                 {"results", Json::array()}, {"pareto_frontier_trials", Json::array()}};
        save_state(output, state, Json(0.0));
    }
    if (state.at("results").size() >= static_cast<std::size_t>(options.trials)) {
        state["status"] = "complete";
        save_state(output, state, python_json_dict_get(state, "total_search_seconds", Json(0)));
        std::cout << "Pareto search already complete: " << output.string() << '\n';
        if (!std::cout) throw std::runtime_error("Cannot write completed Pareto search path");
        return;
    }
    state["status"] = "running";
    save_state(output, state, python_json_dict_get(state, "total_search_seconds", Json(0)));
    std::cout << "Loading " << width << 'x' << height << " source frames once for "
              << options.trials << " trials..." << std::endl;
    if (!std::cout) throw std::runtime_error("Cannot write Pareto source loading message");
    std::vector<std::vector<std::uint8_t>> frames, references;
    {
        BufferedFrameReader buffered(source, width, height, original, original.fps,
            options.input_threads, std::nullopt, "none", 8);
        try {
            while (auto frame = buffered.read()) {
                ProcessInterrupt::check();
                references.push_back(raw_yuv420(*frame, width, height));
                frames.push_back(std::move(*frame));
            }
        } catch (...) {
            buffered.close_checked();
            throw;
        }
        buffered.close_checked();
    }
    if (frames.empty()) throw std::runtime_error("Pareto source yielded no frames");
    IntegerCodec codec(image, &video, options.variant, torch::Device(options.device));
    image.tensors.clear();
    video.tensors.clear();
    const auto initial = uniform_configurations(domains, options.initial_samples, options.seed);
    std::set<decltype(Config{}.key())> seen;
    for (const auto& row : state.at("results")) {
        seen.insert(config_from_json(row.at("config")).key());
    }
    PythonRandom rng(options.seed, state.at("results").size() * 104729ull);
    const MutationOrder mutation_order;
    const Json previous = python_json_dict_get(state, "total_search_seconds", Json(0));
    const auto started = sampled_seconds();
    pareto_detail::ParetoProgress progress(static_cast<std::uint64_t>(options.trials),
                                         state.at("results").size());
    torch::NoGradGuard no_grad;
    try {
        while (state.at("results").size() < static_cast<std::size_t>(options.trials)) {
            ProcessInterrupt::check();
            const auto index = state.at("results").size();
            Config candidate = index < initial.size() ? initial[index]
                : adaptive_configuration(domains, state.at("results"), seen, rng, mutation_order);
            const std::string phase = index < initial.size() ? "uniform" : "genetic";
            seen.insert(candidate.key());
            const auto trial_started = sampled_seconds();
            Json row = {{"trial", index + 1}, {"phase", phase},
                        {"config", candidate.json()}};
            TrialFile temporary;
            try {
                try {
                    codec.set_skip_threshold(candidate.skip_threshold);
                    std::size_t frame_index = 0;
                    const auto encoded = codec.encode([&]() -> std::optional<std::vector<std::uint8_t>> {
                        ProcessInterrupt::check();
                        if (frame_index >= frames.size()) return std::nullopt;
                        return frames[frame_index++];
                    }, temporary.path(), width, height, candidate.qp_i, candidate.qp_p,
                        candidate.reset_interval, candidate.intra_period);
                    const auto size = std::filesystem::file_size(temporary.path());
                    PsnrSink sink(references, width, height);
                    const auto decoded = codec.decode(temporary.path(), encoded.frames, encoded.packets,
                        [&](const std::vector<std::uint8_t>& raw) { sink.write(raw); }, Sps{0, width, height});
                    if (decoded.frames != frames.size()) {
                        throw std::runtime_error("Pareto decoder produced the wrong frame count");
                    }
                    row["status"] = "ok";
                    row["bitstream_bytes"] = size;
                    row.update(sink.result());
                    row["frames"] = encoded.frames;
                    row["packets"] = encoded.packets;
                    row["encode_seconds"] = encoded.encode_seconds;
                    row["decode_seconds"] = decoded.decode_seconds;
                    // Source records trial elapsed while constructing its
                    // result, before the temporary-file finally cleanup.
                    row["trial_seconds"] = seconds_since(trial_started);
                } catch (const std::exception& error) {
                    ProcessInterrupt::check();
                    row = {{"trial", index + 1}, {"phase", phase}, {"status", "failed"},
                           {"config", candidate.json()},
                           {"error", std::string("RuntimeError: ") + error.what()},
                           {"trial_seconds", seconds_since(trial_started)}};
                }
            } catch (...) {
                temporary.remove();
                throw;
            }
            temporary.remove();
            state["results"].push_back(row);
            save_state(output, state, search_elapsed(previous, started));
            progress.update(row, state.at("pareto_frontier_trials").size());
        }
        ProcessInterrupt::check();
        state["status"] = "complete";
        save_state(output, state, search_elapsed(previous, started));
        ProcessInterrupt::check();
    } catch (...) {
        try {
            if (ProcessInterrupt::requested()) {
                state["status"] = "interrupted";
                save_state(output, state, search_elapsed(previous, started));
            }
        } catch (...) {
            progress.close();
            throw;
        }
        progress.close();
        throw;
    }
    progress.close();
    std::cout << "Pareto search complete: " << output.string() << std::endl;
    if (!std::cout) throw std::runtime_error("Cannot write Pareto search completion");
}

}  // namespace uf
