#include "uf/presets.hpp"
#include "uf/json_compat.hpp"
#include "uf/python_json.hpp"
#include "uf/utf8.hpp"

#include <cmath>
#include <initializer_list>
#include <locale>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace uf {

std::filesystem::path bundled_benchmark() {
    const auto name = "uf-int16-htl-144p-pareto.json";
    const auto executable = std::filesystem::read_symlink("/proc/self/exe");
    const auto installed = executable.parent_path() / "benchmarks" / name;
    if (std::filesystem::is_regular_file(installed)) return installed;
    const auto working = std::filesystem::current_path() / "benchmarks" / name;
    if (std::filesystem::is_regular_file(working)) return working;
    throw std::runtime_error("UF PSNR benchmark is missing; generate a HT-L INT16 144p Pareto report with pareto-search and supply UF_PARETO_BENCHMARK when configuring, or place it at benchmarks/uf-int16-htl-144p-pareto.json");
}

TargetSelection select_target_psnr(const std::filesystem::path& benchmark,
                                   double target_psnr,
                                   const nlohmann::json& explicit_constraints,
                                   const nlohmann::json& explicit_requirements) {
    const auto data = read_python_ordered_json_file(benchmark, python_process_text_encoding(), true);
    const auto& settings = data.at("settings");
    TargetSelection selection;
    selection.runtime = settings.at("runtime").get<std::string>();
    selection.variant = settings.at("model_structure").get<std::string>();
    const auto dimensions = settings.at("resolution").get<std::string>();
    const auto arrow = dimensions.rfind(" -> ");
    const auto tail = arrow == std::string::npos ? dimensions : dimensions.substr(arrow + 4);
    std::optional<nlohmann::json> minimum;
    for (std::size_t start = 0;;) {
        const auto end = tail.find('x', start);
        const auto dimension = python_json_integer(PythonInteger(std::string_view(tail).substr(start,
            end == std::string::npos ? std::string::npos : end - start)));
        if (!minimum || json_python_less(dimension, *minimum)) minimum = dimension;
        if (end == std::string::npos) break;
        start = end + 1;
    }
    const auto resolution = python_json_index_int64(*minimum);
    if (resolution < std::numeric_limits<int>::min() || resolution > std::numeric_limits<int>::max())
        throw std::overflow_error("Preset resolution exceeds native int range");
    selection.resolution = static_cast<int>(resolution);
    const nlohmann::json required = {{"runtime", selection.runtime},
        {"model_structure", selection.variant}, {"resolution", selection.resolution}};
    // Python checks required benchmark settings before searching eligible rows.
    for (const auto* field : {"runtime", "model_structure", "resolution"}) {
        if (explicit_requirements.contains(field) &&
            !json_python_equal(explicit_requirements.at(field), required.at(field))) {
            const auto& expected = required.at(field);
            throw std::invalid_argument(std::string("Bundled --target-psnr benchmark requires --") +
                field + " " + (expected.is_string() ? expected.get<std::string>() : python_json_repr(expected)) +
                "; omit --target-psnr to use other settings");
        }
    }
    if (!std::isfinite(target_psnr) || target_psnr <= 0)
        throw std::invalid_argument("--target-psnr must be positive and finite");
    std::vector<const nlohmann::ordered_json*> eligible;
    const nlohmann::json zero(0), target(target_psnr);
    const auto finite = [](const auto& value) {
        const nlohmann::json scalar(value);
        if (!scalar.is_number() && !scalar.is_boolean() && !python_json_extended_integer(scalar))
            throw PythonJsonTypeError("must be real number, not " +
                std::string(json_compat_detail::python_type_name(scalar)));
        return std::isfinite(python_json_as_double(scalar));
    };
    const auto row_less = [](const auto& first, const auto& second) {
        for (const auto* field : {"encode_seconds", "bitstream_bytes", "trial"}) {
            const auto& left = first.at(field);
            const auto& right = second.at(field);
            if (&left == &right || json_python_equal(nlohmann::json(left), nlohmann::json(right))) continue;
            return json_python_less(nlohmann::json(left), nlohmann::json(right));
        }
        return false;
    };
    const auto& results = data.at("results");
    if (!results.is_array()) {
        if (results.is_object() || results.is_string()) {
            if (json_truthy(nlohmann::json(results)))
                throw PythonJsonAttributeError("'str' object has no attribute 'get'");
        } else throw PythonJsonTypeError("'" +
            std::string(json_compat_detail::python_type_name(nlohmann::json(results))) + "' object is not iterable");
    }
    // Python gathers eligible rows before evaluating the min() key. Keep this
    // order so a later filtering error precedes an earlier missing trial key.
    static const nlohmann::ordered_json empty_results = nlohmann::ordered_json::array();
    const auto& rows = results.is_array() ? results : empty_results;
    for (const auto& row : rows) {
        if (python_json_dict_get(nlohmann::json(row), "status") != "ok") continue;
        bool matching = true;
        for (auto it = explicit_constraints.begin(); it != explicit_constraints.end(); ++it) {
            const auto& config = row.at("config");
            if (!json_python_equal(python_json_dict_get(nlohmann::json(config), it.key()), it.value())) {
                matching = false;
                break;
            }
        }
        if (!matching) continue;
        bool all_finite = true;
        for (const auto* field : {"psnr_db", "encode_seconds", "bitstream_bytes"}) {
            if (!finite(row.at(field))) { all_finite = false; break; }
        }
        if (!all_finite) continue;
        const auto& quality = row.at("psnr_db");
        const auto& seconds = row.at("encode_seconds");
        const auto& bytes = row.at("bitstream_bytes");
        if (json_python_less(nlohmann::json(quality), target) ||
            !json_python_less(zero, nlohmann::json(seconds)) ||
            !json_python_less(zero, nlohmann::json(bytes))) continue;
        eligible.push_back(&row);
    }
    if (eligible.empty()) {
        std::ostringstream message;
        message.imbue(std::locale::classic());
        message << "No measured configuration meets --target-psnr " << target_psnr
                << " with the specified parameters; lower the target, relax parameters, "
                   "or omit --target-psnr for manual encoding";
        throw std::invalid_argument(message.str());
    }
    const nlohmann::ordered_json* best = nullptr;
    for (const auto* row : eligible) {
        (void)row->at("trial");  // Evaluate every tuple key, including the first.
        if (!best || row_less(*row, *best)) best = row;
    }
    selection.trial = *best;
    return selection;
}

}  // namespace uf
