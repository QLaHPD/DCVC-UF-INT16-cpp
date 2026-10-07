#include "pareto_state.hpp"

#include "uf/python_integer.hpp"
#include "uf/python_numeric.hpp"
#include "uf/python_json.hpp"
#include "uf/json_compat.hpp"
#include "uf/utf8.hpp"

#include <cstddef>
#include <cmath>
#include <iomanip>
#include <istream>
#include <iterator>
#include <locale>
#include <optional>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace uf::pareto_detail {
namespace {

using Json = nlohmann::json;

std::optional<std::string> integer_seed(const Json& value) {
    if (python_json_extended_integer(value)) return std::string(python_json_extended_integer_decimal(value));
    if (value.is_string()) return PythonInteger(value.get<std::string>()).decimal();
    if (value.is_boolean()) return value.get<bool>() ? "1" : "0";
    if (value.is_number_integer()) return PythonInteger(value.dump()).decimal();
    if (value.is_number_float()) {
        const auto number = value.get<double>();
        if (!std::isfinite(number) || std::trunc(number) != number) return std::nullopt;
        // Fixed precision zero emits the Float64 value's exact integer, even
        // above 2^53. Comparing decimal strings cannot round the CLI seed into
        // a neighboring Float64 value. Python -0.0 compares equal to zero.
        std::ostringstream text;
        text.imbue(std::locale::classic());
        text << std::fixed << std::setprecision(0) << number;
        return PythonInteger(text.str()).decimal();
    }
    return std::nullopt;
}

bool python_equal(const Json& first, const Json& second) {
    return json_python_equal(first, second);
}

void indent(std::ostream& output, std::size_t depth) {
    for (std::size_t index = 0; index < depth; ++index) output << "  ";
}

void emit(std::ostream& output, const Json& value, const Json* seed,
          const std::string& decimal, std::size_t depth) {
    if (&value == seed) {
        check_python_integer_digits(decimal.size() - (decimal.front() == '-'));
        output << decimal;
    } else if (value.is_object()) {
        if (value.empty()) { output << "{}"; return; }
        output << "{\n";
        for (auto item = value.begin(); item != value.end(); ++item) {
            if (item != value.begin()) output << ",\n";
            indent(output, depth + 1);
            output << python_json_string(item.key()) << ": ";
            emit(output, item.value(), seed, decimal, depth + 1);
        }
        output << '\n';
        indent(output, depth);
        output << '}';
    } else if (value.is_array()) {
        if (value.empty()) { output << "[]"; return; }
        output << "[\n";
        for (std::size_t index = 0; index < value.size(); ++index) {
            if (index) output << ",\n";
            indent(output, depth + 1);
            emit(output, value[index], seed, decimal, depth + 1);
        }
        output << '\n';
        indent(output, depth);
        output << ']';
    } else {
        output << python_sorted_json(value);
    }
}

Json normalize_state(Json result) {
    // Keep the existing native random-seed snapshot spelling, while retaining
    // numeric type provenance from the shared parser. A JSON string seed must
    // not impersonate an integer merely because it contains decimal digits.
    if (!result.is_object() || !result.contains("settings") ||
        !result.at("settings").is_object() || !result.at("settings").contains("seed"))
        throw std::invalid_argument("Pareto JSON settings.seed must be a number or Boolean");
    auto& seed = result.at("settings").at("seed");
    if (!seed.is_number() && !seed.is_boolean() && !python_json_extended_integer(seed))
        throw std::invalid_argument("Pareto JSON settings.seed must be a number or Boolean");
    if (seed.is_number_integer() || python_json_extended_integer(seed))
        seed = python_json_int(seed).decimal();
    return result;
}

}  // namespace

Json read_state(std::istream& input) {
    std::string text{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    if (input.bad()) throw std::runtime_error("Cannot read Pareto JSON");
    return normalize_state(parse_python_json(
        decode_process_text(std::move(text), python_process_text_encoding())));
}

Json read_state_file(const std::filesystem::path& path) {
    // Parsing happens inside the owned file scope; seed normalization happens
    // after checked close, as the reference validates state after its with.
    return normalize_state(read_python_json_file(path, python_process_text_encoding(), false));
}

void write_state(std::ostream& output, const Json& state) {
    const auto& seed = state.at("settings").at("seed");
    const auto protected_seed = seed.is_string() ? &seed : nullptr;
    const auto decimal = protected_seed ? PythonInteger(seed.get<std::string>()).decimal() : std::string{};
    emit(output, state, protected_seed, decimal, 0);
    output << '\n';
}

bool same_settings(const Json& saved, const Json& expected) {
    if (!saved.is_object() || !expected.is_object() ||
        !saved.contains("seed") || !expected.contains("seed")) return false;
    const auto saved_integer = integer_seed(saved.at("seed"));
    const auto expected_integer = integer_seed(expected.at("seed"));
    if (!saved_integer || !expected_integer || *saved_integer != *expected_integer) return false;
    auto comparable = saved;
    comparable["seed"] = expected.at("seed");
    return python_equal(comparable, expected);
}

}  // namespace uf::pareto_detail
