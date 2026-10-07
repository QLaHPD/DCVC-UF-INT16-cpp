// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#include "uf/evaluation_json.hpp"
#include "uf/python_json.hpp"
#include "uf/utf8.hpp"
#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <istream>
#include <iterator>
#include <locale>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace uf::evaluation {
namespace {

void write_value(const Json& value, std::ostream& output, int digits, int indent, std::size_t depth) {
    if (value.is_object() || value.is_array()) {
        const bool object = value.is_object();
        output << (object ? '{' : '[');
        if (!value.empty()) {
            std::size_t index = 0;
            for (auto item = value.begin(); item != value.end(); ++item) {
                if (index++) output << ',';
                if (indent >= 0) output << '\n' << std::string((depth + 1) * indent, ' ');
                if (object) output << python_json_string(item.key()) << (indent >= 0 ? ": " : ":");
                write_value(item.value(), output, digits, indent, depth + 1);
            }
            if (indent >= 0) output << '\n' << std::string(depth * indent, ' ');
        }
        output << (object ? '}' : ']');
    } else if (value.is_number_float()) {
        const auto number = value.get<double>();
        if (digits < 0) output << python_json_float(number);
        // common.dump_json replaces the float formatter with format(o, '.Nf'),
        // including its lowercase nan/inf tokens. These are intentionally not
        // accepted by json.loads; do not silently reinterpret the source format.
        else if (!std::isfinite(number)) output << python_float_repr(number);
        else {
            std::ostringstream formatted;
            formatted.imbue(std::locale::classic());
            formatted << std::fixed << std::setprecision(digits) << number;
            output << formatted.str();
        }
    } else output << python_sorted_json(nlohmann::json(value));
}

}  // namespace

Json read_json(std::istream& input) {
    const std::string text((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    if (input.bad()) throw std::runtime_error("Cannot read evaluation JSON");
    return parse_python_ordered_json(text);
}

Json read_json_file(const std::filesystem::path& path) {
    return read_python_ordered_json_file(path, python_process_text_encoding(), false);
}

void write_json(const Json& value, std::ostream& output, int float_digits, int indent) {
    if (float_digits < -1 || indent < -1) throw std::invalid_argument("Invalid evaluation JSON formatting");
    write_value(value, output, float_digits, indent, 0);
    if (!output) throw std::runtime_error("Cannot write evaluation JSON");
}

void write_json_file(const Json& value, const std::filesystem::path& path,
                     int float_digits, int indent) {
    std::ofstream output(path, std::ios::trunc);
    if (!output) throw std::runtime_error("Cannot create evaluation JSON: " + path.string());
    bool close_attempted = false;
    const auto close = [&]() {
        close_attempted = true;
        output.clear();
        output.close();
        if (!output) throw std::runtime_error("Cannot close evaluation JSON: " + path.string());
    };
    try {
        write_json(value, output, float_digits, indent);
        close();
    } catch (...) {
        if (!close_attempted) close();
        throw;
    }
}

}  // namespace uf::evaluation
