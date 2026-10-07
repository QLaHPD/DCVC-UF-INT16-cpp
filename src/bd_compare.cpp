// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#include "uf/bd_metrics.hpp"
#include "uf/cli.hpp"
#include "uf/evaluation_json.hpp"
#include "uf/process.hpp"
#include "uf/python_integer.hpp"
#include "uf/python_json.hpp"
#include "uf/json_compat.hpp"
#include "rd_plot.hpp"
#include "pareto_random.hpp"
#include "numpy_reductions.hpp"

#include <nlohmann/json.hpp>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace uf::evaluation {
namespace {

struct Options {
    std::string compare_between = "class", compare_frame_type = "default", base_method;
    std::string output_path = "stdout", plot_path, plot_scheme;
    std::vector<std::string> log_paths, metrics{"psnr"}, excluded;
    bool plot = true, help = false;
};
struct SequenceNameSet {
    std::set<std::string> members;
    std::vector<std::string> order;
};
using SequenceNames = std::vector<std::pair<std::string, SequenceNameSet>>;
using LogMethods = std::vector<std::pair<std::string, std::string>>;

void help(std::ostream& output) {
    cli::help(output, "compare-bd-rate");
}

Options parse_options(int argc, char** argv) {
    Options options;
    cli::Options values;
    try { values = cli::parse(argc, argv); }
    catch (const cli::HelpRequested&) { options.help = true; return options; }
    options.base_method = values.at("__base_method");
    options.log_paths = cli::list(values, "__log_paths");
    if (values.count("__compare_between")) options.compare_between = values.at("__compare_between");
    if (values.count("__compare_frame_type")) options.compare_frame_type = values.at("__compare_frame_type");
    if (values.count("__output_path")) options.output_path = values.at("__output_path");
    if (values.count("__plot_path")) options.plot_path = values.at("__plot_path");
    if (values.count("__distortion_metrics")) options.metrics = cli::list(values, "__distortion_metrics");
    if (values.count("__rate_exclude")) options.excluded = cli::list(values, "__rate_exclude");
    if (values.count("__plot_rd_curve")) options.plot = values.at("__plot_rd_curve") == "1";
    options.plot_scheme = values.count("__plot_scheme") ? values.at("__plot_scheme")
        : options.compare_between == "class" ? "combined" : "separate";
    return options;
}

void object(const Json& value, const char* kind) {
    if (!value.is_object()) throw std::invalid_argument(std::string("Comparison ") + kind + " must be a JSON object");
}

std::vector<std::string> sorted_keys(const Json& value) {
    std::vector<std::string> keys;
    keys.reserve(value.size());
    for (auto entry = value.begin(); entry != value.end(); ++entry) keys.push_back(entry.key());
    std::sort(keys.begin(), keys.end());
    return keys;
}

double number(const Json& value) {
    if (!value.is_boolean() && !value.is_number() && !python_json_extended_integer(value))
        throw std::invalid_argument("Comparison frame statistic must be numeric");
    return python_json_as_double(nlohmann::json(value));
}

double multiply(double first, double second) {
    volatile double product = first * second;
    return product;
}

// Python retains integer/Boolean sums and products exactly, converting only
// when a floating operand or the floating accumulator requires it. Native JSON
// retains exact integers, including private extended JSON numeric payloads.
class ComparisonNumber {
public:
    ComparisonNumber() = default;
    explicit ComparisonNumber(const Json& value) {
        if (value.is_number_float()) { floating_ = true; real_ = value.get<double>(); }
        else if (value.is_boolean() || value.is_number_integer() || python_json_extended_integer(value))
            integer_ = python_json_int(nlohmann::json(value));
        else throw std::invalid_argument("Comparison frame statistic must be numeric");
    }

    ComparisonNumber& operator+=(const ComparisonNumber& other) {
        if (!floating_ && !other.floating_) integer_ += other.integer_;
        else {
            // Evaluate conversions before changing this object's type/state.
            const auto first = as_double(), second = other.as_double();
            real_ = first + second;
            floating_ = true;
        }
        return *this;
    }

    ComparisonNumber operator+(const ComparisonNumber& other) const {
        auto result = *this;
        result += other;
        return result;
    }

    double product(const ComparisonNumber& other) const {
        if (!floating_ && !other.floating_) return (integer_ * other.integer_).as_double();
        return multiply(as_double(), other.as_double());
    }

    bool zero() const { return floating_ ? real_ == 0.0 : integer_.decimal() == "0"; }
    double as_double() const { return floating_ ? real_ : integer_.as_double(); }

private:
    PythonInteger integer_;
    double real_ = 0;
    bool floating_ = false;
};

LogMethods comparison_log_paths(const Options& options) {
    if (options.log_paths.size() % 2) {
        throw std::invalid_argument("Log paths must include both the method name and the corresponding log path");
    }
    LogMethods methods;
    for (std::size_t index = 0; index < options.log_paths.size(); index += 2) {
        const auto found = std::find_if(methods.begin(), methods.end(),
            [&](const auto& item) { return item.first == options.log_paths[index]; });
        if (found == methods.end()) methods.emplace_back(options.log_paths[index], options.log_paths[index + 1]);
        else found->second = options.log_paths[index + 1];
    }
    if (std::none_of(methods.begin(), methods.end(),
                    [&](const auto& item) { return item.first == options.base_method; })) {
        throw std::invalid_argument("Log paths must include the base method");
    }
    return methods;
}

Json load_results(const Options& options, const LogMethods& methods, SequenceNames& names) {
    const std::set<std::string> excluded(options.excluded.begin(), options.excluded.end());
    Json results = Json::object();
    bool consistent = true;
    for (const auto& method : methods) {
        ProcessInterrupt::check();
        const auto input = read_json_file(method.second);
        object(input, "log");
        auto& method_results = results[method.first] = Json::object();
        for (auto dataset = input.begin(); dataset != input.end(); ++dataset) {
            object(dataset.value(), "dataset");
            std::vector<std::string> keys;
            keys.reserve(dataset.value().size());
            for (auto sequence = dataset.value().begin(); sequence != dataset.value().end(); ++sequence)
                keys.push_back(sequence.key());
            const std::set<std::string> sequences(keys.begin(), keys.end());
            const auto known = std::find_if(names.begin(), names.end(),
                [&](const auto& item) { return item.first == dataset.key(); });
            if (known == names.end()) names.emplace_back(dataset.key(),
                SequenceNameSet{sequences, pareto_detail::string_set_order(keys)});
            else if (known->second.members != sequences) consistent = false;
            auto& dataset_results = method_results[dataset.key()] = Json::object();
            for (auto sequence = dataset.value().begin(); sequence != dataset.value().end(); ++sequence) {
                ProcessInterrupt::check();
                object(sequence.value(), "sequence");
                auto& sequence_results = dataset_results[sequence.key()] = Json::object();
                for (const auto& rate : sorted_keys(sequence.value())) {
                    if (!excluded.count(rate)) {
                        object(sequence.value().at(rate), "rate point");
                        sequence_results[rate] = sequence.value().at(rate);
                    }
                }
            }
        }
    }
    if (!consistent) {
        std::cerr << "warning: inconsistency found in the sequences tested in each dataset\n";
        if (!std::cerr) throw std::runtime_error("Cannot write comparison consistency warning");
    }
    return results;
}

Json mean_rate_point(const Json& records, const std::string& metric) {
    ComparisonNumber i_count, p_count;
    double i_bpp = 0, p_bpp = 0, all_bpp = 0;
    double i_quality = 0, p_quality = 0, all_quality = 0;
    const auto i_key = "ave_i_frame_" + metric, p_key = "ave_p_frame_" + metric,
               all_key = "ave_all_frame_" + metric;
    for (const auto& record : records) {
        const ComparisonNumber i(record.at("i_frame_num")), p(record.at("p_frame_num"));
        i_count += i; p_count += p;
        i_bpp += ComparisonNumber(record.at("ave_i_frame_bpp")).product(i);
        i_quality += ComparisonNumber(record.at(i_key)).product(i);
        p_bpp += ComparisonNumber(record.at("ave_p_frame_bpp")).product(p);
        p_quality += ComparisonNumber(record.at(p_key)).product(p);
        const auto total = p + i;
        all_bpp += ComparisonNumber(record.at("ave_all_frame_bpp")).product(total);
        all_quality += ComparisonNumber(record.at(all_key)).product(total);
    }
    const auto all_count = i_count + p_count;
    const auto i_denominator = i_count.zero() ? 1.0 : i_count.as_double();
    const auto p_denominator = p_count.zero() ? 1.0 : p_count.as_double();
    if (all_count.zero()) throw std::invalid_argument("Cannot average a comparison rate point with zero frames");
    const auto all_denominator = all_count.as_double();
    // retrieve_data consumes only these six curve fields. Keep the exact
    // totals above instead of narrowing unused temporary count fields to JSON.
    return Json{{"ave_i_frame_bpp", i_bpp / i_denominator}, {i_key, i_quality / i_denominator},
        {"ave_p_frame_bpp", p_bpp / p_denominator}, {p_key, p_quality / p_denominator},
        {"ave_all_frame_bpp", all_bpp / all_denominator}, {all_key, all_quality / all_denominator}};
}

Json mean_sequences(Json& results, const std::string& metric) {
    Json means = Json::object();
    for (auto method = results.begin(); method != results.end(); ++method) {
        auto& method_means = means[method.key()] = Json::object();
        for (auto dataset = method.value().begin(); dataset != method.value().end(); ++dataset) {
            Json groups = Json::object();
            for (auto& sequence : dataset.value()) {
                for (auto rate = sequence.begin(); rate != sequence.end(); ++rate) {
                    ProcessInterrupt::check();
                    // The reference adds absent metric fields to the original
                    // records before sequence flattening, for every metric.
                    for (const std::string type : {"i", "p", "all"}) {
                        const auto key = "ave_" + type + "_frame_" + metric;
                        if (!rate.value().contains(key)) rate.value()[key] = 0;
                    }
                    if (!groups.contains(rate.key())) groups[rate.key()] = Json::array();
                    groups[rate.key()].push_back(rate.value());
                }
            }
            auto& dataset_means = method_means[dataset.key()] = Json::object();
            for (auto group = groups.begin(); group != groups.end(); ++group) {
                dataset_means[group.key()] = mean_rate_point(group.value(), metric);
            }
        }
    }
    return means;
}

Json flatten_sequences(const Json& results) {
    Json flattened = Json::object();
    for (auto method = results.begin(); method != results.end(); ++method) {
        auto& target = flattened[method.key()] = Json::object();
        for (const auto& dataset : method.value()) {
            for (auto sequence = dataset.begin(); sequence != dataset.end(); ++sequence) {
                // Duplicate sequence names overwrite their earlier dataset's
                // value, retaining the first insertion position, as in Python.
                target[sequence.key()] = sequence.value();
            }
        }
    }
    return flattened;
}

std::vector<double> coordinates(const Json& values, bool allow_none = false) {
    std::vector<double> result;
    result.reserve(values.size());
    for (const auto& value : values) {
        result.push_back(allow_none && value.is_null() ? std::numeric_limits<double>::quiet_NaN() : number(value));
    }
    return result;
}

struct Retrieved {
    Json data = Json::object(), results = Json::object();
};

Retrieved retrieve_data(const Json& logs, const std::string& frame_type,
                        const std::string& base_method, const std::string& metric) {
    Retrieved result;
    for (auto method = logs.begin(); method != logs.end(); ++method) {
        auto& target = result.data[method.key()] = Json::object();
        for (auto dataset = method.value().begin(); dataset != method.value().end(); ++dataset) {
            auto& curve = target[dataset.key()] = Json{{"bpp", Json::array()}, {metric, Json::array()}};
            for (const auto& record : dataset.value()) {
                curve["bpp"].push_back(record.at("ave_" + frame_type + "_frame_bpp"));
                curve[metric].push_back(record.at("ave_" + frame_type + "_frame_" + metric));
            }
        }
    }
    const auto& base = result.data.at(base_method);
    for (auto method = result.data.begin(); method != result.data.end(); ++method) {
        if (method.key() == base_method) continue;
        auto& target = result.results[method.key()] = Json::object();
        for (auto dataset = method.value().begin(); dataset != method.value().end(); ++dataset) {
            ProcessInterrupt::check();
            const auto& curve = dataset.value();
            // Keep the reference's gate order. In particular, an empty anchor
            // fails only after a target with at least three points is found.
            // Compare before converting curve coordinates to Double: Python's
            // positive-integer gate does not overflow on an unbounded integer.
            if (base.contains(dataset.key()) && curve.at("bpp").size() >= 3 &&
                json_python_less(0, nlohmann::json(base.at(dataset.key()).at("bpp").at(0))) &&
                !curve.at(metric).at(0).is_null() && json_python_less(0, nlohmann::json(curve.at(metric).at(0)))) {
                const auto& anchor = base.at(dataset.key());
                target[dataset.key()] = bd_rate(coordinates(anchor.at("bpp")), coordinates(anchor.at(metric)),
                                                coordinates(curve.at("bpp")), coordinates(curve.at(metric)), true);
            }
        }
    }
    return result;
}

std::size_t character_count(const std::string& text) {
    return std::count_if(text.begin(), text.end(), [](unsigned char value) { return (value & 0xc0) != 0x80; });
}
std::string padded(const std::string& text, std::size_t width) {
    const auto count = character_count(text);
    return text + std::string(width > count ? width - count : 0, ' ');
}
std::string formatted_number(double value) {
    if (std::isnan(value)) return "nan";
    if (std::isinf(value)) return value < 0 ? "-inf" : "inf";
    std::ostringstream output;
    output.imbue(std::locale::classic());
    output << std::fixed << std::setprecision(1) << value;
    return output.str();
}
bool special(const std::string& name) { return !name.empty() && name.front() == '*'; }

void print_results(std::ostream& output, std::vector<std::string> columns, Json results,
                   bool rotate = false, bool print_overall = false) {
    if (results.empty()) return;
    if (rotate) {
        Json rotated = Json::object();
        std::set<std::string> new_columns;
        for (const auto& dataset : columns) {
            auto& row = rotated[dataset] = Json::object();
            for (auto method = results.begin(); method != results.end(); ++method) {
                if (method.value().contains(dataset)) row[method.key()] = method.value().at(dataset);
                new_columns.insert(method.key());
            }
        }
        if (print_overall) {
            rotated["* Overall"] = Json::object(); rotated["* Average"] = Json::object();
            for (auto method = results.begin(); method != results.end(); ++method) {
                if (method.value().contains("* Overall")) {
                    rotated["* Overall"][method.key()] = method.value().at("* Overall");
                    rotated["* Average"][method.key()] = method.value().at("* Average");
                }
            }
        }
        columns.assign(new_columns.begin(), new_columns.end());
        results = std::move(rotated);
    }
    columns.erase(std::remove_if(columns.begin(), columns.end(), special), columns.end());
    std::sort(columns.begin(), columns.end());
    auto rows = sorted_keys(results);
    rows.erase(std::remove_if(rows.begin(), rows.end(), special), rows.end());
    if (print_overall && rotate) { rows.push_back("* Overall"); rows.push_back("* Average"); }
    else if (print_overall) { columns.push_back("* Overall"); columns.push_back("* Average"); }
    std::size_t row_width = 0;
    for (auto row = results.begin(); row != results.end(); ++row) row_width = std::max(row_width, character_count(row.key()));
    row_width += 2;
    // Python builds a complete line before print. Numeric conversion failures
    // must not publish a partially rendered row.
    std::string line(row_width, ' ');
    for (const auto& column : columns) line += ' ' + padded(column, character_count(column) + 2);
    output << line << '\n';
    if (!output) throw std::runtime_error("Cannot write comparison table header");
    for (const auto& row : rows) {
        line = padded(row, row_width);
        for (const auto& column : columns) {
            const auto width = character_count(column) + 2;
            if (!results.at(row).contains(column)) line += std::string(width + 1, ' ');
            else line += ' ' + padded(formatted_number(number(results.at(row).at(column))), width);
        }
        output << line << '\n';
        if (!output) throw std::runtime_error("Cannot write comparison table row");
    }
}

void print_sequence_results(std::ostream& output, const std::vector<std::string>& datasets,
                            const SequenceNames& sequence_names, const Json& sequence_results, const Json& class_results) {
    for (const auto& dataset : datasets) {
        ProcessInterrupt::check();
        output << "---- " << dataset << " ----\n";
        if (!output) throw std::runtime_error("Cannot write comparison dataset heading");
        const auto found = std::find_if(sequence_names.begin(), sequence_names.end(),
                                        [&](const auto& entry) { return entry.first == dataset; });
        const auto& names = found->second.order;
        Json averages = Json::object();
        for (const auto& method : sorted_keys(sequence_results)) {
            std::vector<double> values;
            for (const auto& sequence : names) {
                if (sequence_results.at(method).contains(sequence)) {
                    values.push_back(number(sequence_results.at(method).at(sequence)));
                }
            }
            if (!values.empty()) averages[method] = (0.0 + numpy_detail::pairwise_sum(
                values.data(), static_cast<std::int64_t>(values.size()))) / values.size();
        }
        Json printable = Json::object();
        for (const auto& method : sorted_keys(class_results)) {
            if (!class_results.at(method).contains(dataset)) continue;
            auto& row = printable[method] = Json::object();
            for (const auto& sequence : names) {
                if (sequence_results.at(method).contains(sequence)) row[sequence] = sequence_results.at(method).at(sequence);
                row["* Overall"] = class_results.at(method).at(dataset);
                row["* Average"] = averages.at(method);
            }
        }
        print_results(output, names, std::move(printable), true, true);
    }
    output << '\n';
    if (!output) throw std::runtime_error("Cannot write comparison sequence separator");
}

void plot_curves(const std::vector<std::string>& names, const Json& data, const Options& options,
                 const std::string& frame_type, const std::string& metric) {
    if (options.plot_scheme == "combined" && names.size() > 7) {
        std::cerr << "warning: plotting in combined mode with more than 7 datasets/sequences is not supported\n";
        if (!std::cerr) throw std::runtime_error("Cannot write comparison plot warning");
        return;
    }
    const auto prefix = std::filesystem::path(options.plot_path) / (frame_type + "_frame");
    std::vector<RdPanel> panels;
    for (const auto& name : names) {
        RdPanel panel{name, {}};
        for (auto method = data.begin(); method != data.end(); ++method) {
            if (method.value().contains(name)) {
                const auto& curve = method.value().at(name);
                panel.series.push_back({method.key(), coordinates(curve.at("bpp"), true),
                                         coordinates(curve.at(metric), true)});
            }
        }
        if (options.plot_scheme == "separate") {
            write_rd_plot({panel}, prefix.string() + "_" + metric + "_" + name + ".png", false);
        } else panels.push_back(std::move(panel));
    }
    if (options.plot_scheme == "combined") write_rd_plot(panels, prefix.string() + "_" + metric + ".png", true);
}

void compare(Json& logs, const SequenceNames& names, const Options& options,
             const std::string& metric, std::ostream& output) {
    const auto classes = mean_sequences(logs, metric);
    std::vector<std::string> datasets;
    for (const auto& dataset : names) datasets.push_back(dataset.first);
    std::sort(datasets.begin(), datasets.end());
    Json sequences;
    if (options.compare_between == "sequence") sequences = flatten_sequences(logs);
    const auto frame_types = options.compare_frame_type == "default"
        ? std::vector<std::string>{"i", "p", "all"} : std::vector<std::string>{options.compare_frame_type};
    for (const auto& type : frame_types) {
        output << "============" << std::string(3 - type.size(), ' ') << type << " frame============\n";
        if (!output) throw std::runtime_error("Cannot write comparison frame heading");
        auto frame = retrieve_data(classes, type, options.base_method, metric);
        if (options.compare_between == "sequence") {
            auto sequence = retrieve_data(sequences, type, options.base_method, metric);
            print_sequence_results(output, datasets, names, sequence.results, frame.results);
            frame = std::move(sequence);
        } else print_results(output, datasets, frame.results);
        if (options.plot) {
            std::vector<std::string> plot_names = datasets;
            if (options.compare_between == "sequence") {
                plot_names.clear();
                // Keep each first-seen dataset's CPython set iteration order.
                for (const auto& dataset : names) plot_names.insert(plot_names.end(),
                    dataset.second.order.begin(), dataset.second.order.end());
            }
            plot_curves(plot_names, frame.data, options, type, metric);
        }
    }
}

}  // namespace

int run_bd_rate_command(int argc, char** argv) {
    Options options;
    try { options = parse_options(argc, argv); }
    catch (const cli::ParseError& error) {
        help(std::cerr);
        std::cerr << "error: " << error.what() << '\n';
        return 2;
    }
    if (options.help) { help(std::cout); return 0; }
    const ProcessInterrupt interruption;
    std::ofstream file;
    std::ostream* output = &std::cout;
    const auto separator = options.output_path.find_last_of('.');
    const auto suffix = options.output_path.substr(separator == std::string::npos ? 0 : separator + 1);
    // The reference redirects only txt/TXT. Other suffixes, including csv,
    // continue to print its text tables to stdout.
    if (options.output_path != "stdout" && (suffix == "txt" || suffix == "TXT")) {
        if (options.output_path.find('\0') != std::string::npos) {
            throw std::invalid_argument("Comparison output path contains NUL");
        }
        file.open(options.output_path, std::ios::trunc);
        if (!file) throw std::runtime_error("Cannot open comparison output: " + options.output_path);
        output = &file;
    }
    // Source validates pairs/anchor after output redirection and before
    // creating the plot directory or opening any input log.
    const auto methods = comparison_log_paths(options);
    if (!options.plot_path.empty()) {
        if (options.plot_path.find('\0') != std::string::npos) {
            throw std::invalid_argument("Comparison plot path contains NUL");
        }
        // common.create_folder checks exists, including an existing regular
        // file. Such a path fails later only if plotting actually uses it.
        std::error_code existence_error;
        if (!std::filesystem::exists(options.plot_path, existence_error)) {
            std::filesystem::create_directories(options.plot_path);
        }
    }
    SequenceNames names;
    auto logs = load_results(options, methods, names);
    for (const auto& metric : options.metrics) {
        ProcessInterrupt::check();
        auto title = metric;
        std::transform(title.begin(), title.end(), title.begin(), [](unsigned char value) { return std::toupper(value); });
        *output << "result for: " << title << '\n';
        if (!*output) throw std::runtime_error("Cannot write comparison metric heading");
        compare(logs, names, options, metric, *output);
    }
    if (file.is_open()) {
        file.close();
        if (!file) throw std::runtime_error("Cannot close comparison output");
    }
    return 0;
}

}  // namespace uf::evaluation
