#pragma once

#include "uf/python_integer.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace uf {

struct ParetoOptions {
    std::filesystem::path input_file;
    std::filesystem::path output_json;
    std::filesystem::path prepared_i;
    std::filesystem::path prepared_p;
    std::string variant{"htl"};
    std::string device{"cuda:0"};
    int resolution{144};
    int input_threads{1};
    int trials{2000};
    int initial_samples{800};
    PythonInteger seed{20260912};
    std::vector<double> skip_thresholds{0, 0.05, 0.10, 0.15, 0.20};
    bool resume{true};
};

void run_pareto_search(const ParetoOptions& options);

}  // namespace uf
