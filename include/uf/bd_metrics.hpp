// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#pragma once

#include <vector>

namespace uf::evaluation {

// bd-metric 0.9.0 procedures: cubic least squares or 100-point PCHIP
// integration. The reference independently sorts each PCHIP coordinate.
double bd_rate(const std::vector<double>& base_rate, const std::vector<double>& base_quality,
               const std::vector<double>& rate, const std::vector<double>& quality, bool piecewise = false);
double bd_psnr(const std::vector<double>& base_rate, const std::vector<double>& base_quality,
               const std::vector<double>& rate, const std::vector<double>& quality, bool piecewise = false);

int run_bd_rate_command(int argc, char** argv);

}  // namespace uf::evaluation
