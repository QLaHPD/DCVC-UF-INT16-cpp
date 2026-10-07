// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace uf::evaluation {

struct RdSeries {
    std::string label;
    std::vector<double> rate, quality;
};
struct RdPanel {
    std::string title;
    std::vector<RdSeries> series;
};
void write_rd_plot(const std::vector<RdPanel>& panels, const std::filesystem::path& output, bool combined);

}  // namespace uf::evaluation
