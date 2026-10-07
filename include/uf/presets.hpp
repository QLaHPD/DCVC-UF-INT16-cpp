#pragma once

#include <filesystem>
#include <nlohmann/json.hpp>
#include <string>

namespace uf {

struct TargetSelection {
    std::string runtime;
    std::string variant;
    int resolution{};
    nlohmann::ordered_json trial;
};

std::filesystem::path bundled_benchmark();
TargetSelection select_target_psnr(const std::filesystem::path& benchmark,
                                   double target_psnr,
                                   const nlohmann::json& explicit_constraints,
                                   const nlohmann::json& explicit_requirements = nlohmann::json::object());

}  // namespace uf
