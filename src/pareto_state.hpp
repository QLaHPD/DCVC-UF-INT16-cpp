#pragma once

#include <nlohmann/json.hpp>

#include <filesystem>
#include <iosfwd>

namespace uf::pareto_detail {

// Keep integer settings.seed values as canonical decimal strings inside C++,
// and as unquoted integers in the Python-compatible file. nlohmann::json would
// otherwise lose seeds outside its 64-bit range. Saved Float64/Boolean seeds
// retain their JSON types when they compare equal to the requested integer.
nlohmann::json read_state(std::istream& input);
// Path.open/json.load scope: read/parse and explicitly check close before
// native seed normalization or resumed settings validation.
nlohmann::json read_state_file(const std::filesystem::path& path);
void write_state(std::ostream& output, const nlohmann::json& state);

// Compare settings using Python's JSON scalar/container equality. Seeds
// supplied by the CLI are integers. Python also considers an existing
// Boolean or exactly equal Float64 seed equal to that integer, without rounding
// a large integer through Float64. Preserve the saved seed's JSON type.
bool same_settings(const nlohmann::json& saved, const nlohmann::json& expected);

}  // namespace uf::pareto_detail
