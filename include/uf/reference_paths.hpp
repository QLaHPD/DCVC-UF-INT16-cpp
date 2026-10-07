#pragma once

#include <filesystem>

namespace uf {

// Resolve the reference checkout's resources without assuming the executable's
// build-directory depth. Working-directory resources retain first precedence.
std::filesystem::path reference_path(const std::filesystem::path& relative);
std::filesystem::path checkpoint_root();

}  // namespace uf
