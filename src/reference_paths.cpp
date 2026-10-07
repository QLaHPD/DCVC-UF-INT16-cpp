#include "uf/reference_paths.hpp"
#include "uf/setup_build_config.hpp"

#include <string>
#include <system_error>
#include <vector>

namespace uf {
namespace {

std::filesystem::path configured_reference() {
    return std::filesystem::path(std::string(setup_build::source_root)).parent_path()
        / "DCVC-UF-INT16";
}

std::vector<std::filesystem::path> candidates(const std::filesystem::path& relative) {
    std::vector<std::filesystem::path> result{relative};
    if (relative.is_absolute()) return result;

    // Both the old build directory and the UF preset's build/uf directory may
    // be used. Walking parents also supports a relocated pair of checkouts.
    const auto append_parents = [&](std::filesystem::path directory) {
        while (!directory.empty()) {
            result.push_back(directory / "DCVC-UF-INT16" / relative);
            const auto parent = directory.parent_path();
            if (parent == directory) break;
            directory = parent;
        }
    };
    append_parents(std::filesystem::current_path().parent_path());
    std::error_code error;
    const auto executable = std::filesystem::read_symlink("/proc/self/exe", error);
    if (!error) append_parents(executable.parent_path());
    result.push_back(configured_reference() / relative);
    return result;
}

}  // namespace

std::filesystem::path reference_path(const std::filesystem::path& relative) {
    for (const auto& candidate : candidates(relative)) {
        std::error_code error;
        if (std::filesystem::exists(candidate, error)) return candidate;
    }
    return relative;
}

std::filesystem::path checkpoint_root() {
    for (const auto& candidate : candidates("checkpoints")) {
        std::error_code error;
        if (std::filesystem::is_directory(candidate, error)) return std::filesystem::absolute(candidate);
    }
    return configured_reference() / "checkpoints";
}

}  // namespace uf
