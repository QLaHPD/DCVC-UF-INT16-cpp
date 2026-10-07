#pragma once

#include "uf/prepared.hpp"

#include <filesystem>
#include <map>
#include <optional>
#include <string>

namespace uf {

struct PreparationResult {
    std::filesystem::path path;
    std::string variant;
    std::string identity;
};

struct ResolvedPreparedModel {
    std::filesystem::path path;
    PreparedModel data;
};

using CheckpointState = std::map<std::string, torch::Tensor>;

// Copy checkpoint weights into fresh CPU real tensors using the process default
// floating dtype, as Python's load_state_dict does for newly constructed modules.
// Optional entropy tables follow Python's default-dtype/explicit-Float32 casts
// before inference parameters are converted to FP16 on CUDA. Loading does not
// reject non-finite weights; integer quantization owns its finite-input check.
CheckpointState load_float_checkpoint(const std::filesystem::path& checkpoint,
                                      bool add_entropy_tables = false,
                                      const std::optional<std::string>& variant = std::nullopt);

PreparationResult prepare_checkpoint(const std::filesystem::path& checkpoint,
                                     const std::string& variant,
                                     std::optional<std::filesystem::path> output = std::nullopt);

// Resolve the same artifact location without retaining its tensors. Existing
// artifacts are validated by the calling workflow's load_prepared/model load;
// missing explicit paths are prepared at that path, as in Python's resolver.
std::filesystem::path prepared_artifact_path(
    const std::filesystem::path& checkpoint, const std::string& variant,
    std::optional<std::filesystem::path> path = std::nullopt,
    const std::optional<std::string>& expected_source = std::nullopt);

// Match Python resolve_prepared: an existing explicit artifact is self-contained;
// a missing artifact is prepared at the requested path using the float checkpoint.
ResolvedPreparedModel resolve_prepared(
    const std::filesystem::path& checkpoint, const std::string& variant,
    std::optional<std::filesystem::path> path = std::nullopt,
    const std::optional<std::string>& expected_identity = std::nullopt,
    const std::optional<std::string>& expected_source = std::nullopt);

}  // namespace uf
