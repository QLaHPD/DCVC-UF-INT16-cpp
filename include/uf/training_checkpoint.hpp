// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#pragma once

#include "uf/training_common.hpp"
#include "uf/training_adamw.hpp"

#include <filesystem>
#include <optional>
#include <vector>

namespace uf::training {

struct ExistingTraining {
    std::int64_t begin_epoch = 0;
    // A saved Python None means no optimizer restore. Resume metadata can
    // remain populated after failed network restoration, as in common.py.
    std::optional<torch::IValue> optimizer;
    std::optional<std::filesystem::path> status_path;
    bool pretrained = false;
};

// Last two entries in ascending stable binary64 getmtime order.
std::vector<std::filesystem::path> latest_status_paths(const std::filesystem::path& save_dir);
// Python get_state_dict: map storage to CPU, unwrap state_dict/net, strip module.
CheckpointState read_training_state(const std::filesystem::path& checkpoint);
ExistingTraining load_existing_weights(const std::filesystem::path& save_dir, torch::nn::Module& model,
    int rank = -1, const std::optional<std::filesystem::path>& pretrain_path = std::nullopt);
torch::IValue adamw_state_dict(const AdamW& optimizer);
// Cast nested parameter state, retain unmapped entries, then normalize scalar
// counters after committing state/groups. Loading does not check moment shapes.
void load_adamw_state(AdamW& optimizer, const torch::IValue& state);
void save_training_checkpoint(const std::filesystem::path& save_dir, const torch::nn::Module& model);
void save_training_status(const std::filesystem::path& save_dir, const torch::nn::Module& model,
                          const AdamW& optimizer, std::int64_t epoch);

}  // namespace uf::training
