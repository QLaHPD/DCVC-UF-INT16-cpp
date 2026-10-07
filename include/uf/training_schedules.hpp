// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace uf::training {

struct ImageTrainingStage {
    std::int64_t reference_epoch;
    double learning_rate;
    std::int64_t patch_width, patch_height;
};

struct VideoTrainingStage {
    std::int64_t reference_epoch;
    double learning_rate;
    std::int64_t frame_count;
    bool cascaded;
    std::int64_t patch_width, patch_height;
    bool use_checkpoint() const noexcept { return cascaded && patch_width > 256; }
};

std::vector<ImageTrainingStage> image_training_strategy();
std::vector<VideoTrainingStage> video_training_strategy(
    const std::string& scheduling, std::int64_t frame_delay, const std::string& model_structure);
// Nonnegative epochs clamp to the final row. Negative epochs index from the
// end, following rows[min(len(rows) - 1, epoch)], with checked lower bounds.
ImageTrainingStage image_training_stage(std::int64_t epoch);
VideoTrainingStage video_training_stage(std::int64_t epoch, const std::string& scheduling,
    std::int64_t frame_delay, const std::string& model_structure);
std::vector<double> training_lambdas(const std::vector<double>& endpoints, std::int64_t qp_count);

}  // namespace uf::training
