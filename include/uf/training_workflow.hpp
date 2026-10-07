// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#pragma once

#include "uf/training_distributed.hpp"
#include "uf/training_video_model.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace uf::training {

struct CommonTrainingOptions {
    std::int64_t batch_size = 16;
    // Keep the signed CLI value until runtime validation, like argparse's int.
    std::int64_t num_workers = 4;
    std::vector<double> lambdas;
    std::filesystem::path save_dir, train_dataset;
};

struct ImageTrainingOptions : CommonTrainingOptions {
    std::int64_t epochs = 104;
};

struct VideoTrainingOptions : CommonTrainingOptions {
    std::int64_t epochs = 100;
    std::filesystem::path model_path_i;
    std::string model_structure = "ld", training_scheduling = "stage0";
    std::optional<std::filesystem::path> pretrain_path;
};

using CheckpointingProvider = std::function<void(VideoTrainingModel&, bool)>;

void train_image(const ImageTrainingOptions& options, TrainingRank rank = {});
void train_video(const VideoTrainingOptions& options, TrainingRank rank = {},
                 CheckpointingProvider checkpointing = {});

// argc/argv use the dcvc-uf subcommand at argv[1]. The public command launches
// fresh executable ranks on all visible GPUs when more than one is present.
// Private worker invocations receive their rank/world size as arguments.
int run_training_command(int argc, char** argv, bool video, bool private_worker = false);

}  // namespace uf::training
