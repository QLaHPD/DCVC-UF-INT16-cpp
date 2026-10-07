// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#pragma once

#include "uf/training_datasets.hpp"
#include "uf/training_image_model.hpp"
#include "uf/training_video_model.hpp"
#include "uf/training_adamw.hpp"

#include <functional>
#include <iostream>
#include <optional>

namespace uf::training {

struct EpochContext {
    int rank = -1;
    std::int64_t world_size = 1;
    // Fresh iterator for this epoch. The loader owns shuffling/prefetch,
    // drop-last collation and worker teardown.
    std::function<std::optional<DatasetSample>()> next_batch;
    // Native DDP calls these around forward and before backward so its
    // parameter hooks can reduce gradients through LibTorch's Reducer.
    std::function<void()> before_forward;
    std::function<void(const torch::Tensor&)> prepare_backward;
    // Required for multiple ranks; ensure averaged gradients before clipping.
    // An autograd reducer may already have completed the actual reduction.
    std::function<void(torch::nn::Module&)> reduce_gradients;
    // Optional provider override. Without it, the scheduler sets the graph's
    // native non-reentrant recomputation flag at the Python module boundaries.
    std::function<void(VideoTrainingModel&, bool)> configure_checkpointing;
};

struct EpochResult {
    std::uint64_t batches = 0;
    std::uint64_t updates = 0;
    std::uint64_t skipped_updates = 0;
};

EpochResult train_image_epoch(DmciTrainingImpl& model, ImageFolder& dataset,
    AdamW& optimizer, std::int64_t epoch, const EpochContext& context,
    std::ostream& log = std::cout);
EpochResult train_video_epoch(VideoTrainingModel& model, DmciTrainingImpl& intra_model,
    VideoFolder& dataset, AdamW& optimizer, std::int64_t epoch,
    const std::string& scheduling, const EpochContext& context, std::ostream& log = std::cout);

}  // namespace uf::training
