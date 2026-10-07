// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#pragma once

#include "uf/training_datasets.hpp"

#include <memory>
#include <optional>

namespace uf::training {

struct TrainingLoaderOptions {
    std::int64_t batch_size = 16;  // Total over all ranks, like the Python CLI.
    std::size_t workers = 4;
    std::size_t prefetch_factor = 2;
    int rank = -1;
    std::int64_t world_size = 1;
    torch::Device device{torch::kCPU};
    bool pin_memory = true;
};

// Image/VideoFolder operations use fresh executable worker processes with
// private CPython random streams and one Torch CPU thread each. The generic
// callback interface uses threads because callbacks cannot be sent through exec.
// Batches are assigned round-robin and delivered in sampler order.
// Datasets must outlive the loader; reset/next belong to one controller thread.
class TrainingDataLoader {
public:
    using SampleReader = std::function<DatasetSample(std::int64_t, const RandomBelow&)>;
    TrainingDataLoader(std::size_t dataset_size, SampleReader reader, TrainingLoaderOptions options = {});
    TrainingDataLoader(ImageFolder& dataset, TrainingLoaderOptions options = {});
    TrainingDataLoader(VideoFolder& dataset, TrainingLoaderOptions options = {});
    ~TrainingDataLoader();
    TrainingDataLoader(const TrainingDataLoader&) = delete;
    TrainingDataLoader& operator=(const TrainingDataLoader&) = delete;

    // Ends the previous epoch and records DistributedSampler's signed epoch. Starting
    // workers/sampling is lazy, after the epoch function changes patch settings.
    void reset(std::int64_t epoch);
    std::optional<DatasetSample> next();
    std::size_t batch_count() const;
    std::int64_t local_batch_size() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Private executable entry point; descriptor 3 is the controller socket.
int run_training_loader_worker();

}  // namespace uf::training
