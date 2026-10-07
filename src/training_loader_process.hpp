// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#pragma once

#include "uf/training_datasets.hpp"

#include <exception>
#include <functional>
#include <memory>

namespace uf::training::loader_detail {

struct ProcessBatch {
    std::size_t id;
    DatasetSample tensors;
    std::exception_ptr error;
};

// The controller thread owns lifetime; one transport thread calls fetch/check.
// stop() wakes socket waits without closing/reusing their descriptors.
class ProcessWorker {
public:
    ProcessWorker(const nlohmann::json& dataset, std::size_t worker,
                  std::int64_t base_seed, std::int64_t batch_size);
    ~ProcessWorker();
    ProcessWorker(const ProcessWorker&) = delete;
    ProcessWorker& operator=(const ProcessWorker&) = delete;
    ProcessBatch fetch(std::size_t id, const std::vector<std::int64_t>& indices,
                       const std::function<void()>& check);
    void check_alive();
    void stop(bool interrupt) noexcept;
    bool reap();
    void kill() noexcept;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace uf::training::loader_detail
