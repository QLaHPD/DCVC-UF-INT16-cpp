// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <memory>
#include <string>

namespace uf::training {

struct TrainingRank {
    int rank = -1;
    int world_size = 1;
    std::string master_address = "localhost";
    std::uint16_t master_port = 12355;
};

bool training_nccl_available() noexcept;

// Pure C++ c10d ProcessGroupNCCL and the same Reducer used by Python DDP.
// A rank of -1 is a local trainer with no process group or parameter hooks.
// The model must keep its parameter identities/device after attach_model().
class TrainingDistributed {
public:
    explicit TrainingDistributed(TrainingRank rank = {});
    ~TrainingDistributed();
    TrainingDistributed(const TrainingDistributed&) = delete;
    TrainingDistributed& operator=(const TrainingDistributed&) = delete;
    void barrier();
    void attach_model(torch::nn::Module& model, bool find_unused_parameters,
                      const std::string& module_name);
    void before_forward();
    void prepare_backward(const torch::Tensor& loss);
    void after_backward();
    // Clean shutdown after the final save. Exceptional destruction aborts
    // communicators so surviving ranks do not wait for a failed trainer.
    void finish();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace uf::training
