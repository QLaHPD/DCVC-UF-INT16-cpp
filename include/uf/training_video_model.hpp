// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#pragma once

#include "uf/training_common.hpp"

#include <memory>

namespace uf::training {

struct TrainingStep {
    torch::Tensor loss;
    std::optional<LossInfo> info;
};

// The wrapper itself is not an nn.Module: the root state names remain exactly
// those of DMC, with no additional registration prefix.
class VideoTrainingModel {
public:
    explicit VideoTrainingModel(std::string model_structure);
    torch::nn::Module& module();
    const torch::nn::Module& module() const;
    std::shared_ptr<torch::nn::Module> module_ptr() const;
    const std::string& model_structure() const noexcept;
    std::int64_t frame_delay() const noexcept;
    void set_use_checkpointing(bool enabled = true);
    void clear_dpb();
    void add_reference(const torch::Tensor& frame);
    // Graph QP selection/broadcasting follows the underlying LibTorch operators.
    TrainingStep forward(const torch::Tensor& input, const torch::Tensor& qp,
        const torch::Tensor& lambdas, bool get_loss_info = false, std::int64_t current_poc = 0);
    TrainingStep forward(const std::vector<torch::Tensor>& groups, const torch::Tensor& qp,
        const torch::Tensor& lambdas, bool get_loss_info = false, std::int64_t current_poc = 0);

private:
    std::string structure_;
    std::shared_ptr<CompressionTrainingImpl> graph_;
};

}  // namespace uf::training
