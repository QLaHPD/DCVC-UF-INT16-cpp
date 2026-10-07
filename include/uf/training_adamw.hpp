// Native interface for the installed Python AdamW algorithm.
// Algorithm attribution and BSD terms: licenses/PyTorch.txt.
#pragma once

#include <ATen/core/ivalue.h>
#include <torch/optim/optimizer.h>
#include <torch/types.h>

#include <optional>
#include <tuple>
#include <utility>
#include <vector>

namespace uf::training {

struct AdamWOptions : torch::optim::OptimizerCloneableOptions<AdamWOptions> {
    explicit AdamWOptions(double learning_rate = 1e-3) : lr_(learning_rate) {}
    TORCH_ARG(double, lr) = 1e-3;
    TORCH_ARG(torch::Tensor, tensor_lr) = {};
public:
    using Betas = std::tuple<double, double>;
    TORCH_ARG(Betas, betas) = std::make_tuple(0.9, 0.999);
    TORCH_ARG(double, eps) = 1e-8;
    TORCH_ARG(double, weight_decay) = 1e-2;
    TORCH_ARG(bool, amsgrad) = false;
    TORCH_ARG(bool, maximize) = false;
    TORCH_ARG(std::optional<bool>, foreach) = std::nullopt;
    TORCH_ARG(bool, capturable) = false;
    TORCH_ARG(bool, differentiable) = false;
    TORCH_ARG(std::optional<bool>, fused) = std::nullopt;
    // Preserve additional checkpoint param-group fields such as param_names.
    TORCH_ARG(c10::IValue, extra_options) = {};

public:
    double get_lr() const override;
    // The Python epoch schedules replace a saved tensor LR with a float.
    void set_lr(double learning_rate) override;
};

struct AdamWParamState : torch::optim::OptimizerCloneableParamState<AdamWParamState> {
    TORCH_ARG(torch::Tensor, step) = {};
    TORCH_ARG(torch::Tensor, exp_avg) = {};
    TORCH_ARG(torch::Tensor, exp_avg_sq) = {};
    TORCH_ARG(torch::Tensor, max_exp_avg_sq) = {};
    TORCH_ARG(c10::IValue, extra_state) = {};
};

class AdamW : public torch::optim::Optimizer {
public:
    explicit AdamW(std::vector<torch::Tensor> parameters, AdamWOptions options = AdamWOptions{});
    explicit AdamW(std::vector<torch::optim::OptimizerParamGroup> groups,
                   AdamWOptions options = AdamWOptions{});
    torch::Tensor step(LossClosure closure = nullptr) override;

    // Python GradScaler supplies these only to a fused optimizer. The root
    // trainers do not use AMP; native AMP callers can supply the same scalars.
    void amp_scaling(std::optional<torch::Tensor> grad_scale,
                     std::optional<torch::Tensor> found_inf);

    // Python Optimizer retains saved state whose keys are not parameter IDs.
    // Keep that checkpoint data separately from LibTorch's TensorImpl map.
    const c10::IValue& unmapped_state() const noexcept { return unmapped_state_; }
    void unmapped_state(c10::IValue value) { unmapped_state_ = std::move(value); }

private:
    std::optional<torch::Tensor> grad_scale_, found_inf_;
    c10::IValue unmapped_state_;
};

// AdamW's scalar counters use Float64 only for a Float64 process default;
// fused counters always use Float32, matching the installed Python helper.
torch::ScalarType adamw_step_dtype(bool fused);

}  // namespace uf::training
