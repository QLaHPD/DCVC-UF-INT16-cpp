// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#pragma once

#include <torch/torch.h>

#include <functional>
#include <stdexcept>
#include <vector>

namespace uf::training {

class ActivationCheckpointError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

using CheckpointInputs = std::vector<torch::Tensor>;
using CheckpointFunction = std::function<CheckpointInputs(const CheckpointInputs&)>;
using CheckpointTensorFunction = std::function<torch::Tensor(const CheckpointInputs&)>;

// This library registers its native autograd Engine factory during process
// initialization, before autograd creates worker threads. Its only override
// supplies C++ saved-tensor hooks in checkpoint scopes. Embedding a Python
// engine or replacing the factory after startup is outside this runtime.
void initialize_activation_checkpointing();

// Non-reentrant activation recomputation with preserve_rng_state=False.
// Callback captures should own only module children/non-tensor configuration;
// tensor inputs belong in arguments so SavedVariable can track their lifetime
// and input versions without retaining the checkpoint's output graph.
CheckpointInputs checkpoint_activations(CheckpointFunction function, const CheckpointInputs& arguments);
torch::Tensor checkpoint_tensor(CheckpointTensorFunction function, const CheckpointInputs& arguments);

// CkptModule's flag is ordinary configuration, not a state-dictionary entry.
class ActivationCheckpointModule : public torch::nn::Module {
public:
    void set_use_checkpointing(bool enabled = true);
    bool use_checkpointing() const noexcept { return use_checkpointing_; }
protected:
    torch::Tensor run_tensor(CheckpointTensorFunction function, const CheckpointInputs& arguments) const;
    CheckpointInputs run_tensors(CheckpointFunction function, const CheckpointInputs& arguments) const;
private:
    bool use_checkpointing_ = false;
};

}  // namespace uf::training
