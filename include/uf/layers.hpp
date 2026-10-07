#pragma once

#include "uf/prepared.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace uf {

// Dispatch the same ATen operators as Python F.pixel_shuffle/unshuffle.
// Shape checks, leading batch dimensions and output layouts belong to LibTorch.
torch::Tensor pixel_shuffle(const torch::Tensor& value, std::int64_t factor);
torch::Tensor pixel_unshuffle(const torch::Tensor& value, std::int64_t factor);

// Executes the module topology used by the Python INT16 Layers interpreter.
// Convolution parameters are looked up by their checkpoint module path.
class Layers {
public:
    // Retain device tensor handles and copy topology metadata. The prepared
    // container is only needed during construction; CPU handles share storage.
    explicit Layers(const PreparedModel& prepared, torch::Device device = torch::kCPU);

    // Match [qp:qp+1, :, None, None], including clamped slices and trailing
    // dimensions. QP addition is bounded at Int64 max before slicing.
    torch::Tensor parameter(const std::string& name, std::int64_t qp) const;
    const torch::Tensor& value(const std::string& name) const;
    // Feature/hyper/spatial/image-prior wrappers concatenate multiple inputs
    // on the channel axis; video prior fusion applies its separate QP rule.
    torch::Tensor run(const std::string& module, const std::vector<torch::Tensor>& args) const;
    std::vector<torch::Tensor> reconstruct(const torch::Tensor& feature,
                                           bool for_reset = false) const;

private:
    std::string variant_;
    std::map<std::string, torch::Tensor> values_;

    const torch::Tensor& tensor(const std::string& name) const;
    bool has(const std::string& name) const;
    bool has_child(const std::string& name) const;
    bool residual_shortcut(const std::string& name) const;
    torch::Tensor run_simple(const std::string& name, const torch::Tensor& input,
                             bool shortcut = false) const;
    torch::Tensor run_depth(const std::string& name, const torch::Tensor& input,
                            bool shortcut) const;
    torch::Tensor run_conv(const std::string& name, const torch::Tensor& input) const;
};

}  // namespace uf
