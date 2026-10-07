#pragma once

#include <torch/torch.h>

#include <vector>

// These symbols adapt the Python UF INT16 native extension. This header is
// shared by callers and CUDA definitions; the pybind module is excluded.
// All convolution variants use native int geometry and Int32 indexing limits.
torch::Tensor uf_conv(const torch::Tensor&, const torch::Tensor&,
                      const c10::optional<torch::Tensor>&,
                      int, int, int, int, int);
torch::Tensor uf_conv_generic(const torch::Tensor&, const torch::Tensor&,
                              const c10::optional<torch::Tensor>&,
                              int, int, int, int, int);
torch::Tensor uf_conv_baseline(const torch::Tensor&, const torch::Tensor&,
                               const c10::optional<torch::Tensor>&,
                               int, int, int, int, int);
torch::Tensor uf_add(const std::vector<torch::Tensor>&);
torch::Tensor uf_mul(const torch::Tensor&, const torch::Tensor&);
torch::Tensor uf_lut(const torch::Tensor&, const torch::Tensor&);
torch::Tensor uf_bytes(const torch::Tensor&);
torch::Tensor uf_wsilu4(const torch::Tensor&, const torch::Tensor&);
