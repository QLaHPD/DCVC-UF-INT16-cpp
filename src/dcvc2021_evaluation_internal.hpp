// Copyright (c) Microsoft Corporation. Licensed under the MIT License.
#pragma once
#include "uf/evaluation_json.hpp"
#include <torch/torch.h>

namespace uf::dcvc2021 {
double evaluation_psnr(const torch::Tensor& reconstruction, const torch::Tensor& original);
double evaluation_msssim(const torch::Tensor& reconstruction, const torch::Tensor& original);
uf::evaluation::Json evaluate_document(const uf::evaluation::Json& point, bool use_cuda, int worker_index);
}  // namespace uf::dcvc2021
