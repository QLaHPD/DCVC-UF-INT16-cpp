// Copyright (c) Microsoft Corporation. Licensed under the MIT License.
#pragma once
#include "uf/dcvc_hem.hpp"
namespace uf::hem::detail {
std::vector<std::int32_t> integers(const torch::Tensor& input);
torch::Tensor floating(const std::vector<std::int32_t>& values);
torch::nn::LeakyReLU leaky(double slope = 0.01);
torch::nn::Sequential prior(std::int64_t in, std::int64_t middle, std::int64_t out);
torch::Tensor scalar(double value);
std::uint32_t dimension(std::int64_t value);
}  // namespace uf::hem::detail
