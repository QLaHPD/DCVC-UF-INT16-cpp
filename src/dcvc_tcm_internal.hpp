// Copyright (c) Microsoft Corporation. Licensed under the MIT License.
#pragma once
#include "uf/dcvc_tcm.hpp"

namespace uf::tcm::detail {
torch::nn::Conv2d conv(std::int64_t in, std::int64_t out, std::int64_t kernel = 3, std::int64_t stride = 1);
torch::nn::ConvTranspose2d deconv(std::int64_t in, std::int64_t out, std::int64_t stride = 2);
torch::nn::Sequential subpel(std::int64_t in, std::int64_t out);
torch::nn::LeakyReLU leaky(double slope = 0.01);
std::vector<std::int32_t> integers(const torch::Tensor& input);
torch::Tensor floating(const std::vector<std::int32_t>& input);
CdfTables make_tables(const torch::Tensor& pmf, const torch::Tensor& tails,
                      const torch::Tensor& lengths, const torch::Tensor& offsets);
}  // namespace uf::tcm::detail
