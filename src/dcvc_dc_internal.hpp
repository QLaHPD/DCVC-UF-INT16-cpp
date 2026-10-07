// Copyright (c) Microsoft Corporation. Licensed under the MIT License.
#pragma once
#include "uf/dcvc_dc.hpp"
namespace uf::dc::detail {
torch::Tensor fine_scales(const torch::Tensor& scale);
std::uint32_t dimension(std::int64_t value);
std::uint8_t quality_index(std::int64_t index);
}  // namespace uf::dc::detail
