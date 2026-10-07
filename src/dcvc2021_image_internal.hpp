#pragma once
#include "uf/dcvc2021_image.hpp"
namespace uf::dcvc2021::image::detail {
void move_module(torch::nn::Module& module, std::optional<torch::Device> device,
                 std::optional<torch::Dtype> dtype, bool non_blocking);
}
