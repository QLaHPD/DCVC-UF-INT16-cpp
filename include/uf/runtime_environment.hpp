#pragma once

#include <nlohmann/json.hpp>

#include <string>

namespace uf {

// Record the linked Torch/CUDA build and selected device for an archive.
// This helper creates no tensors and loads no checkpoints/models.
nlohmann::json runtime_environment(const std::string& device);

}  // namespace uf
