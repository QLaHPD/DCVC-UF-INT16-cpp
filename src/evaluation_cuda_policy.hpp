#pragma once

#include <optional>

namespace uf::evaluation {

// Installed Torch's NVML device-count policy for CUDA_VISIBLE_DEVICES.
// A missing result requests the CUDA runtime fallback, including MIG masks.
// This performs discovery only when called by the application at runtime.
std::optional<int> evaluation_nvml_devices();

}  // namespace uf::evaluation
