#pragma once

namespace uf::detail {

// Apply the Torch part of src.utils.common.set_torch_env on every codec
// construction, before creating model tensors or selecting a CUDA stream.
// Codec inference does not use NumPy's random generator.
void set_inference_environment();

}  // namespace uf::detail
