#pragma once

#include "uf/prepare.hpp"

namespace uf {

// Mirrors state_dict names/shapes in the reference image and video modules.
// Source tensor dtype is converted by the loader's copy into real parameters.
void validate_checkpoint_state(const CheckpointState& state, const std::string& variant);

}  // namespace uf
