#include "inference_environment.hpp"

#include <ATen/Context.h>
#include <torch/utils.h>

#include <cstdlib>
#include <stdexcept>

namespace uf::detail {

void set_inference_environment() {
    if (setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8", 1) != 0) {
        throw std::runtime_error("Cannot configure deterministic UF inference");
    }
    auto& context = at::globalContext();
    context.setUserEnabledCuDNN(true);
    context.setBenchmarkCuDNN(false);
    context.setDeterministicAlgorithms(true, false);
    torch::manual_seed(0);
    torch::set_num_threads(1);
    context.setDeterministicFillUninitializedMemory(false);
}

}  // namespace uf::detail
