#include "uf/runtime_environment.hpp"
#include "uf/setup_build_config.hpp"

#include <ATen/detail/CUDAHooksInterface.h>
#include <torch/types.h>
#include <torch/version.h>
#ifdef UF_HAS_ENVIRONMENT_CUDA
#include <cuda_runtime_api.h>
#endif

#include <stdexcept>
#include <string>

namespace uf {

nlohmann::json runtime_environment(const std::string& device) {
    auto version = std::string(setup_build::torch_package_version);
    if (version.empty()) version = TORCH_VERSION;
    nlohmann::json cuda = nullptr;
    if (setup_build::torch_cuda_version_known) {
        if (!setup_build::torch_cuda_version.empty()) cuda = std::string(setup_build::torch_cuda_version);
    } else {
        const auto& hooks = at::detail::getCUDAHooks();
        if (hooks.hasCUDART()) {
            const auto compiled = hooks.versionCUDART();
            cuda = std::to_string(compiled / 1000) + "." + std::to_string((compiled % 1000) / 10);
        }
    }
    const torch::Device selected(device);
    std::string gpu;
    if (selected.is_cpu()) {
        gpu = "CPU reference";
    } else if (selected.is_cuda()) {
#ifdef UF_HAS_ENVIRONMENT_CUDA
        const auto check = [](cudaError_t status) {
            if (status != cudaSuccess) {
                throw std::runtime_error(std::string("Cannot inspect archive CUDA device: ") + cudaGetErrorString(status));
            }
        };
        int index = selected.has_index() ? static_cast<int>(selected.index()) : -1;
        if (index < 0) check(cudaGetDevice(&index));
        cudaDeviceProp properties{};
        check(cudaGetDeviceProperties(&properties, index));
        gpu = properties.name;
#else
        throw std::runtime_error("Archive CUDA device metadata requires CUDA runtime development files at build time");
#endif
    } else {
        throw std::invalid_argument("UF archive environment requires a CPU or CUDA device");
    }
    return {{"torch", version}, {"cuda", cuda}, {"gpu", gpu}};
}

}  // namespace uf
