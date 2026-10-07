#include "uf/setup.hpp"
#include "uf/setup_build_config.hpp"
#include "uf/fp16_model.hpp"
#include "uf/int16_ops.hpp"
#include "uf/media.hpp"

#include <ATen/Version.h>
#include <torch/cuda.h>
#include <torch/version.h>
#ifdef UF_HAS_SETUP_CUDA
#include <cuda_runtime_api.h>
#endif

#include <algorithm>
#include <ostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace uf {

int report_setup(std::ostream& output, const std::filesystem::path& checkpoints,
                 bool require_native, bool require_int16_native) {
    const auto checked_output = [&output] {
        if (!output) throw std::runtime_error("Cannot write UF setup report");
    };
    const auto torch_version = setup_build::torch_package_version.empty()
        ? std::string_view(TORCH_VERSION) : setup_build::torch_package_version;
    output << "UF C++ checkout: " << setup_build::source_root << '\n';
    checked_output();
    output << "C++: " << setup_build::compiler << " ("
           << std::filesystem::read_symlink("/proc/self/exe").string() << ")\n";
    checked_output();
    output << "LibTorch: " << torch_version << '\n';
    checked_output();
    output << "LibTorch build configuration:\n" << at::show_config() << '\n';
    checked_output();

    const bool cuda = torch::cuda::is_available();
    bool fp16_ready = fp16_compiled() && cuda;
    const bool integer_ready = int16::cuda_compiled() && cuda;
    if (!cuda) {
        output << "CUDA: unavailable\n";
    } else {
#ifdef UF_HAS_SETUP_CUDA
        cudaDeviceProp properties{};
        const auto status = cudaGetDeviceProperties(&properties, 0);
        if (status != cudaSuccess) {
            throw std::runtime_error(std::string("Cannot inspect CUDA device 0: ") +
                                     cudaGetErrorString(status));
        }
        const int capability = properties.major * 10 + properties.minor;
        output << "CUDA: " << properties.name << " (device 0, SM "
               << capability << ")\n";
        // The standalone FP16 model requires the exact GPU target selected
        // at build time, including on machines with multiple GPU types.
        if (fp16_compiled() && capability != fp16_compute_capability()) {
            fp16_ready = false;
            output << "UF FP16 GPU target: mismatch; compiled for SM "
                   << fp16_compute_capability() << '\n';
        }
#else
        output << "CUDA: available (device name requires CUDA runtime headers at build time)\n";
#endif
    }
    checked_output();
    output << "Arithmetic: FP16 default; explicit --runtime int16 selects the prepared integer runtime\n"
           << "UF entropy: linked native rANS encoder/decoder\n"
           << "UF FP16 proxies: " << (fp16_ready ? "OK" : "NOT READY");
    if (!fp16_compiled()) output << " (backend was not compiled)";
    else if (!cuda) output << " (CUDA unavailable)";
    output << '\n' << "UF INT16 CUDA: " << (integer_ready ? "OK" : "NOT READY");
    if (!int16::cuda_compiled()) output << " (kernels were not compiled)";
    else if (!cuda) output << " (CUDA unavailable)";
    output << '\n' << "Integer arithmetic: " << int16::arithmetic_id << '\n';
    if (int16::cuda_compiled()) {
        output << "Integer kernel revision: " << int16::kernel_revision << '\n';
    }
    checked_output();

    const std::filesystem::path cutlass{std::string(setup_build::cutlass_root)};
    if (std::filesystem::exists(cutlass / "include/cutlass/cutlass.h")) {
        ChildPipe git({"git", "-C", cutlass.string(), "rev-parse", "--short", "HEAD"});
        auto revision = git.read_all();
        const auto first = revision.find_first_not_of(" \t\n\r\f\v");
        const auto last = revision.find_last_not_of(" \t\n\r\f\v");
        revision = first == std::string::npos ? std::string{}
                                            : revision.substr(first, last - first + 1);
        output << "CUTLASS: " << revision << " (" << cutlass.string() << ")\n";
    } else {
        output << "CUTLASS: missing (" << cutlass.string() << "); see DEPENDENCIES.md\n";
    }
    checked_output();

    std::vector<std::string> names;
    for (const auto& entry : std::filesystem::directory_iterator(checkpoints)) {
        const auto name = entry.path().filename().string();
        if (entry.is_regular_file() && name != ".gitkeep" && name != "README.md") {
            names.push_back(name);
        }
    }
    std::sort(names.begin(), names.end());
    output << "Checkpoint directory: " << checkpoints.string() << '\n'
           << "Checkpoint files: ";
    if (names.empty()) output << "awaiting user-provided models";
    for (std::size_t index = 0; index < names.size(); ++index) {
        if (index) output << ", ";
        output << names[index];
    }
    output << "\nBackend readiness checks do not establish encode/decode or cross-device parity.\n";
    checked_output();
    return (require_native && !fp16_ready) || (require_int16_native && !integer_ready) ? 1 : 0;
}

}  // namespace uf
