#include "uf/fp16_model.hpp"
#include "uf/archive.hpp"
#include "uf/prepare.hpp"

#include <torch/cuda.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <utility>

#ifdef UF_HAS_FP16
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include "dmci_proxy.h"
#include "dmc_hts_proxy.h"
#include "dmc_htl_proxy.h"
#include "dmc_ld_proxy.h"
#endif

namespace uf {

torch::Tensor fp16_parameter_to_device(const torch::Tensor& value, const torch::Device& device) {
    const auto format = value.dim() == 4 ? torch::MemoryFormat::ChannelsLast
                                        : torch::MemoryFormat::Contiguous;
    // A separate contiguous(channels_last) is a no-op for ambiguous 1x1
    // shapes. An explicit-format copy produces the leading dimension used
    // by CUTLASS (weight.stride(3)) while preserving the parameter values.
    return value.to(device, torch::kFloat16, false, true, format);
}

struct Fp16Model::Impl {
    std::string variant;
    torch::Device device;
#ifdef UF_HAS_FP16
    std::unique_ptr<DMCIProxy> image;
    std::unique_ptr<DMCHTSProxy> hts;
    std::unique_ptr<DMCHTLProxy> htl;
    std::unique_ptr<DMCLDProxy> ld;
#endif
    Impl(std::string selected, torch::Device target)
        : variant(std::move(selected)), device(std::move(target)) {}
};

bool fp16_compiled() {
#ifdef UF_HAS_FP16
    return true;
#else
    return false;
#endif
}

int fp16_compute_capability() {
#ifdef UF_HAS_FP16
    return UF_FP16_SM;
#else
    return 0;
#endif
}

bool fp16_available() {
#ifdef UF_HAS_FP16
    return torch::cuda::is_available();
#else
    return false;
#endif
}

Fp16Model::Fp16Model(const std::filesystem::path& checkpoint, std::string variant,
                     torch::Device device, double skip_threshold,
                     const std::optional<std::string>& expected_sha256)
    : impl_(std::make_unique<Impl>(std::move(variant), std::move(device))) {
    if (!impl_->device.is_cuda() ||
        (impl_->variant != "image" && impl_->variant != "hts" &&
         impl_->variant != "htl" && impl_->variant != "ld")) {
        throw std::invalid_argument("Invalid UF FP16 model settings; FP16 requires CUDA");
    }
#ifdef UF_HAS_FP16
    if (!fp16_available()) throw std::runtime_error("CUDA is unavailable for FP16 inference");
    if (!impl_->device.has_index()) {
        impl_->device = torch::Device(torch::kCUDA, c10::cuda::current_device());
    }
    c10::cuda::CUDAGuard guard(impl_->device);
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, impl_->device.index()) != cudaSuccess ||
        properties.major * 10 + properties.minor != UF_FP16_SM) {
        throw std::runtime_error("UF FP16 kernels were built for a different GPU compute capability");
    }
    torch::NoGradGuard no_grad;
    const auto source_sha = sha256_file(checkpoint);
    if (expected_sha256 && source_sha != *expected_sha256) {
        throw std::runtime_error("UF FP16 checkpoint differs from archive identity");
    }
    auto state = load_float_checkpoint(checkpoint, true, impl_->variant);
    if (sha256_file(checkpoint) != source_sha) {
        throw std::runtime_error("UF FP16 checkpoint changed while loading");
    }
    for (auto& [name, tensor] : state) {
        if (!tensor.is_floating_point()) continue;  // CDFs remain CPU INT32.
        tensor = fp16_parameter_to_device(tensor, impl_->device);
    }
    // Match the Python binding: extract Double, then narrow to the proxy's
    // Float32 argument without a finite/range restriction. In particular,
    // test_video's max(0, value) still permits positive infinity.
    const auto threshold = static_cast<float>(skip_threshold);
    if (impl_->variant == "image") {
        impl_->image = std::make_unique<DMCIProxy>();
        impl_->image->set_param(state, threshold);
    } else if (impl_->variant == "hts") {
        impl_->hts = std::make_unique<DMCHTSProxy>();
        impl_->hts->set_param(state, threshold);
    } else if (impl_->variant == "htl") {
        impl_->htl = std::make_unique<DMCHTLProxy>();
        impl_->htl->set_param(state, threshold);
    } else {
        impl_->ld = std::make_unique<DMCLDProxy>();
        impl_->ld->set_param(state, threshold);
    }
#else
    (void)checkpoint;
    (void)skip_threshold;
    (void)expected_sha256;
    throw std::runtime_error("UF FP16 native inference was not enabled in this build");
#endif
}

Fp16Model::~Fp16Model() {
#ifdef UF_HAS_FP16
    if (impl_ && impl_->device.has_index()) {
        c10::cuda::CUDAGuard guard(impl_->device);
        impl_.reset();
    }
#endif
}

Fp16Packet Fp16Model::compress(const torch::Tensor& input, int qp,
                              int padding_bottom, int padding_right,
                              bool reset_feature_memory) {
    if (qp < 0 || qp >= 64 || padding_bottom < 0 || padding_right < 0 ||
        input.dim() != 4 || input.size(0) != 1 ||
        input.size(1) != (impl_->variant == "image" || impl_->variant == "ld" ? 3 : 24) ||
        input.device() != impl_->device || input.scalar_type() != torch::kFloat16) {
        throw std::invalid_argument("Invalid UF FP16 input tensor or QP");
    }
#ifdef UF_HAS_FP16
    c10::cuda::CUDAGuard guard(impl_->device);
    torch::NoGradGuard no_grad;
    Fp16Packet result;
    int parallel = 0;
    if (impl_->image) {
        std::tie(result.payload, result.reconstruction, parallel) =
            impl_->image->compress(input, qp, padding_bottom, padding_right);
    } else if (impl_->hts) {
        std::tie(result.payload, parallel) = impl_->hts->compress(
            input, qp, reset_feature_memory, padding_bottom, padding_right);
    } else if (impl_->htl) {
        std::tie(result.payload, parallel) = impl_->htl->compress(
            input, qp, reset_feature_memory, padding_bottom, padding_right);
    } else {
        std::tie(result.payload, parallel) = impl_->ld->compress(
            input, qp, reset_feature_memory, padding_bottom, padding_right);
    }
    if (result.payload.empty() || parallel < 1 || parallel > 8) {
        throw std::runtime_error("UF FP16 model returned an invalid coded packet");
    }
    result.entropy_parallel = static_cast<std::uint8_t>(parallel);
    return result;
#else
    (void)reset_feature_memory;
    throw std::runtime_error("UF FP16 native inference is unavailable");
#endif
}

std::vector<torch::Tensor> Fp16Model::decompress(
    const std::vector<std::uint8_t>& payload, std::uint32_t width,
    std::uint32_t height, int qp, int entropy_parallel, bool reset_feature_memory) {
    if (payload.empty() || payload.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        !width || !height ||
        width > static_cast<std::uint32_t>(std::numeric_limits<int>::max() - 64) ||
        height > static_cast<std::uint32_t>(std::numeric_limits<int>::max() - 64) ||
        qp < 0 || qp >= 64 || entropy_parallel < 1 || entropy_parallel > 8) {
        throw std::invalid_argument("Invalid UF FP16 coded packet");
    }
#ifdef UF_HAS_FP16
    c10::cuda::CUDAGuard guard(impl_->device);
    torch::NoGradGuard no_grad;
    if (impl_->image) return {impl_->image->decompress(payload, qp, height, width, entropy_parallel)};
    if (impl_->hts) return impl_->hts->decompress(
        payload, qp, height, width, entropy_parallel, reset_feature_memory);
    if (impl_->htl) return impl_->htl->decompress(
        payload, qp, height, width, entropy_parallel, reset_feature_memory);
    return {impl_->ld->decompress(payload, qp, height, width, entropy_parallel, reset_feature_memory)};
#else
    (void)reset_feature_memory;
    throw std::runtime_error("UF FP16 native inference is unavailable");
#endif
}

void Fp16Model::add_reference(const torch::Tensor& frame, bool apply_adaptor) {
    if (impl_->variant == "image" || frame.dim() != 4 || frame.size(0) != 1 ||
        frame.size(1) != 3 || frame.device() != impl_->device ||
        frame.scalar_type() != torch::kFloat16) {
        throw std::invalid_argument("Invalid UF FP16 reference frame");
    }
#ifdef UF_HAS_FP16
    c10::cuda::CUDAGuard guard(impl_->device);
    torch::NoGradGuard no_grad;
    if (impl_->hts) impl_->hts->add_ref_feature_from_frame(frame, apply_adaptor);
    else if (impl_->htl) impl_->htl->add_ref_feature_from_frame(frame, apply_adaptor);
    else impl_->ld->add_ref_feature_from_frame(frame, apply_adaptor);
#else
    (void)apply_adaptor;
    throw std::runtime_error("UF FP16 native inference is unavailable");
#endif
}

}  // namespace uf
