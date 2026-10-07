#include "uf/int16_ops.hpp"
#include "uf/process.hpp"
#include <ATen/TensorIndexing.h>
#include <ATen/ops/matmul.h>
#include <torch/nn/functional/padding.h>
#ifdef UF_HAS_CUDA
#include "uf/cuda_ops.hpp"
#endif

#include <limits>
#include <stdexcept>

namespace uf::int16 {
namespace {

torch::Tensor parameter_values(const torch::Tensor& value) {
    auto values = value.detach().to(torch::kCPU).to(torch::kFloat64);
    if (!torch::isfinite(values).all().item<bool>()) {
        throw std::invalid_argument("Cannot prepare non-finite checkpoint parameters");
    }
    return values;
}

}  // namespace

bool cuda_compiled() {
#ifdef UF_HAS_CUDA
    return true;
#else
    return false;
#endif
}

torch::Tensor saturate(const torch::Tensor& value) {
    return torch::clamp(value, -32768, 32767).to(torch::kInt16);
}

torch::Tensor quantize_parameter(const torch::Tensor& value, double scale) {
    const auto values = parameter_values(value);
    return saturate(torch::round(values * scale).to(torch::kInt64));
}

torch::Tensor quantize_parameter(const torch::Tensor& value, const torch::Tensor& scale) {
    const auto values = parameter_values(value);
    return saturate(torch::round(values * scale).to(torch::kInt64));
}

torch::Tensor divide_round(const torch::Tensor& value, std::int64_t denominator) {
    const auto wide = value.to(torch::kInt64);
    // Python denominator // 2 floors negative odd integers. Let torch.div
    // handle zero and other divisor failures as in the tensor overload.
    const auto half = denominator / 2 - (denominator < 0 && denominator % 2 != 0 ? 1 : 0);
    const auto magnitude = torch::div(wide.abs() + half, denominator,
                                      "floor");
    return torch::where(wide < 0, -magnitude, magnitude);
}

torch::Tensor divide_round(const torch::Tensor& value, const torch::Tensor& denominator) {
    const auto wide = value.to(torch::kInt64);
    const auto half = torch::div(denominator, 2, "floor");
    const auto magnitude = torch::div(wide.abs() + half, denominator, "floor");
    return torch::where(wide < 0, -magnitude, magnitude);
}

torch::Tensor add(const std::vector<torch::Tensor>& values) {
    if (values.empty()) throw std::invalid_argument("INT16 add requires an input");
#ifdef UF_HAS_CUDA
    if (values[0].is_cuda() && values.size() >= 2 && values.size() <= 4) {
        return uf_add(values);
    }
#endif
    auto sum = values[0].to(torch::kInt64);
    for (std::size_t i = 1; i < values.size(); ++i) sum = sum + values[i].to(torch::kInt64);
    return saturate(sum);
}

torch::Tensor multiply(const torch::Tensor& value, const torch::Tensor& scale) {
#ifdef UF_HAS_CUDA
    if (value.is_cuda()) return uf_mul(value, scale);
#endif
    return saturate(divide_round(value.to(torch::kInt64) * scale.to(torch::kInt64),
                                  feature_scale));
}

torch::Tensor reciprocal(const torch::Tensor& value) {
    auto wide = value.to(torch::kInt64).clamp_min(feature_scale / 2);
    auto numerator = torch::full_like(wide, feature_scale * feature_scale);
    return saturate(divide_round(numerator, wide));
}

torch::Tensor symbols(const torch::Tensor& value) {
    return divide_round(value, feature_scale).clamp(-128, 127).to(torch::kInt8);
}

torch::Tensor from_symbols(const torch::Tensor& value) {
    return saturate(value.to(torch::kInt64) * feature_scale);
}

torch::Tensor from_bytes(const torch::Tensor& value) {
#ifdef UF_HAS_CUDA
    if (value.is_cuda()) return uf_bytes(value);
#endif
    return saturate(divide_round((2 * value.to(torch::kInt64) - 255) * feature_scale,
                                  510));
}

torch::Tensor lookup(const torch::Tensor& value, const torch::Tensor& table) {
#ifdef UF_HAS_CUDA
    if (value.is_cuda()) return uf_lut(value, table);
#endif
    // Python lut[x.long()+32768] indexes the first table dimension. Keep
    // index shape, trailing table axes, negative-index and empty semantics.
    const auto indexes = value.to(torch::kInt64) + 32768;
    return table.index({indexes});
}

torch::Tensor wsilu4(const torch::Tensor& value, const torch::Tensor& table) {
#ifdef UF_HAS_CUDA
    if (value.is_cuda()) return uf_wsilu4(value, table);
#endif
    const auto activated = lookup(value, table);
    using at::indexing::None;
    using at::indexing::Slice;
    // Match out[:, index::4], including trailing axes and the add helper's
    // broadcasting when channel counts do not divide evenly into four slices.
    return add({activated.index({Slice(), Slice(0, None, 4)}),
                activated.index({Slice(), Slice(1, None, 4)}),
                activated.index({Slice(), Slice(2, None, 4)}),
                activated.index({Slice(), Slice(3, None, 4)})});
}

torch::Tensor conv2d_reference(const torch::Tensor& input, const torch::Tensor& weight,
                               const torch::Tensor& bias, std::int64_t stride_h,
                               std::int64_t stride_w, std::int64_t pad_h,
                               std::int64_t pad_w, std::int64_t groups) {
    if (!input.device().is_cpu() || !weight.device().is_cpu()) {
        throw std::invalid_argument("Reference convolution requires CPU tensors");
    }
    if (input.dim() != 4 || weight.dim() != 4 || input.scalar_type() != torch::kInt16 ||
        weight.scalar_type() != torch::kInt16 || stride_h <= 0 || stride_w <= 0 || groups <= 0) {
        throw std::invalid_argument("Unsupported INT16 convolution arguments");
    }
    ProcessInterrupt::check();
    namespace F = torch::nn::functional;
    const auto padded = F::pad(input, F::PadFuncOptions({pad_w, pad_w, pad_h, pad_h}));
    const auto kernel_h = weight.size(2), kernel_w = weight.size(3);
    auto patches = padded.unfold(2, kernel_h, stride_h).unfold(3, kernel_w, stride_w);
    const auto batch = patches.size(0), channels = patches.size(1);
    const auto height = patches.size(2), width = patches.size(3);
    const auto output_channels = weight.size(0);
    if (height && width > std::numeric_limits<std::int64_t>::max() / height) {
        throw std::overflow_error("INT16 reference convolution spatial size exceeds Int64");
    }
    patches = patches.reshape({batch, groups, channels / groups, height, width, kernel_h, kernel_w})
        .permute({0, 1, 3, 4, 2, 5, 6}).reshape({batch, groups, height * width, -1}).to(torch::kInt64);
    const auto kernels = weight.reshape({groups, output_channels / groups, -1})
        .transpose(1, 2).to(torch::kInt64);
    ProcessInterrupt::check();
    auto result = torch::matmul(patches, kernels)
        .reshape({batch, groups, height, width, output_channels / groups})
        .permute({0, 1, 4, 2, 3}).reshape({batch, output_channels, height, width});
    result = divide_round(result, weight_scale);
    if (bias.defined()) {
        using at::indexing::None;
        using at::indexing::Slice;
        result += bias.to(torch::kInt64).index({None, Slice(), None, None});
    }
    auto output = saturate(result);
    ProcessInterrupt::check();
    return output;
}

torch::Tensor conv2d(const torch::Tensor& input, const torch::Tensor& weight,
                     const torch::Tensor& bias, std::int64_t stride_h,
                     std::int64_t stride_w, std::int64_t pad_h,
                     std::int64_t pad_w, std::int64_t groups) {
    if (input.device().is_cpu()) {
        return conv2d_reference(input, weight, bias, stride_h, stride_w, pad_h, pad_w, groups);
    }
    if (input.dim() != 4 || weight.dim() != 4 || input.scalar_type() != torch::kInt16 ||
        weight.scalar_type() != torch::kInt16 || stride_h <= 0 || stride_w <= 0 ||
        pad_h < 0 || pad_w < 0 || groups <= 0) {
        throw std::invalid_argument("Unsupported INT16 convolution arguments");
    }
#ifdef UF_HAS_CUDA
    if (input.is_cuda()) {
        const auto native_max = std::numeric_limits<int>::max();
        if (stride_h > native_max || stride_w > native_max || pad_h > native_max ||
            pad_w > native_max || groups > native_max) {
            throw std::overflow_error("INT16 CUDA convolution arguments exceed native int range");
        }
        const auto optional_bias = bias.defined()
            ? c10::optional<torch::Tensor>(bias) : c10::nullopt;
        return uf_conv(input, weight, optional_bias, static_cast<int>(stride_h),
                       static_cast<int>(stride_w), static_cast<int>(pad_h),
                       static_cast<int>(pad_w), static_cast<int>(groups));
    }
#endif
    // Retain the CPU-copy fallback for non-CPU tensors when CUDA is not built.
    const auto device = input.device();
    const auto b = bias.defined() ? bias.to(torch::kCPU) : torch::Tensor{};
    return conv2d_reference(input.to(torch::kCPU), weight.to(torch::kCPU), b,
                            stride_h, stride_w, pad_h, pad_w, groups).to(device);
}

}  // namespace uf::int16
