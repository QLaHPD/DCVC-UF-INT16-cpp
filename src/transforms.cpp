// Copyright (c) Microsoft Corporation. Licensed under the MIT License.
// Spline zoom adapts SciPy 1.15.3; retained BSD notice: licenses/SciPy.txt.
#include "uf/transforms.hpp"
#include "uf_splines.hpp"
#include <ATen/TensorIndexing.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>
namespace uf::transforms {
namespace {
constexpr double kr = 0.2126, kg = 0.7152, kb = 0.0722;
// NumPy materializes every ufunc result in the array's dtype. In particular,
// a Float32 division must divide rather than multiply by an ATen reciprocal.
// Float16 arrays compute a ufunc in Float32 and store its rounded Half result.
template<class Number> struct ArrayArithmetic {
    using Compute = std::conditional_t<std::is_same_v<Number, double>, double, float>;
    static Number add(Number first, Number second) {
        volatile Compute result = static_cast<Compute>(first) + static_cast<Compute>(second);
        return static_cast<Number>(result);
    }
    static Number subtract(Number first, Number second) {
        volatile Compute result = static_cast<Compute>(first) - static_cast<Compute>(second);
        return static_cast<Number>(result);
    }
    static Number multiply(Number first, Number second) {
        volatile Compute result = static_cast<Compute>(first) * static_cast<Compute>(second);
        return static_cast<Number>(result);
    }
    static Number divide(Number first, Number second) {
        volatile Compute result = static_cast<Compute>(first) / static_cast<Compute>(second);
        return static_cast<Number>(result);
    }
};
double round_even(double value) {
    if (!std::isfinite(value) || value == 0) return value;
    const auto magnitude = std::abs(value), lower = std::floor(magnitude);
    const auto fraction = magnitude - lower;
    const auto rounded = lower + (fraction > 0.5 || (fraction == 0.5 && std::fmod(lower, 2.0) != 0.0) ? 1.0 : 0.0);
    return std::copysign(rounded, value);
}
template<class Number> torch::Tensor array_color(const torch::Tensor& input, bool quant) {
    using Math = ArrayArithmetic<Number>;
    const auto source = input.contiguous();
    auto output = torch::empty(source.sizes(), source.options());
    const auto* rgb = source.data_ptr<Number>();
    auto* yuv = output.data_ptr<Number>();
    const auto half = static_cast<Number>(0.5), zero = static_cast<Number>(0), one = static_cast<Number>(1);
    const auto wr = static_cast<Number>(kr), wg = static_cast<Number>(kg), wb = static_cast<Number>(kb);
    const auto cb_denominator = static_cast<Number>(1 - kb), cr_denominator = static_cast<Number>(1 - kr);
    const auto scale = static_cast<Number>(255.0);
    for (std::int64_t index = 0; index < source.numel(); index += 3) {
        const auto r = rgb[index], g = rgb[index + 1], b = rgb[index + 2];
        const auto y = Math::add(Math::add(Math::multiply(wr, r), Math::multiply(wg, g)), Math::multiply(wb, b));
        const auto cb = Math::add(Math::divide(Math::multiply(half, Math::subtract(b, y)), cb_denominator), half);
        const auto cr = Math::add(Math::divide(Math::multiply(half, Math::subtract(r, y)), cr_denominator), half);
        const std::array<Number, 3> values{{y, cb, cr}};
        for (int channel = 0; channel < 3; ++channel) {
            auto value = std::clamp(values[channel], zero, one);
            if (quant) value = Math::divide(static_cast<Number>(round_even(
                static_cast<double>(Math::multiply(value, scale)))), scale);
            yuv[index + channel] = value;
        }
    }
    return output;
}
template<class Number> void copy_as_double(const torch::Tensor& input, torch::Tensor& output) {
    const auto* source = input.data_ptr<Number>();
    auto* destination = output.data_ptr<double>();
    const auto height = input.size(1), width = input.size(2);
    const auto strides = input.strides();
    for (std::int64_t channel = 0; channel < input.size(0); ++channel)
        for (std::int64_t row = 0; row < height; ++row)
            for (std::int64_t column = 0; column < width; ++column)
                destination[(channel * height + row) * width + column] = static_cast<double>(
                    source[channel * strides[0] + row * strides[1] + column * strides[2]]);
}
torch::Tensor spline_coefficients(const torch::Tensor& input) {
    auto result = torch::empty(input.sizes(), input.options().dtype(torch::kFloat64));
    // Read arbitrary array strides directly, including extended unsigned dtypes
    // for which LibTorch 2.5 does not supply all indexing/copy kernels.
    switch (input.scalar_type()) {
        case torch::kBool: copy_as_double<bool>(input, result); break;
        case torch::kUInt8: copy_as_double<std::uint8_t>(input, result); break;
        case torch::kInt8: copy_as_double<std::int8_t>(input, result); break;
        case torch::kInt16: copy_as_double<std::int16_t>(input, result); break;
        case torch::kInt32: copy_as_double<std::int32_t>(input, result); break;
        case torch::kInt64: copy_as_double<std::int64_t>(input, result); break;
        case at::ScalarType::UInt16: copy_as_double<std::uint16_t>(input, result); break;
        case at::ScalarType::UInt32: copy_as_double<std::uint32_t>(input, result); break;
        case at::ScalarType::UInt64: copy_as_double<std::uint64_t>(input, result); break;
        case torch::kFloat32: copy_as_double<float>(input, result); break;
        case torch::kFloat64: copy_as_double<double>(input, result); break;
        default: throw std::invalid_argument("Unsupported SciPy array dtype");
    }
    return result;
}
struct ArrayType { char kind; int bits; };
ArrayType array_type(torch::ScalarType type) {
    switch (type) {
        case torch::kBool: return {'b', 1};
        case torch::kUInt8: return {'u', 8};
        case at::ScalarType::UInt16: return {'u', 16};
        case at::ScalarType::UInt32: return {'u', 32};
        case at::ScalarType::UInt64: return {'u', 64};
        case torch::kInt8: return {'i', 8};
        case torch::kInt16: return {'i', 16};
        case torch::kInt32: return {'i', 32};
        case torch::kInt64: return {'i', 64};
        case torch::kFloat16: return {'f', 16};
        case torch::kFloat32: return {'f', 32};
        case torch::kFloat64: return {'f', 64};
        case torch::kComplexFloat: return {'c', 32};
        case torch::kComplexDouble: return {'c', 64};
        default: throw std::invalid_argument("Unsupported NumPy array dtype");
    }
}
torch::ScalarType concatenate_type(torch::ScalarType first, torch::ScalarType second) {
    const auto a = array_type(first), b = array_type(second);
    if (first == second || b.kind == 'b') return first;
    if (a.kind == 'b') return second;
    if (a.kind == 'f' || a.kind == 'c' || b.kind == 'f' || b.kind == 'c') {
        int bits = 0;
        for (const auto type : {a, b}) {
            if (type.kind == 'f' || type.kind == 'c') bits = std::max(bits, type.bits);
            else bits = std::max(bits, type.bits <= 8 ? 16 : type.bits <= 16 ? 32 : 64);
        }
        if (a.kind == 'c' || b.kind == 'c') return bits > 32 ? torch::kComplexDouble : torch::kComplexFloat;
        return bits > 32 ? torch::kFloat64 : bits > 16 ? torch::kFloat32 : torch::kFloat16;
    }
    int bits = std::max(a.bits, b.bits);
    const bool signed_result = a.kind == 'i' || b.kind == 'i';
    if (a.kind != b.kind) {
        const auto unsigned_bits = a.kind == 'u' ? a.bits : b.bits;
        const auto signed_bits = a.kind == 'i' ? a.bits : b.bits;
        if (unsigned_bits >= signed_bits) {
            if (unsigned_bits == 64) return torch::kFloat64;
            bits = unsigned_bits * 2;
        }
    }
    if (signed_result)
        return bits == 8 ? torch::kInt8 : bits == 16 ? torch::kInt16 : bits == 32 ? torch::kInt32 : torch::kInt64;
    return bits == 8 ? torch::kUInt8 : bits == 16 ? at::ScalarType::UInt16 : bits == 32 ? at::ScalarType::UInt32 : at::ScalarType::UInt64;
}
template<class Number> struct RealNumber { using Type = Number; };
template<class Number> struct RealNumber<c10::complex<Number>> { using Type = Number; };
template<class Output, class Input> Output array_cast(Input value) {
    if constexpr (c10::is_complex<Output>::value) {
        using Real = typename RealNumber<Output>::Type;
        if constexpr (c10::is_complex<Input>::value)
            return {static_cast<Real>(value.real()), static_cast<Real>(value.imag())};
        else return {static_cast<Real>(value), Real{}};
    } else if constexpr (c10::is_complex<Input>::value) {
        // Concatenation never selects a real output for a complex input.
        throw std::logic_error("Complex-to-real NumPy concatenation");
    } else return static_cast<Output>(value);
}
template<class Output, class Input> void append_array(const torch::Tensor& input, Output* output) {
    const auto* source = input.data_ptr<Input>();
    const auto strides = input.strides();
    std::int64_t index = 0;
    for (std::int64_t channel = 0; channel < input.size(0); ++channel)
        for (std::int64_t row = 0; row < input.size(1); ++row)
            for (std::int64_t column = 0; column < input.size(2); ++column)
                output[index++] = array_cast<Output>(source[channel * strides[0] + row * strides[1] + column * strides[2]]);
}
#define UF_ARRAY_TYPES(APPLY) \
    APPLY(torch::kBool, bool) \
    APPLY(torch::kUInt8, std::uint8_t) \
    APPLY(torch::kInt8, std::int8_t) \
    APPLY(torch::kInt16, std::int16_t) \
    APPLY(torch::kInt32, std::int32_t) \
    APPLY(torch::kInt64, std::int64_t) \
    APPLY(at::ScalarType::UInt16, std::uint16_t) \
    APPLY(at::ScalarType::UInt32, std::uint32_t) \
    APPLY(at::ScalarType::UInt64, std::uint64_t) \
    APPLY(torch::kFloat16, c10::Half) \
    APPLY(torch::kFloat32, float) \
    APPLY(torch::kFloat64, double) \
    APPLY(torch::kComplexFloat, c10::complex<float>) \
    APPLY(torch::kComplexDouble, c10::complex<double>)
template<class Output> void append_array(const torch::Tensor& input, Output* output) {
    switch (input.scalar_type()) {
#define UF_APPEND_CASE(TYPE, NUMBER) case TYPE: append_array<Output, NUMBER>(input, output); break;
        UF_ARRAY_TYPES(UF_APPEND_CASE)
#undef UF_APPEND_CASE
        default: throw std::invalid_argument("Unsupported NumPy array dtype");
    }
}
torch::Tensor concatenate_planes(const torch::Tensor& y, const torch::Tensor& uv) {
    if (!y.defined() || !y.device().is_cpu() || y.dim() != 3 || y.size(1) != uv.size(1) || y.size(2) != uv.size(2))
        throw std::invalid_argument("Y and upsampled UV arrays require matching CPU spatial dimensions");
    const auto type = concatenate_type(y.scalar_type(), uv.scalar_type());
    auto result = torch::empty({y.size(0) + uv.size(0), y.size(1), y.size(2)}, y.options().dtype(type));
    // NumPy promotes integer/floating arrays differently from torch.cat. Copy
    // via native scalar conversion, keeping all 64 integer bits when applicable.
    switch (type) {
#define UF_CONCAT_CASE(TYPE, NUMBER) case TYPE: { \
            auto* output = result.data_ptr<NUMBER>(); \
            append_array(y, output); \
            if (uv.numel()) append_array(uv, output + y.numel()); \
            break; }
        UF_ARRAY_TYPES(UF_CONCAT_CASE)
#undef UF_CONCAT_CASE
        default: throw std::invalid_argument("Unsupported NumPy concatenation dtype");
    }
    return result;
}
#undef UF_ARRAY_TYPES
std::int64_t mirror(std::int64_t index, std::int64_t size) {
    if (size <= 1) return 0;
    const auto period = 2 * size - 2;
    index %= period;
    if (index < 0) index += period;
    return index < size ? index : period - index;
}
struct Knot {
    std::array<std::int64_t, 6> indexes{};
    std::array<double, 6> weights{};
    bool outside = false;
};
std::vector<Knot> knots(std::int64_t input, std::int64_t output, int order) {
    std::vector<Knot> result(static_cast<std::size_t>(output));
    const auto scale = output > 1 ? static_cast<double>(input - 1) / (output - 1) : 1.0;
    for (std::int64_t index = 0; index < output; ++index) {
        auto& knot = result[static_cast<std::size_t>(index)];
        const auto coordinate = static_cast<double>(index) * scale;
        // Constant mode checks the coordinate before mirroring its footprint.
        if (coordinate < 0 || coordinate > input - 1) { knot.outside = true; continue; }
        const auto start = static_cast<std::int64_t>(std::floor(order & 1 ? coordinate : coordinate + 0.5)) - order / 2;
        for (int offset = 0; offset <= order; ++offset)
            knot.indexes[offset] = mirror(start + offset, input);
        if (order) detail::get_spline_interpolation_weights(coordinate, order, knot.weights.data());
        else knot.weights[0] = 1;
    }
    return result;
}
void prefilter(torch::Tensor& coefficients, int order) {
    auto* data = coefficients.data_ptr<double>();
    const std::array<std::int64_t, 3> sizes{{coefficients.size(0), coefficients.size(1), coefficients.size(2)}};
    const std::array<std::int64_t, 3> strides{{sizes[1] * sizes[2], sizes[2], 1}};
    double poles[detail::MAX_SPLINE_FILTER_POLES]; int count = 0;
    detail::get_filter_poles(order, &count, poles);
    // scipy.ndimage.spline_filter traverses axes 0, 1, 2 in Float64, including
    // the channel axis whose zoom factor is one.
    for (int axis = 0; axis < 3; ++axis) {
        const auto length = sizes[axis], stride = strides[axis];
        if (length <= 1) continue;
        std::vector<double> line(static_cast<std::size_t>(length));
        const auto blocks = coefficients.numel() / (length * stride);
        for (std::int64_t block = 0; block < blocks; ++block) {
            for (std::int64_t offset = 0; offset < stride; ++offset) {
                const auto base = block * length * stride + offset;
                for (std::int64_t index = 0; index < length; ++index) line[index] = data[base + index * stride];
                detail::apply_filter(line.data(), length, poles, count, detail::NI_EXTEND_CONSTANT);
                for (std::int64_t index = 0; index < length; ++index) data[base + index * stride] = line[index];
            }
        }
    }
}
template<class Number> void cast_samples(const torch::Tensor& input, torch::Tensor& output) {
    const auto* source = input.data_ptr<double>();
    auto* destination = output.data_ptr<Number>();
    for (std::int64_t i = 0; i < input.numel(); ++i) {
        auto value = source[i];
        // SciPy stores a npy_bool (unsigned byte) after truncating the sample,
        // rather than applying a floating-point nonzero predicate.
        if constexpr (std::is_same_v<Number, bool>) destination[i] = std::trunc(value) != 0;
        else {
            value += value > 0 ? 0.5 : -0.5;
            if (value <= static_cast<double>(std::numeric_limits<Number>::lowest())) destination[i] = std::numeric_limits<Number>::lowest();
            else if (value >= static_cast<double>(std::numeric_limits<Number>::max())) destination[i] = std::numeric_limits<Number>::max();
            else destination[i] = static_cast<Number>(value);
        }
    }
}
torch::Tensor cast_integer_zoom(const torch::Tensor& input, torch::ScalarType type) {
    auto output = torch::empty(input.sizes(), input.options().dtype(type));
    switch (type) {
        case torch::kBool: cast_samples<bool>(input, output); break;
        case torch::kUInt8: cast_samples<std::uint8_t>(input, output); break;
        case torch::kInt8: cast_samples<std::int8_t>(input, output); break;
        case torch::kInt16: cast_samples<std::int16_t>(input, output); break;
        case torch::kInt32: cast_samples<std::int32_t>(input, output); break;
        case torch::kInt64: cast_samples<std::int64_t>(input, output); break;
        case at::ScalarType::UInt16: cast_samples<std::uint16_t>(input, output); break;
        case at::ScalarType::UInt32: cast_samples<std::uint32_t>(input, output); break;
        case at::ScalarType::UInt64: cast_samples<std::uint64_t>(input, output); break;
        default: throw std::invalid_argument("Unsupported SciPy integer output dtype");
    }
    return output;
}
torch::Tensor zoom_chroma(const torch::Tensor& uv, int order) {
    if (!uv.defined() || !uv.device().is_cpu() || uv.dim() != 3 || uv.size(0) != 2)
        throw std::invalid_argument("UF chroma zoom requires a CPU 2xHxW array");
    if (order < 0 || order > 5) throw std::invalid_argument("Spline order must be between zero and five");
    const auto type = uv.scalar_type();
    if (uv.is_complex()) {
        if (type != torch::kComplexFloat && type != torch::kComplexDouble)
            throw std::invalid_argument("Unsupported SciPy complex array dtype");
        return torch::complex(zoom_chroma(torch::real(uv), order), zoom_chroma(torch::imag(uv), order));
    }
    if (!c10::isIntegralType(type, true) && type != torch::kFloat32 && type != torch::kFloat64)
        throw std::invalid_argument("Unsupported SciPy array dtype");
    const auto h = uv.size(1), w = uv.size(2);
    if (h > std::numeric_limits<std::int64_t>::max() / 2 || w > std::numeric_limits<std::int64_t>::max() / 2)
        throw std::overflow_error("UF chroma zoom dimensions exceed native integers");
    auto result = torch::zeros({2, 2 * h, 2 * w}, uv.options().dtype(torch::kFloat64));
    if (!h || !w) return result.to(uv.scalar_type());
    // NI_ZoomShift accumulates Double samples even for order zero; preserve
    // that conversion and addition instead of repeating the original bytes.
    // This also gives prefilter independent storage without mutating the input.
    auto coefficients = spline_coefficients(uv.detach());
    if (order > 1) prefilter(coefficients, order);
    const auto channels = knots(2, 2, order), rows = knots(h, 2 * h, order), columns = knots(w, 2 * w, order);
    const auto* data = coefficients.data_ptr<double>();
    auto* output = result.data_ptr<double>();
    for (std::int64_t channel = 0; channel < 2; ++channel) {
        const auto& ck = channels[channel];
        for (std::int64_t row = 0; row < 2 * h; ++row) {
            const auto& rk = rows[row];
            for (std::int64_t column = 0; column < 2 * w; ++column) {
                const auto& wk = columns[column]; double value = 0;
                if (!ck.outside && !rk.outside && !wk.outside) {
                    for (int c = 0; c <= order; ++c)
                        for (int y = 0; y <= order; ++y)
                            for (int x = 0; x <= order; ++x) {
                                auto term = data[(ck.indexes[c] * h + rk.indexes[y]) * w + wk.indexes[x]];
                                if (order) { term *= ck.weights[c]; term *= rk.weights[y]; term *= wk.weights[x]; }
                                value += term;
                            }
                }
                output[(channel * 2 * h + row) * 2 * w + column] = value;
            }
        }
    }
    if (uv.is_floating_point()) return result.to(type);
    return cast_integer_zoom(result, type);
}
}  // namespace
torch::Tensor rgb2ycbcr(const torch::Tensor& rgb, bool is_bgr) {
    const auto parts = rgb.chunk(3, -3);
    const auto r = parts.at(is_bgr ? 2 : 0), g = parts.at(1), b = parts.at(is_bgr ? 0 : 2);
    const auto y = kr * r + kg * g + kb * b;
    const auto cb = 0.5 * (b - y) / (1 - kb) + 0.5;
    const auto cr = 0.5 * (r - y) / (1 - kr) + 0.5;
    return torch::cat({y, cb, cr}, -3).clamp(0, 1);
}
torch::Tensor rgb2ycbcr_np(const torch::Tensor& rgb, bool quant) {
    if (!rgb.device().is_cpu() || rgb.dim() != 3 || rgb.size(2) != 3 || rgb.size(0) % 2 || rgb.size(1) % 2)
        throw std::invalid_argument("RGB array requires CPU HxWx3 data and even spatial dimensions");
    const auto input = rgb.detach();
    switch (input.scalar_type()) {
        case torch::kFloat16: return array_color<c10::Half>(input, quant);
        case torch::kFloat32: return array_color<float>(input, quant);
        case torch::kFloat64: return array_color<double>(input, quant);
        default:
            if (!c10::isIntegralType(input.scalar_type(), true))
                throw std::invalid_argument("RGB array requires a NumPy-compatible real dtype");
            return array_color<double>(spline_coefficients(input), quant);
    }
}
torch::Tensor ycbcr2rgb(const torch::Tensor& ycbcr, bool is_bgr, bool clamp) {
    const auto parts = ycbcr.to(torch::kFloat32).chunk(3, -3);
    const auto r = parts.at(0) + (2 - 2 * kr) * (parts.at(2) - 0.5);
    const auto b = parts.at(0) + (2 - 2 * kb) * (parts.at(1) - 0.5);
    const auto g = (parts.at(0) - kr * r - kb * b) / kg;
    auto result = is_bgr ? torch::cat({b, g, r}, -3) : torch::cat({r, g, b}, -3);
    if (clamp) result = result.clamp(0, 1);
    return result.to(ycbcr.scalar_type());
}
Upsampled ycbcr420_to_444_np(const torch::Tensor& y, const torch::Tensor& uv, int order, bool separate) {
    const auto chroma = zoom_chroma(uv, order);
    if (separate) return Yuv420{y, chroma};
    return concatenate_planes(y.detach(), chroma);
}
torch::Tensor ycbcr420_to_444_np(const torch::Tensor& y, const torch::Tensor& uv, int order) {
    return std::get<torch::Tensor>(ycbcr420_to_444_np(y, uv, order, false));
}
Yuv420 yuv_444_to_420(const torch::Tensor& yuv) {
    using at::indexing::None;
    using at::indexing::Slice;
    const auto y = yuv.index({Slice(), Slice(None, 1), Slice(), Slice()});
    const auto uv = yuv.index({Slice(), Slice(1, None), Slice(), Slice()});
    return {y, torch::nn::functional::avg_pool2d(uv,
        torch::nn::functional::AvgPool2dFuncOptions(2).stride(2))};
}
}  // namespace uf::transforms
