// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
// Default Module state loading follows PyTorch; BSD notice: licenses/PyTorch.txt.
#include "uf/training_common.hpp"
#include "uf/process.hpp"
#include "uf/python_integer.hpp"
#include "python_integer_ratio.hpp"
#include <ATen/TensorIndexing.h>
#include <c10/core/Scalar.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace uf::training {
namespace {

std::array<torch::Tensor, 3> convert_hba(std::array<torch::Tensor, 3> values,
                                      std::optional<torch::ScalarType> dtype) {
    if (dtype) for (auto& value : values) value = value.to(*dtype);
    return values;
}

std::string fixed_number(double value, bool with_sign = false) {
    // Python's fixed formatter ignores a NaN sign; complex imaginary parts
    // always have an explicit sign, including +nan and +inf.
    if (std::isnan(value)) return with_sign ? "+nan" : "nan";
    if (std::isinf(value)) return std::signbit(value) ? "-inf" : (with_sign ? "+inf" : "inf");
    static_assert(sizeof(double) == sizeof(std::uint64_t) &&
                  std::numeric_limits<double>::is_iec559 &&
                  std::numeric_limits<double>::radix == 2 &&
                  std::numeric_limits<double>::digits == 53 &&
                  std::numeric_limits<double>::max_exponent == 1024,
                  "UF loss formatting requires IEEE binary64");
    std::uint64_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const auto field = static_cast<unsigned>((bits >> 52) & 0x7ff);
    auto significand = bits & ((std::uint64_t{1} << 52) - 1);
    int exponent = -1074;
    if (field) {
        significand |= std::uint64_t{1} << 52;
        exponent = static_cast<int>(field) - 1023 - 52;
    }

    std::string digits;
    if (exponent >= 0) {
        auto integer = PythonInteger::from_unsigned(significand) * PythonInteger(100000);
        for (int bit = 0; bit < exponent; ++bit) integer = integer * PythonInteger(2);
        digits = integer.decimal();
    } else {
        // A 53-bit significand times 10^5 fits in 70 bits. Round the exact
        // scaled binary ratio to even without host float/locale formatting.
        using Integer = python_detail::RatioInteger;
        const auto scaled = static_cast<Integer>(significand) * 100000;
        const auto shift = -exponent;
        Integer rounded = 0;
        if (shift < 128) {
            rounded = scaled >> shift;
            const auto remainder = scaled & ((Integer{1} << shift) - 1);
            const auto half = Integer{1} << (shift - 1);
            if (remainder > half || (remainder == half && (rounded & 1))) ++rounded;
        }
        do {
            digits.push_back(static_cast<char>('0' + rounded % 10));
            rounded /= 10;
        } while (rounded);
        std::reverse(digits.begin(), digits.end());
    }
    if (digits.size() < 6) digits.insert(0, 6 - digits.size(), '0');
    digits.insert(digits.size() - 5, 1, '.');
    if (std::signbit(value)) digits.insert(digits.begin(), '-');
    else if (with_sign) digits.insert(digits.begin(), '+');
    return digits;
}

std::string values_string(const torch::Tensor& values) {
    if (values.dim() == 0) throw std::invalid_argument("Iteration over a zero-dimensional tensor");
    std::ostringstream output;
    // generate_str iterates the leading axis and calls item() on each entry;
    // flattening would silently accept entries with more than one element.
    for (std::int64_t index = 0; index < values.size(0); ++index) {
        if (index) output << "  ";
        const auto value = values.select(0, index).item();
        if (value.isComplex()) {
            const auto number = value.toComplexDouble();
            output << fixed_number(number.real()) << fixed_number(number.imag(), true) << 'j';
        } else if (value.isIntegral(false)) {
            const auto integer = value.type() == at::ScalarType::UInt64
                ? PythonInteger::from_unsigned(value.toUInt64()) : PythonInteger(value.toLong());
            output << fixed_number(integer.as_double());
        } else output << fixed_number(value.toDouble());
    }
    return output.str() + "  ";
}

}  // namespace

BitEstimatorImpl::BitEstimatorImpl(std::int64_t rates, std::int64_t channels) {
    // Python torch.empty uses the process's default floating dtype.
    const auto options = torch::TensorOptions();
    h = register_parameter("h", torch::empty({rates, channels, 4}, options).normal_(0, 0.01));
    b = register_parameter("b", torch::empty({rates, channels, 4}, options).normal_(0, 0.01));
    a = register_parameter("a", torch::empty({rates, channels, 3}, options).normal_(0, 0.01));
}

std::array<torch::Tensor, 3> BitEstimatorImpl::get_hba(const torch::Tensor& qp,
    std::optional<torch::ScalarType> dtype) const {
    return convert_hba({h.index_select(0, qp), b.index_select(0, qp), a.index_select(0, qp)}, dtype);
}

std::array<torch::Tensor, 3> BitEstimatorImpl::get_hba(std::int64_t qp,
    std::optional<torch::ScalarType> dtype) const {
    // Python's arbitrary-precision qp+1 can exceed Int64. At its upper bound
    // both positive slice endpoints already clamp to the tensor's length.
    const auto end = qp == std::numeric_limits<std::int64_t>::max() ? qp : qp + 1;
    return convert_hba({h.slice(0, qp, end), b.slice(0, qp, end), a.slice(0, qp, end)}, dtype);
}

torch::Tensor BitEstimatorImpl::forward(const torch::Tensor& input, const torch::Tensor& qp) {
    const auto values = get_hba(qp);
    return bit_estimator_z_probability(input, values[0], values[1], values[2]);
}

torch::Tensor BitEstimatorImpl::forward_scalar(const torch::Tensor& input, std::int64_t qp) {
    const auto values = get_hba(qp);
    return bit_estimator_z_probability(input, values[0], values[1], values[2]);
}

torch::Tensor BitEstimatorImpl::probability(const torch::Tensor& input, const torch::Tensor& qp) {
    const auto values = get_hba(qp, input.scalar_type());
    return bit_estimator_z_mass(input, values[0], values[1], values[2]);
}

torch::Tensor BitEstimatorImpl::probability(const torch::Tensor& input, std::int64_t qp) {
    const auto values = get_hba(qp, input.scalar_type());
    return bit_estimator_z_mass(input, values[0], values[1], values[2]);
}

Loss rate_distortion_loss(const RateDistortion& result, const torch::Tensor& lambdas) {
    auto costs = lambdas * result.mse + result.bpp;
    return {costs, costs.mean()};
}

LossInfo loss_info(const RateDistortion& result, const Loss& loss, std::int64_t pixels) {
    torch::NoGradGuard no_grad;
    return {{"bpp_y", values_string(result.bits_y / pixels)},
            {"bpp_z", values_string(result.bits_z / pixels)},
            {"mse", values_string(result.mse)}, {"losses", values_string(loss.losses)}};
}

CheckpointState export_training_state(const torch::nn::Module& module) {
    torch::NoGradGuard no_grad;
    CheckpointState result;
    // state_dict keeps detached views and all registered paths, including
    // shared children. Moving/cloning each tensor here would lose aliases.
    const std::function<void(const torch::nn::Module&, const std::string&)> collect =
        [&](const torch::nn::Module& current, const std::string& prefix) {
            for (const auto& parameter : current.named_parameters(false)) {
                ProcessInterrupt::check();
                result.insert_or_assign(prefix + parameter.key(), parameter.value().detach());
            }
            for (const auto& buffer : current.named_buffers(false)) {
                ProcessInterrupt::check();
                result.insert_or_assign(prefix + buffer.key(), buffer.value().detach());
            }
            for (const auto& child : current.named_children()) {
                ProcessInterrupt::check();
                if (child.value()) collect(*child.value(), prefix + child.key() + ".");
            }
        };
    collect(module, "");
    return result;
}

void load_training_state(torch::nn::Module& module, const CheckpointState& state, bool strict) {
    torch::NoGradGuard no_grad;
    std::vector<std::string> errors;
    std::set<std::string> expected;
    // Python copies each module's local parameters/buffers before visiting
    // children. Shape/key/copy errors are collected without rolling back
    // matching tensors, which matters when resume tries an older status.
    const auto copy = [&](const auto& entries, const std::string& prefix) {
        for (const auto& entry : entries) {
            ProcessInterrupt::check();
            const auto name = prefix + entry.key();
            expected.insert(name);
            const auto found = state.find(name);
            if (found == state.end()) {
                if (strict) errors.push_back("Missing training tensor: " + name);
                continue;
            }
            auto source = found->second;
            // The default Module loader accepts historical length-one state
            // for scalar parameters. An empty source still fails at select.
            if (entry.value().dim() == 0 && source.dim() == 1) source = source.select(0, 0);
            if (source.sizes() != entry.value().sizes()) {
                errors.push_back("Training tensor shape mismatch: " + name);
                continue;
            }
            try {
                // copy_ performs the default dtype/device conversion while
                // preserving the destination parameter and its grad flag.
                entry.value().copy_(source);
            } catch (const std::exception& error) {
                errors.push_back("Cannot copy training tensor " + name + ": " + error.what());
            }
        }
    };
    std::function<void(torch::nn::Module&, const std::string&)> load;
    load = [&](torch::nn::Module& current, const std::string& prefix) {
        copy(current.named_parameters(false), prefix);
        copy(current.named_buffers(false), prefix);
        for (const auto& child : current.named_children()) {
            ProcessInterrupt::check();
            if (child.value()) load(*child.value(), prefix + child.key() + ".");
        }
    };
    load(module, "");
    if (strict) {
        for (const auto& entry : state) {
            ProcessInterrupt::check();
            if (!expected.count(entry.first)) errors.push_back("Unexpected training tensor: " + entry.first);
        }
    }
    if (!errors.empty()) {
        std::ostringstream message;
        message << "Errors loading training state:";
        for (const auto& error : errors) message << "\n\t" << error;
        throw std::runtime_error(message.str());
    }
}

CompressionTrainingImpl::CompressionTrainingImpl(std::int64_t channels) {
    bit_estimator_z = register_module("bit_estimator_z", BitEstimator(qp_count, channels));
}

void CompressionTrainingImpl::initialize_weights() {
    torch::NoGradGuard no_grad;
    // include_self=false is valid during construction, before a module holder
    // has installed the root module's shared_ptr.
    for (const auto& module : modules(false)) {
        if (auto* convolution = module->as<torch::nn::Conv2d>()) {
            torch::nn::init::xavier_normal_(convolution->weight, 1.0);
            if (convolution->bias.defined()) torch::nn::init::constant_(convolution->bias, 0.0);
        }
    }
}

torch::Tensor CompressionTrainingImpl::add_noise(const torch::Tensor& input) {
    return input + torch::empty_like(input).uniform_(-0.5, 0.5).detach();
}

torch::Tensor CompressionTrainingImpl::select_q(const torch::Tensor& scales, const torch::Tensor& qp) {
    // Python's index_select_dim0 passes None through before reading index.
    // [:, :, None, None] inserts after the first two dimensions and retains
    // any trailing axes; it also rejects a selected tensor of rank below two.
    if (!scales.defined()) return scales;
    return scales.index_select(0, qp).index({at::indexing::Slice(), at::indexing::Slice(),
                                          at::indexing::None, at::indexing::None});
}

torch::Tensor CompressionTrainingImpl::probabilities_to_bits(const torch::Tensor& probabilities) {
    const auto dtype = probabilities.scalar_type();
    auto bits = lower_bound(probabilities.to(torch::kFloat32), 1e-6).log() * (-1.0 / std::log(2.0));
    return lower_bound(bits, 0.0).to(dtype);
}

PriorResult CompressionTrainingImpl::process_mask(const torch::Tensor& y,
    const torch::Tensor& scales, const torch::Tensor& means, const torch::Tensor& mask) {
    const auto scale = scales * mask;
    const auto mean = means * mask;
    const auto residual = (y - mean) * mask;
    const auto rounded = quantize(residual);
    return {residual, rounded, rounded + mean, scale};
}

torch::Tensor CompressionTrainingImpl::mse(const torch::Tensor& input,
                                          const torch::Tensor& reconstruction) {
    const auto errors = mse_yuv_rgb(input, reconstruction);
    return mse_weighted_average(errors.first, errors.second, input.size(2) * input.size(3));
}

torch::Tensor CompressionTrainingImpl::y_bits(const torch::Tensor& residual,
                                             const torch::Tensor& scales) {
    return probabilities_to_bits(gaussian_probability(residual, scales));
}

torch::Tensor CompressionTrainingImpl::z_bits(const torch::Tensor& residual, const torch::Tensor& qp) {
    return probabilities_to_bits(bit_estimator_z->probability(residual, qp));
}

std::vector<torch::Tensor> CompressionTrainingImpl::masks(const torch::Tensor& latent, int parts) {
    return masks(latent, parts, latent.device());
}

std::vector<torch::Tensor> CompressionTrainingImpl::masks(const torch::Tensor& latent, int parts,
                                                        const torch::Device& device) {
    if (latent.dim() != 4 || (parts != 2 && parts != 4) || latent.size(1) % parts) {
        throw std::invalid_argument("Invalid spatial prior tensor or partition count");
    }
    const auto batch = latent.size(0), channels = latent.size(1);
    const auto height = latent.size(2), width = latent.size(3);
    const MaskKey key{batch, channels, height, width, parts};
    if (const auto found = masks_.find(key); found != masks_.end()) return found->second;
    torch::NoGradGuard no_grad;
    // Match get_mask_2x/get_mask_4x: default CPU Boolean factories, multiply
    // each micro-mask by ones, concatenate, then move each complete mask.
    const auto options = torch::TensorOptions().dtype(torch::kBool);
    const auto ones = torch::ones({batch, channels / parts, height, width}, options);
    std::vector<torch::Tensor> micro;
    const auto add = [&](std::vector<std::int64_t> pattern) {
        auto tile = torch::tensor(pattern, options).reshape({2, 2});
        // Tensor extents are nonnegative. These ceil halves avoid H+1/W+1
        // overflow while retaining Python's arbitrary-precision result.
        tile = tile.repeat({height / 2 + height % 2, width / 2 + width % 2})
            .slice(0, 0, height).slice(1, 0, width);
        micro.push_back(tile.unsqueeze(0).unsqueeze(0));
    };
    std::vector<torch::Tensor> result;
    if (parts == 2) {
        add({1, 0, 0, 1});
        add({0, 1, 1, 0});
        result.push_back(torch::cat({ones * micro[0], ones * micro[1]}, 1).to(device));
        result.push_back(torch::cat({ones * micro[1], ones * micro[0]}, 1).to(device));
    } else {
        add({1, 0, 0, 0}); add({0, 1, 0, 0}); add({0, 0, 1, 0}); add({0, 0, 0, 1});
        result.push_back(torch::cat({ones * micro[0], ones * micro[1],
                                     ones * micro[2], ones * micro[3]}, 1).to(device));
        result.push_back(torch::cat({ones * micro[3], ones * micro[2],
                                     ones * micro[1], ones * micro[0]}, 1).to(device));
        result.push_back(torch::cat({ones * micro[2], ones * micro[3],
                                     ones * micro[0], ones * micro[1]}, 1).to(device));
        result.push_back(torch::cat({ones * micro[1], ones * micro[0],
                                     ones * micro[3], ones * micro[2]}, 1).to(device));
    }
    masks_.emplace(key, result);
    return result;
}

PriorResult CompressionTrainingImpl::prior_two(const torch::Tensor& input,
    const torch::Tensor& parameters, const Combine& spatial_prior) {
    ProcessInterrupt::check();
    const auto separated = parameters.chunk(3, 1);
    const auto q_dec = lower_bound(separated.at(0), 0.5);
    const auto y = input * (1.0 / q_dec);
    const auto& scales = separated.at(1);
    const auto partition = masks(y, 2, parameters.device());
    const auto first = process_mask(y, scales, separated.at(2), partition[0]);
    const auto means = spatial_prior(first.reconstruction, parameters);
    const auto second = process_mask(y, scales, means, partition[1]);
    return {first.residual + second.residual, first.quantized + second.quantized,
            (first.reconstruction + second.reconstruction) * q_dec, first.scales + second.scales};
}

PriorResult CompressionTrainingImpl::prior_four(const torch::Tensor& input,
    const torch::Tensor& q_encoder, const torch::Tensor& q_decoder, const torch::Tensor& parameters,
    const Transform& reduction, const std::array<Combine, 3>& adaptors, const Transform& spatial_prior,
    bool has_scales) {
    ProcessInterrupt::check();
    torch::Tensor y, q_dec, scales, means;
    if (q_encoder.defined()) {
        has_scales = true;
        const auto separated = parameters.chunk(2, 1);
        scales = separated.at(0); means = separated.at(1);
        y = input * q_encoder; q_dec = q_decoder;
    } else {
        const auto separated = parameters.chunk(3, 1);
        q_dec = lower_bound(separated.at(0), 0.5);
        scales = separated.at(1); means = separated.at(2);
        y = input * (1.0 / q_dec);
    }
    const auto common = reduction(parameters);
    const auto partition = masks(y, 4, common.device());
    std::array<PriorResult, 4> result;
    result[0] = process_mask(y, scales, means, partition[0]);
    auto reconstructed = result[0].reconstruction;
    for (std::size_t phase = 1; phase < result.size(); ++phase) {
        ProcessInterrupt::check();
        const auto prediction = spatial_prior(adaptors[phase - 1](reconstructed, common));
        if (has_scales) {
            const auto separated = prediction.chunk(2, 1);
            scales = separated.at(0); means = separated.at(1);
        } else means = prediction;
        result[phase] = process_mask(y, scales, means, partition[phase]);
        reconstructed = reconstructed + result[phase].reconstruction;
    }
    return {(result[0].residual + result[1].residual) + (result[2].residual + result[3].residual),
            (result[0].quantized + result[1].quantized) + (result[2].quantized + result[3].quantized),
            reconstructed * q_dec,
            (result[0].scales + result[1].scales) + (result[2].scales + result[3].scales)};
}

}  // namespace uf::training
