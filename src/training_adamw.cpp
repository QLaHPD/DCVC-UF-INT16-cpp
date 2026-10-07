// Adapted from torch/optim/adamw.py and optimizer.py in the installed
// PyTorch 2.5.0a0+872d972e41.nv24.08 distribution.
// Retained copyright notices and BSD terms: licenses/PyTorch.txt.
#include "uf/training_adamw.hpp"
#include "uf/process.hpp"

#include <ATen/Functions.h>
#include <c10/core/DefaultDtype.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#ifdef UF_HAS_TRAINING_CUDA
#include <c10/cuda/CUDAGraphsC10Utils.h>
#include <torch/cuda.h>
#endif

#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

namespace uf::training {
namespace {

struct Batch {
    std::vector<torch::Tensor> parameters, gradients, means, squares, maxima, steps;
};

bool automatic_foreach(const Batch& batch, const AdamWOptions& options) {
    if (options.differentiable() || (options.tensor_lr().defined() && !options.capturable())) return false;
    return std::all_of(batch.parameters.begin(), batch.parameters.end(), [](const torch::Tensor& parameter) {
        const auto device = parameter.device().type();
        return device == c10::DeviceType::CUDA || device == c10::DeviceType::XPU ||
               device == c10::DeviceType::PrivateUse1;
    });
}

void capturable_parameter(const torch::Tensor& parameter, const torch::Tensor& step, bool allow_xla) {
    const auto device = parameter.device().type();
    if (device != step.device().type() ||
        !(device == c10::DeviceType::CUDA || device == c10::DeviceType::PrivateUse1 ||
          (allow_xla && device == c10::DeviceType::XLA))) {
        throw std::runtime_error("Capturable AdamW parameters and steps must share a supported device type");
    }
}

void reject_amp(const std::optional<torch::Tensor>& scale, const std::optional<torch::Tensor>& infinity) {
    if (scale || infinity) throw std::runtime_error("AdamW AMP scalars require the fused update path");
}

// Retain parameter order within each device/dtype group. State tensors stay
// attached to their parameters; only local Tensor handles are regrouped.
std::vector<Batch> grouped(const Batch& input) {
    using Key = std::tuple<int, int, int>;
    std::map<Key, std::size_t> indices;
    std::vector<Batch> output;
    for (std::size_t index = 0; index < input.parameters.size(); ++index) {
        const auto& parameter = input.parameters[index];
        const Key key{static_cast<int>(parameter.device().type()), parameter.device().index(),
                      static_cast<int>(parameter.scalar_type())};
        const auto added = indices.emplace(key, output.size());
        if (added.second) output.emplace_back();
        auto& batch = output[added.first->second];
        batch.parameters.push_back(parameter);
        batch.gradients.push_back(input.gradients[index]);
        batch.means.push_back(input.means[index]);
        batch.squares.push_back(input.squares[index]);
        batch.steps.push_back(input.steps[index]);
        if (!input.maxima.empty()) batch.maxima.push_back(input.maxima[index]);
    }
    return output;
}

void real_views(Batch& batch) {
    for (std::size_t index = 0; index < batch.parameters.size(); ++index) {
        if (!batch.parameters[index].is_complex()) continue;
        batch.parameters[index] = at::view_as_real(batch.parameters[index]);
        batch.gradients[index] = at::view_as_real(batch.gradients[index]);
        batch.means[index] = at::view_as_real(batch.means[index]);
        batch.squares[index] = at::view_as_real(batch.squares[index]);
        if (!batch.maxima.empty()) batch.maxima[index] = at::view_as_real(batch.maxima[index]);
    }
}

void single_tensor(Batch batch, const AdamWOptions& options,
                   const std::optional<torch::Tensor>& scale,
                   const std::optional<torch::Tensor>& infinity) {
    reject_amp(scale, infinity);
    const auto beta1 = std::get<0>(options.betas()), beta2 = std::get<1>(options.betas());
    for (std::size_t index = 0; index < batch.parameters.size(); ++index) {
        ProcessInterrupt::check();
        if (options.capturable()) capturable_parameter(batch.parameters[index], batch.steps[index], true);
        if (batch.parameters[index].is_complex()) {
            batch.parameters[index] = at::view_as_real(batch.parameters[index]);
            batch.gradients[index] = at::view_as_real(batch.gradients[index]);
            batch.means[index] = at::view_as_real(batch.means[index]);
            batch.squares[index] = at::view_as_real(batch.squares[index]);
            if (options.amsgrad()) batch.maxima[index] = at::view_as_real(batch.maxima[index]);
        }
        auto parameter = batch.parameters[index];
        const auto gradient = options.maximize() ? batch.gradients[index].neg() : batch.gradients[index];
        auto mean = batch.means[index], square = batch.squares[index], step = batch.steps[index];
        step.add_(1);
        if (options.tensor_lr().defined()) parameter.mul_(1.0 - options.tensor_lr() * options.weight_decay());
        else parameter.mul_(1.0 - options.lr() * options.weight_decay());
        mean.lerp_(gradient, 1.0 - beta1);
        square.mul_(beta2).addcmul_(gradient, gradient, 1.0 - beta2);
        if (options.capturable() || options.differentiable()) {
            const auto correction1 = 1.0 - at::pow(beta1, step);
            const auto correction2 = 1.0 - at::pow(beta2, step);
            const auto size = options.tensor_lr().defined()
                ? options.tensor_lr() / correction1 : options.lr() / correction1;
            const auto negative_size = size.neg();
            torch::Tensor denominator;
            if (options.amsgrad()) {
                auto previous = options.differentiable() ? batch.maxima[index].clone() : batch.maxima[index];
                batch.maxima[index].copy_(at::maximum(previous, square));
                denominator = batch.maxima[index].sqrt() / (correction2.sqrt() * negative_size);
            } else denominator = square.sqrt() / (correction2.sqrt() * negative_size);
            denominator.add_(options.eps() / negative_size);
            parameter.addcdiv_(mean, denominator);
        } else {
            const auto count = step.item<double>();
            const auto correction1 = 1.0 - std::pow(beta1, count);
            const auto correction2 = std::sqrt(1.0 - std::pow(beta2, count));
            torch::Tensor denominator;
            if (options.amsgrad()) {
                at::maximum_out(batch.maxima[index], batch.maxima[index], square);
                denominator = batch.maxima[index].sqrt().div(correction2).add_(options.eps());
            } else denominator = square.sqrt().div(correction2).add_(options.eps());
            const auto negative_size = options.tensor_lr().defined()
                ? options.tensor_lr().div(correction1).neg().item<double>() : -(options.lr() / correction1);
            parameter.addcdiv_(mean, denominator, negative_size);
        }
    }
}

void foreach_tensor(const Batch& input, const AdamWOptions& options,
                    const std::optional<torch::Tensor>& scale,
                    const std::optional<torch::Tensor>& infinity) {
    if (input.parameters.empty()) return;
    if (options.tensor_lr().defined() && !options.capturable()) {
        throw std::runtime_error("AdamW foreach with a tensor learning rate requires capturable=True");
    }
    if (options.capturable()) {
        for (std::size_t index = 0; index < input.parameters.size(); ++index) {
            capturable_parameter(input.parameters[index], input.steps[index], false);
        }
    }
    if (options.differentiable()) throw std::runtime_error("AdamW foreach operations do not support autograd");
    reject_amp(scale, infinity);
    const auto beta1 = std::get<0>(options.betas()), beta2 = std::get<1>(options.betas());
    for (auto batch : grouped(input)) {
        ProcessInterrupt::check();
        real_views(batch);
        if (options.maximize()) batch.gradients = at::_foreach_neg(batch.gradients);
        if (batch.steps.front().is_cpu()) {
            at::_foreach_add_(batch.steps, at::scalar_tensor(1.0, torch::TensorOptions()
                .dtype(c10::get_default_dtype_as_scalartype()).device(torch::kCPU)), 1.0);
        } else at::_foreach_add_(batch.steps, 1);
        if (options.weight_decay() != 0) {
            if (options.tensor_lr().defined()) {
                at::_foreach_mul_(batch.parameters, 1.0 - options.tensor_lr() * options.weight_decay());
            } else at::_foreach_mul_(batch.parameters, 1.0 - options.lr() * options.weight_decay());
        }
        at::_foreach_lerp_(batch.means, batch.gradients, 1.0 - beta1);
        at::_foreach_mul_(batch.squares, beta2);
        at::_foreach_addcmul_(batch.squares, batch.gradients, batch.gradients, 1.0 - beta2);
        batch.gradients.clear();
        if (options.capturable()) {
            auto negative_size = at::_foreach_pow(beta1, batch.steps);
            auto correction2 = at::_foreach_pow(beta2, batch.steps);
            at::_foreach_sub_(negative_size, 1);
            at::_foreach_sub_(correction2, 1);
            at::_foreach_neg_(correction2);
            if (options.tensor_lr().defined()) at::_foreach_div_(negative_size, options.tensor_lr());
            else at::_foreach_div_(negative_size, options.lr());
            at::_foreach_reciprocal_(negative_size);
            at::_foreach_sqrt_(correction2);
            if (options.amsgrad()) at::_foreach_maximum_(batch.maxima, batch.squares);
            auto denominator = at::_foreach_sqrt(options.amsgrad() ? batch.maxima : batch.squares);
            at::_foreach_div_(denominator, correction2);
            at::_foreach_add_(denominator, options.eps());
            at::_foreach_div_(denominator, negative_size);
            at::_foreach_addcdiv_(batch.parameters, batch.means, denominator);
        } else {
            std::vector<at::Scalar> negative_sizes, correction2;
            for (const auto& step : batch.steps) {
                const auto count = step.item<double>();
                negative_sizes.emplace_back((options.lr() / (1.0 - std::pow(beta1, count))) * -1.0);
                correction2.emplace_back(std::sqrt(1.0 - std::pow(beta2, count)));
            }
            if (options.amsgrad()) at::_foreach_maximum_(batch.maxima, batch.squares);
            auto denominator = at::_foreach_sqrt(options.amsgrad() ? batch.maxima : batch.squares);
            at::_foreach_div_(denominator, correction2);
            at::_foreach_add_(denominator, options.eps());
            at::_foreach_addcdiv_(batch.parameters, batch.means, denominator, negative_sizes);
        }
    }
}

void fused_tensor(const Batch& input, const AdamWOptions& options,
                  const std::optional<torch::Tensor>& scale,
                  const std::optional<torch::Tensor>& infinity) {
    if (input.parameters.empty()) return;
    if (options.differentiable()) throw std::runtime_error("Fused AdamW does not support differentiable=True");
    const auto beta1 = std::get<0>(options.betas()), beta2 = std::get<1>(options.betas());
    std::map<std::string, torch::Tensor> rates, scales, infinities;
    for (auto batch : grouped(input)) {
        ProcessInterrupt::check();
        const auto device = batch.parameters.front().device();
        if (device.type() == c10::DeviceType::MPS && (scale || infinity)) {
            throw std::runtime_error("MPS fused AdamW does not support AMP scalars");
        }
        const auto on_device = [&](const std::optional<torch::Tensor>& tensor,
                                   std::map<std::string, torch::Tensor>& cache) -> std::optional<torch::Tensor> {
            if (!tensor) return std::nullopt;
            const auto found = cache.find(device.str());
            if (found != cache.end()) return found->second;
            return cache.emplace(device.str(), tensor->to(device, true)).first->second;
        };
        const auto device_scale = on_device(scale, scales), device_infinity = on_device(infinity, infinities);
        at::_foreach_add_(batch.steps, 1);
        if (options.tensor_lr().defined() && !options.tensor_lr().is_cpu()) {
            const auto rate = on_device(options.tensor_lr(), rates).value();
            at::_fused_adamw_(batch.parameters, batch.gradients, batch.means, batch.squares,
                batch.maxima, batch.steps, rate, beta1, beta2, options.weight_decay(), options.eps(),
                options.amsgrad(), options.maximize(), device_scale, device_infinity);
        } else {
            const auto rate = options.tensor_lr().defined() ? options.tensor_lr().item<double>() : options.lr();
            at::_fused_adamw_(batch.parameters, batch.gradients, batch.means, batch.squares,
                batch.maxima, batch.steps, rate, beta1, beta2, options.weight_decay(), options.eps(),
                options.amsgrad(), options.maximize(), device_scale, device_infinity);
        }
        if (device_infinity) {
            const std::vector<torch::Tensor> skipped(batch.steps.size(), *device_infinity);
            at::_foreach_sub_(batch.steps, skipped);
        }
    }
}

void validate_defaults(const AdamWOptions& options) {
    if (!(options.get_lr() >= 0 && options.eps() >= 0 && options.weight_decay() >= 0 &&
          std::get<0>(options.betas()) >= 0 && std::get<0>(options.betas()) < 1 &&
          std::get<1>(options.betas()) >= 0 && std::get<1>(options.betas()) < 1)) {
        throw std::invalid_argument("Invalid AdamW hyperparameters");
    }
    if (options.tensor_lr().defined() && options.foreach().value_or(false) && !options.capturable()) {
        throw std::invalid_argument("AdamW foreach with a tensor learning rate requires capturable=True");
    }
    if (options.fused().value_or(false) &&
        (options.differentiable() || options.foreach().value_or(false))) {
        throw std::invalid_argument("Fused AdamW cannot enable foreach or differentiable updates");
    }
}

bool empty_state(const AdamWParamState& state) {
    return !state.step().defined() && !state.exp_avg().defined() && !state.exp_avg_sq().defined() &&
        !state.max_exp_avg_sq().defined() &&
        (!state.extra_state().isGenericDict() || state.extra_state().toGenericDict().empty());
}

}  // namespace

double AdamWOptions::get_lr() const { return tensor_lr().defined() ? tensor_lr().item<double>() : lr(); }
void AdamWOptions::set_lr(double rate) { lr(rate); tensor_lr(torch::Tensor{}); }

torch::ScalarType adamw_step_dtype(bool fused) {
    return !fused && c10::get_default_dtype_as_scalartype() == torch::kFloat64 ? torch::kFloat64 : torch::kFloat32;
}

AdamW::AdamW(std::vector<torch::Tensor> parameters, AdamWOptions options)
    : AdamW(std::vector<torch::optim::OptimizerParamGroup>{
          torch::optim::OptimizerParamGroup(std::move(parameters))}, std::move(options)) {}

AdamW::AdamW(std::vector<torch::optim::OptimizerParamGroup> groups, AdamWOptions options)
    : torch::optim::Optimizer(std::move(groups), std::make_unique<AdamWOptions>(options)) {
    validate_defaults(options);
    if (options.fused().value_or(false)) {
        for (const auto& group : param_groups()) for (const auto& parameter : group.params()) {
            const auto device = parameter.device().type();
            if (!parameter.is_floating_point() ||
                !(device == c10::DeviceType::MPS || device == c10::DeviceType::CUDA ||
                  device == c10::DeviceType::XPU || device == c10::DeviceType::CPU ||
                  device == c10::DeviceType::PrivateUse1)) {
                throw std::invalid_argument("Fused AdamW requires floating parameters on a supported device");
            }
        }
    }
}

void AdamW::amp_scaling(std::optional<torch::Tensor> scale, std::optional<torch::Tensor> infinity) {
    grad_scale_ = std::move(scale);
    found_inf_ = std::move(infinity);
}

torch::Tensor AdamW::step(LossClosure closure) {
    const torch::AutoGradMode gradients(dynamic_cast<const AdamWOptions&>(defaults()).differentiable());
#ifdef UF_HAS_TRAINING_CUDA
    if (torch::cuda::is_available() &&
        c10::cuda::currentStreamCaptureStatusMayInitCtx() != c10::cuda::CaptureStatus::None &&
        !std::all_of(param_groups().begin(), param_groups().end(), [](const auto& group) {
            return dynamic_cast<const AdamWOptions&>(group.options()).capturable();
        })) {
        throw std::runtime_error("AdamW step CUDA graph capture requires capturable=True in every parameter group");
    }
#endif
    torch::Tensor loss;
    if (closure) { const torch::AutoGradMode enable(true); loss = closure(); }
    for (auto& group : param_groups()) {
        ProcessInterrupt::check();
        const auto& options = dynamic_cast<const AdamWOptions&>(group.options());
        Batch batch;
        for (const auto& parameter : group.params()) {
            if (!parameter.grad().defined()) continue;
            if (parameter.grad().is_sparse()) throw std::runtime_error("AdamW does not support sparse gradients");
            auto& entry = state()[parameter.unsafeGetTensorImpl()];
            if (!entry) entry = std::make_unique<AdamWParamState>();
            auto& values = dynamic_cast<AdamWParamState&>(*entry);
            if (empty_state(values)) {
                const auto device = options.capturable() || options.fused().value_or(false)
                    ? parameter.device() : torch::Device(torch::kCPU);
                values.step(at::zeros({}, torch::TensorOptions().dtype(adamw_step_dtype(options.fused().value_or(false))).device(device)));
                values.exp_avg(at::zeros_like(parameter, {}, at::MemoryFormat::Preserve));
                values.exp_avg_sq(at::zeros_like(parameter, {}, at::MemoryFormat::Preserve));
                if (options.amsgrad()) values.max_exp_avg_sq(at::zeros_like(parameter, {}, at::MemoryFormat::Preserve));
            }
            if (!values.step().defined() || !values.exp_avg().defined() || !values.exp_avg_sq().defined() ||
                (options.amsgrad() && !values.max_exp_avg_sq().defined())) {
                throw std::runtime_error("Incomplete AdamW parameter state");
            }
            if (options.differentiable() && values.step().requires_grad()) {
                throw std::runtime_error("Differentiable AdamW does not support requires_grad on its step counter");
            }
            if (options.foreach().value_or(false) && options.tensor_lr().defined() && !options.capturable()) {
                throw std::runtime_error("AdamW foreach with a tensor learning rate requires capturable=True");
            }
            batch.parameters.push_back(parameter);
            batch.gradients.push_back(parameter.grad());
            batch.means.push_back(values.exp_avg());
            batch.squares.push_back(values.exp_avg_sq());
            batch.steps.push_back(values.step());
            if (options.amsgrad()) batch.maxima.push_back(values.max_exp_avg_sq());
        }
        const bool fused = options.fused().value_or(false);
        const bool foreach = options.foreach().value_or(!options.fused().has_value() && automatic_foreach(batch, options));
        if (fused) fused_tensor(batch, options, grad_scale_, found_inf_);
        else if (foreach) foreach_tensor(batch, options, grad_scale_, found_inf_);
        else single_tensor(std::move(batch), options, grad_scale_, found_inf_);
    }
    return loss;
}

}  // namespace uf::training
