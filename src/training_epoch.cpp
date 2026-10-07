// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#include "uf/training_epoch.hpp"
#include "uf/training_schedules.hpp"
#include "uf/process.hpp"

#include <c10/core/InferenceMode.h>
#include <ATen/Functions.h>
#include <ATen/native/ForeachUtils.h>
#include <torch/utils.h>

#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace uf::training {
namespace {

using Clock = std::chrono::system_clock;

void validate_context(const EpochContext& context) {
    if (!context.next_batch || context.world_size <= 0 ||
        (context.rank >= 0 && context.rank >= context.world_size) ||
        (context.rank < 0 && context.world_size != 1)) {
        throw std::invalid_argument("Invalid native training epoch context");
    }
    if (context.rank >= 0 && context.world_size > 1 && !context.reduce_gradients) {
        throw std::invalid_argument("Distributed training requires gradient reduction before clipping");
    }
}

torch::Device module_device(const torch::nn::Module& model) {
    const auto parameters = model.parameters();
    if (parameters.empty()) throw std::invalid_argument("Training model has no parameters");
    return parameters.front().device();
}

DatasetSample to_device(DatasetSample sample, const torch::Device& device) {
    for (auto& tensor : sample) {
        ProcessInterrupt::check();
        tensor = tensor.to(device);
    }
    return sample;
}

void set_learning_rate(AdamW& optimizer, double rate) {
    for (auto& group : optimizer.param_groups()) group.options().set_lr(rate);
}

bool gradient_foreach_device(const torch::Device& device) {
    return device.is_cpu() || device.is_cuda() ||
        device.type() == c10::DeviceType::XPU || device.type() == c10::DeviceType::PrivateUse1;
}

// Follow the installed Python clip_grad_norm_ default (L2, foreach=None,
// error_if_nonfinite=False). The LibTorch frontend helper uses individual
// Tensor::norm calls instead of Python's foreach/vector-norm path.
// Algorithm attribution and BSD terms: licenses/PyTorch.txt.
double clip_training_gradients(const std::vector<torch::Tensor>& parameters, double max_norm) {
    torch::NoGradGuard no_grad;
    at::native::nested_optional_tensorvec_t gradients(1);
    gradients.front().reserve(parameters.size());
    for (const auto& parameter : parameters) {
        if (parameter.grad().defined()) gradients.front().emplace_back(parameter.grad());
    }
    if (gradients.front().empty()) return 0.0;
    const auto first_device = gradients.front().front()->device();
    // Reuse the installed grouping helper, including its iteration order.
    const auto grouped = at::native::_group_tensors_by_first_tensors_device_and_dtype(gradients, false);
    struct Group {
        torch::Device device;
        std::vector<torch::Tensor> gradients;
        bool foreach;
    };
    std::vector<Group> groups;
    groups.reserve(grouped.size());
    std::vector<torch::Tensor> norms;
    norms.reserve(gradients.front().size());
    for (const auto& entry : grouped) {
        ProcessInterrupt::check();
        Group group{entry.first.first, {}, gradient_foreach_device(entry.first.first)};
        const auto& tensors = entry.second.first.front();
        group.gradients.reserve(tensors.size());
        for (const auto& tensor : tensors) group.gradients.push_back(tensor.value());
        if (group.foreach) {
            const auto group_norms = at::_foreach_norm(group.gradients, 2.0);
            norms.insert(norms.end(), group_norms.begin(), group_norms.end());
        } else {
            for (const auto& gradient : group.gradients) {
                norms.push_back(at::linalg_vector_norm(gradient, 2.0));
            }
        }
        groups.push_back(std::move(group));
    }
    // Python stacks the individual norms even when there is only one gradient.
    for (auto& norm : norms) norm = norm.to(first_device);
    const auto total_norm = at::linalg_vector_norm(at::stack(norms), 2.0);
    const auto coefficient = at::clamp_max(max_norm / (total_norm + 1e-6), 1.0);
    for (const auto& group : groups) {
        ProcessInterrupt::check();
        const auto on_device = coefficient.to(group.device);
        if (group.foreach) at::_foreach_mul_(group.gradients, on_device);
        else {
            for (const auto& gradient : group.gradients) gradient.mul_(on_device);
        }
    }
    // Both reference trainers call .item() after clipping, then skip the
    // optimizer update for NaN/Inf without undoing the gradient scaling.
    return total_norm.item<double>();
}

std::uint64_t progress_numerator(std::uint64_t index, std::int64_t batch_size, std::int64_t world_size) {
    if (batch_size <= 0 || world_size <= 0) throw std::invalid_argument("Invalid training batch size");
    const auto batch = static_cast<std::uint64_t>(batch_size), world = static_cast<std::uint64_t>(world_size);
    if (index > std::numeric_limits<std::uint64_t>::max() / batch ||
        index * batch > std::numeric_limits<std::uint64_t>::max() / world) {
        throw std::overflow_error("Training progress counter overflow");
    }
    return index * batch * world;
}

void progress(std::ostream& log, double elapsed, std::int64_t epoch, std::uint64_t index,
    std::int64_t batch_size, std::int64_t world_size, std::size_t dataset_size,
    const LossInfo& info, const AdamW& optimizer) {
    if (!dataset_size || optimizer.param_groups().empty()) throw std::invalid_argument("Invalid training progress context");
    const auto numerator = progress_numerator(index, batch_size, world_size);
    const auto flags = log.flags();
    const auto precision = log.precision();
    log << std::fixed << std::setprecision(3) << "Time: " << elapsed << " seconds, Train epoch " << epoch
        << ": [" << numerator << " / " << dataset_size << " (" << std::setprecision(0)
        << (100.0 * numerator / dataset_size) << "%)]"
        << " MSE: " << info.at("mse") << " | Bpp y: " << info.at("bpp_y")
        << " | Bpp z: " << info.at("bpp_z") << " | Losses: " << info.at("losses")
        << " | lr: " << std::scientific << std::setprecision(1)
        << optimizer.param_groups().front().options().get_lr() << '\n';
    log.flags(flags);
    log.precision(precision);
    if (!log) throw std::runtime_error("Cannot write training progress");
}

bool update(torch::nn::Module& model, AdamW& optimizer, const torch::Tensor& loss,
    double max_norm, const EpochContext& context, std::ostream& log) {
    ProcessInterrupt::check();
    if (context.prepare_backward) context.prepare_backward(loss);
    optimizer.zero_grad();
    loss.backward();
    ProcessInterrupt::check();
    if (context.reduce_gradients) context.reduce_gradients(model);
    const auto total_norm = clip_training_gradients(model.parameters(), max_norm);
    if (!std::isfinite(total_norm)) {
        if (context.rank <= 0) {
            log << "non-finite norm, skip this batch\n";
            if (!log) throw std::runtime_error("Cannot write training skipped-update message");
        }
        return false;
    }
    ProcessInterrupt::check();
    optimizer.step();
    return true;
}

}  // namespace

EpochResult train_image_epoch(DmciTrainingImpl& model, ImageFolder& dataset,
    AdamW& optimizer, std::int64_t epoch, const EpochContext& context, std::ostream& log) {
    validate_context(context);
    const auto stage = image_training_stage(epoch);
    model.train();
    const auto device = module_device(model);
    set_learning_rate(optimizer, stage.learning_rate);
    dataset.set_patch_size(stage.patch_width, stage.patch_height);
    EpochResult result;
    auto started = Clock::now();
    for (std::uint64_t index = 0;; ++index) {
        ProcessInterrupt::check();
        auto sample = context.next_batch();
        if (!sample) break;
        auto batch = to_device(std::move(*sample), device);
        if (batch.size() != 3 || batch[0].dim() != 4) {
            throw std::invalid_argument("Image training batch must contain image, QP and lambda tensors");
        }
        const auto batch_size = batch[0].size(0);
        const auto print_info = context.rank <= 0 && index % 200 == 0;
        if (context.before_forward) context.before_forward();
        auto output = model.forward(batch[0], batch[1], batch[2], print_info);
        ++result.batches;
        if (!update(model, optimizer, output.loss.mean, 0.1, context, log)) {
            ++result.skipped_updates;
            if (index == std::numeric_limits<std::uint64_t>::max()) {
                throw std::overflow_error("Training batch counter overflow");
            }
            continue;
        }
        ++result.updates;
        if (print_info) {
            const auto now = Clock::now();
            progress(log, std::chrono::duration<double>(now - started).count(), epoch, index,
                     batch_size, context.world_size, dataset.size(), output.info.value(), optimizer);
            started = now;
        }
        if (index == std::numeric_limits<std::uint64_t>::max()) throw std::overflow_error("Training batch counter overflow");
    }
    return result;
}

EpochResult train_video_epoch(VideoTrainingModel& model, DmciTrainingImpl& intra_model,
    VideoFolder& dataset, AdamW& optimizer, std::int64_t epoch,
    const std::string& scheduling, const EpochContext& context, std::ostream& log) {
    validate_context(context);
    const auto stage = video_training_stage(epoch, scheduling, model.frame_delay(), model.model_structure());
    model.module().train();
    intra_model.eval();
    const auto device = module_device(model.module());
    if (module_device(intra_model) != device) throw std::invalid_argument("Intra and video training models must share a device");
    set_learning_rate(optimizer, stage.learning_rate);
    if (context.configure_checkpointing) context.configure_checkpointing(model, stage.use_checkpoint());
    else model.set_use_checkpointing(stage.use_checkpoint());
    dataset.set_frame_count(stage.frame_count);
    dataset.set_patch_size(stage.patch_width, stage.patch_height);
    EpochResult result;
    auto started = Clock::now();
    for (std::uint64_t index = 0;; ++index) {
        ProcessInterrupt::check();
        auto sample = context.next_batch();
        if (!sample) break;
        auto batch = to_device(std::move(*sample), device);
        if (batch.size() < 4 || batch[0].dim() != 4) {
            throw std::invalid_argument("Video training batch must contain reference, frame groups, QP and lambda tensors");
        }
        const auto batch_size = batch[0].size(0);
        const auto frame_fields = batch.size() - 2;
        const auto frame_stop = stage.cascaded ? std::size_t(2) : frame_fields;
        const auto& qp = batch[frame_fields];
        const auto& lambdas = batch[frame_fields + 1];
        const auto print_info = context.rank <= 0 && index % (stage.frame_count >= 8 ? 100 : 200) == 0;
        auto reference = batch[0];
        if (stage.frame_count > 1 + model.frame_delay() || stage.cascaded) {
            c10::InferenceMode inference;
            reference = intra_model.reconstruct(reference, qp);
        }
        model.clear_dpb();
        model.add_reference(reference);
        std::optional<LossInfo> last_info;
        for (std::size_t frame_index = 1; frame_index < frame_stop; ++frame_index) {
            ProcessInterrupt::check();
            TrainingStep output;
            if (context.before_forward) context.before_forward();
            if (stage.cascaded) {
                std::vector<torch::Tensor> groups(batch.begin() + 1, batch.end() - 2);
                output = model.forward(groups, qp, lambdas, print_info, static_cast<std::int64_t>(frame_index));
            } else {
                output = model.forward(batch[frame_index], qp, lambdas, print_info,
                                       static_cast<std::int64_t>(frame_index));
            }
            last_info = std::move(output.info);
            if (update(model.module(), optimizer, output.loss, 0.2, context, log)) ++result.updates;
            else ++result.skipped_updates;
        }
        ++result.batches;
        if (print_info) {
            const auto now = Clock::now();
            progress(log, std::chrono::duration<double>(now - started).count(), epoch, index,
                     batch_size, context.world_size, dataset.size(), last_info.value(), optimizer);
            started = now;
        }
        if (index == std::numeric_limits<std::uint64_t>::max()) throw std::overflow_error("Training batch counter overflow");
    }
    return result;
}

}  // namespace uf::training
