// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#include "uf/training_workflow.hpp"
#include "uf/training_checkpoint.hpp"
#include "uf/training_datasets.hpp"
#include "uf/training_epoch.hpp"
#include "uf/training_image_model.hpp"
#include "uf/training_loader.hpp"
#include "uf/training_recompute.hpp"
#include "uf/training_schedules.hpp"
#include "uf/process.hpp"

#include <ATen/Context.h>
#include <c10/core/DeviceGuard.h>
#include <torch/cuda.h>

#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>

namespace uf::training {
namespace {

void validate_options(const CommonTrainingOptions& options, const TrainingRank& rank) {
    // An empty dataset root joins to relative description/sample paths, as in
    // Python's os.path.join('', name). It is still a supplied CLI argument.
    if (options.save_dir.empty() || options.batch_size <= 0 ||
        options.num_workers <= 0 || rank.world_size <= 0 ||
        (rank.rank < 0 && rank.world_size != 1) || rank.rank >= rank.world_size) {
        throw std::invalid_argument("Invalid native training options");
    }
    if (static_cast<std::uint64_t>(options.num_workers) > std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error("Training worker count exceeds the native size range");
    }
    if (rank.rank >= 0 && options.batch_size % rank.world_size != 0) {
        throw std::invalid_argument("Training batch size must be divisible by world size");
    }
}

void initialize_environment() {
    initialize_activation_checkpointing();
    auto& context = at::globalContext();
    context.setUserEnabledCuDNN(true);
    context.setBenchmarkCuDNN(true);
    context.setAllowTF32CuBLAS(true);
    context.setAllowTF32CuDNN(true);
}

torch::Device training_device(const TrainingRank& rank) {
    const auto count = torch::cuda::device_count();
    if (rank.rank >= 0) {
        if (static_cast<std::size_t>(rank.world_size) != count ||
            rank.rank > std::numeric_limits<c10::DeviceIndex>::max()) {
            throw std::invalid_argument("Training ranks must match all visible CUDA devices");
        }
        return torch::Device(torch::kCUDA, static_cast<c10::DeviceIndex>(rank.rank));
    }
    return count ? torch::Device(torch::kCUDA, 0) : torch::Device(torch::kCPU);
}

void print_device(const TrainingRank& rank, const torch::Device& device) {
    if (rank.rank <= 0) {
        std::cout << "cuda device count: " << torch::cuda::device_count() << '\n';
        if (!std::cout) throw std::runtime_error("Cannot write training device count");
    }
    std::cout << "rank: " << rank.rank << ", current device: " << device.str() << '\n';
    if (!std::cout) throw std::runtime_error("Cannot write training device selection");
}

void initialize_save_directory(const std::filesystem::path& path, const TrainingRank& rank,
                               TrainingDistributed& distributed) {
    ProcessInterrupt::check();
    if (rank.rank <= 0) std::filesystem::create_directories(path);
    // Ensure peers can inspect the directory on a first training run.
    distributed.barrier();
}

TrainingLoaderOptions loader_options(const CommonTrainingOptions& options,
                                    const TrainingRank& rank, const torch::Device& device) {
    TrainingLoaderOptions loader;
    loader.batch_size = options.batch_size;
    loader.workers = static_cast<std::size_t>(options.num_workers);
    loader.rank = rank.rank;
    loader.world_size = rank.world_size;
    loader.device = device;
    return loader;
}

EpochContext epoch_context(const TrainingRank& rank, TrainingDataLoader& loader,
                           TrainingDistributed& distributed) {
    EpochContext context;
    context.rank = rank.rank;
    context.world_size = rank.world_size;
    context.next_batch = [&loader] { return loader.next(); };
    context.before_forward = [&distributed] { distributed.before_forward(); };
    context.prepare_backward = [&distributed](const torch::Tensor& loss) { distributed.prepare_backward(loss); };
    context.reduce_gradients = [&distributed](torch::nn::Module&) { distributed.after_backward(); };
    return context;
}

}  // namespace

void train_image(const ImageTrainingOptions& options, TrainingRank rank) {
    validate_options(options, rank);
    initialize_environment();
    const auto device = training_device(rank);
    const c10::DeviceGuard device_guard(device);
    TrainingDistributed distributed(rank);
    initialize_save_directory(options.save_dir, rank, distributed);
    ImageFolder dataset(options.train_dataset, 256, 256, CompressionTrainingImpl::qp_count,
                        training_lambdas(options.lambdas, CompressionTrainingImpl::qp_count));
    DmciTraining model;
    const auto existing = load_existing_weights(options.save_dir, *model, rank.rank);
    print_device(rank, device);
    model->to(device);
    distributed.attach_model(*model, false, "DMCI");
    AdamW optimizer(model->parameters(), AdamWOptions(1e-4));
    if (existing.optimizer) load_adamw_state(optimizer, *existing.optimizer);
    TrainingDataLoader loader(dataset, loader_options(options, rank, device));
    const auto context = epoch_context(rank, loader, distributed);
    for (auto epoch = existing.begin_epoch; epoch < options.epochs; ++epoch) {
        ProcessInterrupt::check();
        loader.reset(epoch);
        train_image_epoch(*model, dataset, optimizer, epoch, context);
        ProcessInterrupt::check();
        if (rank.rank <= 0) save_training_status(options.save_dir, *model, optimizer, epoch);
    }
    ProcessInterrupt::check();
    if (rank.rank <= 0) save_training_checkpoint(options.save_dir, *model);
    distributed.finish();
}

void train_video(const VideoTrainingOptions& options, TrainingRank rank, CheckpointingProvider checkpointing) {
    validate_options(options, rank);
    if (options.model_path_i.empty()) throw std::invalid_argument("Video training requires model_path_i");
    const auto delay = options.model_structure == "ld" ? 1 : 8;
    // Validate choices before reading data or changing the training directory.
    video_training_strategy(options.training_scheduling, delay, options.model_structure);
    initialize_environment();
    const auto device = training_device(rank);
    const c10::DeviceGuard device_guard(device);
    TrainingDistributed distributed(rank);
    initialize_save_directory(options.save_dir, rank, distributed);
    VideoFolder dataset(options.train_dataset, 256, 256, CompressionTrainingImpl::qp_count,
                        training_lambdas(options.lambdas, CompressionTrainingImpl::qp_count), 5, delay);
    // Preserve the reference construction/RNG order: read intra state, build
    // and restore DMCI, then construct the predictive graph.
    const auto intra_state = read_training_state(options.model_path_i);
    DmciTraining intra;
    load_training_state(*intra, intra_state);
    intra->eval();
    VideoTrainingModel model(options.model_structure);
    const auto existing = load_existing_weights(options.save_dir, model.module(), rank.rank, options.pretrain_path);
    print_device(rank, device);
    model.module().to(device);
    intra->to(device);
    distributed.attach_model(model.module(), true, "DMC");
    AdamW optimizer(model.module().parameters(), AdamWOptions(1e-4));
    if (existing.optimizer) load_adamw_state(optimizer, *existing.optimizer);
    TrainingDataLoader loader(dataset, loader_options(options, rank, device));
    auto context = epoch_context(rank, loader, distributed);
    context.configure_checkpointing = std::move(checkpointing);
    for (auto epoch = existing.begin_epoch; epoch < options.epochs; ++epoch) {
        ProcessInterrupt::check();
        loader.reset(epoch);
        train_video_epoch(model, *intra, dataset, optimizer, epoch, options.training_scheduling, context);
        ProcessInterrupt::check();
        if (rank.rank <= 0) save_training_status(options.save_dir, model.module(), optimizer, epoch);
    }
    ProcessInterrupt::check();
    if (rank.rank <= 0) save_training_checkpoint(options.save_dir, model.module());
    distributed.finish();
}

}  // namespace uf::training
