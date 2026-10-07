// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#include "uf/training_loader.hpp"
#include "uf/process.hpp"
#include "pareto_random.hpp"
#include "training_loader_process.hpp"

#include <ATen/CPUGeneratorImpl.h>
#include <ATen/core/Generator.h>
#include <c10/core/DeviceGuard.h>
#include <c10/util/thread_name.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <exception>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace uf::training {
namespace {

std::size_t local_sample_count(std::size_t count, const TrainingLoaderOptions& options) {
    if (options.rank < 0) return count;
    const auto world = static_cast<std::size_t>(options.world_size);
    return count / world + (count % world != 0);
}

std::vector<std::int64_t> sample_indices(std::size_t count, const TrainingLoaderOptions& options,
                                        std::int64_t epoch) {
    const auto seed = options.rank >= 0 ? static_cast<std::uint64_t>(epoch)
        : static_cast<std::uint64_t>(torch::empty({}, torch::TensorOptions()
              .dtype(torch::kInt64).device(torch::kCPU)).random_().item<std::int64_t>());
    auto generator = at::make_generator<at::CPUGeneratorImpl>(seed);
    const auto shuffled = torch::randperm(static_cast<std::int64_t>(count), generator,
        torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
    const auto* values = shuffled.data_ptr<std::int64_t>();
    std::vector<std::int64_t> indices;
    const auto samples = local_sample_count(count, options);
    if (samples > indices.max_size()) throw std::overflow_error("Training sampler exceeds addressable memory");
    indices.reserve(samples);
    if (options.rank < 0) {
        if (count) indices.assign(values, values + count);
    } else {
        const auto world = static_cast<std::size_t>(options.world_size);
        if (samples && world > std::numeric_limits<std::size_t>::max() / samples) {
            throw std::overflow_error("Distributed training sampler size overflow");
        }
        // DistributedSampler repeats a prefix to ceil(N/world)*world, then
        // takes rank::world. Modulo selects the same padded entries directly.
        for (std::size_t index = 0; index < samples; ++index) {
            indices.push_back(values[(static_cast<std::size_t>(options.rank) + index * world) % count]);
        }
    }
    return indices;
}

}  // namespace

struct TrainingDataLoader::Impl {
    struct Task { std::size_t id, offset; };
    struct Batch {
        std::size_t id;
        DatasetSample tensors;
        std::exception_ptr error;
    };

    const std::size_t dataset_size;
    SampleReader reader;
    std::function<nlohmann::json()> snapshotter;
    TrainingLoaderOptions options;
    std::int64_t batch_size;
    std::int64_t epoch = 0, base_seed = 0;
    const std::size_t batches, prefetch_capacity;
    const bool pin;
    bool started = false, finished = false;
    std::atomic<bool> stopping{false};
    std::vector<std::int64_t> indices;
    std::size_t sent = 0, received = 0;
    std::mutex mutex;
    std::condition_variable changed;
    std::vector<std::deque<Task>> tasks;
    std::deque<Batch> unpinned;
    std::map<std::size_t, Batch> ready;
    std::exception_ptr fatal;
    std::vector<std::thread> workers;
    std::vector<std::unique_ptr<loader_detail::ProcessWorker>> processes;
    std::thread pin_worker;

    static std::int64_t checked_batch_size(std::size_t count, const SampleReader& reader,
                                          const TrainingLoaderOptions& options) {
        if (!reader || options.batch_size <= 0 || !options.workers || !options.prefetch_factor ||
            options.world_size <= 0 || (options.rank >= 0 && options.rank >= options.world_size) ||
            (options.rank < 0 && options.world_size != 1) ||
            static_cast<std::uint64_t>(options.world_size) > std::numeric_limits<std::size_t>::max() ||
            count > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()) ||
            (options.rank < 0 && !count)) {
            throw std::invalid_argument("Invalid native training loader options or dataset size");
        }
        if (options.rank >= 0 && options.batch_size % options.world_size != 0) {
            throw std::invalid_argument("Training batch size must be divisible by world size");
        }
        if (options.device.is_cuda() && options.device.index() < 0) {
            throw std::invalid_argument("Training loader needs the indexed model CUDA device");
        }
        if (options.workers > std::numeric_limits<std::size_t>::max() / options.prefetch_factor) {
            throw std::overflow_error("Training prefetch count overflow");
        }
        const auto size = options.rank >= 0 ? options.batch_size / options.world_size : options.batch_size;
        if (static_cast<std::uint64_t>(size) > std::numeric_limits<std::size_t>::max()) {
            throw std::overflow_error("Training batch exceeds addressable memory");
        }
        return size;
    }

    Impl(std::size_t count, SampleReader sample_reader, TrainingLoaderOptions loader_options,
         std::function<nlohmann::json()> snapshot = {})
        : dataset_size(count), reader(std::move(sample_reader)), snapshotter(std::move(snapshot)),
          options(std::move(loader_options)),
          batch_size(checked_batch_size(dataset_size, reader, options)),
          batches(local_sample_count(count, options) / static_cast<std::size_t>(batch_size)),
          prefetch_capacity(options.workers * options.prefetch_factor),
          pin(options.pin_memory && options.device.is_cuda()) {}

    ~Impl() { shutdown(); }

    void dispatch() {
        if (sent >= batches) return;
        tasks[sent % options.workers].push_back({sent, sent * static_cast<std::size_t>(batch_size)});
        ++sent;
    }

    void fail(std::exception_ptr error) {
        {
            const std::lock_guard<std::mutex> lock(mutex);
            if (stopping.load()) return;
            if (!fatal) fatal = std::move(error);
            stopping.store(true);
        }
        changed.notify_all();
    }

    void worker_loop(std::size_t worker) {
        try {
            torch::NoGradGuard no_grad;
            pareto_detail::PythonRandom random(base_seed, worker);
            const RandomBelow draw = [&](std::uint64_t upper) { return random.below(upper); };
            for (;;) {
                Task task;
                {
                    std::unique_lock<std::mutex> lock(mutex);
                    changed.wait_for(lock, std::chrono::milliseconds(100),
                        [&] { return stopping.load() || !tasks[worker].empty(); });
                    if (stopping.load()) return;
                    if (tasks[worker].empty()) {
                        lock.unlock();
                        // Detect an idle process dying even before it receives
                        // another task, as the Python worker liveness check does.
                        if (snapshotter) processes[worker]->check_alive();
                        continue;
                    }
                    task = tasks[worker].front();
                    tasks[worker].pop_front();
                }
                Batch batch{task.id, {}, {}};
                try {
                    if (snapshotter) {
                        std::vector<std::int64_t> selected;
                        selected.reserve(static_cast<std::size_t>(batch_size));
                        for (std::int64_t index = 0; index < batch_size; ++index) {
                            selected.push_back(indices[task.offset + static_cast<std::size_t>(index)]);
                        }
                        const auto check = [this] {
                            ProcessInterrupt::check();
                            if (stopping.load()) throw std::runtime_error("Training loader transport stopped");
                        };
                        auto result = processes[worker]->fetch(task.id, selected, check);
                        batch.tensors = std::move(result.tensors);
                        batch.error = std::move(result.error);
                    } else {
                        std::vector<DatasetSample> samples;
                        samples.reserve(static_cast<std::size_t>(batch_size));
                        for (std::int64_t index = 0; index < batch_size; ++index) {
                            if (stopping.load()) return;
                            ProcessInterrupt::check();
                            samples.push_back(reader(indices[task.offset + static_cast<std::size_t>(index)], draw));
                        }
                        batch.tensors = collate_samples(samples);
                    }
                } catch (...) {
                    if (stopping.load()) return;
                    // Remote dataset exceptions are returned in result.error.
                    // A transport/protocol/dead-process error is loader-wide.
                    if (snapshotter) throw;
                    batch.error = std::current_exception();
                }
                {
                    const std::lock_guard<std::mutex> lock(mutex);
                    if (stopping.load()) return;
                    if (pin) unpinned.push_back(std::move(batch));
                    else ready.emplace(batch.id, std::move(batch));
                }
                changed.notify_all();
            }
        } catch (...) { if (!stopping.load()) fail(std::current_exception()); }
    }

    void pin_loop() {
        try {
            // The reference applies this in its pinning thread too.
            torch::set_num_threads(1);
            c10::setThreadName("pt_data_pin");
            torch::NoGradGuard no_grad;
            const c10::DeviceGuard device(options.device);
            for (;;) {
                Batch batch;
                {
                    std::unique_lock<std::mutex> lock(mutex);
                    changed.wait(lock, [&] { return stopping.load() || !unpinned.empty(); });
                    if (stopping.load()) return;
                    batch = std::move(unpinned.front());
                    unpinned.pop_front();
                }
                if (!batch.error) {
                    try {
                        for (auto& tensor : batch.tensors) {
                            if (stopping.load()) return;
                            ProcessInterrupt::check();
                            tensor = tensor.pin_memory();
                        }
                    } catch (...) { batch.error = std::current_exception(); }
                }
                {
                    const std::lock_guard<std::mutex> lock(mutex);
                    if (stopping.load()) return;
                    ready.emplace(batch.id, std::move(batch));
                }
                changed.notify_all();
            }
        } catch (...) { fail(std::current_exception()); }
    }

    void start() {
        if (started) return;
        ProcessInterrupt::check();
        // BaseDataLoaderIter consumes its worker base seed before the lazy
        // RandomSampler draws a separate seed from the global Torch generator.
        base_seed = torch::empty({}, torch::TensorOptions().dtype(torch::kInt64)
            .device(torch::kCPU)).random_().item<std::int64_t>();
        indices = sample_indices(dataset_size, options, epoch);
        indices.resize(batches * static_cast<std::size_t>(batch_size));  // DataLoader drop_last=True.
        tasks.resize(options.workers);
        stopping.store(false);
        sent = received = 0;
        fatal = {};
        started = true;
        try {
            if (snapshotter) {
                // Snapshot after the epoch has updated patch/frame settings.
                // Each child restores the parent's already-read manifest.
                const auto snapshot = snapshotter();
                processes.reserve(options.workers);
                for (std::size_t index = 0; index < options.workers; ++index) {
                    ProcessInterrupt::check();
                    processes.push_back(std::make_unique<loader_detail::ProcessWorker>(
                        snapshot, index, base_seed, batch_size));
                }
            }
            if (pin) pin_worker = std::thread([this] { pin_loop(); });
            workers.reserve(options.workers);
            for (std::size_t index = 0; index < options.workers; ++index) {
                workers.emplace_back([this, index] { worker_loop(index); });
            }
            {
                const std::lock_guard<std::mutex> lock(mutex);
                for (std::size_t index = 0; index < std::min(batches, prefetch_capacity); ++index) dispatch();
            }
            changed.notify_all();
        } catch (...) {
            shutdown();
            throw;
        }
    }

    void shutdown() {
        bool abort;
        {
            const std::lock_guard<std::mutex> lock(mutex);
            abort = !finished || received != batches || ProcessInterrupt::requested() || static_cast<bool>(fatal);
            stopping.store(true);
        }
        changed.notify_all();
        for (auto& process : processes) process->stop(abort);
        for (auto& worker : workers) if (worker.joinable()) worker.join();
        if (pin_worker.joinable()) pin_worker.join();
        workers.clear();
        // All workers receive EOF/TERM together. Wait for the reference's
        // five-second worker shutdown interval, then kill/reap survivors.
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        for (;;) {
            bool done = true;
            for (auto& process : processes) {
                try { if (!process->reap()) done = false; }
                catch (...) { done = false; }
            }
            if (done || std::chrono::steady_clock::now() >= deadline) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        for (auto& process : processes) process->kill();
        processes.clear();
        tasks.clear();
        unpinned.clear();
        ready.clear();
        indices.clear();
        started = false;
    }

    std::optional<DatasetSample> next() {
        if (finished) return std::nullopt;
        start();
        if (received == batches) {
            finished = true;
            shutdown();
            return std::nullopt;
        }
        Batch batch;
        {
            std::unique_lock<std::mutex> lock(mutex);
            for (;;) {
                ProcessInterrupt::check();
                if (fatal) std::rethrow_exception(fatal);
                if (const auto found = ready.find(received); found != ready.end()) {
                    batch = std::move(found->second);
                    ready.erase(found);
                    ++received;
                    dispatch();  // Replace one consumed task, including a failed batch.
                    break;
                }
                if (stopping.load()) throw std::runtime_error("Training loader stopped while fetching a batch");
                changed.wait_for(lock, std::chrono::milliseconds(100));
            }
        }
        changed.notify_all();
        if (batch.error) std::rethrow_exception(batch.error);
        return std::move(batch.tensors);
    }
};

TrainingDataLoader::TrainingDataLoader(std::size_t size, SampleReader reader, TrainingLoaderOptions options)
    : impl_(std::make_unique<Impl>(size, std::move(reader), std::move(options))) {}

TrainingDataLoader::TrainingDataLoader(ImageFolder& dataset, TrainingLoaderOptions options)
    : impl_(std::make_unique<Impl>(dataset.size(),
        [&dataset](std::int64_t index, const RandomBelow& random) { return dataset.get(index, random); },
        std::move(options), [&dataset] { return dataset.worker_snapshot(); })) {}

TrainingDataLoader::TrainingDataLoader(VideoFolder& dataset, TrainingLoaderOptions options)
    : impl_(std::make_unique<Impl>(dataset.size(),
        [&dataset](std::int64_t index, const RandomBelow& random) { return dataset.get(index, random); },
        std::move(options), [&dataset] { return dataset.worker_snapshot(); })) {}

TrainingDataLoader::~TrainingDataLoader() = default;

void TrainingDataLoader::reset(std::int64_t epoch) {
    impl_->shutdown();
    impl_->epoch = epoch;
    impl_->finished = false;
}

std::optional<DatasetSample> TrainingDataLoader::next() { return impl_->next(); }
std::size_t TrainingDataLoader::batch_count() const { return impl_->batches; }
std::int64_t TrainingDataLoader::local_batch_size() const { return impl_->batch_size; }

}  // namespace uf::training
