// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#include "uf/training_recompute.hpp"
#include "uf/process.hpp"

#include <ATen/SavedTensorHooks.h>
#include <ATen/ThreadLocalState.h>
#include <ATen/autocast_mode.h>
#include <c10/core/DeviceGuard.h>
#include <torch/csrc/autograd/engine.h>
#include <torch/csrc/autograd/graph_task.h>
#include <torch/csrc/autograd/saved_variable.h>
#include <torch/csrc/autograd/saved_variable_hooks.h>

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>

namespace uf::training {
namespace {

struct Frame;
struct Replay;

struct TensorMetadata {
    std::vector<std::int64_t> sizes;
    torch::ScalarType dtype;
    torch::Device device;
    explicit TensorMetadata(const torch::Tensor& tensor)
        : sizes(tensor.sizes().vec()), dtype(tensor.scalar_type()), device(tensor.device()) {}
    bool matches(const torch::Tensor& tensor) const {
        return tensor.sizes().vec() == sizes && tensor.scalar_type() == dtype && tensor.device() == device;
    }
};

struct Slot {
    const std::size_t index;
    const TensorMetadata metadata;
    struct Value { torch::Tensor tensor; bool consumed = false; };
    std::mutex mutex;
    std::map<std::int64_t, Value> values;
    Slot(std::size_t id, const torch::Tensor& tensor) : index(id), metadata(tensor) {}
    void save(std::int64_t task, const torch::Tensor& tensor) {
        if (!metadata.matches(tensor)) {
            throw ActivationCheckpointError("Recomputed activation has different shape, dtype or device at saved slot " +
                                            std::to_string(index));
        }
        const std::lock_guard<std::mutex> lock(mutex);
        if (!values.emplace(task, Value{tensor, false}).second) {
            throw ActivationCheckpointError("Recomputation saved an activation slot twice");
        }
    }
    torch::Tensor take(std::int64_t task) {
        const std::lock_guard<std::mutex> lock(mutex);
        const auto found = values.find(task);
        if (found == values.end()) throw ActivationCheckpointError("Recomputation did not populate the requested activation");
        if (found->second.consumed) throw ActivationCheckpointError("Checkpoint activation was unpacked twice in one backward task");
        found->second.consumed = true;
        auto tensor = std::move(found->second.tensor);
        found->second.tensor = torch::Tensor{};
        return tensor;
    }
    void erase(std::int64_t task) {
        const std::lock_guard<std::mutex> lock(mutex);
        values.erase(task);
    }
};

struct Replay {
    const std::int64_t task;
    // Weak entries ensure that destroying a SavedVariable immediately frees
    // its recomputed value, including values unused in a partial backward.
    std::vector<std::weak_ptr<Slot>> slots;
    std::mutex mutex;
    std::condition_variable changed;
    bool running = false, complete = false;
    std::thread::id owner;
    std::exception_ptr error;
    std::size_t packed = 0;
    explicit Replay(std::int64_t id) : task(id) {}
};

struct StopReplay final : std::exception {
    const Replay* replay;
    explicit StopReplay(const Replay* value) : replay(value) {}
    const char* what() const noexcept override { return "Required checkpoint activations recomputed"; }
};

struct HookScope {
    std::shared_ptr<Frame> frame;
    std::shared_ptr<Replay> replay;
};
thread_local HookScope* current_hooks = nullptr;

class HookGuard {
public:
    explicit HookGuard(HookScope& scope) : previous_(current_hooks) { current_hooks = &scope; }
    ~HookGuard() { current_hooks = previous_; }
    HookGuard(const HookGuard&) = delete;
    HookGuard& operator=(const HookGuard&) = delete;
private:
    HookScope* previous_;
};

struct AutocastState {
    const bool cache = at::autocast::is_autocast_cache_enabled();
    const std::array<bool, 2> enabled{at::autocast::is_autocast_enabled(at::kCPU),
                                      at::autocast::is_autocast_enabled(at::kCUDA)};
    const std::array<at::ScalarType, 2> dtype{at::autocast::get_autocast_dtype(at::kCPU),
                                              at::autocast::get_autocast_dtype(at::kCUDA)};
};

class AutocastGuard {
public:
    explicit AutocastGuard(const AutocastState& state) {
        for (std::size_t index = 0; index < devices_.size(); ++index) {
            at::autocast::set_autocast_dtype(devices_[index], state.dtype[index]);
            at::autocast::set_autocast_enabled(devices_[index], state.enabled[index]);
            at::autocast::increment_nesting();
        }
        at::autocast::set_autocast_cache_enabled(state.cache);
    }
    ~AutocastGuard() {
        for (std::size_t index = devices_.size(); index-- > 0;) {
            if (at::autocast::decrement_nesting() == 0) at::autocast::clear_cache();
            at::autocast::set_autocast_dtype(devices_[index], previous_.dtype[index]);
            at::autocast::set_autocast_enabled(devices_[index], previous_.enabled[index]);
        }
        at::autocast::set_autocast_cache_enabled(previous_.cache);
    }
    AutocastGuard(const AutocastGuard&) = delete;
    AutocastGuard& operator=(const AutocastGuard&) = delete;
private:
    const AutocastState previous_;
    const std::array<at::DeviceType, 2> devices_{at::kCPU, at::kCUDA};
};

struct Frame : std::enable_shared_from_this<Frame> {
    CheckpointFunction function;
    at::ThreadLocalState forward_state;
    const AutocastState autocast;
    std::vector<torch::autograd::SavedVariable> arguments;
    std::optional<torch::Device> device;
    std::mutex mutex;
    std::vector<std::weak_ptr<Slot>> slots;
    std::map<std::int64_t, std::shared_ptr<Replay>> replays;

    Frame(CheckpointFunction callback, const CheckpointInputs& inputs) : function(std::move(callback)) {
        arguments.reserve(inputs.size());
        for (const auto& input : inputs) {
            if (!device && input.defined()) device = input.device();
            // Construct these before installing this frame's hooks. A nested
            // checkpoint's inputs consequently belong to its parent's scope.
            arguments.emplace_back(input, false);
        }
        forward_state.set_grad_mode(true);
    }

    std::shared_ptr<Slot> record(const torch::Tensor& tensor) {
        ProcessInterrupt::check();
        const std::lock_guard<std::mutex> lock(mutex);
        if (slots.size() == slots.max_size()) throw std::overflow_error("Checkpoint saved-slot count overflow");
        auto slot = std::make_shared<Slot>(slots.size(), tensor);
        slots.push_back(slot);
        if (!device) device = tensor.device();
        return slot;
    }

    void clear_replay(std::int64_t task, const std::shared_ptr<Replay>& expected) {
        {
            const std::lock_guard<std::mutex> lock(mutex);
            const auto found = replays.find(task);
            if (found != replays.end() && found->second == expected) replays.erase(found);
        }
        for (const auto& weak : expected->slots) if (auto slot = weak.lock()) slot->erase(task);
    }

    std::shared_ptr<Replay> replay_for(std::int64_t task, bool in_backward) {
        std::shared_ptr<Replay> replay;
        {
            const std::lock_guard<std::mutex> lock(mutex);
            if (in_backward) {
                const auto found = replays.find(task);
                if (found != replays.end()) return found->second;
            }
            replay = std::make_shared<Replay>(task);
            replay->slots = slots;
            if (in_backward) replays.emplace(task, replay);
        }
        if (in_backward) {
            const std::weak_ptr<Frame> weak = shared_from_this();
            torch::autograd::Engine::get_default_engine().queue_callback([weak, task, replay] {
                if (auto frame = weak.lock()) frame->clear_replay(task, replay);
            });
        }
        return replay;
    }

    void recompute(const std::shared_ptr<Replay>& replay) {
        {
            std::unique_lock<std::mutex> lock(replay->mutex);
            while (replay->running) {
                if (replay->owner == std::this_thread::get_id()) {
                    throw ActivationCheckpointError("Checkpoint recursively requested its own unfinished recomputation");
                }
                ProcessInterrupt::check();
                replay->changed.wait_for(lock, std::chrono::milliseconds(100));
            }
            if (replay->error) std::rethrow_exception(replay->error);
            if (replay->complete) return;
            replay->running = true;
            replay->owner = std::this_thread::get_id();
        }
        std::exception_ptr failure;
        try {
            ProcessInterrupt::check();
            CheckpointInputs inputs;
            inputs.reserve(arguments.size());
            // Unpack before entering the replay hook scope: nested parent
            // recomputation must populate the parent's saved inputs first.
            for (const auto& argument : arguments) inputs.push_back(argument.unpack());
            const c10::OptionalDeviceGuard device_guard(device);
            const at::ThreadLocalStateGuard thread_state(forward_state);
            const AutocastGuard autocast_guard(autocast);
            const torch::AutoGradMode enable_grad(true);
            HookScope scope{shared_from_this(), replay};
            const HookGuard hooks(scope);
            try { function(inputs); }
            catch (const StopReplay& stop) { if (stop.replay != replay.get()) throw; }
            if (replay->packed != replay->slots.size()) {
                throw ActivationCheckpointError("Recomputation saved a different number of activation tensors");
            }
            ProcessInterrupt::check();
        } catch (...) { failure = std::current_exception(); }
        {
            const std::lock_guard<std::mutex> lock(replay->mutex);
            replay->error = failure;
            replay->complete = !failure;
            replay->running = false;
        }
        replay->changed.notify_all();
        if (failure) std::rethrow_exception(failure);
    }

    torch::Tensor unpack(const std::shared_ptr<Slot>& slot) {
        const auto graph_task = torch::autograd::get_current_graph_task_id();
        const bool in_backward = graph_task >= 0;
        static std::atomic<std::int64_t> outside_task{-2};
        std::int64_t task = graph_task;
        if (!in_backward) {
            auto next = outside_task.load();
            for (;;) {
                if (next == std::numeric_limits<std::int64_t>::min()) {
                    throw std::overflow_error("Checkpoint temporary-task counter overflow");
                }
                if (outside_task.compare_exchange_weak(next, next - 1)) { task = next; break; }
            }
        }
        auto replay = replay_for(task, in_backward);
        try {
            recompute(replay);
            auto tensor = slot->take(task);
            if (!in_backward) clear_replay(task, replay);
            return tensor;
        } catch (...) {
            if (!in_backward) clear_replay(task, replay);
            throw;
        }
    }
};

class OriginalHooks final : public torch::autograd::SavedVariableHooks {
public:
    explicit OriginalHooks(std::shared_ptr<Frame> frame) : frame_(std::move(frame)) {}
    void call_pack_hook(const torch::Tensor& tensor) override { slot_ = frame_->record(tensor); }
    torch::Tensor call_unpack_hook() override { return frame_->unpack(slot_); }
private:
    const std::shared_ptr<Frame> frame_;
    std::shared_ptr<Slot> slot_;
};

class ReplayHooks final : public torch::autograd::SavedVariableHooks {
public:
    explicit ReplayHooks(std::shared_ptr<Replay> replay) : replay_(std::move(replay)) {}
    void call_pack_hook(const torch::Tensor& tensor) override {
        ProcessInterrupt::check();
        const auto index = replay_->packed;
        if (index >= replay_->slots.size()) {
            throw ActivationCheckpointError("Recomputation saved more activation tensors than forward");
        }
        packed_ = tensor.requires_grad() ? tensor.detach() : tensor;
        if (auto slot = replay_->slots[index].lock()) slot->save(replay_->task, packed_);
        ++replay_->packed;
        // Match non-reentrant checkpoint early stopping. Outputs after the
        // last required save do not need to be reconstructed during backward.
        if (replay_->packed == replay_->slots.size()) throw StopReplay(replay_.get());
    }
    torch::Tensor call_unpack_hook() override { return packed_; }
private:
    const std::shared_ptr<Replay> replay_;
    torch::Tensor packed_;
};

class CheckpointEngine final : public torch::autograd::Engine {
public:
    ~CheckpointEngine() override { stop(); }
    std::unique_ptr<torch::autograd::SavedVariableHooks> get_default_saved_variable_hooks() override {
        if (!current_hooks) return nullptr;
        if (!at::SavedTensorDefaultHooks::is_enabled()) {
            const auto& message = at::SavedTensorDefaultHooks::get_disabled_error_message();
            throw ActivationCheckpointError(message.value_or("Saved tensor hooks are disabled"));
        }
        if (current_hooks->replay) return std::make_unique<ReplayHooks>(current_hooks->replay);
        return std::make_unique<OriginalHooks>(current_hooks->frame);
    }
};

torch::autograd::Engine& checkpoint_engine() {
    static CheckpointEngine engine;
    return engine;
}

struct InstallEngine {
    InstallEngine() {
        // Register before main, just as the Python frontend registers its
        // Engine factory before autograd's first use. Do not create/use the
        // base Engine or change factories after worker threads have started.
        torch::autograd::set_default_engine_stub(&checkpoint_engine);
    }
};
const InstallEngine installed_engine;

}  // namespace

void initialize_activation_checkpointing() {
    if (&torch::autograd::Engine::get_default_engine() != &checkpoint_engine()) {
        throw ActivationCheckpointError("Native checkpointing requires its process-start autograd Engine factory");
    }
}

CheckpointInputs checkpoint_activations(CheckpointFunction function, const CheckpointInputs& arguments) {
    if (!function) throw std::invalid_argument("Activation checkpoint callback is empty");
    ProcessInterrupt::check();
    if (!torch::GradMode::is_enabled()) return function(arguments);
    initialize_activation_checkpointing();
    if (!at::SavedTensorDefaultHooks::is_enabled()) {
        throw ActivationCheckpointError(at::SavedTensorDefaultHooks::get_disabled_error_message()
            .value_or("Saved tensor hooks are disabled"));
    }
    auto frame = std::make_shared<Frame>(std::move(function), arguments);
    HookScope scope{frame, {}};
    const HookGuard hooks(scope);
    return frame->function(arguments);
}

torch::Tensor checkpoint_tensor(CheckpointTensorFunction function, const CheckpointInputs& arguments) {
    if (!function) throw std::invalid_argument("Activation checkpoint callback is empty");
    auto results = checkpoint_activations([function = std::move(function)](const CheckpointInputs& inputs) {
        return CheckpointInputs{function(inputs)};
    }, arguments);
    if (results.size() != 1) throw ActivationCheckpointError("Expected one checkpoint output tensor");
    return std::move(results.front());
}

void ActivationCheckpointModule::set_use_checkpointing(bool enabled) {
    if (enabled) initialize_activation_checkpointing();
    use_checkpointing_ = enabled;
}

torch::Tensor ActivationCheckpointModule::run_tensor(CheckpointTensorFunction function,
                                                      const CheckpointInputs& arguments) const {
    if (!use_checkpointing_) return function(arguments);
    return checkpoint_tensor(std::move(function), arguments);
}

CheckpointInputs ActivationCheckpointModule::run_tensors(CheckpointFunction function,
                                                         const CheckpointInputs& arguments) const {
    if (!use_checkpointing_) return function(arguments);
    return checkpoint_activations(std::move(function), arguments);
}

}  // namespace uf::training
