// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#include "uf/training_distributed.hpp"
#include "uf/process.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef UF_HAS_TRAINING_NCCL
#include <torch/csrc/distributed/c10d/GroupRegistry.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/PrefixStore.hpp>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>
#include <torch/csrc/distributed/c10d/comm.hpp>
#include <torch/csrc/distributed/c10d/logger.hpp>
#include <torch/csrc/distributed/c10d/reducer.hpp>
#endif

namespace uf::training {
namespace {

void check_rank(const TrainingRank& rank) {
    if (rank.world_size <= 0 || (rank.rank < 0 && rank.world_size != 1) ||
        (rank.rank >= 0 && (rank.rank >= rank.world_size ||
                           rank.rank > std::numeric_limits<c10::DeviceIndex>::max())) ||
        rank.master_address.empty() || !rank.master_port) {
        throw std::invalid_argument("Invalid native training rank or rendezvous options");
    }
}

#ifdef UF_HAS_TRAINING_NCCL
constexpr std::size_t broadcast_bucket_bytes = 250 * 1024 * 1024;

class GroupRegistration {
public:
    GroupRegistration() = default;
    ~GroupRegistration() { try { reset(); } catch (...) {} }
    GroupRegistration(const GroupRegistration&) = delete;
    GroupRegistration& operator=(const GroupRegistration&) = delete;
    void attach(const c10::intrusive_ptr<c10d::ProcessGroup>& group) {
        if (registered_) throw std::logic_error("Training process group is already registered");
        c10d::register_process_group("0", group);
        registered_ = true;
    }
    void reset() {
        if (registered_) {
            c10d::unregister_process_group("0");
            registered_ = false;
        }
    }
private:
    bool registered_ = false;
};

void await_work(const c10::intrusive_ptr<c10d::Work>& work) {
    while (!work->isCompleted()) {
        ProcessInterrupt::check();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ProcessInterrupt::check();
    // WorkNCCL::wait establishes CUDA stream dependencies and reports errors.
    if (!work->wait()) throw std::runtime_error("Native training collective was aborted");
}
#endif

}  // namespace

bool training_nccl_available() noexcept {
#ifdef UF_HAS_TRAINING_NCCL
    return true;
#else
    return false;
#endif
}

struct TrainingDistributed::Impl {
    TrainingRank rank;
    bool finished = false, find_unused = false;
    torch::nn::Module* model = nullptr;
#ifdef UF_HAS_TRAINING_NCCL
    c10::intrusive_ptr<c10d::Store> store;
    c10::intrusive_ptr<c10d::ProcessGroupNCCL> backend;
    c10::intrusive_ptr<c10d::ProcessGroup> group;
    GroupRegistration registration;
    std::shared_ptr<c10d::Reducer> reducer;
    std::shared_ptr<c10d::Logger> logger;
    std::atomic<bool> stop_monitor{false};
    std::thread interrupt_monitor;
#endif

    explicit Impl(TrainingRank options) : rank(std::move(options)) {
        check_rank(rank);
        if (rank.rank < 0) return;
#ifdef UF_HAS_TRAINING_NCCL
        ProcessInterrupt::check();
        c10d::TCPStoreOptions store_options;
        store_options.port = rank.master_port;
        store_options.numWorkers = static_cast<std::size_t>(rank.world_size);
        store_options.timeout = c10d::kProcessGroupNCCLDefaultTimeout;
        store_options.waitWorkers = true;
        const auto* agent_setting = std::getenv("TORCHELASTIC_USE_AGENT_STORE");
        const bool agent_store = agent_setting && std::string_view(agent_setting) == "True";
        std::string agent_attempt;
        if (agent_store) {
            const auto* attempt = std::getenv("TORCHELASTIC_RESTART_COUNT");
            if (!attempt) throw std::invalid_argument("Agent store requires TORCHELASTIC_RESTART_COUNT");
            agent_attempt = attempt;
            // env:// agent rendezvous always connects as a client and uses
            // TCPStore's defaults, ignoring USE_LIBUV for this branch.
            store_options.isServer = false;
        } else {
            store_options.isServer = rank.rank == 0;
            store_options.multiTenant = true;
            const auto* libuv = std::getenv("USE_LIBUV");
            store_options.useLibUV = !libuv || std::string_view(libuv) == "1";
        }
        auto tcp_store = c10::make_intrusive<c10d::TCPStore>(rank.master_address, store_options);
        store = std::move(tcp_store);
        if (agent_store) store = c10::make_intrusive<c10d::PrefixStore>("/worker/attempt_" + agent_attempt, store);
        // Retain the installed Python frontend's nested PrefixStore arguments;
        // the CUDA backend has its own namespace inside the default group.
        store = c10::make_intrusive<c10d::PrefixStore>("default_pg", store);
        store = c10::make_intrusive<c10d::PrefixStore>("0/", store);
        auto backend_store = c10::make_intrusive<c10d::PrefixStore>("cuda/", store);
        auto nccl_options = c10d::ProcessGroupNCCL::Options::create(false);
        nccl_options->timeout = c10d::kProcessGroupNCCLDefaultTimeout;
        nccl_options->group_name = "0";
        // Python identifies the default group with an empty global-rank list.
        backend = c10::make_intrusive<c10d::ProcessGroupNCCL>(
            backend_store, rank.rank, rank.world_size, std::move(nccl_options));
        group = c10::make_intrusive<c10d::ProcessGroup>(store, rank.rank, rank.world_size,
            c10::make_intrusive<c10d::ProcessGroup::Options>("nccl", c10d::kProcessGroupNCCLDefaultTimeout));
        const c10::intrusive_ptr<c10d::Backend> cuda_backend = backend;
        group->setBackend(c10::DeviceType::CUDA, c10d::ProcessGroup::BackendType::NCCL, cuda_backend);
        group->setGroupName("0");
        group->setGroupDesc("default_pg");
        group->setSequenceNumberForGroup();
        registration.attach(group);
        // Backward can wait inside the autograd engine. Abort NCCL from a
        // separate thread on INT/TERM so that wait can release the controller.
        interrupt_monitor = std::thread([this] {
            while (!stop_monitor.load()) {
                if (ProcessInterrupt::requested()) {
                    try { backend->abort(std::string("UF training interrupted")); } catch (...) {}
                    return;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
        });
#else
        throw std::runtime_error("Distributed training requires UF_BUILD_TRAINING_NCCL and NCCL-enabled LibTorch");
#endif
    }

    ~Impl() {
#ifdef UF_HAS_TRAINING_NCCL
        stop_monitor.store(true);
        if (interrupt_monitor.joinable()) interrupt_monitor.join();
        if (!finished && backend) {
            try { backend->abort(std::string("UF trainer exited before completion")); } catch (...) {}
        }
        // Logger owns a reducer reference; destroy it first. Reducer removes
        // its parameter autograd hooks when its final reference is released.
        logger.reset();
        reducer.reset();
#endif
    }

    void barrier() {
        if (rank.rank < 0) return;
#ifdef UF_HAS_TRAINING_NCCL
        c10d::BarrierOptions options;
        options.device_ids = {rank.rank};
        options.device = torch::Device(torch::kCUDA, static_cast<c10::DeviceIndex>(rank.rank));
        await_work(group->barrier(options));
#endif
    }

    void attach(torch::nn::Module& module, bool unused, const std::string& module_name) {
        if (rank.rank < 0) return;
#ifdef UF_HAS_TRAINING_NCCL
        if (model || finished) throw std::logic_error("Training reducer already attached or finished");
        std::vector<torch::Tensor> parameters, state;
        std::unordered_map<std::size_t, std::string> names;
        for (const auto& entry : module.named_parameters()) {
            const auto& parameter = entry.value();
            if (!parameter.is_cuda() || parameter.device().index() != rank.rank) {
                throw std::invalid_argument("Distributed training parameters must be on the rank CUDA device");
            }
            state.push_back(parameter);
            if (parameter.requires_grad()) {
                names.emplace(parameters.size(), entry.key());
                parameters.push_back(parameter);
            }
        }
        if (parameters.empty()) throw std::invalid_argument("Training reducer has no differentiable parameters");
        for (const auto& buffer : module.buffers()) state.push_back(buffer);
        ProcessInterrupt::check();
        c10d::verify_params_across_processes(group, parameters, std::nullopt);
        {
            torch::NoGradGuard no_grad;
            c10d::broadcast_coalesced(group, state, broadcast_bucket_bytes, 0);
        }
        ProcessInterrupt::check();
        std::vector<bool> sparse(parameters.size(), false);
        const std::vector<std::size_t> limits = unused
            ? std::vector<std::size_t>{c10d::kDefaultFirstBucketBytes, c10d::kDefaultBucketBytesCap}
            : std::vector<std::size_t>{static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())};
        auto assignment = c10d::compute_bucket_assignment_by_size(parameters, limits, sparse);
        auto& buckets = std::get<0>(assignment);
        auto& bucket_limits = std::get<1>(assignment);
        std::reverse(buckets.begin(), buckets.end());
        std::reverse(bucket_limits.begin(), bucket_limits.end());
        reducer = std::make_shared<c10d::Reducer>(std::move(parameters), std::move(buckets),
            bucket_limits, group, std::move(sparse), c10d::kDefaultBucketBytesCap,
            unused, false, std::move(names), c10d::kDefaultFirstBucketBytes);
        logger = std::make_shared<c10d::Logger>(reducer);
        reducer->set_logger(logger);
        logger->set_construction_data_and_log(module_name, {rank.rank}, rank.rank, true, false, false);
        model = &module;
        find_unused = unused;
#else
        (void)module; (void)unused; (void)module_name;
#endif
    }

    void before_forward() {
        if (rank.rank < 0) return;
#ifdef UF_HAS_TRAINING_NCCL
        if (!reducer || finished) throw std::logic_error("Native training reducer is not active");
        ProcessInterrupt::check();
        logger->set_runtime_stats_and_log();
        reducer->prepare_for_forward();
        reducer->rebuild_buckets();
        // Default DDP broadcasts buffers before every forward, including
        // buffers a model replaces after construction.
        const auto buffers = model->buffers();
        if (!buffers.empty()) {
            torch::NoGradGuard no_grad;
            c10d::broadcast_coalesced(group, buffers, broadcast_bucket_bytes, 0);
        }
#endif
    }

    void prepare_backward(const torch::Tensor& loss) {
        if (rank.rank < 0) return;
#ifdef UF_HAS_TRAINING_NCCL
        if (!reducer || finished) throw std::logic_error("Native training reducer is not active");
        // Python's training forward returns only loss and string diagnostics.
        // The scalar loss is consequently the complete tensor output set.
        reducer->prepare_for_backward(find_unused ? std::vector<torch::Tensor>{loss}
                                                  : std::vector<torch::Tensor>{});
#else
        (void)loss;
#endif
    }

    void after_backward() {
        if (rank.rank < 0) return;
#ifdef UF_HAS_TRAINING_NCCL
        if (!reducer || finished) throw std::logic_error("Native training reducer is not active");
        ProcessInterrupt::check();
        // LibTorch's autograd hooks already averaged buckets by world size.
        // Do not reduce again before the epoch's clip_grad_norm_ call.
        reducer->check_finalized();
#endif
    }

    void finish() {
        if (finished) return;
        barrier();
#ifdef UF_HAS_TRAINING_NCCL
        stop_monitor.store(true);
        if (interrupt_monitor.joinable()) interrupt_monitor.join();
        logger.reset();
        reducer.reset();
        if (backend) backend->shutdown();
        registration.reset();
        group.reset();
        backend.reset();
        store.reset();
#endif
        finished = true;
    }
};

TrainingDistributed::TrainingDistributed(TrainingRank rank)
    : impl_(std::make_unique<Impl>(std::move(rank))) {}
TrainingDistributed::~TrainingDistributed() = default;
void TrainingDistributed::barrier() { impl_->barrier(); }
void TrainingDistributed::attach_model(torch::nn::Module& model, bool unused, const std::string& name) {
    impl_->attach(model, unused, name);
}
void TrainingDistributed::before_forward() { impl_->before_forward(); }
void TrainingDistributed::prepare_backward(const torch::Tensor& loss) { impl_->prepare_backward(loss); }
void TrainingDistributed::after_backward() { impl_->after_backward(); }
void TrainingDistributed::finish() { impl_->finish(); }

}  // namespace uf::training
