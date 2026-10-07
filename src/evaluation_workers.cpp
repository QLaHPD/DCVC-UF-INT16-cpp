// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "evaluation_workflow.hpp"
#include "uf/evaluation.hpp"
#include "uf/process.hpp"
#ifdef UF_HAS_DCVC2021
#include "dcvc2021_evaluation_internal.hpp"
#endif
#ifdef UF_HAS_DCVC_TCM
#include "dcvc_tcm_evaluation_internal.hpp"
#endif
#ifdef UF_HAS_DCVC_HEM
#include "dcvc_hem_evaluation_internal.hpp"
#endif
#ifdef UF_HAS_DCVC_DC
#include "uf/dcvc_dc_evaluation.hpp"
#endif

#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstring>
#include <deque>
#include <exception>
#include <functional>
#include <iostream>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>
#include <fcntl.h>
#include <poll.h>
#include <sys/eventfd.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

namespace uf::evaluation {
namespace {

constexpr int worker_socket = 3;
using Check = std::function<void()>;
constexpr std::array<char, 4> magic{'U', 'F', 'E', '1'};

std::runtime_error system_error(const char* message) {
    return std::runtime_error(std::string(message) + ": " + std::strerror(errno));
}
void wait_socket(int socket, short wanted, const Check& check) {
    for (;;) {
        check();
        pollfd descriptor{socket, wanted, 0};
        const auto status = poll(&descriptor, 1, 100);
        if (status < 0 && errno == EINTR) continue;
        if (status < 0) throw system_error("Cannot poll evaluation worker");
        if (descriptor.revents & POLLNVAL) throw std::runtime_error("Invalid evaluation worker socket");
        if (descriptor.revents & wanted) return;
        if (descriptor.revents & (POLLERR | POLLHUP)) {
            if (wanted == POLLIN) return;  // recvmsg identifies EOF.
            throw std::runtime_error("Evaluation worker socket closed");
        }
    }
}

// Transfer a sealed memory file instead of putting a potentially large
// verbose frame report in the socket packet.
void send_document(int socket, const Json& value, const Check& check) {
    check();
    std::ostringstream serialized;
    write_json(value, serialized, -1, -1);
    const auto text = serialized.str();
    FileDescriptor file(memfd_create("uf_evaluation_message", MFD_CLOEXEC | MFD_ALLOW_SEALING));
    if (file.get() < 0) throw system_error("Cannot create evaluation message memory file");
    write_descriptor(file.get(), text.data(), text.size());
    if (fcntl(file.get(), F_ADD_SEALS, F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_WRITE | F_SEAL_SEAL) < 0) {
        throw system_error("Cannot seal evaluation message");
    }
    iovec data{const_cast<char*>(magic.data()), magic.size()};
    alignas(cmsghdr) std::array<unsigned char, CMSG_SPACE(sizeof(int))> ancillary{};
    msghdr message{};
    message.msg_iov = &data; message.msg_iovlen = 1;
    message.msg_control = ancillary.data(); message.msg_controllen = ancillary.size();
    auto* rights = CMSG_FIRSTHDR(&message);
    rights->cmsg_level = SOL_SOCKET; rights->cmsg_type = SCM_RIGHTS; rights->cmsg_len = CMSG_LEN(sizeof(int));
    const auto descriptor = file.get();
    std::memcpy(CMSG_DATA(rights), &descriptor, sizeof(descriptor));
    for (;;) {
        wait_socket(socket, POLLOUT, check);
        const auto count = sendmsg(socket, &message, MSG_DONTWAIT | MSG_NOSIGNAL);
        if (count < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) continue;
        if (count < 0) throw system_error("Cannot send evaluation message");
        if (count != static_cast<ssize_t>(magic.size())) throw std::runtime_error("Incomplete evaluation socket packet");
        return;
    }
}

std::optional<Json> receive_document(int socket, const Check& check) {
    for (;;) {
        wait_socket(socket, POLLIN, check);
        std::array<char, magic.size()> header{};
        iovec data{header.data(), header.size()};
        alignas(cmsghdr) std::array<unsigned char, CMSG_SPACE(2 * sizeof(int))> ancillary{};
        msghdr message{};
        message.msg_iov = &data; message.msg_iovlen = 1;
        message.msg_control = ancillary.data(); message.msg_controllen = ancillary.size();
        std::vector<FileDescriptor> files;
        files.reserve(2);  // Own all installed descriptors before validation.
        const auto count = recvmsg(socket, &message, MSG_DONTWAIT | MSG_CMSG_CLOEXEC);
        if (count < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) continue;
        if (count < 0) throw system_error("Cannot receive evaluation message");
        for (auto* rights = CMSG_FIRSTHDR(&message); rights; rights = CMSG_NXTHDR(&message, rights)) {
            if (rights->cmsg_level == SOL_SOCKET && rights->cmsg_type == SCM_RIGHTS && rights->cmsg_len >= CMSG_LEN(0)) {
                const auto number = (rights->cmsg_len - CMSG_LEN(0)) / sizeof(int);
                for (std::size_t index = 0; index < number; ++index) {
                    int descriptor;
                    std::memcpy(&descriptor, CMSG_DATA(rights) + index * sizeof(int), sizeof(descriptor));
                    files.emplace_back(descriptor);
                }
            }
        }
        if (!count) return std::nullopt;
        if (count != static_cast<ssize_t>(header.size()) || header != magic ||
            (message.msg_flags & (MSG_TRUNC | MSG_CTRUNC)) || files.size() != 1) {
            throw std::runtime_error("Malformed evaluation socket packet");
        }
        struct stat status{};
        if (fstat(files[0].get(), &status) < 0) throw system_error("Cannot inspect evaluation message");
        constexpr int seals = F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_WRITE | F_SEAL_SEAL;
        const auto actual_seals = fcntl(files[0].get(), F_GET_SEALS);
        if (!S_ISREG(status.st_mode) || status.st_size <= 0 || actual_seals < 0 || (actual_seals & seals) != seals ||
            static_cast<std::uint64_t>(status.st_size) > std::numeric_limits<std::size_t>::max()) {
            throw std::runtime_error("Invalid evaluation message memory file");
        }
        std::string text(static_cast<std::size_t>(status.st_size), '\0');
        for (std::size_t position = 0; position < text.size();) {
            check();
            const auto remaining = std::min<std::size_t>(text.size() - position, 64 * 1024);
            const auto read = pread(files[0].get(), text.data() + position, remaining, static_cast<off_t>(position));
            if (read < 0 && errno == EINTR) continue;
            if (read <= 0) throw std::runtime_error("Cannot read evaluation message memory file");
            position += static_cast<std::size_t>(read);
        }
        std::istringstream input(text);
        return read_json(input);
    }
}

struct Worker {
    FileDescriptor socket;
    std::unique_ptr<ChildProcess> process;
    std::optional<std::size_t> point;
    bool ready = false, stopping = false;
};

class Pool {
public:
    ~Pool() { if (!finished_) cancel(); }
    std::vector<std::unique_ptr<Worker>> workers;
    void start(const Json& initialization) {
        const auto executable = std::filesystem::read_symlink("/proc/self/exe").string();
        const auto index = workers.size();
        ProcessInterrupt::check();
        auto worker = std::make_unique<Worker>();
        int descriptors[2];
        if (socketpair(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC | SOCK_NONBLOCK, 0, descriptors) < 0) {
            throw system_error("Cannot create evaluation worker socket");
        }
        worker->socket.reset(descriptors[0]);
        FileDescriptor child(descriptors[1]);
        worker->process = std::make_unique<ChildProcess>(
            std::vector<std::string>{executable, "_evaluation_worker"},
            std::vector<std::pair<int, int>>{{child.get(), worker_socket}}, true);
        child.reset();
        workers.push_back(std::move(worker));
        auto document = initialization;
        document["kind"] = "init"; document["protocol"] = "uf-evaluation-v1";
        document["parent"] = getpid();
        // Python SpawnProcess-N starts at N=1.
        document["worker_index"] = index + 1;
        send_document(workers.back()->socket.get(), document, [] { ProcessInterrupt::check(); });
    }
    void stop(Worker& worker) {
        if (worker.stopping) return;
        send_document(worker.socket.get(), Json{{"kind", "stop"}}, [] { ProcessInterrupt::check(); });
        worker.stopping = true;
        shutdown(worker.socket.get(), SHUT_WR);
    }
    void finish() {
        for (auto& worker : workers) stop(*worker);
        for (;;) {
            ProcessInterrupt::check();
            bool active = false;
            for (auto& worker : workers) {
                const auto status = worker->process->poll();
                if (!status) active = true;
                else if (*status != 0) throw std::runtime_error("Evaluation worker exited unsuccessfully during shutdown");
            }
            if (!active) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        workers.clear();
        finished_ = true;
    }
private:
    bool finished_ = false;
    void cancel() noexcept {
        for (auto& worker : workers) {
            if (worker->socket.get() >= 0) shutdown(worker->socket.get(), SHUT_RDWR);
            if (worker->process) worker->process->signal(SIGTERM);
        }
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        for (;;) {
            bool active = false;
            for (auto& worker : workers) {
                try { if (worker->process && !worker->process->poll_for_cleanup()) active = true; }
                catch (...) { if (worker->process) worker->process->signal(SIGKILL); }
            }
            if (!active || std::chrono::steady_clock::now() >= deadline) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        for (auto& worker : workers) if (worker->process) worker->process->signal(SIGKILL);
        workers.clear();  // ChildProcess reaps any remaining process.
    }
};

// CPython's idle semaphore counts completed calls, including calls whose
// worker has already taken another queued item. Readiness alone is not a permit.
class DocumentExecutor {
public:
    DocumentExecutor(std::size_t worker_limit, std::function<Json(std::size_t)> initialization)
        : worker_limit_(worker_limit), initialization_(std::move(initialization)),
          wake_(eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK)) {
        if (worker_limit_ > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw std::invalid_argument("Evaluation worker count is too large");
        }
        if (wake_.get() < 0) throw system_error("Cannot create evaluation manager wakeup");
    }
    ~DocumentExecutor() {
        if (!finished_) {
            try { finish(false); }
            catch (...) {
                // A wait/allocation failure must not destroy a joinable
                // std::thread during stack unwinding. Ordinary worker errors
                // have already joined in finish().
                if (manager_.joinable()) {
                    {
                        std::lock_guard<std::mutex> lock(mutex_);
                        if (!error_) error_ = std::current_exception();
                        closed_ = true;
                        try { wake(); } catch (...) {}
                    }
                    manager_.join();
                }
            }
        }
    }
    void submit(Json document) {
        ProcessInterrupt::check();
        std::lock_guard<std::mutex> lock(mutex_);
        if (error_) std::rethrow_exception(error_);
        if (closed_) throw std::logic_error("Evaluation executor is closed");
        if (!worker_limit_) throw std::invalid_argument("Evaluation requires at least one worker");
        const auto index = outcomes_.size();
        outcomes_.emplace_back();
        try { pending_.push_back({index, std::move(document)}); }
        catch (...) { outcomes_.pop_back(); throw; }
        try {
            // Exactly one adjustment per submitted point. Even at the worker
            // limit, consume an available permit as ProcessPoolExecutor does.
            if (idle_permits_) --idle_permits_;
            else if (pool_.workers.size() < worker_limit_) {
                pool_.start(initialization_(pool_.workers.size()));
            }
            if (!manager_.joinable()) manager_ = std::thread([this] { manage(); });
            wake();
        } catch (...) {
            error_ = std::current_exception();
            closed_ = true;
            // The manager also polls periodically if a wakeup itself failed.
            changed_.notify_all();
            throw;
        }
    }
    std::vector<Json> finish(bool progress = true) {
        if (finished_) throw std::logic_error("Evaluation executor already finished");
        std::unique_lock<std::mutex> lock(mutex_);
        closed_ = true;
        try { wake(); } catch (...) { if (!error_) error_ = std::current_exception(); }
        std::size_t collected = 0;
        std::optional<std::size_t> failed_point;
        auto collect = [&] {
            const auto previous = collected;
            while (!failed_point && collected < outcomes_.size() && outcomes_[collected].settled) {
                if (outcomes_[collected].failure) failed_point = collected;
                else ++collected;
            }
            if (progress && collected != previous) {
                std::cerr << "\rEvaluated " << collected << '/' << outcomes_.size() << " rate points" << std::flush;
            }
        };
        // No thread is started until the first submission. Empty input never
        // starts a worker, and failed thread creation must still unwind Pool.
        while (manager_.joinable() && !done_) {
            collect();
            changed_.wait_for(lock, std::chrono::milliseconds(100));
            try { ProcessInterrupt::check(); }
            catch (...) {
                if (!error_) error_ = std::current_exception();
                try { wake(); } catch (...) {}
            }
        }
        collect();
        lock.unlock();
        if (manager_.joinable()) manager_.join();
        finished_ = true;
        if (progress && !outcomes_.empty()) std::cerr << '\n';
        if (error_) std::rethrow_exception(error_);
        // Drain all submitted calls before surfacing the first failure in
        // Future.result() order. Transport/initializer failures break the pool.
        if (failed_point) throw std::runtime_error("Evaluation worker: " + *outcomes_[*failed_point].failure);
        if (!progress) return {};  // Unwinding needs shutdown, not a result copy.
        std::vector<Json> results;
        results.reserve(outcomes_.size());
        for (auto& outcome : outcomes_) results.push_back(std::move(outcome.log));
        return results;
    }
private:
    struct Pending { std::size_t index; Json document; };
    struct Outcome { Json log; std::optional<std::string> failure; bool settled = false; };
    Pool pool_;
    std::size_t worker_limit_;
    std::function<Json(std::size_t)> initialization_;
    FileDescriptor wake_;
    std::mutex mutex_;
    std::condition_variable changed_;
    std::thread manager_;
    std::deque<Pending> pending_;
    std::vector<Outcome> outcomes_;
    std::size_t completed_ = 0, idle_permits_ = 0;
    bool closed_ = false, done_ = false, finished_ = false;
    std::exception_ptr error_;

    void wake() {
        const std::uint64_t value = 1;
        for (;;) {
            const auto count = write(wake_.get(), &value, sizeof(value));
            if (count == static_cast<ssize_t>(sizeof(value)) || (count < 0 && errno == EAGAIN)) return;
            if (count < 0 && errno == EINTR) continue;
            throw system_error("Cannot wake evaluation manager");
        }
    }
    void assign() {
        for (auto& pointer : pool_.workers) {
            auto& worker = *pointer;
            if (!worker.ready || worker.point || worker.stopping) continue;
            if (pending_.empty()) {
                if (closed_) pool_.stop(worker);
                continue;
            }
            const auto& point = pending_.front();
            worker.point = point.index;
            send_document(worker.socket.get(), Json{{"kind", "point"}, {"index", point.index},
                {"point", point.document}}, [] { ProcessInterrupt::check(); });
            pending_.pop_front();
        }
    }
    void response(Worker& worker, const std::optional<Json>& message) {
        if (!message) {
            worker.socket.reset();
            if (!worker.stopping) throw std::runtime_error("Evaluation worker disconnected before completing its point");
            return;
        }
        const auto kind = message->at("kind").get<std::string>();
        if (kind == "failed" && !message->contains("index")) {
            throw std::runtime_error("Evaluation worker: " + message->at("error").get<std::string>());
        }
        if (kind == "ready" && !worker.ready && !worker.point) {
            worker.ready = true;
        } else if ((kind == "result" || kind == "failed") && worker.ready && worker.point &&
                   message->at("index").get<std::size_t>() == *worker.point) {
            auto& outcome = outcomes_.at(*worker.point);
            if (kind == "result") outcome.log = message->at("log");
            else outcome.failure = message->at("error").get<std::string>();
            outcome.settled = true;
            worker.point.reset();
            ++completed_;
            ++idle_permits_;
            changed_.notify_all();
        } else throw std::runtime_error("Unexpected evaluation worker response");
    }
    void manage() noexcept {
        try {
            for (;;) {
                ProcessInterrupt::check();
                std::unique_lock<std::mutex> lock(mutex_);
                if (error_) std::rethrow_exception(error_);
                assign();
                if (closed_ && completed_ == outcomes_.size()) {
                    pool_.finish();
                    done_ = true;
                    changed_.notify_all();
                    return;
                }
                // Keep a snapshot while submit() can append a worker. Worker
                // descriptors are closed only by this manager or after join.
                std::vector<pollfd> descriptors{{wake_.get(), POLLIN, 0}};
                for (const auto& worker : pool_.workers) descriptors.push_back({worker->socket.get(), POLLIN, 0});
                lock.unlock();
                const auto count = poll(descriptors.data(), descriptors.size(), 100);
                if (count < 0 && errno == EINTR) continue;
                if (count < 0) throw system_error("Cannot poll evaluation pool");
                lock.lock();
                if (error_) std::rethrow_exception(error_);
                if (descriptors[0].revents & (POLLERR | POLLHUP | POLLNVAL)) {
                    throw std::runtime_error("Evaluation manager wakeup disconnected");
                }
                if (descriptors[0].revents & POLLIN) {
                    std::uint64_t value;
                    ssize_t bytes;
                    do { bytes = read(wake_.get(), &value, sizeof(value)); } while (bytes < 0 && errno == EINTR);
                    if (bytes != static_cast<ssize_t>(sizeof(value)) && !(bytes < 0 && errno == EAGAIN)) {
                        throw system_error("Cannot read evaluation manager wakeup");
                    }
                }
                for (std::size_t index = 1; index < descriptors.size(); ++index) {
                    if (descriptors[index].revents & (POLLIN | POLLHUP | POLLERR | POLLNVAL)) {
                        auto& worker = *pool_.workers[index - 1];
                        response(worker, receive_document(worker.socket.get(), [] { ProcessInterrupt::check(); }));
                    }
                }
                for (const auto& worker : pool_.workers) {
                    const auto status = worker->process->poll();
                    if (status && (!worker->stopping || *status != 0)) {
                        throw std::runtime_error("Evaluation worker exited unexpectedly");
                    }
                }
            }
        } catch (...) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!error_) error_ = std::current_exception();
            done_ = true;
            changed_.notify_all();
        }
    }
};

std::size_t uf_worker_limit(const VideoEvaluationOptions& options) {
    if (options.workers < 1) throw std::invalid_argument("Evaluation requires at least one worker");
    return static_cast<std::size_t>(options.workers);
}
Json uf_initialization(const VideoEvaluationOptions& options, int gpu_count, std::size_t index) {
    if (gpu_count < 1) throw std::invalid_argument("Evaluation workers require a CUDA device");
    const auto gpu = static_cast<int>((static_cast<std::int64_t>(index) + 1) % gpu_count);
    const auto selector = options.cuda_indices ? options.cuda_indices->at(gpu) : gpu;
    return Json{{"backend", "uf"}, {"cuda_selector", std::to_string(selector)},
        {"settings", evaluation_settings_json(options)}};
}

}  // namespace

struct EvaluationExecutor::Impl {
    DocumentExecutor executor;
    Impl(const VideoEvaluationOptions& options, int gpu_count)
        : executor(uf_worker_limit(options), [options, gpu_count](std::size_t index) {
            return uf_initialization(options, gpu_count, index);
        }) {}
};
EvaluationExecutor::EvaluationExecutor(const VideoEvaluationOptions& options, int gpu_count)
    : impl_(std::make_unique<Impl>(options, gpu_count)) {}
EvaluationExecutor::~EvaluationExecutor() = default;
void EvaluationExecutor::submit(const EvaluationPoint& point) { impl_->executor.submit(evaluation_point_json(point)); }
std::vector<Json> EvaluationExecutor::finish() { return impl_->executor.finish(); }

std::vector<Json> evaluate_points(const VideoEvaluationOptions& options,
                                const std::vector<EvaluationPoint>& points, int gpu_count) {
    if (points.empty()) return {};
    EvaluationExecutor executor(options, gpu_count);
    for (const auto& point : points) executor.submit(point);
    return executor.finish();
}

std::vector<Json> evaluate_documents(const std::vector<Json>& initializations,
                                    const std::vector<Json>& points) {
    if (points.empty()) return {};
    DocumentExecutor executor(initializations.size(), [&initializations](std::size_t index) {
        return initializations.at(index);
    });
    for (const auto& point : points) executor.submit(point);
    return executor.finish();
}

int run_evaluation_worker() {
    const ProcessInterrupt interruption;
    FileDescriptor socket(worker_socket);
    const auto parent = getppid();
    if (prctl(PR_SET_PDEATHSIG, SIGTERM) < 0) throw system_error("Cannot watch evaluation worker parent");
    if (getppid() != parent) return 0;
    const Check check = [parent] {
        ProcessInterrupt::check();
        if (getppid() != parent) throw std::runtime_error("Evaluation worker parent exited");
    };
    try {
        const auto init = receive_document(socket.get(), check);
        if (!init) return 0;
        if (init->at("kind") != "init" || init->at("protocol") != "uf-evaluation-v1" ||
            init->at("parent").get<pid_t>() != parent) throw std::runtime_error("Invalid evaluation worker initialization");
        const auto backend = init->value("backend", std::string("uf"));
        std::unique_ptr<EvaluationModels> models;
#ifdef UF_HAS_DCVC_DC
        std::unique_ptr<uf::dc::EvaluationModels> dc_models;
#endif
        std::function<Json(const Json&)> run;
        if (backend == "uf") {
            models = std::make_unique<EvaluationModels>(evaluation_settings(init->at("settings")),
                init->at("cuda_selector").get<std::string>());
            run = [&](const Json& document) {
                const auto point = evaluation_point(document);
                std::filesystem::create_directories(point.bin_folder);
                auto log = models->run(point);
                log["ds_name"] = point.dataset; log["seq"] = point.sequence; log["rate_idx"] = point.rate_index;
                return log;
            };
        } else if (backend == "dcvc2021") {
#ifdef UF_HAS_DCVC2021
            const auto use_cuda = init->at("settings").at("cuda").get<bool>();
            const auto worker_index = init->at("worker_index").get<int>();
            run = [use_cuda, worker_index](const Json& document) {
                return uf::dcvc2021::evaluate_document(document, use_cuda, worker_index);
            };
#else
            throw std::runtime_error("Original DCVC evaluation requires UF_BUILD_DCVC2021");
#endif
        } else if (backend == "dcvc_tcm") {
#ifdef UF_HAS_DCVC_TCM
            const auto use_cuda = init->at("settings").at("cuda").get<bool>();
            const auto worker_index = init->at("worker_index").get<int>();
            run = [use_cuda, worker_index](const Json& document) {
                return uf::tcm::evaluate_document(document, use_cuda, worker_index);
            };
#else
            throw std::runtime_error("TCM evaluation requires UF_BUILD_DCVC_TCM");
#endif
        } else if (backend == "dcvc_hem") {
#ifdef UF_HAS_DCVC_HEM
            const auto use_cuda = init->at("settings").at("cuda").get<bool>();
            const auto worker_index = init->at("worker_index").get<int>();
            run = [use_cuda, worker_index](const Json& document) {
                return uf::hem::evaluate_document(document, use_cuda, worker_index);
            };
#else
            throw std::runtime_error("HEM evaluation requires UF_BUILD_DCVC_HEM");
#endif
        } else if (backend == "dcvc_dc") {
#ifdef UF_HAS_DCVC_DC
            dc_models = std::make_unique<uf::dc::EvaluationModels>(init->at("settings"),
                init->at("worker_index").get<int>());
            run = [&](const Json& document) { return dc_models->run(document); };
#else
            throw std::runtime_error("DC evaluation requires UF_BUILD_DCVC_DC");
#endif
        } else throw std::invalid_argument("Unknown evaluation worker backend");
        send_document(socket.get(), Json{{"kind", "ready"}}, check);
        for (;;) {
            const auto message = receive_document(socket.get(), check);
            if (!message || message->at("kind") == "stop") return 0;
            if (message->at("kind") != "point") throw std::runtime_error("Invalid evaluation worker point");
            const auto index = message->at("index").get<std::size_t>();
            try {
                auto log = run(message->at("point"));
                send_document(socket.get(), Json{{"kind", "result"}, {"index", index}, {"log", std::move(log)}}, check);
            } catch (const std::exception& error) {
                check();
                send_document(socket.get(), Json{{"kind", "failed"}, {"index", index}, {"error", error.what()}}, check);
            }
        }
    } catch (const std::exception& error) {
        if (ProcessInterrupt::requested() || getppid() != parent) return ProcessInterrupt::requested() ? 128 + ProcessInterrupt::requested() : 0;
        try { send_document(socket.get(), Json{{"kind", "failed"}, {"error", error.what()}}, check); } catch (...) {}
        std::cerr << "Evaluation worker initialization/protocol error: " << error.what() << '\n';
        return 1;
    }
}

}  // namespace uf::evaluation
