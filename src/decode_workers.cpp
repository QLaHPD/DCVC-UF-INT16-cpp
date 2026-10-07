#include "uf/decode_workers.hpp"

#include "uf/events.hpp"
#include "uf/process.hpp"
#include "uf/python_json.hpp"
#include "uf/python_path.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <deque>
#include <list>
#include <memory>
#include <optional>
#include <stdexcept>
#include <thread>
#include <utility>
#include <fcntl.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace uf {
namespace {

using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
constexpr const char* protocol = "uf-cpp-decode-worker-v1";
constexpr std::size_t maximum_document = 4u * 1024u * 1024u;
constexpr std::size_t maximum_event = 2u * 1024u * 1024u;
constexpr auto termination_grace = std::chrono::seconds(10);

Json bytes(const std::string& value) {
    return Json::binary(std::vector<std::uint8_t>(value.begin(), value.end()));
}

std::string string_bytes(const Json& value) {
    const auto& data = value.get_binary();
    std::string result(data.size(), '\0');
    if (!data.empty()) std::memcpy(result.data(), data.data(), data.size());
    return result;
}

void close_on_exec(int descriptor) {
    const int flags = fcntl(descriptor, F_GETFD);
    if (flags < 0 || fcntl(descriptor, F_SETFD, flags | FD_CLOEXEC) < 0) {
        throw std::runtime_error("Cannot protect decode worker descriptor");
    }
}

// CBOR byte strings preserve every filename byte and CLI value. The private
// protocol does not require Unix paths to be valid UTF-8 JSON strings.
void write_document(int descriptor, const Json& document) {
    const auto serialized = Json::to_cbor(document);
    if (serialized.size() > maximum_document) throw std::runtime_error("Decode worker document exceeds 4 MiB");
    if (ftruncate(descriptor, 0) < 0 || lseek(descriptor, 0, SEEK_SET) < 0) {
        throw std::runtime_error("Cannot rewind decode worker document");
    }
    write_descriptor(descriptor, serialized.data(), serialized.size());
}

Json read_document(int descriptor) {
    struct stat status{};
    if (fstat(descriptor, &status) < 0 || !S_ISREG(status.st_mode) ||
        status.st_size <= 0 || static_cast<std::uint64_t>(status.st_size) > maximum_document) {
        throw std::runtime_error("Invalid decode worker document");
    }
    std::vector<std::uint8_t> serialized(static_cast<std::size_t>(status.st_size));
    for (std::size_t offset = 0; offset < serialized.size();) {
        ProcessInterrupt::check();
        const auto amount = std::min<std::size_t>(serialized.size() - offset, 64u * 1024u);
        const auto count = pread(descriptor, serialized.data() + offset, amount, static_cast<off_t>(offset));
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) throw std::runtime_error("Cannot read decode worker document");
        offset += static_cast<std::size_t>(count);
    }
    return Json::from_cbor(serialized);
}

Json options_document(const cli::Options& values) {
    auto result = Json::object();
    for (const auto& [key, value] : values) result[key] = bytes(value);
    return result;
}

cli::Options read_options(const Json& document) {
    if (!document.is_object()) throw std::runtime_error("Decode worker options must be an object");
    cli::Options result;
    for (auto item = document.begin(); item != document.end(); ++item) {
        result.emplace(item.key(), string_bytes(item.value()));
    }
    if (!result.count("__input") || !result.count("__output_file")) {
        throw std::runtime_error("Decode worker omitted its input or output");
    }
    return result;
}

bool valid_device(const std::string& device) {
    if (device == "cpu") return true;
    if (device.rfind("cuda:", 0) != 0 || device.size() == 5) return false;
    for (std::size_t index = 5; index < device.size(); ++index) {
        if (device[index] < '0' || device[index] > '9') return false;
    }
    return true;
}

void discard_events(int descriptor) noexcept {
    std::array<char, 8192> buffer{};
    for (int pass = 0; pass < 16; ++pass) {
        const auto count = read(descriptor, buffer.data(), buffer.size());
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) return;
    }
}

struct Active {
    DecodeJob job;
    std::string device, event_tail, protocol_error;
    FileDescriptor input, result, events;
    bool terminal_event{};
    std::optional<Clock::time_point> terminate_at;
    std::unique_ptr<ChildProcess> process;
};

void read_events(Active& worker, bool final = false) {
    if (worker.events.get() < 0) return;
    std::array<char, 8192> buffer{};
    for (std::size_t pass = 0; final || pass < 16; ++pass) {
        ProcessInterrupt::check();
        const auto count = read(worker.events.get(), buffer.data(), buffer.size());
        if (count < 0 && errno == EINTR) continue;
        if (count < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) return;
        if (count < 0) throw std::runtime_error("Cannot read decode worker events");
        if (!count) {
            worker.events.reset();
            if (!worker.event_tail.empty()) throw std::runtime_error("Truncated decode worker event");
            return;
        }
        worker.event_tail.append(buffer.data(), static_cast<std::size_t>(count));
        std::size_t start = 0;
        for (;;) {
            const auto end = worker.event_tail.find('\n', start);
            if (end == std::string::npos) break;
            if (end - start > maximum_event) throw std::runtime_error("Decode worker event exceeds 2 MiB");
            const auto record = parse_python_json(worker.event_tail.substr(start, end - start));
            if (!record.is_object()) throw std::runtime_error("Decode worker event must be an object");
            const auto name = record.value("event", std::string{});
            if (name == "decode_completed" || name == "decode_failed") worker.terminal_event = true;
            deliver_event(record);
            start = end + 1;
        }
        worker.event_tail.erase(0, start);
        if (worker.event_tail.size() > maximum_event) throw std::runtime_error("Decode worker event exceeds 2 MiB");
    }
}

class Pool {
public:
    Pool(int workers, const std::vector<std::string>& devices) {
        if (workers < 1 || devices.empty()) throw std::invalid_argument("Decode workers require a positive count and devices");
        for (const auto& device : devices) {
            if (!valid_device(device)) throw std::invalid_argument("Invalid decode worker device: " + device);
        }
        for (int index = 0; index < workers; ++index) {
            free_.push_back(devices[static_cast<std::size_t>(index) % devices.size()]);
        }
    }

    ~Pool() { if (!finished_) cancel(); }

    bool has_capacity() const { return !free_.empty(); }
    bool running() const { return !active_.empty(); }

    void launch(const DecodeJob& job) {
        ProcessInterrupt::check();
        if (!has_capacity()) throw std::logic_error("No free decode worker slot");
        const auto worker = active_.emplace(active_.end());
        try {
            worker->job = job;
            worker->device = free_.front();
            auto options = job.options;
            options.erase("__input_file");
            options["__input"] = job.input.string();
            options["__output_file"] = job.output.string();
            options["__device"] = worker->device;
            options["__cuda"] = worker->device == "cpu" ? "false" : "true";
            options["__cuda_idx"] = worker->device == "cpu" ? "0" : worker->device.substr(5);
            worker->input = anonymous_file("/dev/shm/uf-decode-job-");
            worker->result = anonymous_file("/dev/shm/uf-decode-result-");
            write_document(worker->input.get(), {{"protocol", protocol}, {"parent", getpid()},
                                                {"options", options_document(options)}});
            auto pipe = process_pipe();
            const int flags = fcntl(pipe[0].get(), F_GETFL);
            if (flags < 0 || fcntl(pipe[0].get(), F_SETFL, flags | O_NONBLOCK) < 0) {
                throw std::runtime_error("Cannot set decode event pipe to nonblocking");
            }
            worker->events = std::move(pipe[0]);
            // Allocate/list the complete owner before starting a process. No
            // throwing allocations remain after ChildProcess successfully spawns.
            worker->process = std::make_unique<ChildProcess>(
                std::vector<std::string>{"/proc/self/exe", "_decode_worker"},
                std::vector<std::pair<int, int>>{{worker->input.get(), 0}, {worker->result.get(), 3}, {pipe[1].get(), 8}},
                true);
            free_.pop_front();
        } catch (...) {
            active_.erase(worker);
            throw;
        }
    }

    void poll() {
        ProcessInterrupt::check();
        for (auto worker = active_.begin(); worker != active_.end();) {
            try { read_events(*worker); }
            catch (const std::exception& error) {
                ProcessInterrupt::check();
                fault(*worker, error.what());
            }
            if (worker->terminate_at && Clock::now() >= *worker->terminate_at) {
                worker->process->signal(SIGKILL);
                worker->terminate_at.reset();
            }
            const auto status = worker->process->poll();
            if (!status) { ++worker; continue; }
            try { read_events(*worker, true); }
            catch (const std::exception& error) {
                ProcessInterrupt::check();
                fault(*worker, error.what());
            }
            DecodeOutcome outcome{worker->job.input, worker->job.output, "failed", worker->protocol_error,
                                  worker->process->returncode().value_or(*status)};
            try {
                const auto result = read_document(worker->result.get());
                if (result.at("protocol") != protocol) throw std::runtime_error("Unsupported decode worker result");
                const auto reported = result.at("status").get<std::string>();
                if (reported != "encoded" && reported != "failed") throw std::runtime_error("Invalid decode worker status");
                const auto error = string_bytes(result.at("error"));
                if (outcome.error.empty()) outcome.error = error;
                if (!*status && worker->protocol_error.empty()) outcome.status = reported;
                if (*status && outcome.error.empty()) outcome.error = "Decode worker exited without success";
            } catch (const std::exception& error) {
                ProcessInterrupt::check();
                outcome.status = "failed";
                if (outcome.error.empty()) outcome.error = error.what();
            }
            if (outcome.status == "encoded") ++report_.decoded;
            else {
                ++report_.failed;
                if (!worker->terminal_event) {
                    event("decode_failed", {{"source", python_path_text(outcome.input)}, {"error", outcome.error}});
                }
            }
            if (*status) {
                event("worker_failed", {{"source", python_path_text(outcome.input)},
                                        {"exitcode", outcome.returncode}, {"error", outcome.error}});
            }
            report_.outcomes.push_back(std::move(outcome));
            free_.push_back(worker->device);
            worker = active_.erase(worker);
        }
    }

    DecodeRun finish() {
        while (running()) {
            poll();
            if (running()) std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        finished_ = true;
        return std::move(report_);
    }

private:
    std::deque<std::string> free_;
    std::list<Active> active_;
    DecodeRun report_;
    bool finished_{};

    void fault(Active& worker, const std::string& error) {
        if (worker.protocol_error.empty()) {
            worker.protocol_error = error;
            worker.terminate_at = Clock::now() + termination_grace;
            worker.process->signal(SIGTERM);
        }
        worker.events.reset();
        worker.event_tail.clear();
    }

    void cancel() noexcept {
        for (auto& worker : active_) if (worker.process) worker.process->signal(SIGTERM);
        const auto deadline = Clock::now() + termination_grace;
        bool running = true;
        while (running && Clock::now() < deadline) {
            running = false;
            for (auto& worker : active_) {
                if (!worker.process) continue;
                // Discard queued events during teardown so a full event pipe
                // cannot prevent a worker from unwinding and removing its stage.
                if (worker.events.get() >= 0) discard_events(worker.events.get());
                try { if (!worker.process->poll_for_cleanup()) running = true; }
                catch (...) { running = true; }
            }
            if (running) std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        for (auto& worker : active_) if (worker.process) worker.process->signal(SIGKILL);
        // ChildProcess destruction reaps any process still alive after grace.
        active_.clear();
    }
};

}  // namespace

int run_decode_worker(const DecodeTask& decode, int input_descriptor,
                      int result_descriptor, int event_descriptor) {
    ProcessInterrupt interruption;
    FileDescriptor input(input_descriptor), result(result_descriptor), events(event_descriptor);
    std::string status = "failed", error;
    bool fatal = false;
    try {
        close_on_exec(input.get());
        close_on_exec(result.get());
        close_on_exec(events.get());
        const auto document = read_document(input.get());
        input.reset();
        if (document.at("protocol") != protocol) throw std::runtime_error("Unsupported decode worker protocol");
        const auto parent = document.at("parent").get<pid_t>();
        if (parent <= 0 || prctl(PR_SET_PDEATHSIG, SIGTERM) < 0 || getppid() != parent) {
            throw std::runtime_error("Decode worker parent is unavailable");
        }
        const auto options = read_options(document.at("options"));
        set_event_descriptor(events.get());
        try {
            const int code = decode(options);
            ProcessInterrupt::check();
            status = code == 0 ? "encoded" : "failed";
        } catch (const std::exception& exception) {
            ProcessInterrupt::check();
            error = std::string(exception.what()).substr(0, 8192);
            event("decode_failed", {{"source", options.at("__input")}, {"error", error}});
        }
    } catch (const std::exception& exception) {
        fatal = true;
        error = std::string(exception.what()).substr(0, 8192);
    } catch (...) {
        fatal = true;
        error = "Unknown decode worker failure";
    }
    set_event_descriptor(-1);
    events.reset();
    try { write_document(result.get(), {{"protocol", protocol}, {"status", status}, {"error", bytes(error)}}); }
    catch (...) { return ProcessInterrupt::requested() ? 128 + ProcessInterrupt::requested() : 1; }
    // Handled file/codec failures are normal process exits, as _decode_entry
    // catches them in Python. Protocol/setup faults remain abnormal exits.
    return fatal ? 1 : 0;
}

DecodeRun decode_jobs(const std::vector<DecodeJob>& jobs, int workers,
                      const std::vector<std::string>& devices) {
    Pool pool(workers, devices);
    std::size_t next = 0;
    while (next < jobs.size()) {
        ProcessInterrupt::check();
        while (next < jobs.size() && pool.has_capacity()) pool.launch(jobs[next++]);
        pool.poll();
        if (next < jobs.size() && !pool.has_capacity()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
    return pool.finish();
}

}  // namespace uf
