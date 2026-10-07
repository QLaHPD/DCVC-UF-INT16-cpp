#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "uf/stream_pipeline.hpp"
#include "uf/process.hpp"
#include "uf/events.hpp"
#include "uf/python_numeric.hpp"
#include "uf/utf8.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <csignal>
#include <limits>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <vector>
#include <fcntl.h>
#include <poll.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace uf {
namespace {

constexpr std::uint64_t state_magic = 0x5546435050535432ull;
struct alignas(64) State {
    std::uint64_t magic{state_magic};
    std::atomic<int> status{SharedStreamState::pending};
    std::atomic<int> near{0};
    pthread_mutex_t policy_mutex{};
    int owner = 0, frames = 8;
    bool measured = false;
};
static_assert(std::atomic<int>::is_always_lock_free,
              "UF process-shared stream state requires lock-free integer atomics");

void mutex_check(int error, const char* operation) {
    if (error) throw std::system_error(error, std::generic_category(), operation);
}

void initialize_policy_mutex(State& state) {
    pthread_mutexattr_t attributes;
    mutex_check(pthread_mutexattr_init(&attributes), "Cannot initialize UF buffer mutex attributes");
    try {
        mutex_check(pthread_mutexattr_setpshared(&attributes, PTHREAD_PROCESS_SHARED),
                    "Cannot share UF buffer mutex");
        mutex_check(pthread_mutexattr_settype(&attributes, PTHREAD_MUTEX_RECURSIVE),
                    "Cannot configure UF recursive buffer mutex");
        mutex_check(pthread_mutex_init(&state.policy_mutex, &attributes),
                    "Cannot initialize UF buffer mutex");
    } catch (...) {
        pthread_mutexattr_destroy(&attributes);
        throw;
    }
    pthread_mutexattr_destroy(&attributes);
}

class PolicyLock {
public:
    explicit PolicyLock(State& state, bool interruptible = true) : mutex_(&state.policy_mutex) {
        for (;;) {
            if (interruptible) ProcessInterrupt::check();
            const auto error = pthread_mutex_trylock(mutex_);
            if (!error) return;
            if (error != EBUSY) mutex_check(error, "Cannot lock UF buffer policy");
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    }
    ~PolicyLock() { pthread_mutex_unlock(mutex_); }
    PolicyLock(const PolicyLock&) = delete;
    PolicyLock& operator=(const PolicyLock&) = delete;
private:
    pthread_mutex_t* mutex_;
};

bool near_download_end(const std::string& line) {
    constexpr const char* prefix = "UF_DOWNLOAD:";
    if (line.rfind(prefix, 0) != 0) return false;
    std::array<std::string, 6> parts;
    std::size_t begin = 12;
    for (std::size_t i = 0; i < parts.size(); ++i) {
        const auto end = line.find(':', begin);
        if ((i + 1 < parts.size()) != (end != std::string::npos)) return false;
        parts[i] = line.substr(begin, end == std::string::npos ? end : end - begin);
        if (end != std::string::npos) begin = end + 1;
    }
    const auto number = [](const std::string& value) -> std::optional<double> {
        try {
            const double result = python_float(value);
            if (std::isfinite(result) && result >= 0) return result;
        } catch (const std::invalid_argument&) {}
        return std::nullopt;
    };
    const auto index = number(parts[0]), count = number(parts[1]);
    const auto bytes = number(parts[2]), total = number(parts[3]), estimate = number(parts[4]);
    if (parts[5] == "finished") return true;
    if (index && count && *count > 0) return *count - *index <= 20;
    const auto size = total && *total > 0 ? total : estimate;
    return size && *size > 0 && bytes && *bytes >= *size * .95;
}

}  // namespace

struct SharedStreamState::Impl {
    FileDescriptor file;
    State* value{};
    Impl(int descriptor, bool create) {
        if (create) {
            file = anonymous_file("/dev/shm/uf-stream-state-");
            if (ftruncate(file.get(), sizeof(State)) != 0) throw std::runtime_error("Cannot size UF stream state");
        } else {
            file.reset(fcntl(descriptor, F_DUPFD_CLOEXEC, 8));
            if (file.get() < 0) throw std::runtime_error("Cannot duplicate UF stream state");
            struct stat info{};
            if (fstat(file.get(), &info) != 0 || !S_ISREG(info.st_mode) || info.st_size != sizeof(State)) {
                throw std::runtime_error("Invalid UF stream state descriptor");
            }
        }
        void* mapping = mmap(nullptr, sizeof(State), PROT_READ | PROT_WRITE, MAP_SHARED, file.get(), 0);
        if (mapping == MAP_FAILED) throw std::runtime_error("Cannot map UF stream state");
        value = static_cast<State*>(mapping);
        if (create) {
            new (value) State;
            try { initialize_policy_mutex(*value); }
            catch (...) {
                value->~State();
                munmap(value, sizeof(State));
                value = nullptr;
                throw;
            }
        } else if (value->magic != state_magic) {
            munmap(value, sizeof(State));
            value = nullptr;
            throw std::runtime_error("Unsupported UF stream state protocol");
        }
    }
    // Peers retain independent mappings. Do not destroy their shared mutex
    // when one mapping closes; storage lives until the last mapping/FD closes.
    ~Impl() { if (value) munmap(value, sizeof(State)); }
};

SharedStreamState::SharedStreamState() : impl_(std::make_unique<Impl>(-1, true)) {}
SharedStreamState::SharedStreamState(int descriptor) : impl_(std::make_unique<Impl>(descriptor, false)) {}
SharedStreamState::~SharedStreamState() = default;
int SharedStreamState::descriptor() const { return impl_->file.get(); }
int SharedStreamState::status() const { return impl_->value->status.load(); }
void SharedStreamState::set_status(int status) { impl_->value->status.store(status); }
bool SharedStreamState::near_end() const { return impl_->value->near.load() != 0; }
void SharedStreamState::set_near_end() { impl_->value->near.store(1); }
void SharedStreamState::claim() {
    const PolicyLock lock(*impl_->value);
    if (!impl_->value->measured && !impl_->value->owner)
        impl_->value->owner = static_cast<int>(getpid());
}
void SharedStreamState::release(int process_id) {
    // Teardown must release ownership after a foreground interrupt as well.
    const PolicyLock lock(*impl_->value, false);
    const int owner = process_id ? process_id : static_cast<int>(getpid());
    if (impl_->value->owner == owner) impl_->value->owner = 0;
}
int SharedStreamState::capacity() const {
    const PolicyLock lock(*impl_->value);
    return impl_->value->frames;
}
std::optional<int> SharedStreamState::observe(std::uint64_t frames, double fps) {
    if (!frames || !std::isfinite(fps) || fps <= 0) return std::nullopt;
    const double elapsed = frames / fps;
    const PolicyLock lock(*impl_->value);
    if (impl_->value->measured || impl_->value->owner != static_cast<int>(getpid()) || elapsed < 5) {
        return std::nullopt;
    }
    const double measured = std::ceil(fps * 2);
    if (measured > std::numeric_limits<int>::max()) throw std::runtime_error("UF stream buffer calibration overflow");
    const int capacity = std::max(1, static_cast<int>(measured));
    impl_->value->frames = capacity;
    impl_->value->measured = true;
    // BufferPolicy.observe publishes while holding its recursive lock. Keep
    // peers from applying the new capacity before this event is submitted.
    event("stream_buffer_calibrated", {{"fps", fps}, {"sample_seconds", elapsed},
        {"buffer_frames", capacity}});
    return capacity;
}

struct PreparedDownload::Impl {
    RemoteCookies cookies;
    std::shared_ptr<SharedStreamState> shared{std::make_shared<SharedStreamState>()};
    FileDescriptor stream, errors, progress;
    std::unique_ptr<ChildProcess> process;
    std::mutex termination_mutex;
    std::atomic<bool> stop_monitor{false}, monitor_finished{false};
    std::thread monitor;
    bool close_attempted{};

    explicit Impl(const RemoteJob& remote) : cookies(remote.cookies) {
        try {
            auto output = process_pipe();
            auto updates = process_pipe();
            errors = python_temporary_anonymous_file();
            FileDescriptor null_input(open("/dev/null", O_RDONLY | O_CLOEXEC));
            if (null_input.get() < 0) throw std::runtime_error("Cannot open null download input");
            auto command = ytdlp_command(remote.ytdlp_binary, cookies.path());
            command.insert(command.end(), {"--progress", "--newline", "--progress-delta", "0.2",
                "--progress-template", "download:UF_DOWNLOAD:%(progress.fragment_index)s:%(progress.fragment_count)s:"
                "%(progress.downloaded_bytes)s:%(progress.total_bytes)s:%(progress.total_bytes_estimate)s:%(progress.status)s",
                "--no-playlist", "-f", python_filesystem_encode(remote.video_format),
                "-o", "-", "--", python_filesystem_encode(remote.url)});
            process = std::make_unique<ChildProcess>(command, std::vector<std::pair<int, int>>{
                {null_input.get(), 0}, {output[1].get(), 1}, {updates[1].get(), 2}}, true, true);
            stream = std::move(output[0]);
            progress = std::move(updates[0]);
            monitor = std::thread([this] { watch(); });
        } catch (...) {
            close();
            throw;
        }
    }

    void terminate_download() {
        // The foreground destructor and an interrupted monitor may both close
        // this process. Serialize termination; cleanup waits ignore INT/TERM.
        std::lock_guard<std::mutex> lock(termination_mutex);
        if (!process || process->poll_for_cleanup()) return;
        process->signal(SIGTERM);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
        while (!process->poll_for_cleanup()) {
            if (std::chrono::steady_clock::now() >= deadline) {
                process->signal(SIGKILL);
                process->wait_for_cleanup();
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    }

    void line(std::string value) {
        const auto trimmed = python_strip(utf8_replace(value));
        if (trimmed.rfind("UF_DOWNLOAD:", 0) == 0) {
            if (near_download_end(trimmed)) shared->set_near_end();
        } else write_descriptor(errors.get(), value.data(), value.size());
    }

    void watch() noexcept {
        struct Finished {
            std::atomic<bool>& flag;
            ~Finished() { flag.store(true); }
        } finished{monitor_finished};
        try {
            std::array<char, 8192> buffer{};
            std::string pending;
            for (;;) {
                if (stop_monitor.load()) break;
                ProcessInterrupt::check();
                pollfd descriptor{progress.get(), POLLIN, 0};
                const int ready = poll(&descriptor, 1, 100);
                if (ready < 0 && errno == EINTR) continue;
                if (ready < 0 || (ready > 0 && (descriptor.revents & (POLLERR | POLLNVAL))))
                    throw std::runtime_error("Cannot poll yt-dlp progress");
                if (!ready) continue;
                ProcessInterrupt::check();
                const auto count = read(progress.get(), buffer.data(), buffer.size());
                if (count < 0 && errno == EINTR) continue;
                if (count < 0) throw std::runtime_error("Cannot read yt-dlp progress");
                if (!count) break;
                pending.append(buffer.data(), static_cast<std::size_t>(count));
                std::size_t start = 0;
                while (true) {
                    const auto end = pending.find('\n', start);
                    if (end == std::string::npos) break;
                    line(pending.substr(start, end - start + 1));
                    start = end + 1;
                }
                pending.erase(0, start);
            }
            if (!pending.empty()) line(std::move(pending));
            const auto status = process->wait_for_cleanup();
            shared->set_status(process->returncode().value_or(status));
        } catch (const std::exception& error) {
            try {
                if (ProcessInterrupt::requested()) terminate_download();
                else { process->signal(SIGKILL); process->wait_for_cleanup(); }
            } catch (...) { process->signal(SIGKILL); }
            try { write_descriptor(errors.get(), error.what(), std::char_traits<char>::length(error.what())); } catch (...) {}
            shared->set_status(1);
        } catch (...) {
            process->signal(SIGKILL);
            try { process->wait_for_cleanup(); } catch (...) {}
            shared->set_status(1);
        }
        shared->set_near_end();
    }

    void finish_monitor() {
        // A descendant may retain stderr after its parent has exited. Give
        // monitoring the reference five seconds, then wake its polling loop.
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (monitor.joinable() && !monitor_finished.load() &&
               std::chrono::steady_clock::now() < deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        stop_monitor.store(true);
        if (monitor.joinable()) monitor.join();
    }

    void close() {
        if (close_attempted) return;
        close_attempted = true;
        terminate_download();
        finish_monitor();
        stream.close_checked("Cannot close prepared yt-dlp stream");
        progress.close_checked("Cannot close prepared yt-dlp progress");
        errors.close_checked("Cannot close prepared yt-dlp errors");
        cookies.close();
    }

    ~Impl() {
        // Safe fallback for abandoned objects or an explicit close that failed
        // before monitoring stopped. Checked cookie cleanup is never retried.
        try { terminate_download(); }
        catch (...) { if (process) process->signal(SIGKILL); }
        finish_monitor();
    }
};

PreparedDownload::PreparedDownload(const RemoteJob& remote) : impl_(std::make_unique<Impl>(remote)) {}
PreparedDownload::~PreparedDownload() = default;
int PreparedDownload::input_descriptor() const { return impl_ ? impl_->stream.get() : -1; }
int PreparedDownload::error_descriptor() const { return impl_ ? impl_->errors.get() : -1; }
std::shared_ptr<SharedStreamState> PreparedDownload::state() const { return impl_ ? impl_->shared : nullptr; }
void PreparedDownload::close() {
    auto owned = std::move(impl_);
    if (owned) owned->close();
}

void drain_prepared_download(int input, int errors, const SharedStreamState& state) {
    // Keep the 1 MiB transfer storage off the stack. The target GCC 11
    // emitted incorrect unwind offsets for this large stack frame, turning
    // a download failure into std::terminate instead of a catchable error.
    std::vector<char> buffer(1024u * 1024u);
    for (;;) {
        ProcessInterrupt::check();
        pollfd descriptor{input, POLLIN, 0};
        const int ready = poll(&descriptor, 1, 100);
        if (ready < 0 && errno == EINTR) continue;
        if (ready < 0 || (ready > 0 && (descriptor.revents & (POLLERR | POLLNVAL))))
            throw std::runtime_error("Cannot poll yt-dlp stream");
        if (!ready) continue;
        ProcessInterrupt::check();
        const auto count = read(input, buffer.data(), buffer.size());
        if (count < 0 && errno == EINTR) continue;
        if (count < 0) throw std::runtime_error("Cannot drain yt-dlp stream");
        if (!count) break;
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(15);
    while (state.status() == SharedStreamState::pending) {
        ProcessInterrupt::check();
        if (std::chrono::steady_clock::now() >= deadline) {
            throw std::runtime_error("yt-dlp did not report completion after stream EOF");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    ProcessInterrupt::check();
    if (state.status() != 0) throw std::runtime_error("yt-dlp stream failed: " + utf8_file_tail(errors, 2500));
}

}  // namespace uf
