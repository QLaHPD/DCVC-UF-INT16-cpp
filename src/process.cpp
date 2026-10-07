#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "uf/process.hpp"
#include "uf/python_path.hpp"
#include "pareto_random.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <csignal>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <fcntl.h>
#include <set>
#include <spawn.h>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

extern char** environ;

namespace uf {

namespace {
std::atomic<std::sig_atomic_t> interrupted{0};
static_assert(std::atomic<std::sig_atomic_t>::is_always_lock_free,
              "UF signal handlers require lock-free signal atomics");
void interrupt_handler(int number) { interrupted.store(number, std::memory_order_relaxed); }

// tempfile/posixpath policy adapted from Python 3.10; retained notice in
// licenses/CPython-3.10.20.txt. Native paths hold POSIX filename bytes.
class TemporaryNames {
public:
    std::string next() {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto pid = getpid();
        if (!random_ || pid_ != pid) {
            random_.emplace();
            pid_ = pid;
        }
        constexpr char alphabet[] = "abcdefghijklmnopqrstuvwxyz0123456789_";
        std::string name;
        name.reserve(8);
        for (int index = 0; index < 8; ++index)
            name.push_back(alphabet[static_cast<std::size_t>(random_->random() * 37.0)]);
        return name;
    }
private:
    std::mutex mutex_;
    std::optional<pareto_detail::PythonRandom> random_;
    pid_t pid_{};
};

TemporaryNames& temporary_names() {
    static TemporaryNames names;
    return names;
}

#ifdef TMP_MAX
constexpr auto maximum_temporary_attempts = TMP_MAX;
#else
constexpr auto maximum_temporary_attempts = 10000;
#endif

std::filesystem::path temporary_absolute(const std::filesystem::path& input) {
    // tempfile uses os.path.abspath, which collapses '..' lexically without
    // resolving links. Path.absolute alone intentionally retains those parts.
    const auto spelling = python_path_absolute(input).native();
    std::size_t cursor = 0;
    while (cursor < spelling.size() && spelling[cursor] == '/') ++cursor;
    const auto slash_count = cursor == 2 ? 2u : (cursor ? 1u : 0u);
    std::vector<std::string_view> parts;
    while (cursor < spelling.size()) {
        auto end = spelling.find('/', cursor);
        if (end == std::string::npos) end = spelling.size();
        const std::string_view part(spelling.data() + cursor, end - cursor);
        if (!part.empty() && part != ".") {
            if (part != ".." || (!slash_count && parts.empty()) ||
                (!parts.empty() && parts.back() == "..")) parts.push_back(part);
            else if (!parts.empty()) parts.pop_back();
        }
        cursor = end;
        while (cursor < spelling.size() && spelling[cursor] == '/') ++cursor;
    }
    std::string result(slash_count, '/');
    for (const auto part : parts) {
        if (!result.empty() && result.back() != '/') result.push_back('/');
        result.append(part.data(), part.size());
    }
    return result.empty() ? std::filesystem::path(".") : std::filesystem::path(result);
}

FileDescriptor open_temporary(const std::filesystem::path& path) {
    if (path.native().find('\0') != std::string::npos)
        throw std::invalid_argument("embedded null byte");
    for (;;) {
        ProcessInterrupt::check();
        const int descriptor = open(path.c_str(), O_RDWR | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
        if (descriptor >= 0) return FileDescriptor(descriptor);
        const int error = errno;
        if (error == EINTR) continue;
        throw std::system_error(error, std::generic_category(), "Cannot create native Python temporary file");
    }
}

void unlink_temporary_probe(const std::filesystem::path& path) {
    if (unlink(path.c_str()) != 0) {
        const int error = errno;
        throw std::system_error(error, std::generic_category(), "Cannot remove native Python temporary probe");
    }
}

std::filesystem::path select_temporary_directory() {
    std::vector<std::filesystem::path> directories;
    for (const auto* name : {"TMPDIR", "TEMP", "TMP"})
        if (const auto* value = std::getenv(name); value && *value) directories.emplace_back(value);
    for (const auto* path : {"/tmp", "/var/tmp", "/usr/tmp"}) directories.emplace_back(path);
    try { directories.push_back(std::filesystem::current_path()); }
    catch (const std::system_error&) { directories.emplace_back("."); }
    // Directory probing and actual file names have independent Random state,
    // as tempfile's local namer and shared _name_sequence do.
    TemporaryNames names;
    for (auto directory : directories) {
        if (directory.native() != ".") directory = temporary_absolute(directory);
        for (int attempt = 0; attempt < 100; ++attempt) {
            ProcessInterrupt::check();
            const auto path = directory / names.next();
            try {
                auto file = open_temporary(path);
                try {
                    try {
                        for (;;) {
                            ProcessInterrupt::check();
                            // Python's probe performs one write and does not
                            // require a full count; retain that return policy.
                            if (write(file.get(), "blat", 4) >= 0) break;
                            const int error = errno;
                            if (error == EINTR) continue;
                            throw std::system_error(error, std::generic_category(), "Cannot write native Python temporary probe");
                        }
                    } catch (...) {
                        file.close_checked("Cannot close native Python temporary probe");
                        throw;
                    }
                    file.close_checked("Cannot close native Python temporary probe");
                } catch (...) {
                    unlink_temporary_probe(path);
                    throw;
                }
                unlink_temporary_probe(path);
                return directory;
            } catch (const std::system_error& error) {
                if (error.code() == std::error_code(EEXIST, std::generic_category())) continue;
                break;
            }
        }
    }
    throw std::system_error(ENOENT, std::generic_category(), "No usable temporary directory found");
}
}  // namespace

struct ProcessInterrupt::Impl {
    struct sigaction previous_int{}, previous_term{};
    Impl() {
        interrupted.store(0, std::memory_order_relaxed);
        struct sigaction action{};
        sigemptyset(&action.sa_mask);
        action.sa_handler = interrupt_handler;
        if (sigaction(SIGINT, &action, &previous_int) != 0) {
            throw std::runtime_error("Cannot install UF interrupt handler");
        }
        if (sigaction(SIGTERM, &action, &previous_term) != 0) {
            sigaction(SIGINT, &previous_int, nullptr);
            throw std::runtime_error("Cannot install UF termination handler");
        }
    }
    ~Impl() {
        sigaction(SIGINT, &previous_int, nullptr);
        sigaction(SIGTERM, &previous_term, nullptr);
    }
};

ProcessInterrupt::ProcessInterrupt() : impl_(std::make_unique<Impl>()) {}
ProcessInterrupt::~ProcessInterrupt() = default;
int ProcessInterrupt::requested() { return interrupted.load(std::memory_order_relaxed); }
void ProcessInterrupt::check() {
    if (requested()) throw std::runtime_error("UF operation interrupted");
}

std::optional<std::filesystem::path> find_executable(const std::string& name) {
    // POSIX shutil.which policy, adapted from CPython 3.10; retained license:
    // licenses/CPython-3.10.20.txt. Do not execute the candidate to inspect it.
    if (name.empty() || name.find('\0') != std::string::npos) return std::nullopt;
    const auto executable = [](const std::string& candidate) {
        struct stat status;
        if (stat(candidate.c_str(), &status) != 0 ||
            access(candidate.c_str(), F_OK | X_OK) != 0) return false;
        // Python checks exists/access/isdir separately, following symlinks.
        return stat(candidate.c_str(), &status) != 0 || !S_ISDIR(status.st_mode);
    };
    if (name.find('/') != std::string::npos) {
        return executable(name) ? std::optional<std::filesystem::path>(name) : std::nullopt;
    }
    const char* environment = std::getenv("PATH");
    std::string path;
    if (environment) {
        path = environment;
    } else {
        // POSIX provides CS_PATH; an unset PATH differs from an empty PATH.
        auto length = confstr(_CS_PATH, nullptr, 0);
        if (!length) return std::nullopt;
        for (;;) {
            path.resize(length);
            const auto actual = confstr(_CS_PATH, path.data(), path.size());
            if (!actual) return std::nullopt;
            if (actual > path.size()) { length = actual; continue; }
            path.resize(actual - 1);
            break;
        }
    }
    if (path.empty()) return std::nullopt;
    std::set<std::string> seen;
    std::size_t begin = 0;
    while (begin <= path.size()) {
        const auto end = path.find(':', begin);
        const auto directory = path.substr(begin,
            end == std::string::npos ? std::string::npos : end - begin);
        if (seen.insert(directory).second) {
            const auto candidate = directory.empty() ? name :
                directory + (directory.back() == '/' ? "" : "/") + name;
            if (executable(candidate)) return std::filesystem::path(candidate);
        }
        if (end == std::string::npos) break;
        begin = end + 1;
    }
    return std::nullopt;
}

FileDescriptor::~FileDescriptor() { reset(); }
FileDescriptor::FileDescriptor(FileDescriptor&& other) noexcept : value_(other.release()) {}
FileDescriptor& FileDescriptor::operator=(FileDescriptor&& other) noexcept {
    if (this != &other) reset(other.release());
    return *this;
}
int FileDescriptor::release() { const int value = value_; value_ = -1; return value; }
void FileDescriptor::close_checked(const std::string& context) {
    const int descriptor = release();
    if (descriptor < 0) return;
    if (close(descriptor) != 0) {
        const int error = errno;
        throw std::system_error(error, std::generic_category(), context);
    }
}
void FileDescriptor::reset(int value) {
    if (value_ >= 0) close(value_);
    value_ = value;
}

std::array<FileDescriptor, 2> process_pipe() {
    int descriptors[2];
    if (pipe2(descriptors, O_CLOEXEC) != 0) throw std::runtime_error("Cannot create UF process pipe");
    return {FileDescriptor(descriptors[0]), FileDescriptor(descriptors[1])};
}

FileDescriptor anonymous_file(const std::string& prefix) {
    const std::filesystem::path spelling(prefix);
    const auto parent = spelling.parent_path().empty()
        ? std::filesystem::path(".") : spelling.parent_path();
    return python_temporary_anonymous_file(spelling.filename().native(), {}, parent);
}

void write_descriptor(int descriptor, const void* bytes, std::size_t size) {
    const auto* data = static_cast<const char*>(bytes);
    std::size_t position = 0;
    while (position < size) {
        ProcessInterrupt::check();
        const auto count = write(descriptor, data + position, size - position);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) throw std::runtime_error("Cannot write UF process descriptor");
        position += static_cast<std::size_t>(count);
    }
}

std::filesystem::path python_temporary_directory() {
    // Function-local static initialization supplies tempfile's once lock. A
    // failed initialization is retried; successful selection remains cached.
    static const auto directory = select_temporary_directory();
    return directory;
}

PythonTemporaryFile python_temporary_file(std::string_view prefix,
    std::string_view suffix, const std::optional<std::filesystem::path>& directory) {
    const auto parent = temporary_absolute(directory ? *directory : python_temporary_directory());
    for (int attempt = 0; attempt < maximum_temporary_attempts; ++attempt) {
        ProcessInterrupt::check();
        auto name = std::string(prefix) + temporary_names().next();
        if (!suffix.empty()) name.append(suffix.data(), suffix.size());
        const auto path = parent / name;
        try {
            auto descriptor = open_temporary(path);
            return {std::move(descriptor), path};
        } catch (const std::system_error& error) {
            if (error.code() != std::error_code(EEXIST, std::generic_category())) throw;
        }
    }
    throw std::system_error(EEXIST, std::generic_category(), "No usable temporary file name found");
}

std::filesystem::path python_temporary_mkdtemp(std::string_view prefix,
    std::string_view suffix, const std::optional<std::filesystem::path>& directory) {
    // Unlike _mkstemp_inner, Python 3.10 mkdtemp does not apply abspath.
    const auto parent = directory ? *directory : python_temporary_directory();
    for (int attempt = 0; attempt < maximum_temporary_attempts; ++attempt) {
        ProcessInterrupt::check();
        auto name = std::string(prefix) + temporary_names().next();
        if (!suffix.empty()) name.append(suffix.data(), suffix.size());
        const auto path = parent / name;
        if (path.native().find('\0') != std::string::npos)
            throw std::invalid_argument("embedded null byte");
        for (;;) {
            ProcessInterrupt::check();
            if (mkdir(path.c_str(), 0700) == 0) return path;
            const int error = errno;
            if (error == EINTR) continue;
            if (error == EEXIST) break;
            throw std::system_error(error, std::generic_category(),
                                    "Cannot create native Python temporary directory");
        }
    }
    throw std::system_error(EEXIST, std::generic_category(), "No usable temporary directory name found");
}

FileDescriptor python_temporary_anonymous_file(std::string_view prefix,
    std::string_view suffix, const std::optional<std::filesystem::path>& directory) {
    const auto parent = directory ? *directory : python_temporary_directory();
    if (parent.native().find('\0') != std::string::npos)
        throw std::invalid_argument("embedded null byte");
#ifdef O_TMPFILE
    // Python only remembers EISDIR (old kernels interpreting O_TMPFILE as
    // O_DIRECTORY). Other OS errors permit a named fallback for this call.
    static std::atomic<bool> unnamed_supported{true};
    if (unnamed_supported.load(std::memory_order_relaxed)) {
        for (;;) {
            ProcessInterrupt::check();
            const int descriptor = open(parent.c_str(),
                O_RDWR | O_EXCL | O_NOFOLLOW | O_CLOEXEC | O_TMPFILE, 0600);
            if (descriptor >= 0) return FileDescriptor(descriptor);
            const int error = errno;
            if (error == EINTR) continue;
            if (error == EISDIR) unnamed_supported.store(false, std::memory_order_relaxed);
            break;
        }
    }
#endif
    auto temporary = python_temporary_file(prefix, suffix, parent);
    // tempfile's opener closes the descriptor on unlink failure before raising;
    // a checked close error can replace the saved unlink error.
    if (unlink(temporary.path.c_str()) != 0) {
        const int error = errno;
        temporary.descriptor.close_checked("Cannot close native Python anonymous temporary file");
        throw std::system_error(error, std::generic_category(),
                                "Cannot unlink native Python anonymous temporary file");
    }
    return std::move(temporary.descriptor);
}

namespace {

void spawn_check(int error) {
    if (error) throw std::runtime_error(std::string("Cannot launch UF process: ") + std::strerror(error));
}

struct SpawnState {
    posix_spawn_file_actions_t actions;
    posix_spawnattr_t attributes;
    SpawnState() {
        spawn_check(posix_spawn_file_actions_init(&actions));
        const int error = posix_spawnattr_init(&attributes);
        if (error) {
            posix_spawn_file_actions_destroy(&actions);
            spawn_check(error);
        }
    }
    ~SpawnState() {
        posix_spawnattr_destroy(&attributes);
        posix_spawn_file_actions_destroy(&actions);
    }
};

}  // namespace

ChildProcess::ChildProcess(const std::vector<std::string>& command,
                            const std::vector<std::pair<int, int>>& descriptors,
                            bool own_process_group, bool start_new_session)
    : group_(own_process_group || start_new_session) {
    if (command.empty() || command.front().empty()) throw std::invalid_argument("Empty UF process command");
    for (const auto& argument : command) {
        if (argument.find('\0') != std::string::npos) {
            throw std::invalid_argument("embedded null byte");
        }
    }
    std::set<int> targets;
    int close_from = 3;
    for (const auto& [source, target] : descriptors) {
        if (source < 0 || target < 0 || target > 63 || !targets.insert(target).second) {
            throw std::invalid_argument("Invalid UF process descriptor mapping");
        }
        close_from = std::max(close_from, target + 1);
    }
    SpawnState state;
    std::vector<FileDescriptor> copies;
    copies.reserve(descriptors.size());
    for (const auto& [source, target] : descriptors) {
        copies.emplace_back(fcntl(source, F_DUPFD_CLOEXEC, 64));
        if (copies.back().get() < 0) throw std::runtime_error("Cannot duplicate UF process input");
        spawn_check(posix_spawn_file_actions_adddup2(&state.actions, copies.back().get(), target));
    }
    for (int descriptor = 3; descriptor < close_from; ++descriptor) {
        if (!targets.count(descriptor)) spawn_check(posix_spawn_file_actions_addclose(&state.actions, descriptor));
    }
    // Linux/glibc provides an atomic close-from action; this project already
    // relies on Linux /dev/shm, flock and /proc/self/exe.
    spawn_check(posix_spawn_file_actions_addclosefrom_np(&state.actions, close_from));
    sigset_t defaults;
    sigemptyset(&defaults);
    // Match Python's restore_signals: restore the signals Python ignores,
    // while preserving other ignored dispositions and the caller's mask.
    // Caught handlers are reset by exec/posix_spawn itself.
    sigaddset(&defaults, SIGPIPE);
#ifdef SIGXFZ
    sigaddset(&defaults, SIGXFZ);
#endif
#ifdef SIGXFSZ
    sigaddset(&defaults, SIGXFSZ);
#endif
    spawn_check(posix_spawnattr_setsigdefault(&state.attributes, &defaults));
    short flags = POSIX_SPAWN_SETSIGDEF;
    if (start_new_session) {
        // The session already supplies its group; do not also request setpgid.
        flags |= POSIX_SPAWN_SETSID;
    } else if (group_) {
        flags |= POSIX_SPAWN_SETPGROUP;
        spawn_check(posix_spawnattr_setpgroup(&state.attributes, 0));
    }
    spawn_check(posix_spawnattr_setflags(&state.attributes, flags));
    std::vector<char*> arguments;
    arguments.reserve(command.size() + 1);
    for (const auto& argument : command) arguments.push_back(const_cast<char*>(argument.c_str()));
    arguments.push_back(nullptr);
    pid_t process = -1;
    spawn_check(posix_spawnp(&process, arguments.front(), &state.actions,
                            &state.attributes, arguments.data(), environ));
    pid_ = process;
}

ChildProcess::~ChildProcess() {
    if (pid_ > 0 && !status_) {
        signal(SIGKILL);
        int status = 0;
        while (waitpid(pid_, &status, 0) < 0 && errno == EINTR) {}
    }
}

void ChildProcess::signal(int number) const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (pid_ > 0 && !status_) kill(group_ ? -pid_ : pid_, number);
}

int ChildProcess::collect(bool block, bool check_interrupt) {
    if (check_interrupt) ProcessInterrupt::check();
    if (block) {
        for (;;) {
            const int result = collect(false, check_interrupt);
            if (result >= 0) return result;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (status_) return *status_;
    int status = 0;
    pid_t result;
    do { result = waitpid(pid_, &status, WNOHANG); }
    while (result < 0 && errno == EINTR);
    if (!result) return -1;
    if (result < 0) throw std::runtime_error("Cannot wait for UF process");
    signal_ = WIFSIGNALED(status) ? WTERMSIG(status) : 0;
    status_ = WIFEXITED(status) ? WEXITSTATUS(status) : (signal_ ? 128 + signal_ : 1);
    return *status_;
}

std::optional<int> ChildProcess::poll() {
    const int result = collect(false);
    return result < 0 ? std::nullopt : std::optional<int>(result);
}
std::optional<int> ChildProcess::poll_for_cleanup() {
    const int result = collect(false, false);
    return result < 0 ? std::nullopt : std::optional<int>(result);
}
std::optional<int> ChildProcess::returncode() const {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (!status_) return std::nullopt;
    return signal_ ? -signal_ : *status_;
}
int ChildProcess::wait() { return collect(true); }
int ChildProcess::wait_for_cleanup() { return collect(true, false); }

}  // namespace uf
