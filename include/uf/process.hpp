#pragma once

#include <array>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <optional>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace uf {

class FileDescriptor {
public:
    explicit FileDescriptor(int value = -1) : value_(value) {}
    ~FileDescriptor();
    FileDescriptor(const FileDescriptor&) = delete;
    FileDescriptor& operator=(const FileDescriptor&) = delete;
    FileDescriptor(FileDescriptor&& other) noexcept;
    FileDescriptor& operator=(FileDescriptor&& other) noexcept;
    int get() const { return value_; }
    int release();
    void reset(int value = -1);
    // Explicit checked close. Relinquish ownership before the syscall so a
    // failure cannot cause RAII or a later call to retry a reused descriptor.
    void close_checked(const std::string& context);
private:
    int value_;
};

std::array<FileDescriptor, 2> process_pipe();
// Native worker exchange scratch: prefix includes its explicit directory.
// Reuses TemporaryFile creation while retaining the caller's RAM location.
FileDescriptor anonymous_file(const std::string& prefix);
void write_descriptor(int descriptor, const void* bytes, std::size_t size);

// Python tempfile on POSIX: cached writable-directory selection and native
// filename bytes. Named files use eight-character, per-PID Random choices,
// exclusive noninheritable creation and mode 0600. Caller owns unlink.
struct PythonTemporaryFile {
    FileDescriptor descriptor;
    std::filesystem::path path;
};
std::filesystem::path python_temporary_directory();
PythonTemporaryFile python_temporary_file(std::string_view prefix,
    std::string_view suffix = {},
    const std::optional<std::filesystem::path>& directory = std::nullopt);
// Python 3.10 mkdtemp preserves an explicitly supplied relative directory;
// mode 0700 and the same name sequence as named files. Caller owns cleanup.
std::filesystem::path python_temporary_mkdtemp(std::string_view prefix,
    std::string_view suffix = {},
    const std::optional<std::filesystem::path>& directory = std::nullopt);
// Python TemporaryFile's POSIX descriptor policy: try O_TMPFILE, then an
// exclusive named file immediately unlinked. Default prefix is Python's tmp.
FileDescriptor python_temporary_anonymous_file(std::string_view prefix = "tmp",
    std::string_view suffix = {},
    const std::optional<std::filesystem::path>& directory = std::nullopt);

// POSIX shutil.which policy: explicit paths, unset/empty PATH, CS_PATH fallback,
// access checks and symlinks. Resolve without executing or loading a candidate.
std::optional<std::filesystem::path> find_executable(const std::string& name);

// A foreground workflow catches INT/TERM so stack unwinding stops its workers,
// metadata producer and warm downloaders before returning to the shell.
class ProcessInterrupt {
public:
    ProcessInterrupt();
    ~ProcessInterrupt();
    ProcessInterrupt(const ProcessInterrupt&) = delete;
    ProcessInterrupt& operator=(const ProcessInterrupt&) = delete;
    static int requested();
    static void check();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Each pair maps a parent descriptor to a child descriptor. All unmapped
// descriptors above stderr are closed before exec; no shell is involved.
// A new session also owns a process group for descendant cancellation.
class ChildProcess {
public:
    ChildProcess(const std::vector<std::string>& command,
                  const std::vector<std::pair<int, int>>& descriptors = {},
                  bool own_process_group = false,
                  bool start_new_session = false);
    ~ChildProcess();
    ChildProcess(const ChildProcess&) = delete;
    ChildProcess& operator=(const ChildProcess&) = delete;
    std::optional<int> poll();
    // Teardown can reap children after the foreground interrupt was recorded.
    std::optional<int> poll_for_cleanup();
    // Python/multiprocessing convention: ordinary exit status or -signal.
    // Available after poll/wait has observed completion; no additional wait.
    std::optional<int> returncode() const;
    int wait();
    // Reap during unwinding even when the interrupt flag is already set.
    int wait_for_cleanup();
    void signal(int number) const noexcept;
    int pid() const { return pid_; }
private:
    int pid_{-1};
    bool group_{};
    std::optional<int> status_;
    int signal_{};
    mutable std::mutex mutex_;
    int collect(bool block, bool check_interrupt = true);
};

}  // namespace uf
