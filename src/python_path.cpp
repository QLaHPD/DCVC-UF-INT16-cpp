// Adapted from CPython 3.10 pathlib/posixpath path and non-strict realpath rules.
// See licenses/CPython-3.10.20.txt. No Python runtime is embedded or invoked.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "uf/python_path.hpp"
#include "uf/process.hpp"
#include "uf/python_json.hpp"
#include "uf/utf8.hpp"

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <dirent.h>
#include <fcntl.h>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>
#include <sys/stat.h>
#include <unistd.h>

namespace uf {
namespace {

// shutil reports replacement by a symlink as OSError without an errno.
class TreeReplacementError : public std::system_error {
public:
    TreeReplacementError() : std::system_error(std::error_code{}, message) {}
    const char* what() const noexcept override { return message; }
private:
    static constexpr const char* message = "Cannot call rmtree on a symbolic link";
};

[[noreturn]] void tree_os_error(int error, const std::filesystem::path& path) {
    throw std::system_error(error, std::generic_category(),
                            "Cannot remove UF directory tree: " + path.string());
}

void reset_temporary_permissions(const std::filesystem::path& path, bool check_interrupt) {
    if (check_interrupt) ProcessInterrupt::check();
    // Linux's installed Python has no working lchmod/chflags. Its
    // _dont_follow_symlinks first checks islink, then calls chmod for nonlinks.
    struct stat status{};
    if (lstat(path.c_str(), &status) == 0 && S_ISLNK(status.st_mode)) return;
    for (;;) {
        if (check_interrupt) ProcessInterrupt::check();
        if (chmod(path.c_str(), 0700) == 0) return;
        const int error = errno;
        if (error == EINTR) continue;
        tree_os_error(error, path);
    }
}

struct TreeErrors {
    const std::filesystem::path* temporary_root;
    bool ignore_errors;
    bool check_interrupt;

    void operator()(int error, const std::filesystem::path& path) const {
        if (temporary_root && (error == EACCES || error == EPERM)) {
            try {
                if (path != *temporary_root)
                    reset_temporary_permissions(path.parent_path(), check_interrupt);
                reset_temporary_permissions(path, check_interrupt);
                if (check_interrupt) ProcessInterrupt::check();
                if (unlink(path.c_str()) == 0) return;
                const int retry_error = errno;
                if (retry_error == EISDIR || retry_error == EACCES || retry_error == EPERM)
                    python_temporary_rmtree(path, ignore_errors, check_interrupt);
                else tree_os_error(retry_error, path);
            } catch (const std::system_error& failure) {
                // Permission repair/retry errors propagate even when cleanup
                // ignores ordinary errors; only FileNotFoundError is suppressed.
                if (failure.code() != std::error_code(ENOENT, std::generic_category())) throw;
            }
            return;
        }
        if (temporary_root && error == ENOENT) return;
        if (!ignore_errors) {
            if (!error) throw TreeReplacementError();
            tree_os_error(error, path);
        }
    }
};

void close_tree_descriptor(FileDescriptor& file, const std::filesystem::path& path) {
    const int descriptor = file.release();
    if (close(descriptor) != 0) tree_os_error(errno, path);
}

FileDescriptor open_tree_directory(int parent, const char* name,
    const std::filesystem::path& path, bool check_interrupt) {
    for (;;) {
        if (check_interrupt) ProcessInterrupt::check();
        const int descriptor = openat(parent, name, O_RDONLY | O_CLOEXEC);
        if (descriptor >= 0) return FileDescriptor(descriptor);
        const int error = errno;
        if (error == EINTR) continue;
        tree_os_error(error, path);
    }
}

struct TreeEntry {
    std::string name;
    unsigned char type;
};

struct TreeListingCloser {
    void operator()(DIR* directory) const noexcept {
        // os.scandir(fd) duplicates the descriptor, then rewinds the shared
        // directory position before closing its listing. closedir is unchecked.
        rewinddir(directory);
        closedir(directory);
    }
};

std::vector<TreeEntry> tree_entries(int descriptor,
    const std::filesystem::path& path, bool check_interrupt) {
    int duplicate;
    for (;;) {
        if (check_interrupt) ProcessInterrupt::check();
        duplicate = fcntl(descriptor, F_DUPFD_CLOEXEC, 0);
        if (duplicate >= 0) break;
        const int error = errno;
        if (error == EINTR) continue;
        tree_os_error(error, path);
    }
    FileDescriptor owner(duplicate);
    DIR* opened = fdopendir(owner.get());
    if (!opened) tree_os_error(errno, path);
    owner.release();  // fdopendir transfers descriptor ownership to closedir.
    std::unique_ptr<DIR, TreeListingCloser> listing(opened);
    std::vector<TreeEntry> entries;
    for (;;) {
        if (check_interrupt) ProcessInterrupt::check();
        errno = 0;
        const auto* entry = readdir(listing.get());
        if (!entry) {
            const int error = errno;
            if (error) tree_os_error(error, path);
            break;
        }
        const std::string name(entry->d_name);
        if (name != "." && name != "..") entries.push_back({name, entry->d_type});
    }
    return entries;
}

void remove_tree_contents(int parent, const std::filesystem::path& path,
                          const TreeErrors& onerror, bool check_interrupt) {
    std::vector<TreeEntry> entries;
    try {
        entries = tree_entries(parent, path, check_interrupt);
    } catch (const std::system_error& error) {
        onerror(error.code().value(), path);
        return;
    }
    for (const auto& entry : entries) {
        if (check_interrupt) ProcessInterrupt::check();
        const auto fullname = path / entry.name;
        struct stat original{};
        bool have_status = false;
        bool directory = entry.type == DT_DIR;
        if (entry.type == DT_UNKNOWN) {
            // DirEntry.is_dir errors are suppressed before the unlink path;
            // a successful unknown-type stat is cached for the later stat call.
            have_status = fstatat(parent, entry.name.c_str(), &original, AT_SYMLINK_NOFOLLOW) == 0;
            directory = have_status && S_ISDIR(original.st_mode);
        }
        if (directory && !have_status) {
            if (fstatat(parent, entry.name.c_str(), &original, AT_SYMLINK_NOFOLLOW) != 0) {
                onerror(errno, fullname);
                continue;
            }
            directory = S_ISDIR(original.st_mode);
        }
        if (!directory) {
            if (unlinkat(parent, entry.name.c_str(), 0) != 0)
                onerror(errno, fullname);
            continue;
        }
        FileDescriptor child;
        try {
            child = open_tree_directory(parent, entry.name.c_str(), fullname, check_interrupt);
        } catch (const std::system_error& error) {
            onerror(error.code().value(), fullname);
            continue;
        }
        try {
            struct stat current{};
            if (fstat(child.get(), &current) != 0) tree_os_error(errno, fullname);
            if (original.st_dev == current.st_dev && original.st_ino == current.st_ino) {
                remove_tree_contents(child.get(), fullname, onerror, check_interrupt);
                try {
                    close_tree_descriptor(child, fullname);
                    if (unlinkat(parent, entry.name.c_str(), AT_REMOVEDIR) != 0)
                        tree_os_error(errno, fullname);
                } catch (const std::system_error& error) {
                    onerror(error.code().value(), fullname);
                }
            } else {
                onerror(0, fullname);
            }
            if (child.get() >= 0) close_tree_descriptor(child, fullname);
        } catch (...) {
            if (child.get() >= 0) close_tree_descriptor(child, fullname);
            throw;
        }
    }
}

using ResolvedLinks = std::map<std::filesystem::path,
                               std::optional<std::filesystem::path>>;

struct PosixParts {
    std::filesystem::path normalized;
    std::string root;
    std::vector<std::string> components;
};

PosixParts posix_parts(const std::filesystem::path& input) {
    PosixParts result;
    result.normalized = python_path_normalize(input);
    const auto& spelling = result.normalized.native();
    if (spelling == ".") return result;
    std::size_t cursor = 0;
    while (cursor < spelling.size() && spelling[cursor] == '/') ++cursor;
    result.root = spelling.substr(0, cursor);
    while (cursor < spelling.size()) {
        auto end = spelling.find('/', cursor);
        if (end == std::string::npos) end = spelling.size();
        result.components.push_back(spelling.substr(cursor, end - cursor));
        cursor = end == spelling.size() ? end : end + 1;
    }
    return result;
}

bool has_path_prefix(const PosixParts& path, const PosixParts& base) {
    return path.root == base.root && base.components.size() <= path.components.size() &&
        std::equal(base.components.begin(), base.components.end(), path.components.begin());
}

std::optional<struct stat> path_status(const std::filesystem::path& input,
                                      bool follow_symlinks) {
    const auto path = python_path_normalize(input);
    if (path.native().find('\0') != std::string::npos) return std::nullopt;
    ProcessInterrupt::check();
    struct stat status{};
    const int result = follow_symlinks ? stat(path.c_str(), &status) : lstat(path.c_str(), &status);
    if (result == 0) return status;
    const int error = errno;
    if (error == ENOENT || error == ENOTDIR || error == EBADF || error == ELOOP)
        return std::nullopt;
    throw std::system_error(error, std::generic_category(),
                            "Cannot inspect UF path: " + path.string());
}

std::pair<std::filesystem::path, bool> join_real_path(
    std::filesystem::path prefix, const std::filesystem::path& rest,
    ResolvedLinks& seen) {
    if (rest.is_absolute()) prefix = "/";
    const auto relative = rest.relative_path();
    for (auto part = relative.begin(); part != relative.end(); ++part) {
        ProcessInterrupt::check();
        if (part->empty() || *part == ".") continue;
        if (*part == "..") {
            if (prefix.empty()) prefix = "..";
            else if (prefix.filename() == "..") prefix = prefix.parent_path() / ".." / "..";
            else prefix = prefix.parent_path();
            continue;
        }
        const auto candidate = prefix / *part;
        struct stat status{};
        if (lstat(candidate.c_str(), &status) != 0 || !S_ISLNK(status.st_mode)) {
            prefix = candidate;
            continue;
        }
        const auto append_remaining = [&](std::filesystem::path value) {
            auto next = part;
            while (++next != relative.end()) value /= *next;
            return value;
        };
        const auto found = seen.find(candidate);
        if (found != seen.end()) {
            if (found->second) { prefix = *found->second; continue; }
            return {append_remaining(candidate), false};
        }
        seen.emplace(candidate, std::nullopt);
        ProcessInterrupt::check();
        auto resolved = join_real_path(prefix, std::filesystem::read_symlink(candidate), seen);
        if (!resolved.second) return {append_remaining(std::move(resolved.first)), false};
        prefix = std::move(resolved.first);
        seen.at(candidate) = prefix;
    }
    return {std::move(prefix), true};
}

}  // namespace

std::string python_path_text(const std::filesystem::path& path) {
    return python_filesystem_decode(path.native());
}

std::filesystem::path python_path_from_text(std::string_view text) {
    return std::filesystem::path(python_filesystem_encode(text));
}

std::filesystem::path python_path_normalize(const std::filesystem::path& path) {
    const auto& spelling = path.native();
    std::size_t cursor = 0;
    while (cursor < spelling.size() && spelling[cursor] == '/') ++cursor;
    std::string result = cursor == 0 ? "" : (cursor == 2 ? "//" : "/");
    while (cursor < spelling.size()) {
        auto end = spelling.find('/', cursor);
        if (end == std::string::npos) end = spelling.size();
        if (end != cursor && !(end == cursor + 1 && spelling[cursor] == '.')) {
            if (!result.empty() && result.back() != '/') result.push_back('/');
            result.append(spelling, cursor, end - cursor);
        }
        cursor = end;
        while (cursor < spelling.size() && spelling[cursor] == '/') ++cursor;
    }
    return result.empty() ? std::filesystem::path(".") : std::filesystem::path(result);
}

std::string python_path_name_text(std::string_view text) {
    // Only ASCII POSIX separators/dot components are structural. Passing text
    // through lexical path normalization preserves every WTF-8 codepoint.
    const auto path = python_path_normalize(std::filesystem::path(std::string(text)));
    return path.native() == "." ? std::string{} : path.filename().native();
}

bool python_path_less(const std::filesystem::path& left,
                      const std::filesystem::path& right) {
    const auto parts = [](const std::filesystem::path& path) {
        // Decode the complete spelling before splitting, preserving the
        // filesystem codec's state and Python's surrogateescaped characters.
        auto parsed = posix_parts(std::filesystem::path(python_path_text(path)));
        if (!parsed.root.empty())
            parsed.components.insert(parsed.components.begin(), std::move(parsed.root));
        return std::move(parsed.components);
    };
    // Canonical UTF-8/WTF-8 byte ordering follows Unicode codepoint ordering.
    // Compare component lists, not the slash-joined spelling.
    return parts(left) < parts(right);
}

std::filesystem::path python_path_absolute(const std::filesystem::path& input) {
    const auto path = python_path_normalize(input);
    if (path.is_absolute()) return path;
    ProcessInterrupt::check();
    return python_path_normalize(std::filesystem::current_path() / path);
}

std::filesystem::path python_path_relative_to(const std::filesystem::path& input,
                                              const std::filesystem::path& parent) {
    const auto path = posix_parts(input);
    const auto base = posix_parts(parent);
    if (!has_path_prefix(path, base)) {
        const auto formatted = base.root.empty() && base.components.empty()
            ? std::string{} : base.normalized.string();
        throw std::invalid_argument(python_json_repr(nlohmann::json(path.normalized.string())) +
            " is not in the subpath of " + python_json_repr(nlohmann::json(formatted)) +
            " OR one path is relative and the other is absolute.");
    }
    std::filesystem::path result;
    for (std::size_t index = base.components.size(); index < path.components.size(); ++index)
        result /= path.components[index];
    return result.empty() ? std::filesystem::path(".") : result;
}

bool python_path_is_relative_to(const std::filesystem::path& path,
                                 const std::filesystem::path& base) {
    return has_path_prefix(posix_parts(path), posix_parts(base));
}

bool python_path_exists(const std::filesystem::path& path) {
    return path_status(path, true).has_value();
}

bool python_path_is_file(const std::filesystem::path& path) {
    const auto status = path_status(path, true);
    return status && S_ISREG(status->st_mode);
}

bool python_path_is_directory(const std::filesystem::path& path) {
    const auto status = path_status(path, true);
    return status && S_ISDIR(status->st_mode);
}

bool python_path_is_symlink(const std::filesystem::path& path) {
    const auto status = path_status(path, false);
    return status && S_ISLNK(status->st_mode);
}

std::int64_t python_path_size(const std::filesystem::path& input) {
    const auto path = python_path_normalize(input);
    if (path.native().find('\0') != std::string::npos)
        throw std::invalid_argument("embedded null byte");
    ProcessInterrupt::check();
    struct stat status{};
    if (stat(path.c_str(), &status) != 0) {
        const int error = errno;
        throw std::system_error(error, std::generic_category(),
                                "Cannot inspect UF path size: " + path.string());
    }
    return static_cast<std::int64_t>(status.st_size);
}

PythonPathIdentity python_path_identity(const std::filesystem::path& input) {
    const auto path = python_path_normalize(input);
    if (path.native().find('\0') != std::string::npos)
        throw std::invalid_argument("embedded null byte");
    ProcessInterrupt::check();
    struct stat status{};
    if (stat(path.c_str(), &status) != 0) {
        const int error = errno;
        throw std::system_error(error, std::generic_category(),
                                "Cannot inspect UF source identity: " + path.string());
    }
    auto timestamp = PythonInteger(static_cast<std::int64_t>(status.st_mtim.tv_sec)) *
                     PythonInteger(1000000000);
    timestamp += PythonInteger(static_cast<std::int64_t>(status.st_mtim.tv_nsec));
    return {static_cast<std::int64_t>(status.st_size), std::move(timestamp)};
}

namespace {
void remove_tree(const std::filesystem::path& input,
                 bool ignore_errors, bool check_interrupt, bool temporary) {
    // TemporaryDirectory passes its original string to shutil, including an
    // empty spelling. Do not turn that into '.' and remove the current directory.
    const auto path = temporary ? input : python_path_normalize(input);
    if (path.native().find('\0') != std::string::npos) {
        if (ignore_errors) return;
        throw std::invalid_argument("embedded null byte");
    }
    if (check_interrupt) ProcessInterrupt::check();
    const TreeErrors onerror{temporary ? &path : nullptr, ignore_errors, check_interrupt};
    struct stat original{};
    if (lstat(path.c_str(), &original) != 0) {
        onerror(errno, path);
        return;
    }
    FileDescriptor directory;
    try {
        directory = open_tree_directory(AT_FDCWD, path.c_str(), path, check_interrupt);
    } catch (const std::system_error& error) {
        onerror(error.code().value(), path);
        return;
    }
    try {
        struct stat current{};
        if (fstat(directory.get(), &current) != 0) tree_os_error(errno, path);
        if (original.st_dev == current.st_dev && original.st_ino == current.st_ino) {
            remove_tree_contents(directory.get(), path, onerror, check_interrupt);
            try {
                close_tree_descriptor(directory, path);
                if (rmdir(path.c_str()) != 0) tree_os_error(errno, path);
            } catch (const std::system_error& error) {
                onerror(error.code().value(), path);
            }
        } else {
            onerror(0, path);
        }
        if (directory.get() >= 0) close_tree_descriptor(directory, path);
    } catch (...) {
        // Keep ownership released after a failed close. CPython's finally can
        // retry a failed os.close; native owners never retry released descriptors.
        if (directory.get() >= 0) close_tree_descriptor(directory, path);
        throw;
    }
}
}  // namespace

void python_path_rmtree(const std::filesystem::path& path,
                        bool ignore_errors, bool check_interrupt) {
    remove_tree(path, ignore_errors, check_interrupt, false);
}

void python_temporary_rmtree(const std::filesystem::path& path,
                             bool ignore_errors, bool check_interrupt) {
    remove_tree(path, ignore_errors, check_interrupt, true);
}

std::filesystem::path python_path_resolve(const std::filesystem::path& input) {
    if (input.native().find('\0') != std::string::npos)
        throw std::invalid_argument("embedded null byte");
    ProcessInterrupt::check();
    ResolvedLinks seen;
    auto joined = join_real_path({}, python_path_normalize(input), seen).first;
    if (joined.empty()) joined = ".";
    auto result = python_path_absolute(joined).lexically_normal();
    if (result.filename().empty() && result != result.root_path()) result = result.parent_path();
    // Path.resolve performs this final loop check even in non-strict mode;
    // other stat errors are ignored until the caller checks/creates its target.
    struct stat status{};
    ProcessInterrupt::check();
    if (stat(result.c_str(), &status) != 0 && errno == ELOOP)
        throw PythonPathLoopError("Symlink loop from " +
            python_json_repr(nlohmann::json(result.string())));
    return result;
}

}  // namespace uf
