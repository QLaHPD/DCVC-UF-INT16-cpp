#pragma once

#include "uf/python_integer.hpp"

#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>

namespace uf {

class PythonPathLoopError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Explicit boundary between native POSIX filename bytes and Python/JSON text.
// These do no filesystem I/O and do not normalize or resolve path components.
std::string python_path_text(const std::filesystem::path& path);
std::filesystem::path python_path_from_text(std::string_view text);

// POSIX Path construction: remove empty/dot components and trailing separators,
// retain '..' and exactly two leading slashes. No filesystem access or resolving.
std::filesystem::path python_path_normalize(const std::filesystem::path& path);

// Pure Path(text).name, preserving Python/JSON text including lone surrogates.
// Does not encode a filename or access the filesystem.
std::string python_path_name_text(std::string_view text);

// POSIX PurePath ordering compares normalized Unicode component lists,
// including a root anchor. Decode filesystem bytes with surrogateescape;
// ordering requires no stat, resolution or other filesystem access.
bool python_path_less(const std::filesystem::path& left,
                      const std::filesystem::path& right);

// Path.absolute() retains '..' and does not resolve links. relative_to and
// is_relative_to compare normalized components/anchors without filesystem I/O.
std::filesystem::path python_path_absolute(const std::filesystem::path& path);
std::filesystem::path python_path_relative_to(const std::filesystem::path& path,
                                              const std::filesystem::path& base);
bool python_path_is_relative_to(const std::filesystem::path& path,
                                 const std::filesystem::path& base);

// Path.exists(): stat follows symlinks; NUL and ENOENT/ENOTDIR/EBADF/ELOOP return
// false. Other operating-system errors propagate.
bool python_path_exists(const std::filesystem::path& path);
bool python_path_is_file(const std::filesystem::path& path);
bool python_path_is_directory(const std::filesystem::path& path);
bool python_path_is_symlink(const std::filesystem::path& path);

// Path.stat().st_size, including nonregular files. NUL/OS errors propagate;
// this lookup does not apply exists/is_file's ignored-error policy.
std::int64_t python_path_size(const std::filesystem::path& path);

// storage.identity: follow links with one stat, retaining signed st_size and
// exact Python st_mtime_ns across the entire native seconds range. Inode and
// device are not part of Python's size/mtime identity comparison.
struct PythonPathIdentity {
    std::int64_t size{};
    PythonInteger mtime_ns;
    bool operator==(const PythonPathIdentity& other) const {
        return size == other.size && mtime_ns.decimal() == other.mtime_ns.decimal();
    }
};
PythonPathIdentity python_path_identity(const std::filesystem::path& path);

// Linux shutil.rmtree directory-descriptor policy: snapshot children, compare
// lstat/fstat identity before recursion, and unlink relative to opened parents.
// ignore_errors handles the maintained onerror boundaries; unexpected resource
// errors and fstat/finally-close failures still propagate. Finally cleanup and
// destructors may disable latched interruption checks after a caught signal.
void python_path_rmtree(const std::filesystem::path& path,
                        bool ignore_errors = false, bool check_interrupt = true);
// Python TemporaryDirectory._rmtree on Linux: ignore missing paths and repair
// permissions before retrying unlink or recursively removing a directory.
// This is distinct from storage's plain shutil.rmtree error policy.
void python_temporary_rmtree(const std::filesystem::path& path,
                             bool ignore_errors = false, bool check_interrupt = true);

// Path.resolve(strict=False): follow symlinks, including dangling targets,
// process parents after resolving links, then check the final path for loops.
// Non-loop stat/lstat failures follow Python's non-strict policy; readlink and
// current-directory errors propagate. Embedded NUL paths are rejected.
std::filesystem::path python_path_resolve(const std::filesystem::path& path);

}  // namespace uf
