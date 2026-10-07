// POSIX path and regular-file rules follow CPython 3.10 pathlib; see
// licenses/CPython-3.10.20.txt.
#include "uf/shared_work.hpp"
#include "uf/json_compat.hpp"
#include "uf/process.hpp"
#include "uf/python_json.hpp"
#include "uf/python_path.hpp"
#include "uf/utf8.hpp"

#include <openssl/evp.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <charconv>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>
#include <fcntl.h>
#include <sys/random.h>
#include <sys/stat.h>
#include <unistd.h>

namespace uf {

// Python leases retain their pool and consult its channel cache when marking
// completion. Share that cache's lifetime with native leases as well.
struct SharedPipelineState {
    std::mutex mutex;
    std::map<std::filesystem::path, std::string> ids;
};

namespace {

using Json = nlohmann::json;

std::string canonical_json(const Json& value) {
    return python_canonical_json(value);
}

std::string sha256_text(const std::string& value) {
    std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
    unsigned int size = 0;
    if (EVP_Digest(value.data(), value.size(), digest.data(), &size,
                   EVP_sha256(), nullptr) != 1 || size != 32) {
        throw std::runtime_error("Cannot hash UF shared-work identity");
    }
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (unsigned int i = 0; i < size; ++i) {
        output << std::setw(2) << static_cast<unsigned>(digest[i]);
    }
    return output.str();
}

std::string shared_job_key(const std::string& native_id) {
    // The maintained helper hashes str.encode('utf-8'), not filename bytes.
    // Decode native CLI/path spellings first; lone surrogate escapes then fail
    // at the same strict UTF-8 identity boundary instead of hashing raw bytes.
    return sha256_text(python_utf8_encode(python_filesystem_decode(native_id)));
}

std::string random_token() {
    std::array<std::uint8_t, 16> bytes{};
    std::size_t received = 0;
    while (received < bytes.size()) {
        ProcessInterrupt::check();
        const auto count = getrandom(bytes.data() + received, bytes.size() - received, 0);
        if (count < 0) {
            const int error = errno;
            if (error == EINTR) continue;
            throw std::system_error(error, std::generic_category(), "Cannot generate shared-work token");
        }
        if (count == 0) throw std::runtime_error("Cannot generate UF shared-work token");
        received += static_cast<std::size_t>(count);
    }
    // uuid.uuid4().hex sets the version and RFC 4122 variant before formatting.
    bytes[6] = static_cast<unsigned char>((bytes[6] & 0x0f) | 0x40);
    bytes[8] = static_cast<unsigned char>((bytes[8] & 0x3f) | 0x80);
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (const auto byte : bytes) output << std::setw(2) << static_cast<int>(byte);
    return output.str();
}

std::string now_iso() {
    const auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm utc{};
    gmtime_r(&now, &utc);
    std::ostringstream output;
    output << std::put_time(&utc, "%Y-%m-%dT%H:%M:%S") << "+00:00";
    return output.str();
}

void write_all(int fd, const std::string& value) {
    std::size_t offset = 0;
    while (offset < value.size()) {
        ProcessInterrupt::check();
        const auto count = write(fd, value.data() + offset, value.size() - offset);
        if (count < 0) {
            const int error = errno;
            if (error == EINTR) continue;
            throw std::system_error(error, std::generic_category(), "Cannot write shared-work state");
        }
        offset += static_cast<std::size_t>(count);
    }
    for (;;) {
        ProcessInterrupt::check();
        if (fsync(fd) == 0) break;
        const int error = errno;
        if (error == EINTR) continue;
        throw std::system_error(error, std::generic_category(), "Cannot sync shared-work state");
    }
}

FileDescriptor open_shared_file(const std::filesystem::path& path, int flags,
                                mode_t mode, int directory = AT_FDCWD) {
    const auto filename = path.string();
    if (filename.find('\0') != std::string::npos)
        throw std::invalid_argument("Embedded NUL in shared-work file path");
    for (;;) {
        ProcessInterrupt::check();
        const int descriptor = openat(directory, path.c_str(), flags | O_CLOEXEC, mode);
        if (descriptor >= 0) return FileDescriptor(descriptor);
        const int error = errno;
        if (error == EINTR) continue;
        throw std::system_error(error, std::generic_category(),
                                "Cannot open shared-work file: " + filename);
    }
}

void close_shared_file(FileDescriptor& file) {
    const int descriptor = file.release();
    if (close(descriptor) != 0) {
        const int error = errno;
        throw std::system_error(error, std::generic_category(), "Cannot close shared-work file");
    }
}

void write_json_and_close(FileDescriptor& file, const Json& value) {
    try {
        write_all(file.get(), canonical_json(value) + "\n");
        close_shared_file(file);
    } catch (...) {
        // Python finally/with closes can replace their active exception.
        if (file.get() >= 0) close_shared_file(file);
        throw;
    }
}

// The maintained UF wrapper rereads pipeline.json after a protocol error.
// A failed read/parse or absent config preserves the original diagnostic.
std::string pipeline_differences(const std::filesystem::path& path, const Json& config) {
    try {
        const auto document = read_python_ordered_json_file(
            path, python_process_text_encoding(), true);
        if (!document.is_object() || !document.contains("config")) return {};
        const auto& previous = document.at("config");
        // Python's .keys() on a non-dict raises AttributeError, which the UF
        // diagnostic wrapper does not catch. Do not suppress a shape error.
        if (!previous.is_object() || !config.is_object())
            throw std::runtime_error("Shared pipeline configuration must be an object");
        const Json defaults = {{"opus_frame_ms", 20}, {"opus_complexity", 10},
                               {"opus_vbr", "on"}};
        std::set<std::string> keys;
        for (auto item = previous.begin(); item != previous.end(); ++item) keys.insert(item.key());
        for (auto item = config.begin(); item != config.end(); ++item) keys.insert(item.key());
        std::string differences;
        for (const auto& key : keys) {
            const Json fallback = defaults.contains(key) ? defaults.at(key) : Json(nullptr);
            const Json saved = previous.contains(key) ? Json(previous.at(key)) : fallback;
            const Json& requested = config.contains(key) ? config.at(key) : fallback;
            if (json_python_equal(saved, requested)) continue;
            differences += "; " + key + ": saved=";
            differences += previous.contains(key) ? python_json_repr(previous.at(key))
                                                 : python_json_repr(fallback);
            differences += ", requested=" + python_json_repr(requested);
        }
        return differences;
    } catch (const nlohmann::json::exception&) {
        return {};
    } catch (const std::system_error&) {
        return {};
    } catch (const std::invalid_argument&) {
        return {};
    }
}

[[noreturn]] void pipeline_error(const std::filesystem::path& path, const Json& config,
                                const std::string& message) {
    throw std::runtime_error(message + pipeline_differences(path, config));
}

void atomic_json(const std::filesystem::path& path, const Json& value) {
    std::filesystem::create_directories(path.parent_path().empty()
        ? std::filesystem::path(".") : path.parent_path());
    const auto temporary = path.parent_path() /
        ("." + path.filename().string() + ".tmp." + std::to_string(getpid()) +
         "." + random_token());
    const auto remove_temporary = [&] {
        if (unlink(temporary.c_str()) != 0) {
            const int error = errno;
            if (error != ENOENT)
                throw std::system_error(error, std::generic_category(),
                    "Cannot remove shared-work temporary: " + temporary.string());
        }
    };
    try {
        // Path.open('xb') creates with mode 0666, subject to the process umask.
        auto file = open_shared_file(temporary, O_WRONLY | O_CREAT | O_EXCL, 0666);
        write_json_and_close(file, value);
        if (rename(temporary.c_str(), path.c_str()) != 0) {
            const int error = errno;
            throw std::system_error(error, std::generic_category(),
                                    "Cannot publish shared-work JSON: " + path.string());
        }
    } catch (...) {
        remove_temporary();
        throw;
    }
    remove_temporary();
}

std::filesystem::path canonical_channel(const std::filesystem::path& channel) {
    return python_path_resolve(channel);
}

bool shared_artifact_name(const std::string& name) {
    // On POSIX, Path(name).name == name admits the empty string and '..', but
    // removes '.' and every spelling containing a directory separator.
    return name != "." && name.find('/') == std::string::npos;
}

bool shared_artifact_complete(const std::filesystem::path& channel,
                              const std::string& name) {
    const auto artifact = python_path_normalize(channel / name);
    // Path.is_file() returns false for NUL in a path instead of truncating it.
    if (artifact.native().find('\0') != std::string::npos) return false;
    struct stat metadata{};
    ProcessInterrupt::check();
    if (stat(artifact.c_str(), &metadata) != 0) {
        const int error = errno;
        if (error == ENOENT || error == ENOTDIR || error == EBADF || error == ELOOP)
            return false;
        throw std::system_error(error, std::generic_category(),
                                "Cannot inspect shared-work artifact: " + artifact.string());
    }
    if (!S_ISREG(metadata.st_mode)) return false;
    // The maintained helper performs a second stat for size. Its failures
    // propagate to mark_done, while completed_artifacts catches them.
    ProcessInterrupt::check();
    if (stat(artifact.c_str(), &metadata) != 0) {
        const int error = errno;
        throw std::system_error(error, std::generic_category(),
                                "Cannot inspect shared-work artifact size: " + artifact.string());
    }
    return metadata.st_size > 0;
}

std::string cached_pipeline_id(const std::shared_ptr<SharedPipelineState>& state,
                               const std::filesystem::path& channel) {
    std::lock_guard<std::mutex> guard(state->mutex);
    const auto found = state->ids.find(channel);
    if (found == state->ids.end())
        throw std::runtime_error("shared pipeline was not initialized for " + channel.string());
    return found->second;
}

double age_seconds(const std::filesystem::path& path) {
    struct timespec now{};
    if (clock_gettime(CLOCK_REALTIME, &now) != 0) {
        const int error = errno;
        throw std::system_error(error, std::generic_category(), "Cannot read shared-work clock");
    }
    // CPython time.time() converts checked epoch nanoseconds to binary64 before
    // division, except for whole seconds. Adapted from CPython pytime.c;
    // see licenses/CPython-3.10.20.txt. The clock is read before the path stat.
    constexpr std::int64_t nanoseconds = 1000000000;
    constexpr auto maximum = std::numeric_limits<std::int64_t>::max();
    constexpr auto minimum = std::numeric_limits<std::int64_t>::min();
    if (now.tv_sec > maximum / nanoseconds || now.tv_sec < minimum / nanoseconds)
        throw std::overflow_error("timestamp too large to convert to C _PyTime_t");
    auto ticks = static_cast<std::int64_t>(now.tv_sec) * nanoseconds;
    if (ticks > maximum - now.tv_nsec)
        throw std::overflow_error("timestamp too large to convert to C _PyTime_t");
    ticks += now.tv_nsec;
    volatile double current;
    if (ticks % nanoseconds == 0) current = static_cast<double>(ticks / nanoseconds);
    else {
        current = static_cast<double>(ticks);
        current /= 1e9;
    }
    struct stat metadata{};
    if (stat(path.c_str(), &metadata) != 0) {
        const int error = errno;
        throw std::system_error(error, std::generic_category(),
                                "Cannot stat shared claim: " + path.string());
    }
    // stat.st_mtime has a different conversion: seconds + fraction first.
    const auto modified = static_cast<double>(metadata.st_mtim.tv_sec) +
                          static_cast<double>(metadata.st_mtim.tv_nsec) * 1e-9;
    return std::max(0.0, current - modified);
}

std::string claim_string(const nlohmann::ordered_json& value) {
    return value.is_string() ? value.get_ref<const std::string&>() : python_json_repr(value);
}

PythonInteger claim_pid(const Json& value) {
    // The reference applies int(state.get('pid') or 0) after the I/O try.
    return json_truthy(value) ? python_json_int(value) : PythonInteger{};
}

}  // namespace

std::string SharedClaimInfo::owner_text() const {
    const auto location = hostname.empty() ? std::string("unknown-host") : hostname;
    return pid.decimal() == "0" ? location : location + " pid " + pid.decimal();
}

SharedLease::SharedLease(std::filesystem::path channel, std::filesystem::path root,
                         std::filesystem::path claim, std::string video_id,
                         std::string token, std::string owner, std::string hostname,
                         std::shared_ptr<SharedPipelineState> pipeline_state, int directory_fd,
                         bool recovered_stale)
    : channel_(std::move(channel)), root_(std::move(root)),
      claim_(std::move(claim)), video_id_(std::move(video_id)),
      token_(std::move(token)), owner_(std::move(owner)),
      hostname_(std::move(hostname)), pipeline_state_(std::move(pipeline_state)),
      directory_fd_(directory_fd), recovered_stale_(recovered_stale) {}

SharedLease::~SharedLease() {
    // Explicit finally paths already attempted release, even if it failed.
    if (release_attempted_) return;
    try { release(); } catch (...) {}
}

bool SharedLease::owns_path(bool check_interrupt) const {
    if (closed_ || directory_fd_ < 0) return false;
    struct stat opened{}, current{};
    if (fstat(directory_fd_, &opened) != 0 || stat(claim_.c_str(), &current) != 0 ||
        opened.st_dev != current.st_dev || opened.st_ino != current.st_ino) return false;
    try {
        const auto state = read_python_json_file(claim_ / "state.json", "UTF-8", false,
                                                check_interrupt);
        if (!state.is_object())
            throw std::invalid_argument("Shared claim state must be an object");
        const auto token = state.find("token");
        return token != state.end() && token->is_string() &&
               token->get_ref<const std::string&>() == token_;
    } catch (const nlohmann::json::parse_error&) {
        return false;
    } catch (const std::system_error&) {
        return false;
    }
}

void SharedLease::assert_owned() const {
    if (!owns_path()) {
        throw std::runtime_error("shared claim for " + python_json_repr(Json(python_filesystem_decode(video_id_))) +
            " was lost; refusing to publish its output");
    }
}

void SharedLease::write_state(const Json& progress) {
    std::lock_guard<std::mutex> guard(state_mutex_);
    const Json state = {{"schema_version", 1}, {"video_id", python_filesystem_decode(video_id_)},
        {"token", token_}, {"owner", python_filesystem_decode(owner_)}, {"hostname", python_filesystem_decode(hostname_)},
        {"pid", getpid()}, {"updated_at", now_iso()},
        {"progress", json_truthy(progress) ? progress : Json::object()}};
    const auto temporary = ".state.tmp." + token_;
    auto file = open_shared_file(temporary, O_WRONLY | O_CREAT | O_TRUNC, 0644, directory_fd_);
    write_json_and_close(file, state);
    if (renameat(directory_fd_, temporary.c_str(), directory_fd_, "state.json") != 0) {
        const int error = errno;
        throw std::system_error(error, std::generic_category(), "Cannot renew shared claim");
    }
}

void SharedLease::initialize() {
    write_state({{"status", "claimed"}});
    assert_owned();
    heartbeat_thread_ = std::thread([this] { heartbeat_loop(); });
}

void SharedLease::heartbeat_loop() {
    std::unique_lock<std::mutex> lock(mutex_);
    while (!wake_.wait_for(lock, std::chrono::seconds(20), [this] { return stopping_; })) {
        lock.unlock();
        try {
            heartbeat();
        } catch (const std::exception&) {
            lock.lock();
            heartbeat_error_ = std::current_exception();
            return;
        }
        lock.lock();
    }
}

void SharedLease::heartbeat(const Json& progress) {
    assert_owned();
    write_state(progress);
    assert_owned();
}

void SharedLease::check() const {
    std::exception_ptr error;
    {
        std::lock_guard<std::mutex> guard(mutex_);
        error = heartbeat_error_;
    }
    if (error) {
        try { std::rethrow_exception(error); }
        catch (const std::exception& failure) {
            throw std::runtime_error("Lost shared-work heartbeat: " + std::string(failure.what()));
        }
    }
    assert_owned();
}

void SharedLease::mark_done(const std::vector<std::string>& artifact_names) {
    assert_owned();
    std::set<std::string> names(artifact_names.begin(), artifact_names.end());
    for (const auto& name : names) {
        if (!shared_artifact_name(name))
            throw std::runtime_error("invalid shared-work artifact name: " + python_json_repr(Json(python_filesystem_decode(name))));
        const auto artifact = python_path_normalize(channel_ / name);
        if (!shared_artifact_complete(channel_, name))
            throw std::runtime_error("cannot mark missing or empty artifact complete: " + artifact.string());
    }
    const auto pipeline_id = cached_pipeline_id(pipeline_state_, canonical_channel(channel_));
    std::vector<std::string> text_names;
    text_names.reserve(names.size());
    for (const auto& name : names) text_names.push_back(python_filesystem_decode(name));
    atomic_json(root_ / "done" / (shared_job_key(video_id_) + ".json"),
        {{"schema_version", 1}, {"pipeline_id", pipeline_id},
         {"video_id", python_filesystem_decode(video_id_)}, {"completed_at", now_iso()},
         {"owner", python_filesystem_decode(owner_)}, {"artifacts", text_names}});
}

void SharedLease::release() {
    if (closed_) return;
    {
        std::lock_guard<std::mutex> guard(mutex_);
        stopping_ = true;
    }
    wake_.notify_all();
    if (heartbeat_thread_.joinable()) heartbeat_thread_.join();
    // Directory cleanup starts only once the heartbeat cannot touch the lease.
    release_attempted_ = true;
    const auto close_directory = [&] {
        if (close(directory_fd_) != 0) {
            const int error = errno;
            throw std::system_error(error, std::generic_category(), "Cannot close shared claim directory");
        }
        closed_ = true;
        directory_fd_ = -1;
    };
    try {
        // The process flag is already handled by the body's exception. Cleanup
        // must still read ownership and close/unlink without throwing it again.
        if (owns_path(false)) {
            const auto temporary = ".state.tmp." + token_;
            const std::array<const char*, 2> entries{{"state.json", temporary.c_str()}};
            for (const auto* name : entries) {
                if (unlinkat(directory_fd_, name, 0) != 0) {
                    const int error = errno;
                    if (error != ENOENT)
                        throw std::system_error(error, std::generic_category(),
                            std::string("Cannot remove shared claim entry: ") + name);
                }
            }
            for (int attempt = 0; attempt < 3; ++attempt) {
                if (rmdir(claim_.c_str()) == 0) break;
                const int error = errno;
                if (error == ENOENT) break;
                if (error != ENOTEMPTY && error != EEXIST)
                    throw std::system_error(error, std::generic_category(),
                                            "Cannot remove shared claim directory");
                if (attempt < 2) std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        }
    } catch (...) {
        close_directory();
        throw;
    }
    close_directory();
}

SharedWorkPool::SharedWorkPool(WorkKind kind, double lease_seconds,
                               std::string instance_name)
    : kind_(kind), lease_seconds_(lease_seconds) {
    if (lease_seconds_ < 30) {
        throw std::invalid_argument("shared-work lease must be at least 30 seconds");
    }
    std::array<char, 256> host{};
    if (gethostname(host.data(), host.size() - 1) != 0) {
        throw std::runtime_error("Cannot read UF shared-work hostname");
    }
    hostname_ = host.data();
    instance_id_ = (instance_name.empty() ? hostname_ : instance_name) +
                   ":" + std::to_string(getpid()) + ":" + random_token();
    pipeline_state_ = std::make_shared<SharedPipelineState>();
}

std::filesystem::path SharedWorkPool::root(const std::filesystem::path& channel) const {
    return python_path_normalize(channel / (kind_ == WorkKind::video
        ? ".dcvc-uf-shared-work" : ".dcvc-uf-image-work"));
}

std::filesystem::path SharedWorkPool::claim_path(
    const std::filesystem::path& channel, const std::string& video_id) const {
    return root(channel) / "claims" / shared_job_key(video_id);
}

std::filesystem::path SharedWorkPool::done_path(
    const std::filesystem::path& channel, const std::string& video_id) const {
    return root(channel) / "done" / (shared_job_key(video_id) + ".json");
}

std::string SharedWorkPool::ensure_pipeline(const std::filesystem::path& channel,
                                            const Json& config) {
    const auto canonical = canonical_channel(channel);
    const auto location = root(canonical);
    std::filesystem::create_directories(location / "claims");
    std::filesystem::create_directories(location / "done");
    std::filesystem::create_directories(location / "stale");
    const auto expected = sha256_text(canonical_json(config));
    const auto path = location / "pipeline.json";
    const Json payload = {{"schema_version", 1}, {"pipeline_id", expected},
                          {"created_at", now_iso()}, {"config", config}};
    FileDescriptor file;
    try {
        file = open_shared_file(path, O_WRONLY | O_CREAT | O_EXCL, 0644);
    } catch (const std::system_error& error) {
        if (error.code() != std::error_code(EEXIST, std::generic_category())) throw;
    }
    if (file.get() >= 0) write_json_and_close(file, payload);
    Json existing;
    bool readable = false;
    for (int attempt = 0; attempt < 50; ++attempt) {
        try {
            existing = read_python_json_file(path, "UTF-8", false);
            readable = true;
            break;
        } catch (const nlohmann::json::parse_error&) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        } catch (const std::system_error& error) {
            if (error.code() != std::error_code(ENOENT, std::generic_category())) throw;
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    }
    if (!readable || existing.is_null())
        pipeline_error(path, config, "could not read shared pipeline state: " + path.string());
    if (!existing.is_object())
        throw std::invalid_argument("Shared pipeline state must be an object");
    const auto saved_id = existing.find("pipeline_id");
    if (saved_id == existing.end() || !saved_id->is_string() ||
        saved_id->get_ref<const std::string&>() != expected)
        pipeline_error(path, config,
            "shared pipeline settings differ for " + canonical.string() +
            "; use the same models and encode options on every machine, or use a separate output directory");
    {
        std::lock_guard<std::mutex> guard(pipeline_state_->mutex);
        pipeline_state_->ids.insert_or_assign(canonical, expected);
    }
    return expected;
}

std::string SharedWorkPool::pipeline_id_for(const std::filesystem::path& channel) const {
    return cached_pipeline_id(pipeline_state_, canonical_channel(channel));
}

std::optional<std::vector<std::string>> SharedWorkPool::completed_artifacts(
    const std::filesystem::path& channel, const std::string& video_id) const {
    const auto path = done_path(channel, video_id);
    Json record;
    try {
        record = read_python_json_file(path, "UTF-8", false);
    } catch (const nlohmann::json::parse_error&) {
        return std::nullopt;
    } catch (const std::system_error&) {
        return std::nullopt;
    }
    // Python record.get() and cache lookup run outside its read-error catches.
    if (!record.is_object())
        throw std::invalid_argument("Shared completion state must be an object");
    const auto saved_id = record.find("pipeline_id");
    const auto expected_id = pipeline_id_for(channel);
    if (saved_id == record.end() || !saved_id->is_string() ||
        saved_id->get_ref<const std::string&>() != expected_id) return std::nullopt;
    const auto names = record.find("artifacts");
    if (names == record.end() || !names->is_array() || names->empty()) return std::nullopt;
    for (const auto& value : *names) {
        if (!value.is_string()) return std::nullopt;
        const auto& name = value.get_ref<const std::string&>();
        if (!shared_artifact_name(name)) return std::nullopt;
        try {
            if (!shared_artifact_complete(channel, python_filesystem_encode(name))) return std::nullopt;
        } catch (const std::system_error&) {
            return std::nullopt;
        }
    }
    // As with tuple(names), construct the result only after all checks pass.
    // Preserve disk order and duplicates; do not sort or deduplicate this read.
    std::vector<std::string> artifacts;
    artifacts.reserve(names->size());
    for (const auto& value : *names) artifacts.push_back(python_filesystem_encode(value.get_ref<const std::string&>()));
    return artifacts;
}

std::optional<SharedClaimInfo> SharedWorkPool::read_claim(
    const std::filesystem::path& channel, const std::string& video_id) const {
    const auto path = claim_path(channel, video_id) / "state.json";
    nlohmann::ordered_json state;
    double age;
    try {
        state = read_python_ordered_json_file(path, "UTF-8", false);
        age = age_seconds(path);
    } catch (const nlohmann::json::parse_error&) {
        return std::nullopt;
    } catch (const std::system_error&) {
        return std::nullopt;
    }
    // Python state.get()/str()/int() conversions run outside its I/O catches.
    if (!state.is_object())
        throw std::invalid_argument("Shared claim state must be an object");
    SharedClaimInfo info;
    info.video_id = state.contains("video_id") ? claim_string(state.at("video_id")) : python_filesystem_decode(video_id);
    info.owner = state.contains("owner") ? claim_string(state.at("owner")) : std::string{};
    info.hostname = state.contains("hostname") ? claim_string(state.at("hostname")) : std::string{};
    info.pid = claim_pid(state.contains("pid") ? Json(state.at("pid")) : Json(nullptr));
    info.age_seconds = age;
    if (state.contains("progress") && state.at("progress").is_object())
        info.progress = state.at("progress");
    return info;
}

std::unique_ptr<SharedLease> SharedWorkPool::try_acquire(
    const std::filesystem::path& channel, const std::string& video_id,
    std::optional<SharedClaimInfo>* owner) {
    const auto canonical = canonical_channel(channel);
    pipeline_id_for(canonical);
    const auto location = root(canonical);
    const auto claim = claim_path(canonical, video_id);
    const auto key = claim.filename().string();
    const auto token = random_token();
    bool recovered_stale = false;
    for (int attempt = 0; attempt < 4; ++attempt) {
        std::error_code error;
        if (std::filesystem::create_directory(claim, error)) {
            auto directory = open_shared_file(claim, O_RDONLY | O_DIRECTORY, 0);
            auto lease = std::unique_ptr<SharedLease>(new SharedLease(canonical, location,
                claim, video_id, token, instance_id_, hostname_, pipeline_state_, directory.get(),
                recovered_stale));
            directory.release();
            try { lease->initialize(); }
            catch (...) { lease->release(); throw; }
            if (owner) owner->reset();
            return lease;
        }
        if (error && error != std::errc::file_exists) {
            throw std::runtime_error("Cannot create UF shared claim directory");
        }
        auto info = read_claim(canonical, video_id);
        const auto state = claim / "state.json";
        double age;
        try {
            age = age_seconds(state);
        } catch (const std::system_error& error) {
            if (error.code() != std::error_code(ENOENT, std::generic_category())) throw;
            try {
                age = age_seconds(claim);
            } catch (const std::system_error& missing) {
                if (missing.code() != std::error_code(ENOENT, std::generic_category())) throw;
                continue;
            }
        }
        if (age <= lease_seconds_) {
            if (owner) *owner = std::move(info);
            return nullptr;
        }
        const auto stale = location / "stale" / (key + "." + random_token());
        std::filesystem::rename(claim, stale, error);
        if (error == std::errc::no_such_file_or_directory) continue;
        if (error) {
            if (owner) *owner = std::move(info);
            return nullptr;
        }
        std::filesystem::remove_all(stale, error);
        recovered_stale = true;
    }
    auto info = read_claim(canonical, video_id);
    if (owner) *owner = std::move(info);
    return nullptr;
}

}  // namespace uf
