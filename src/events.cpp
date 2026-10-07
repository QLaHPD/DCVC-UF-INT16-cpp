#include "uf/events.hpp"
#include "uf/process.hpp"
#include "uf/python_json.hpp"
#include "uf/utf8.hpp"

#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <utility>
#include <fcntl.h>
#include <unistd.h>

namespace uf {
namespace {

std::mutex presentation_mutex, write_mutex;
EventSink presentation;
std::atomic<int> event_descriptor{-1};

std::string utc_time() {
    const auto now = std::chrono::system_clock::now();
    const auto micros = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
    // datetime.isoformat(timespec='auto') omits a zero microsecond fraction.
    // Floor the seconds boundary so the remainder also works before 1970.
    auto seconds_count = micros / 1000000;
    auto fraction = micros % 1000000;
    if (fraction < 0) { --seconds_count; fraction += 1000000; }
    const auto seconds = static_cast<std::time_t>(seconds_count);
    std::tm parts{};
    if (!gmtime_r(&seconds, &parts)) throw std::runtime_error("Cannot format UF event time");
    std::ostringstream result;
    result << std::put_time(&parts, "%Y-%m-%dT%H:%M:%S");
    if (fraction) result << '.' << std::setfill('0') << std::setw(6) << fraction;
    result << "+00:00";
    return result.str();
}

std::string event_line(const nlohmann::json& record) {
    return python_compact_json(record, true) + "\n";
}

void write_all(int descriptor, const std::string& text) {
    std::size_t offset = 0;
    while (offset < text.size()) {
        const auto count = write(descriptor, text.data() + offset, text.size() - offset);
        if (count < 0 && errno == EINTR) {
            ProcessInterrupt::check();
            continue;
        }
        if (count <= 0) throw std::runtime_error("Cannot write UF event record");
        offset += static_cast<std::size_t>(count);
    }
}

std::optional<std::string> environment(const char* name) {
    if (const char* value = std::getenv(name)) return std::string(value);
    return std::nullopt;
}

void restore(const char* name, const std::optional<std::string>& value) noexcept {
    if (value) setenv(name, value->c_str(), 1); else unsetenv(name);
}

void deliver_record(const nlohmann::json& record, const std::string* serialized) {
    EventSink sink;
    {
        std::lock_guard<std::mutex> lock(presentation_mutex);
        sink = presentation;
    }
    if (sink) { sink(record); return; }
    std::lock_guard<std::mutex> lock(write_mutex);
    std::string formatted;
    if (!serialized) { formatted = event_line(record); serialized = &formatted; }
    const auto descriptor = event_descriptor.load();
    if (descriptor >= 0) write_all(descriptor, *serialized);
    else {
        // Python print(..., flush=True) raises on delivery failure. The
        // default ostream mask only records errors, so inspect each boundary.
        std::cout << *serialized;
        if (!std::cout.good()) throw std::runtime_error("Cannot write UF event record");
        std::cout.flush();
        if (!std::cout.good()) throw std::runtime_error("Cannot flush UF event record");
    }
}

}  // namespace

void set_event_sink(EventSink sink) {
    std::lock_guard<std::mutex> lock(presentation_mutex);
    presentation = std::move(sink);
}
void set_event_descriptor(int descriptor) { event_descriptor.store(descriptor); }

void deliver_event(const nlohmann::json& record) {
    deliver_record(record, nullptr);
}

void event(const std::string& name, nlohmann::json fields) {
    if (!fields.is_object()) throw std::invalid_argument("UF event fields must be an object");
    // Python constructs defaults before unpacking **fields; caller values
    // replace time/event, but a nonempty UF_JOB_KEY then replaces job_key.
    auto timestamp = utc_time();
    if (!fields.contains("time")) fields["time"] = std::move(timestamp);
    if (!fields.contains("event")) fields["event"] = name;
    if (const char* key = std::getenv("UF_JOB_KEY"); key && *key)
        fields["job_key"] = python_filesystem_decode(key);
    // storage.event serializes before delivery, then appends the progress
    // log. A log failure must not suppress the already delivered event.
    const auto line = event_line(fields);
    deliver_record(fields, &line);
    if (const char* log = std::getenv("UF_PROGRESS_LOG"); log && *log) {
        int descriptor;
        do {
            descriptor = open(log, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
            if (descriptor < 0 && errno == EINTR) ProcessInterrupt::check();
        } while (descriptor < 0 && errno == EINTR);
        FileDescriptor file(descriptor);
        if (file.get() < 0) throw std::system_error(errno, std::generic_category(), "Cannot open UF progress log");
        try {
            ssize_t count;
            do {
                count = write(file.get(), line.data(), line.size());
                if (count < 0 && errno == EINTR) ProcessInterrupt::check();
            } while (count < 0 && errno == EINTR);
            // os.write returns a short count without retrying its remainder;
            // storage.event ignores that count. Transport writes above retain
            // their separate complete-record policy.
            if (count < 0) throw std::system_error(errno, std::generic_category(), "Cannot write UF progress log");
        } catch (...) {
            file.close_checked("Cannot close UF progress log");
            throw;
        }
        file.close_checked("Cannot close UF progress log");
    }
}

struct JobEvents::Impl {
    std::optional<std::string> previous_key{environment("UF_JOB_KEY")};
    std::optional<std::string> previous_log{environment("UF_PROGRESS_LOG")};
    Impl(const std::string& key, const std::filesystem::path& log) {
        if (setenv("UF_JOB_KEY", key.c_str(), 1) != 0) throw std::runtime_error("Cannot set UF job key");
        if (setenv("UF_PROGRESS_LOG", log.c_str(), 1) != 0) {
            restore("UF_JOB_KEY", previous_key);
            throw std::runtime_error("Cannot set UF progress log");
        }
    }
    ~Impl() {
        restore("UF_JOB_KEY", previous_key);
        restore("UF_PROGRESS_LOG", previous_log);
    }
};

JobEvents::JobEvents(std::string key, const std::filesystem::path& log)
    : impl_(std::make_unique<Impl>(key, log)) {}
JobEvents::~JobEvents() = default;

}  // namespace uf
