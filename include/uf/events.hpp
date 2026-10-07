#pragma once

#include <nlohmann/json.hpp>

#include <filesystem>
#include <functional>
#include <memory>
#include <string>

namespace uf {

using EventSink = std::function<void(const nlohmann::json&)>;

// Parent presentation can install a sink; freshly exec'd workers write records
// to a dedicated inherited pipe. Log records use the Python UF field names.
void set_event_sink(EventSink sink);
void set_event_descriptor(int descriptor);
void deliver_event(const nlohmann::json& record);
// Serialize, deliver, then append UF_PROGRESS_LOG with checked finally close.
// An append/close failure propagates after delivery, following storage.event.
// Failed stdout writes/flushes propagate before logging; pipe EINTR retries
// check the foreground interrupt. Presentation sinks supply their own errors.
// Caller fields override default time/event; nonempty UF_JOB_KEY overrides job_key.
void event(const std::string& name, nlohmann::json fields = nlohmann::json::object());

class JobEvents {
public:
    JobEvents(std::string job_key, const std::filesystem::path& progress_log);
    ~JobEvents();
    JobEvents(const JobEvents&) = delete;
    JobEvents& operator=(const JobEvents&) = delete;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace uf
