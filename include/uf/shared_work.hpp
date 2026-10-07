#pragma once

#include "uf/python_integer.hpp"

#include <nlohmann/json.hpp>

#include <condition_variable>
#include <exception>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace uf {

enum class WorkKind { video, image };

struct SharedPipelineState;

struct SharedClaimInfo {
    std::string video_id, owner, hostname;
    PythonInteger pid;
    double age_seconds = 0.0;
    nlohmann::ordered_json progress = nlohmann::ordered_json::object();

    std::string owner_text() const;
};

class SharedWorkBusy : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class SharedLease {
public:
    ~SharedLease();
    SharedLease(const SharedLease&) = delete;
    SharedLease& operator=(const SharedLease&) = delete;

    void check() const;
    void heartbeat(const nlohmann::json& progress = nlohmann::json::object());
    void mark_done(const std::vector<std::string>& artifact_names);
    bool recovered_stale() const noexcept { return recovered_stale_; }
    void release();

private:
    friend class SharedWorkPool;
    SharedLease(std::filesystem::path channel, std::filesystem::path root,
                std::filesystem::path claim, std::string video_id,
                std::string token, std::string owner, std::string hostname,
                std::shared_ptr<SharedPipelineState> pipeline_state, int directory_fd,
                bool recovered_stale);
    void initialize();
    bool owns_path(bool check_interrupt = true) const;
    void assert_owned() const;
    void write_state(const nlohmann::json& progress);
    void heartbeat_loop();

    std::filesystem::path channel_, root_, claim_;
    std::string video_id_, token_, owner_, hostname_;
    std::shared_ptr<SharedPipelineState> pipeline_state_;
    int directory_fd_{-1};
    bool closed_ = false;
    bool release_attempted_ = false;
    bool recovered_stale_ = false;
    mutable std::mutex mutex_;
    std::mutex state_mutex_;
    std::condition_variable wake_;
    std::thread heartbeat_thread_;
    bool stopping_{};
    std::exception_ptr heartbeat_error_;
};

// Run the UF body and checked release in Python try/finally order. A release
// failure replaces an active body exception; destructors do not retry it.
template<class Operation>
auto with_shared_lease(SharedLease& lease, Operation&& operation)
    -> std::invoke_result_t<Operation> {
    using Result = std::invoke_result_t<Operation>;
    if constexpr (std::is_void_v<Result>) {
        try { std::forward<Operation>(operation)(); }
        catch (...) { lease.release(); throw; }
        lease.release();
    } else {
        Result result = [&]() -> Result {
            try { return std::forward<Operation>(operation)(); }
            catch (...) { lease.release(); throw; }
        }();
        lease.release();
        return std::forward<Result>(result);
    }
}

class SharedWorkPool {
public:
    explicit SharedWorkPool(WorkKind kind, double lease_seconds = 900,
                            std::string instance_name = {});
    std::string ensure_pipeline(const std::filesystem::path& channel,
                                const nlohmann::json& config);
    std::string pipeline_id_for(const std::filesystem::path& channel) const;
    std::filesystem::path root(const std::filesystem::path& channel) const;
    std::filesystem::path claim_path(const std::filesystem::path& channel,
                                    const std::string& video_id) const;
    std::filesystem::path done_path(const std::filesystem::path& channel,
                                   const std::string& video_id) const;
    // Missing/invalid records return nullopt; successful lookup retains order
    // and duplicates. UTF-8, object-shape and uninitialized-cache errors propagate.
    std::optional<std::vector<std::string>> completed_artifacts(
        const std::filesystem::path& channel, const std::string& video_id) const;
    std::optional<SharedClaimInfo> read_claim(const std::filesystem::path& channel,
                                             const std::string& video_id) const;
    std::unique_ptr<SharedLease> try_acquire(const std::filesystem::path& channel,
                                            const std::string& video_id,
                                            std::optional<SharedClaimInfo>* owner = nullptr);

private:
    WorkKind kind_;
    double lease_seconds_;
    std::string hostname_, instance_id_;
    std::shared_ptr<SharedPipelineState> pipeline_state_;
};

}  // namespace uf
