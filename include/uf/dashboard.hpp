#pragma once

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace uf {

struct DashboardAction {
    std::string kind;
    std::optional<std::string> channel;  // Python/JSON text; decode to a native path.
};

// Consume structured events in plain mode or a responsive terminal dashboard.
// The presentation thread queues commands; only the workflow deletes originals.
class Dashboard {
public:
    Dashboard(std::filesystem::path storage, int workers = 1, std::string mode = "auto");
    ~Dashboard();
    Dashboard(const Dashboard&) = delete;
    Dashboard& operator=(const Dashboard&) = delete;
    bool interactive() const;
    std::vector<DashboardAction> poll_actions();
private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

}  // namespace uf
