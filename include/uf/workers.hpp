#pragma once

#include "uf/workflow.hpp"
#include "uf/image_workflow.hpp"

#include <memory>
#include <set>

namespace uf {

class PreparedDownload;

struct EncodeOutcome {
    std::string status;
    std::string source;
    std::filesystem::path manifest;
    std::string error;
};

struct EncodeRun {
    std::vector<EncodeOutcome> outcomes;
    std::vector<std::filesystem::path> manifests;
    std::set<std::filesystem::path> failed_channels;
    std::size_t encoded{}, resumed{}, busy{}, failed{};
};

struct ImageOutcome {
    std::string status;
    std::filesystem::path source, archive;
    std::string error;
};

struct ImageRun {
    std::vector<ImageOutcome> outcomes;
    std::set<std::filesystem::path> failed_channels;
    std::size_t encoded{}, resumed{}, busy{}, failed{};
    bool worker_failed{};
};

// One fresh process owns the entire sequential image pass and its lazy codec.
// Events are drained while it runs; failures preserve every affected channel.
ImageRun encode_image_pass(const std::vector<ImageEncodeOptions>& jobs);
int run_image_worker(int input_descriptor = 0, int result_descriptor = 3);

// Launch a fresh executable per archive, as Python's spawn workers do.
// Worker communication uses anonymous RAM files and inherited descriptors.
class EncodeScheduler {
public:
    EncodeScheduler(int workers, std::vector<std::string> devices, bool streaming = false);
    ~EncodeScheduler();
    EncodeScheduler(const EncodeScheduler&) = delete;
    EncodeScheduler& operator=(const EncodeScheduler&) = delete;
    bool has_capacity() const;
    std::size_t running() const;
    void submit(const EncodeOptions& job, std::shared_ptr<PreparedDownload> prepared = {});
    void poll();
    bool near_end() const;
    EncodeRun finish();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Private executable entry point; fd 0 contains a job, fd 3 receives its result.
int run_encode_worker(int input_descriptor = 0, int result_descriptor = 3);

}  // namespace uf
