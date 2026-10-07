#pragma once

#include "uf/remote.hpp"

#include <cstdint>
#include <memory>
#include <optional>

namespace uf {

// Anonymous tmpfs memory inherited across freshly exec'd archive workers.
// A process-shared recursive mutex coordinates calibration ownership/capacity;
// Linux lock-free integer atomics publish download status and near-end state.
class SharedStreamState {
public:
    static constexpr int pending = 1000000;
    SharedStreamState();
    explicit SharedStreamState(int descriptor);
    ~SharedStreamState();
    SharedStreamState(const SharedStreamState&) = delete;
    SharedStreamState& operator=(const SharedStreamState&) = delete;
    int descriptor() const;
    int status() const;
    void set_status(int status);
    bool near_end() const;
    void set_near_end();
    void claim();
    void release(int process_id = 0);
    int capacity() const;
    // Successful calibration updates capacity/measured and submits its event
    // within the same policy lock. Event errors propagate after state updates.
    std::optional<int> observe(std::uint64_t frames, double fps);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Starts extraction/network work into an unread bounded OS pipe. The parent
// monitors yt-dlp stderr and passes descriptors into an archive worker.
class PreparedDownload {
public:
    explicit PreparedDownload(const RemoteJob& remote);
    ~PreparedDownload();
    PreparedDownload(const PreparedDownload&) = delete;
    PreparedDownload& operator=(const PreparedDownload&) = delete;
    int input_descriptor() const;
    int error_descriptor() const;
    std::shared_ptr<SharedStreamState> state() const;
    // Idempotent parent-side close, including shared aliases retained elsewhere.
    // Inherited worker descriptors and shared state keep their own lifetimes.
    // Explicit descriptor/cookie cleanup errors propagate to the caller.
    void close();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

void drain_prepared_download(int input, int errors, const SharedStreamState& state);

}  // namespace uf
