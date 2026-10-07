#include "uf/frame_buffer.hpp"
#include "uf/process.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

namespace uf {

struct BufferedFrames::Impl {
    struct Item {
        std::optional<std::vector<std::uint8_t>> frame;
        std::exception_ptr error;
    };
    struct State {
        Read source;
        Cancel cancel;
        std::size_t capacity;
        bool stop{}, terminal{}, consumed_end{}, producer_done{}, threaded{};
        std::exception_ptr fatal;
        std::mutex mutex;
        std::condition_variable ready, space, done;
        std::deque<Item> queue;

        State(Read input, Cancel interrupt, std::size_t frames)
            : source(std::move(input)), cancel(std::move(interrupt)), capacity(frames) {}
    };
    std::shared_ptr<State> state;
    std::thread producer;
    bool close_attempted{};

    Impl(Read input, Cancel interrupt, std::size_t frames, std::size_t bytes, bool dynamic)
        : state(std::make_shared<State>(std::move(input), std::move(interrupt), frames)) {
        if (!state->source || !state->cancel || !bytes) throw std::invalid_argument("Invalid UF frame buffer settings");
        if (!dynamic) state->capacity = std::min(state->capacity,
            std::max<std::size_t>(1, (8u * 1024u * 1024u) / bytes));
        state->threaded = state->capacity != 0;
        if (state->capacity) producer = std::thread([owned = state] {
            try { produce(owned); }
            catch (...) {
                const std::lock_guard<std::mutex> lock(owned->mutex);
                owned->fatal = std::current_exception();
                owned->terminal = true;
            }
            {
                const std::lock_guard<std::mutex> lock(owned->mutex);
                owned->producer_done = true;
            }
            owned->ready.notify_all();
            owned->done.notify_all();
        });
    }

    static void produce(const std::shared_ptr<State>& state) {
        for (;;) {
            {
                std::lock_guard<std::mutex> lock(state->mutex);
                if (state->stop) return;
            }
            Item item;
            try { item.frame = state->source(); }
            catch (...) { item.error = std::current_exception(); }
            const bool last = !item.frame || item.error;
            std::unique_lock<std::mutex> lock(state->mutex);
            state->space.wait(lock, [&] { return state->stop || state->queue.size() < state->capacity; });
            if (state->stop) return;
            state->queue.push_back(std::move(item));
            if (last) state->terminal = true;
            state->ready.notify_all();
            if (last) return;
        }
    }

    void close_checked() {
        if (close_attempted) return;
        close_attempted = true;
        {
            std::lock_guard<std::mutex> lock(state->mutex);
            state->stop = true;
        }
        state->ready.notify_all();
        state->space.notify_all();
        // Kill abandoned input before joining, as FrameReader.close does.
        try { state->cancel(); }
        catch (...) {
            // Source close stops here if process cleanup fails. Retain the
            // worker's resources without waiting on a failed cancellation.
            if (producer.joinable()) producer.detach();
            throw;
        }
        if (producer.joinable()) {
            std::unique_lock<std::mutex> lock(state->mutex);
            const bool ended = state->done.wait_for(lock, std::chrono::seconds(10),
                [&] { return state->producer_done; });
            lock.unlock();
            if (!ended) {
                // The worker owns State and its callbacks. Callers retain
                // their reader through those callbacks until it exits.
                producer.detach();
                throw std::runtime_error("Input prefetch thread did not stop");
            }
            producer.join();
        }
    }

    ~Impl() {
        try { close_checked(); } catch (...) {}
        // A native join/detach failure must not leave a joinable thread for
        // std::thread's destructor. State remains owned by a live worker.
        if (producer.joinable()) {
            try { producer.detach(); } catch (...) { producer.join(); }
        }
    }
};

BufferedFrames::BufferedFrames(Read source, Cancel cancel, std::size_t frames,
                                std::size_t bytes, bool adaptive)
    : impl_(std::make_unique<Impl>(std::move(source), std::move(cancel), frames, bytes, adaptive)) {}
BufferedFrames::~BufferedFrames() = default;

std::optional<std::vector<std::uint8_t>> BufferedFrames::read() {
    ProcessInterrupt::check();
    auto& state = *impl_->state;
    std::unique_lock<std::mutex> lock(state.mutex);
    if (state.stop) throw std::invalid_argument("Frame reader is closed");
    if (state.consumed_end) return std::nullopt;
    if (!impl_->producer.joinable()) {
        lock.unlock();
        return state.source();
    }
    while (!state.stop && state.queue.empty()) {
        ProcessInterrupt::check();
        if (state.fatal) std::rethrow_exception(state.fatal);
        state.ready.wait_for(lock, std::chrono::milliseconds(100));
    }
    ProcessInterrupt::check();
    if (state.stop) throw std::invalid_argument("Frame reader is closed");
    auto item = std::move(state.queue.front());
    state.queue.pop_front();
    state.space.notify_all();
    if (item.error || !item.frame) state.consumed_end = true;
    lock.unlock();
    if (item.error) std::rethrow_exception(item.error);
    return std::move(item.frame);
}

void BufferedFrames::prime() {
    ProcessInterrupt::check();
    if (!impl_->producer.joinable()) return;
    auto& state = *impl_->state;
    std::unique_lock<std::mutex> lock(state.mutex);
    while (!state.stop && !state.terminal && state.queue.size() < state.capacity) {
        ProcessInterrupt::check();
        state.ready.wait_for(lock, std::chrono::milliseconds(100));
    }
    ProcessInterrupt::check();
}

void BufferedFrames::set_capacity(std::size_t frames) {
    if (!impl_->state->threaded || !frames) {
        throw std::invalid_argument("Adaptive UF buffering requires a positive threaded queue");
    }
    auto& state = *impl_->state;
    std::lock_guard<std::mutex> lock(state.mutex);
    state.capacity = frames;
    state.space.notify_all();
    state.ready.notify_all();
}

void BufferedFrames::close_checked() { impl_->close_checked(); }

BufferedFrameReader::BufferedFrameReader(const std::filesystem::path& source,
    std::uint32_t width, std::uint32_t height, const MediaInfo& original,
    std::optional<double> fps, int threads, std::optional<int> input_fd,
    std::string hwaccel, std::size_t requested_frames, bool adaptive)
    : reader_(std::make_shared<FfmpegFrameReader>(source, width, height, original,
                                                fps, threads, input_fd, std::move(hwaccel))) {
    try {
        buffer_ = std::make_unique<BufferedFrames>(
            [owned = reader_] { return owned->read(); },
            [owned = reader_] { owned->stop_for_cleanup(); }, requested_frames,
            static_cast<std::size_t>(width) * height * 3, adaptive);
    } catch (...) {
        reader_->close_checked();
        throw;
    }
}

BufferedFrameReader::~BufferedFrameReader() {
    try { close_checked(); } catch (...) {}
}

std::optional<std::vector<std::uint8_t>> BufferedFrameReader::read() {
    if (closed_) throw std::invalid_argument("Frame reader is closed");
    return buffer_->read();
}

void BufferedFrameReader::prime() { buffer_->prime(); }
void BufferedFrameReader::set_capacity(std::size_t frames) { buffer_->set_capacity(frames); }

void BufferedFrameReader::close_checked() {
    if (closed_) return;
    closed_ = true;
    buffer_->close_checked();
    reader_->close_checked();
}

}  // namespace uf
