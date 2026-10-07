#include "uf/workers.hpp"
#include "uf/process.hpp"
#include "uf/shared_work.hpp"
#include "uf/stream_pipeline.hpp"
#include "uf/events.hpp"
#include "uf/python_json.hpp"
#include "uf/python_path.hpp"
#include "uf/utf8.hpp"

#include <nlohmann/json.hpp>

#include <array>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <deque>
#include <iostream>
#include <list>
#include <stdexcept>
#include <thread>
#include <utility>
#include <fcntl.h>
#include <unistd.h>

namespace uf {
namespace {

template <typename T>
nlohmann::json nullable(const std::optional<T>& value) {
    return value ? nlohmann::json(*value) : nlohmann::json(nullptr);
}

template <typename T>
std::optional<T> optional_value(const nlohmann::json& value, const char* key) {
    if (!value.contains(key) || value.at(key).is_null()) return std::nullopt;
    return value.at(key).get<T>();
}

nlohmann::json job_json(const EncodeOptions& job) {
    nlohmann::json result = {
        {"protocol", "uf-cpp-archive-worker-v1"},
        {"event_pipe", true},
        {"input_file", python_path_text(job.input_file)}, {"output_root", python_path_text(job.output_root)},
        {"relative_path", job.relative_path ? nlohmann::json(python_filesystem_decode(*job.relative_path)) : nlohmann::json(nullptr)},
        {"prepared_i", python_path_text(job.prepared_i)}, {"prepared_p", python_path_text(job.prepared_p)},
        {"checkpoint_i", python_path_text(job.checkpoint_i)}, {"checkpoint_p", python_path_text(job.checkpoint_p)},
        {"runtime", job.runtime}, {"variant", job.variant}, {"device", job.device},
        {"qp_i", job.qp_i}, {"qp_p", job.qp_p}, {"reset_interval", job.reset_interval},
        {"intra_period", job.intra_period}, {"resolution", nullable(job.resolution)},
        {"fps", nullable(job.fps)}, {"max_frames", nullable(job.max_frames)},
        {"skip_threshold", job.skip_threshold}, {"audio", job.audio},
        {"opus_channels", job.opus_channels}, {"opus_bitrate", job.opus_bitrate},
        {"opus_frame_ms", job.opus_frame_ms}, {"opus_complexity", job.opus_complexity},
        {"opus_vbr", job.opus_vbr}, {"input_threads", job.input_threads},
        {"prefetch_frames", job.prefetch_frames}, {"shared_instance", job.shared_instance},
        {"hwaccel", job.hwaccel},
        {"shared_lease_seconds", job.shared_lease_seconds}, {"remote", nullptr},
        {"stream_state", job.streaming.has_value()},
        {"prepared_download", job.streaming && job.streaming->stream >= 0}
    };
    if (job.remote) {
        const auto& remote = *job.remote;
        result["remote"] = {
            {"video_id", remote.video_id}, {"channel_id", remote.channel_id},
            {"url", remote.url}, {"video_format", remote.video_format},
            {"audio_format", nullable(remote.audio_format)},
            {"media", {{"width", remote.media.width}, {"height", remote.media.height},
                       {"fps", remote.media.fps}, {"duration", remote.media.duration},
                       {"audio", remote.media.audio}}},
            {"public_metadata", remote.public_metadata}, {"manifest", python_path_text(remote.manifest)},
            {"ytdlp_binary", python_filesystem_decode(remote.ytdlp_binary)},
            {"cookies", remote.cookies ? nlohmann::json(python_path_text(*remote.cookies)) : nlohmann::json(nullptr)},
            {"write_thumbnail", remote.write_thumbnail}, {"thumbnail_codec", remote.thumbnail_codec},
            {"thumbnail_qp", remote.thumbnail_qp}
        };
    }
    return result;
}

EncodeOptions read_job(const nlohmann::json& value) {
    if (value.at("protocol") != "uf-cpp-archive-worker-v1") {
        throw std::invalid_argument("Unsupported UF archive worker protocol");
    }
    EncodeOptions job;
    job.input_file = python_path_from_text(value.at("input_file").get_ref<const std::string&>());
    job.output_root = python_path_from_text(value.at("output_root").get_ref<const std::string&>());
    job.relative_path = optional_value<std::string>(value, "relative_path");
    if (job.relative_path) *job.relative_path = python_filesystem_encode(*job.relative_path);
    job.prepared_i = python_path_from_text(value.at("prepared_i").get_ref<const std::string&>());
    job.prepared_p = python_path_from_text(value.at("prepared_p").get_ref<const std::string&>());
    job.checkpoint_i = python_path_from_text(value.at("checkpoint_i").get_ref<const std::string&>());
    job.checkpoint_p = python_path_from_text(value.at("checkpoint_p").get_ref<const std::string&>());
    job.runtime = value.at("runtime").get<std::string>();
    job.variant = value.at("variant").get<std::string>();
    job.device = value.at("device").get<std::string>();
    job.qp_i = value.at("qp_i").get<int>();
    job.qp_p = value.at("qp_p").get<int>();
    job.reset_interval = value.at("reset_interval").get<int>();
    job.intra_period = value.at("intra_period").get<int>();
    job.resolution = optional_value<int>(value, "resolution");
    job.fps = optional_value<double>(value, "fps");
    job.max_frames = optional_value<std::uint64_t>(value, "max_frames");
    job.skip_threshold = value.at("skip_threshold").get<double>();
    job.audio = value.at("audio").get<std::string>();
    job.opus_channels = value.at("opus_channels").get<std::string>();
    job.opus_bitrate = value.at("opus_bitrate").get<std::string>();
    job.opus_frame_ms = value.at("opus_frame_ms");
    if (!job.opus_frame_ms.is_number() && !python_json_extended_integer(job.opus_frame_ms))
        throw std::invalid_argument("Opus frame duration must be numeric");
    job.opus_complexity = value.at("opus_complexity").get<int>();
    job.opus_vbr = value.at("opus_vbr").get<std::string>();
    job.input_threads = value.at("input_threads").get<int>();
    job.prefetch_frames = value.at("prefetch_frames").get<int>();
    job.hwaccel = value.at("hwaccel").get<std::string>();
    job.shared_instance = value.at("shared_instance").get<std::string>();
    job.shared_lease_seconds = value.at("shared_lease_seconds").get<double>();
    if (value.at("stream_state").get<bool>()) {
        const bool prepared = value.at("prepared_download").get<bool>();
        job.streaming = StreamInputs{prepared ? 4 : -1, prepared ? 5 : -1, 6, 7};
    }
    if (!value.at("remote").is_null()) {
        const auto& item = value.at("remote");
        RemoteJob remote;
        remote.video_id = item.at("video_id").get<std::string>();
        remote.channel_id = item.at("channel_id").get<std::string>();
        remote.url = item.at("url").get<std::string>();
        remote.video_format = item.at("video_format").get<std::string>();
        remote.audio_format = optional_value<std::string>(item, "audio_format");
        const auto& media = item.at("media");
        remote.media = {media.at("width").get<std::uint32_t>(), media.at("height").get<std::uint32_t>(),
                        media.at("fps").get<double>(), media.at("duration").get<double>(),
                        media.at("audio").get<bool>()};
        remote.public_metadata = item.at("public_metadata");
        remote.manifest = python_path_from_text(item.at("manifest").get_ref<const std::string&>());
        remote.ytdlp_binary = python_filesystem_encode(item.at("ytdlp_binary").get_ref<const std::string&>());
        if (const auto cookies = optional_value<std::string>(item, "cookies")) remote.cookies = python_path_from_text(*cookies);
        remote.write_thumbnail = item.at("write_thumbnail").get<bool>();
        remote.thumbnail_codec = item.at("thumbnail_codec").get<std::string>();
        remote.thumbnail_qp = item.at("thumbnail_qp").get<int>();
        job.remote = std::move(remote);
    }
    if (job.streaming && !job.remote) throw std::invalid_argument("UF stream descriptors require a remote job");
    return job;
}

std::string read_descriptor(int descriptor, std::size_t maximum) {
    std::string result;
    std::array<char, 16384> buffer{};
    for (;;) {
        const auto count = read(descriptor, buffer.data(), buffer.size());
        if (count < 0 && errno == EINTR) continue;
        if (count < 0) throw std::runtime_error("Cannot read UF worker message");
        if (!count) return result;
        if (static_cast<std::size_t>(count) > maximum - result.size()) {
            throw std::runtime_error("UF worker message exceeds its size limit");
        }
        result.append(buffer.data(), static_cast<std::size_t>(count));
    }
}

std::filesystem::path job_channel(const EncodeOptions& job) {
    return job.remote ? job.remote->manifest.parent_path() : std::filesystem::absolute(job.output_root);
}

std::string job_key(const EncodeOptions& job) {
    return job.remote ? job.remote->manifest.string()
        : (std::filesystem::absolute(job.output_root) / (job.input_file.stem().string() + ".uf.json")).string();
}

}  // namespace

int run_encode_worker(int input_descriptor, int result_descriptor) {
    nlohmann::json result = {{"status", "failed"}, {"manifest", ""}, {"error", ""}};
    FileDescriptor events;
    std::unique_ptr<ProcessInterrupt> interruption;
    try {
        const auto document = parse_python_json(read_descriptor(input_descriptor, 256u * 1024u * 1024u));
        if (document.value("event_pipe", false)) {
            events.reset(8);
            set_event_descriptor(events.get());
        }
        const auto job = read_job(document);
        // The Python prepared worker installs a TERM handler before opening
        // inherited download descriptors, so its encode stack can unwind.
        if (job.streaming && job.streaming->stream >= 0)
            interruption = std::make_unique<ProcessInterrupt>();
        FileDescriptor stream(job.streaming ? job.streaming->stream : -1);
        FileDescriptor errors(job.streaming ? job.streaming->errors : -1);
        FileDescriptor state(job.streaming ? job.streaming->state : -1);
        FileDescriptor buffer(job.streaming ? job.streaming->buffer : -1);
        bool resumed = false;
        const auto manifest = encode_local(job, &resumed);
        result["status"] = resumed ? "resumed" : "encoded";
        result["manifest"] = python_path_text(manifest);
    } catch (const SharedWorkBusy& error) {
        result["status"] = "busy";
        result["error"] = error.what();
    } catch (const std::exception& error) {
        result["error"] = std::string(error.what()).substr(0, 8192);
    } catch (...) {
        result["error"] = "Unknown UF archive worker error";
    }
    set_event_descriptor(-1);
    const auto message = python_compact_json(result, true) + "\n";
    write_descriptor(result_descriptor, message.data(), message.size());
    return result.at("status") == "failed" ? 1 : 0;
}

struct EncodeScheduler::Impl {
    struct Active {
        EncodeOptions job;
        std::string device;
        FileDescriptor input, result;
        FileDescriptor events;
        std::string pending_events;
        bool terminal_event{};
        std::shared_ptr<PreparedDownload> prepared;
        std::shared_ptr<SharedStreamState> near;
        std::unique_ptr<ChildProcess> process;
    };
    std::deque<std::string> free;
    std::shared_ptr<SharedStreamState> buffer;
    std::list<Active> active;
    EncodeRun run;
    bool finished{};

    Impl(int workers, const std::vector<std::string>& devices, bool streaming) {
        if (workers < 1 || devices.empty()) throw std::invalid_argument("UF workers require a positive count and devices");
        if (streaming) buffer = std::make_shared<SharedStreamState>();
        for (int index = 0; index < workers; ++index) free.push_back(devices[static_cast<std::size_t>(index) % devices.size()]);
    }

    ~Impl() {
        for (auto& worker : active) {
            if (!worker.process) continue;
            worker.process->signal(SIGTERM);
            if (worker.prepared) try { worker.prepared->close(); } catch (...) {}
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
            for (;;) {
                // Consume queued events during teardown so workers can finish
                // cleanup without waiting for space in their event pipe.
                if (worker.events.get() >= 0) {
                    std::array<char, 8192> bytes{};
                    for (int i = 0; i < 16; ++i) {
                        const auto count = read(worker.events.get(), bytes.data(), bytes.size());
                        if (count < 0 && errno == EINTR) continue;
                        if (count <= 0) break;
                    }
                }
                try { if (worker.process->poll_for_cleanup()) break; }
                catch (...) { worker.process->signal(SIGKILL); break; }
                if (std::chrono::steady_clock::now() >= deadline) {
                    worker.process->signal(SIGKILL);
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
            const int pid = worker.process->pid();
            // Reap before releasing calibration ownership or inherited state.
            worker.process.reset();
            // Destructor teardown keeps cleaning remaining children if a
            // damaged shared mutex rejects the ownership release. Normal
            // collection below uses the checked release path.
            if (buffer) try { buffer->release(pid); } catch (...) {}
        }
        active.clear();
    }

    void launch(const EncodeOptions& settings, std::shared_ptr<PreparedDownload> prepared) {
        ProcessInterrupt::check();
        Active worker;
        worker.job = settings;
        worker.job.streaming.reset();
        if (prepared && (!settings.remote || !buffer)) {
            throw std::invalid_argument("UF prepared downloads require a streaming scheduler");
        }
        if (buffer && settings.remote) {
            worker.prepared = std::move(prepared);
            worker.near = worker.prepared ? worker.prepared->state() : std::make_shared<SharedStreamState>();
            if (!worker.near) throw std::invalid_argument("UF prepared download is already closed");
            worker.job.streaming = StreamInputs{worker.prepared ? 4 : -1, worker.prepared ? 5 : -1, 6, 7};
        }
        worker.device = free.front();
        worker.job.device = worker.device;
        worker.input = anonymous_file("/dev/shm/uf-worker-job-");
        worker.result = anonymous_file("/dev/shm/uf-worker-result-");
        auto events = process_pipe();
        const int flags = fcntl(events[0].get(), F_GETFL);
        if (flags < 0 || fcntl(events[0].get(), F_SETFL, flags | O_NONBLOCK) != 0) {
            throw std::runtime_error("Cannot set UF event pipe to nonblocking");
        }
        const auto message = python_compact_json(job_json(worker.job));
        write_descriptor(worker.input.get(), message.data(), message.size());
        if (lseek(worker.input.get(), 0, SEEK_SET) < 0) throw std::runtime_error("Cannot rewind UF worker input");
        std::vector<std::pair<int, int>> descriptors{{worker.input.get(), 0}, {worker.result.get(), 3}};
        descriptors.emplace_back(events[1].get(), 8);
        if (worker.near) {
            descriptors.emplace_back(worker.near->descriptor(), 6);
            descriptors.emplace_back(buffer->descriptor(), 7);
        }
        if (worker.prepared) {
            descriptors.emplace_back(worker.prepared->input_descriptor(), 4);
            descriptors.emplace_back(worker.prepared->error_descriptor(), 5);
        }
        worker.process = std::make_unique<ChildProcess>(
            std::vector<std::string>{"/proc/self/exe", "_encode_worker"},
            descriptors, true);
        worker.events = std::move(events[0]);
        active.push_back(std::move(worker));
        free.pop_front();
    }

    void read_events(Active& worker) {
        std::array<char, 8192> bytes{};
        for (;;) {
            const auto count = read(worker.events.get(), bytes.data(), bytes.size());
            if (count < 0 && errno == EINTR) continue;
            if (count < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) break;
            if (count < 0) throw std::runtime_error("Cannot read UF worker events");
            if (!count) break;
            worker.pending_events.append(bytes.data(), static_cast<std::size_t>(count));
            std::size_t start = 0;
            for (;;) {
                const auto end = worker.pending_events.find('\n', start);
                if (end == std::string::npos) break;
                const auto record = parse_python_json(worker.pending_events.substr(start, end - start));
                const auto name = record.value("event", std::string{});
                if (record.value("job_key", std::string{}) == python_filesystem_decode(job_key(worker.job)) &&
                    (name == "encode_completed" || name == "already_encoded" || name == "encode_failed" || name == "claimed_by_peer")) {
                    worker.terminal_event = true;
                }
                deliver_event(record);
                start = end + 1;
            }
            worker.pending_events.erase(0, start);
            if (worker.pending_events.size() > 2u * 1024u * 1024u) throw std::runtime_error("UF worker event exceeds 2 MiB");
        }
    }

    void poll() {
        ProcessInterrupt::check();
        for (auto worker = active.begin(); worker != active.end();) {
            read_events(*worker);
            const auto status = worker->process->poll();
            if (!status) { ++worker; continue; }
            read_events(*worker);
            EncodeOutcome outcome;
            outcome.source = worker->job.remote ? worker->job.remote->url : worker->job.input_file.string();
            outcome.status = "failed";
            try {
                if (lseek(worker->result.get(), 0, SEEK_SET) < 0) throw std::runtime_error("Cannot rewind UF worker result");
                const auto result = parse_python_json(read_descriptor(worker->result.get(), 1024u * 1024u));
                outcome.status = result.at("status").get<std::string>();
                outcome.error = result.at("error").get<std::string>();
                outcome.manifest = python_path_from_text(result.at("manifest").get_ref<const std::string&>());
                if ((outcome.status == "encoded" || outcome.status == "resumed") && outcome.manifest.empty()) {
                    throw std::runtime_error("UF worker omitted its archive path");
                }
                if ((outcome.status != "encoded" && outcome.status != "resumed" &&
                     outcome.status != "busy" && outcome.status != "failed") ||
                    (*status && outcome.status != "failed")) {
                    throw std::runtime_error("UF worker returned an inconsistent result");
                }
            } catch (const std::exception& error) {
                outcome.status = "failed";
                outcome.error = std::string(error.what()) + " (worker exit " + std::to_string(*status) + ")";
            }
            if (outcome.status == "encoded" || outcome.status == "resumed") {
                if (outcome.status == "encoded") ++run.encoded; else ++run.resumed;
                run.manifests.push_back(outcome.manifest);
            } else {
                if (outcome.status == "busy") ++run.busy; else ++run.failed;
                run.failed_channels.insert(job_channel(worker->job));
            }
            if (!worker->terminal_event) {
                event(outcome.status == "busy" ? "claimed_by_peer" : outcome.status == "failed" ? "encode_failed"
                    : outcome.status == "resumed" ? "already_encoded" : "encode_completed",
                    {{"source", worker->job.remote ? outcome.source : python_filesystem_decode(outcome.source)},
                     {"archive", python_path_text(outcome.manifest)},
                     {"job_key", python_filesystem_decode(job_key(worker->job))}, {"error", outcome.error}});
            }
            if (*status) {
                event("worker_failed", {{"source", worker->job.remote ? outcome.source : python_filesystem_decode(outcome.source)},
                    {"job_key", python_filesystem_decode(job_key(worker->job))},
                    {"exitcode", worker->process->returncode().value_or(*status)}, {"error", outcome.error}});
            }
            run.outcomes.push_back(std::move(outcome));
            if (buffer) buffer->release(worker->process->pid());
            if (worker->prepared) worker->prepared->close();
            free.push_back(worker->device);
            worker = active.erase(worker);
        }
    }
};

EncodeScheduler::EncodeScheduler(int workers, std::vector<std::string> devices, bool streaming)
    : impl_(std::make_unique<Impl>(workers, devices, streaming)) {}
EncodeScheduler::~EncodeScheduler() = default;
bool EncodeScheduler::has_capacity() const { return !impl_->free.empty(); }
std::size_t EncodeScheduler::running() const { return impl_->active.size(); }
void EncodeScheduler::poll() { impl_->poll(); }
bool EncodeScheduler::near_end() const {
    for (const auto& worker : impl_->active) if (worker.near && worker.near->near_end()) return true;
    return false;
}

void EncodeScheduler::submit(const EncodeOptions& job, std::shared_ptr<PreparedDownload> prepared) {
    if (impl_->finished) throw std::logic_error("UF archive scheduler is finished");
    while (!has_capacity()) {
        poll();
        if (!has_capacity()) std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    impl_->launch(job, std::move(prepared));
}

EncodeRun EncodeScheduler::finish() {
    if (impl_->finished) throw std::logic_error("UF archive scheduler is already finished");
    while (running()) {
        poll();
        if (running()) std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    impl_->finished = true;
    return std::move(impl_->run);
}

std::vector<std::filesystem::path> encode_folder(
    const std::filesystem::path& base_root, const std::filesystem::path& output_root,
    const std::vector<std::string>& channel_ids, bool recursive,
    const EncodeOptions& settings, EncodeRun* report) {
    EncodeScheduler scheduler(settings.workers,
        settings.devices.empty() ? std::vector<std::string>{settings.device} : settings.devices);
    const auto jobs = discover_local_jobs(base_root, output_root, channel_ids, recursive, settings);
    event("run_discovered", {{"total", jobs.size()}});
    for (const auto& job : jobs) {
        scheduler.submit(job);
    }
    auto run = scheduler.finish();
    auto manifests = run.manifests;
    if (report) *report = std::move(run);
    else if (run.failed) throw std::runtime_error("One or more UF archive workers failed");
    return manifests;
}

}  // namespace uf
