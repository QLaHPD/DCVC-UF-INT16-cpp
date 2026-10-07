#include "uf/workers.hpp"
#include "uf/events.hpp"
#include "uf/process.hpp"
#include "uf/shared_work.hpp"
#include "uf/python_json.hpp"
#include "uf/python_path.hpp"
#include "uf/utf8.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <functional>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <fcntl.h>
#include <unistd.h>

namespace uf {
namespace {

constexpr std::size_t maximum_message = 2u * 1024u * 1024u;
constexpr const char* image_protocol = "uf-cpp-image-pass-v1";

class Messages {
public:
    explicit Messages(int descriptor) : descriptor_(descriptor) {}
    std::optional<nlohmann::json> next() {
        std::string line;
        for (;;) {
            ProcessInterrupt::check();
            if (start_ == end_) {
                ssize_t count;
                do { count = read(descriptor_, buffer_.data(), buffer_.size()); }
                while (count < 0 && errno == EINTR);
                if (count < 0) throw std::runtime_error("Cannot read UF image worker message");
                if (!count) {
                    if (!line.empty()) throw std::runtime_error("Truncated UF image worker message");
                    return {};
                }
                start_ = 0;
                end_ = static_cast<std::size_t>(count);
            }
            const auto newline = std::find(buffer_.begin() + start_, buffer_.begin() + end_, '\n');
            const auto stop = static_cast<std::size_t>(newline - buffer_.begin());
            if (stop - start_ > maximum_message - line.size()) {
                throw std::runtime_error("UF image worker message exceeds 2 MiB");
            }
            line.append(buffer_.data() + start_, stop - start_);
            start_ = stop;
            if (newline != buffer_.begin() + end_) {
                ++start_;
                return parse_python_json(line);
            }
        }
    }
private:
    int descriptor_;
    std::array<char, 16384> buffer_{};
    std::size_t start_{}, end_{};
};

void write_message(int descriptor, const nlohmann::json& document, bool replace_invalid = false) {
    const auto line = python_compact_json(document, replace_invalid) + "\n";
    if (line.size() - 1 > maximum_message) throw std::runtime_error("UF image worker message exceeds 2 MiB");
    write_descriptor(descriptor, line.data(), line.size());
}

nlohmann::json job_document(const ImageEncodeOptions& job) {
    return {
        {"source", python_path_text(job.source)}, {"target", python_path_text(job.target)},
        {"prepared_image", python_path_text(job.prepared_image)}, {"checkpoint_image", python_path_text(job.checkpoint_image)},
        {"runtime", job.runtime}, {"device", job.device}, {"qp", job.qp},
        {"skip_threshold", job.skip_threshold}, {"shared_instance", job.shared_instance},
        {"shared_lease_seconds", job.shared_lease_seconds},
        {"source_base", job.source_base ? nlohmann::json(python_path_text(*job.source_base)) : nlohmann::json(nullptr)}
    };
}

ImageEncodeOptions read_job(const nlohmann::json& document) {
    ImageEncodeOptions job;
    job.source = python_path_from_text(document.at("source").get_ref<const std::string&>());
    job.target = python_path_from_text(document.at("target").get_ref<const std::string&>());
    job.prepared_image = python_path_from_text(document.at("prepared_image").get_ref<const std::string&>());
    job.checkpoint_image = python_path_from_text(document.at("checkpoint_image").get_ref<const std::string&>());
    job.runtime = document.at("runtime").get<std::string>();
    job.device = document.at("device").get<std::string>();
    job.qp = document.at("qp").get<int>();
    job.skip_threshold = document.at("skip_threshold").get<double>();
    job.shared_instance = document.at("shared_instance").get<std::string>();
    job.shared_lease_seconds = document.at("shared_lease_seconds").get<double>();
    if (!document.at("source_base").is_null()) job.source_base = python_path_from_text(document.at("source_base").get_ref<const std::string&>());
    return job;
}

bool known_status(const std::string& status) {
    return status == "encoded" || status == "resumed" || status == "busy" || status == "failed";
}

bool terminal_event(const std::string& name) {
    return name == "encode_completed" || name == "already_encoded" ||
           name == "encode_failed" || name == "claimed_by_peer";
}

void drain_events(int descriptor, std::string& pending, std::set<std::string>& terminal) {
    std::array<char, 8192> bytes{};
    for (;;) {
        ProcessInterrupt::check();
        const auto count = read(descriptor, bytes.data(), bytes.size());
        if (count < 0 && errno == EINTR) continue;
        if (count < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) return;
        if (count < 0) throw std::runtime_error("Cannot read UF image worker events");
        if (!count) {
            if (!pending.empty()) throw std::runtime_error("Truncated UF image worker event");
            return;
        }
        pending.append(bytes.data(), static_cast<std::size_t>(count));
        std::size_t start = 0;
        for (;;) {
            const auto end = pending.find('\n', start);
            if (end == std::string::npos) break;
            if (end - start > maximum_message) throw std::runtime_error("UF image worker event exceeds 2 MiB");
            const auto record = parse_python_json(pending.substr(start, end - start));
            if (terminal_event(record.value("event", std::string{}))) {
                terminal.insert(python_filesystem_encode(record.value("job_key", std::string{})));
            }
            deliver_event(record);
            start = end + 1;
        }
        pending.erase(0, start);
        if (pending.size() > maximum_message) throw std::runtime_error("UF image worker event exceeds 2 MiB");
    }
}

ImageEncodeOptions normalized(ImageEncodeOptions job) {
    const auto absolute = [](const std::filesystem::path& path) {
        return std::filesystem::absolute(path).lexically_normal();
    };
    job.source = absolute(job.source);
    job.target = absolute(job.target);
    if (!job.prepared_image.empty()) job.prepared_image = absolute(job.prepared_image);
    if (!job.checkpoint_image.empty()) job.checkpoint_image = absolute(job.checkpoint_image);
    if (job.source_base) job.source_base = absolute(*job.source_base);
    return job;
}

}  // namespace

int run_image_worker(int input_descriptor, int result_descriptor) {
    FileDescriptor events;
    std::uint64_t count = 0;
    bool complete = false;
    std::string fatal_error;
    try {
        Messages input(input_descriptor);
        const auto header = input.next();
        if (!header || header->at("protocol") != image_protocol) {
            throw std::invalid_argument("Unsupported UF image worker protocol");
        }
        count = header->at("jobs").get<std::uint64_t>();
        if (header->value("event_pipe", false)) {
            events.reset(8);
            set_event_descriptor(events.get());
        }
        write_message(result_descriptor, {{"protocol", image_protocol}, {"kind", "begin"}, {"jobs", count}});
        ImageEncoder encoder;
        for (std::uint64_t index = 0; index < count; ++index) {
            ProcessInterrupt::check();
            const auto document = input.next();
            if (!document) throw std::runtime_error("Image worker input omitted a job");
            const auto job = read_job(*document);
            nlohmann::json result = {
                {"kind", "outcome"}, {"index", index}, {"status", "failed"}, {"archive", ""}, {"error", ""}
            };
            try {
                bool resumed = false;
                const auto archive = encoder.encode(job, &resumed);
                result["status"] = resumed ? "resumed" : "encoded";
                result["archive"] = python_path_text(archive);
            } catch (const SharedWorkBusy& error) {
                result["status"] = "busy";
                result["error"] = std::string(error.what()).substr(0, 8192);
            } catch (const std::exception& error) {
                ProcessInterrupt::check();
                result["error"] = std::string(error.what()).substr(0, 8192);
            }
            // Per-job records preserve completed outcomes if the worker later
            // dies and avoid a whole-batch JSON document/size limit.
            write_message(result_descriptor, result, true);
        }
        if (input.next()) throw std::runtime_error("Unexpected trailing image worker input");
        complete = true;
    } catch (const std::exception& error) {
        fatal_error = std::string(error.what()).substr(0, 8192);
    } catch (...) {
        fatal_error = "Unknown UF image pass worker error";
    }
    set_event_descriptor(-1);
    events.reset();
    write_message(result_descriptor, {
        {"kind", "end"}, {"jobs", count}, {"complete", complete}, {"error", fatal_error}
    }, true);
    // Ordinary image failures/busy claims are recorded and the pass continues,
    // just as encode_images returns its failed-channel set without crashing.
    return complete ? 0 : 1;
}

ImageRun encode_image_pass(const std::vector<ImageEncodeOptions>& settings) {
    ImageRun run;
    if (settings.empty()) return run;
    ProcessInterrupt::check();
    std::vector<ImageEncodeOptions> jobs;
    jobs.reserve(settings.size());
    for (const auto& job : settings) jobs.push_back(normalized(job));

    FileDescriptor input = anonymous_file("/dev/shm/uf-image-worker-job-");
    FileDescriptor result = anonymous_file("/dev/shm/uf-image-worker-result-");
    auto events = process_pipe();
    const int flags = fcntl(events[0].get(), F_GETFL);
    if (flags < 0 || fcntl(events[0].get(), F_SETFL, flags | O_NONBLOCK) != 0) {
        throw std::runtime_error("Cannot set UF image event pipe to nonblocking");
    }
    write_message(input.get(), {{"protocol", image_protocol}, {"event_pipe", true}, {"jobs", jobs.size()}});
    for (const auto& job : jobs) {
        ProcessInterrupt::check();
        write_message(input.get(), job_document(job));
    }
    if (lseek(input.get(), 0, SEEK_SET) < 0) throw std::runtime_error("Cannot rewind UF image worker input");
    // Declared last: interruption destroys/kills the process group before any
    // of its parent IPC resources are closed.
    ChildProcess process({"/proc/self/exe", "_image_worker"},
        {{input.get(), 0}, {result.get(), 3}, {events[1].get(), 8}}, true);
    events[1].reset();
    std::set<std::string> terminal;
    std::string pending, worker_error;
    int exitcode = 1;
    try {
        for (;;) {
            drain_events(events[0].get(), pending, terminal);
            if (const auto status = process.poll()) {
                exitcode = *status;
                drain_events(events[0].get(), pending, terminal);
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    } catch (const std::exception& error) {
        ProcessInterrupt::check();
        worker_error = error.what();
        process.signal(SIGKILL);
        exitcode = process.wait();
    }

    run.outcomes.resize(jobs.size());
    for (std::size_t index = 0; index < jobs.size(); ++index) {
        run.outcomes[index].source = jobs[index].source;
        run.outcomes[index].status = "failed";
        run.outcomes[index].error = "Image worker stopped before reporting this job";
    }
    bool finished = false;
    try {
        if (lseek(result.get(), 0, SEEK_SET) < 0) throw std::runtime_error("Cannot rewind UF image worker result");
        Messages output(result.get());
        const auto header = output.next();
        if (!header || header->at("protocol") != image_protocol || header->at("kind") != "begin" ||
            header->at("jobs").get<std::uint64_t>() != jobs.size()) {
            throw std::runtime_error("Invalid UF image worker result header");
        }
        std::size_t expected_index = 0;
        for (;;) {
            const auto document = output.next();
            if (!document) throw std::runtime_error("Missing UF image worker result footer");
            if (document->at("kind") == "end") {
                finished = document->at("complete").get<bool>();
                if (document->at("jobs").get<std::uint64_t>() != jobs.size() ||
                    (finished && expected_index != jobs.size()) || output.next()) {
                    throw std::runtime_error("Inconsistent UF image worker result footer");
                }
                if (!finished) worker_error = document->value("error", std::string("Image worker did not finish"));
                break;
            }
            if (document->at("kind") != "outcome" || expected_index >= jobs.size() ||
                document->at("index").get<std::uint64_t>() != expected_index) {
                throw std::runtime_error("Inconsistent UF image worker result index");
            }
            ImageOutcome outcome;
            outcome.source = jobs[expected_index].source;
            outcome.status = document->at("status").get<std::string>();
            outcome.archive = python_path_from_text(document->at("archive").get_ref<const std::string&>());
            outcome.error = document->at("error").get<std::string>();
            const bool success = outcome.status == "encoded" || outcome.status == "resumed";
            if (!known_status(outcome.status) || (success && outcome.archive != jobs[expected_index].target) ||
                (!success && !outcome.archive.empty())) {
                throw std::runtime_error("Invalid UF image worker outcome");
            }
            run.outcomes[expected_index++] = std::move(outcome);
        }
    } catch (const std::exception& error) {
        ProcessInterrupt::check();
        finished = false;
        worker_error = std::string(error.what()) + " (worker exit " + std::to_string(exitcode) + ")";
    }
    run.worker_failed = exitcode != 0 || !finished || !worker_error.empty();
    if (run.worker_failed && worker_error.empty()) worker_error = "Image worker exited with code " + std::to_string(exitcode);
    for (std::size_t index = 0; index < jobs.size(); ++index) {
        auto& outcome = run.outcomes[index];
        if (outcome.status == "encoded") ++run.encoded;
        else if (outcome.status == "resumed") ++run.resumed;
        else if (outcome.status == "busy") ++run.busy;
        else {
            ++run.failed;
            if (run.worker_failed && outcome.error == "Image worker stopped before reporting this job") {
                outcome.error = worker_error;
            }
        }
        if (run.worker_failed || outcome.status == "failed" || outcome.status == "busy") {
            run.failed_channels.insert(jobs[index].target.parent_path());
        }
        const auto key = jobs[index].target.string();
        if (!terminal.count(key)) {
            event(outcome.status == "busy" ? "claimed_by_peer" : outcome.status == "failed" ? "encode_failed"
                : outcome.status == "resumed" ? "already_encoded" : "encode_completed",
                {{"source", python_path_text(outcome.source)}, {"archive", python_path_text(jobs[index].target)},
                 {"job_key", python_filesystem_decode(key)}, {"error", outcome.error}});
        }
    }
    if (run.worker_failed) {
        event("worker_failed", {{"job_key", "image-pass"}, {"jobs", jobs.size()},
              {"exitcode", exitcode}, {"error", worker_error}});
    }
    event("image_run_completed", {{"encoded", run.encoded}, {"resumed", run.resumed},
        {"busy", run.busy}, {"failed", run.failed}, {"worker_failed", run.worker_failed}});
    return run;
}

}  // namespace uf
