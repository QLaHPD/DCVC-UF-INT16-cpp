#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "uf/workflow.hpp"
#include "uf/archive.hpp"
#include "uf/image_workflow.hpp"
#include "uf/integer_codec.hpp"
#include "uf/codec.hpp"
#include "uf/frame_buffer.hpp"
#include "uf/media.hpp"
#include "uf/prepared.hpp"
#include "uf/shared_work.hpp"
#include "uf/stream_pipeline.hpp"
#include "uf/events.hpp"
#include "uf/json_compat.hpp"
#include "uf/process.hpp"
#include "uf/runtime_environment.hpp"
#include "uf/python_path.hpp"
#include "uf/python_json.hpp"
#include "uf/utf8.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <tuple>
#include <vector>
#include <linux/limits.h>
#include <sys/stat.h>
#include <sys/xattr.h>
#include <fcntl.h>
#include <unistd.h>

namespace uf {
namespace {

class DirectoryStage {
public:
    DirectoryStage(const std::filesystem::path& parent, const std::string& prefix) {
        if (parent == "/dev/shm") {
            (void)memory_stage_root();
        }
        path_ = python_temporary_mkdtemp(prefix, {}, parent);
    }
    ~DirectoryStage() {
        if (!retained_ && !path_.empty()) {
            try { python_path_rmtree(path_, true, false); } catch (...) {}
        }
    }
    const std::filesystem::path& path() const { return path_; }
    void retain() { retained_ = true; }
    void remove() {
        if (path_.empty()) return;
        python_path_rmtree(path_);
        path_.clear();
    }

private:
    std::filesystem::path path_;
    bool retained_{};
};

void close_publication_file(FileDescriptor& file, const std::filesystem::path& path) {
    const int owned = file.release();
    if (close(owned) != 0) {
        const int error = errno;
        throw std::system_error(error, std::generic_category(),
                                "Cannot close UF publication file: " + path.string());
    }
}

void fsync_path(const std::filesystem::path& input, bool directory = false) {
    const auto path = python_path_normalize(input);
    if (path.native().find('\0') != std::string::npos)
        throw std::invalid_argument("embedded null byte");
    int descriptor;
    for (;;) {
        ProcessInterrupt::check();
        descriptor = open(path.c_str(), O_RDONLY | O_CLOEXEC | (directory ? O_DIRECTORY : 0));
        if (descriptor >= 0) break;
        const int error = errno;
        if (error == EINTR) continue;
        throw std::system_error(error, std::generic_category(),
                                "Cannot open UF output for sync: " + path.string());
    }
    FileDescriptor file(descriptor);
    try {
        for (;;) {
            ProcessInterrupt::check();
            if (fsync(file.get()) == 0) break;
            const int error = errno;
            if (error == EINTR) continue;
            throw std::system_error(error, std::generic_category(),
                                    "Cannot sync UF output: " + path.string());
        }
        close_publication_file(file, path);
    } catch (...) {
        if (file.get() >= 0) close_publication_file(file, path);
        throw;
    }
}

void write_journal(const std::filesystem::path& input, const nlohmann::json& data) {
    write_python_json_file_atomic(input, data);
}

void copy_completed_artifact(const std::filesystem::path& source,
                             const std::filesystem::path& target,
                             SharedLease& lease) {
    copy_completed_archive_file(source, target, [&] { lease.check(); });
}

nlohmann::json media_json(const MediaInfo& info) {
    return {{"width", info.width}, {"height", info.height}, {"fps", info.fps},
            {"duration", info.duration}, {"audio", info.audio}};
}

nlohmann::json pipeline_json(const EncodeOptions& options,
                             const CodecModels& models) {
    nlohmann::json result = {
        {"codec", "dcvc-uf"}, {"archive_version", 1}, {"runtime", options.runtime},
        {"variant", options.variant}, {"image_sha256", models.image_sha256},
        {"video_sha256", models.video_sha256.value()}, {"qp_i", options.qp_i},
        {"qp_p", options.qp_p},
        {"fps", options.fps ? nlohmann::json(*options.fps) : nlohmann::json(nullptr)},
        {"resolution", options.resolution ? nlohmann::json(*options.resolution) : nlohmann::json(nullptr)},
        {"reset_interval", options.reset_interval}, {"intra_period", options.intra_period},
        {"max_frames", options.max_frames ? nlohmann::json(*options.max_frames) : nlohmann::json(nullptr)},
        {"audio", options.audio}, {"opus_channels", options.opus_channels},
        {"opus_bitrate", options.opus_bitrate}
    };
    if (models.integer_image && models.integer_video) {
        result["integer"] = {{"arithmetic", models.integer_image->arithmetic},
                             {"image", models.integer_image->identity},
                             {"video", models.integer_video->identity}};
    }
    if (options.opus_frame_ms != 20) result["opus_frame_ms"] = options.opus_frame_ms;
    if (options.opus_complexity != 10) result["opus_complexity"] = options.opus_complexity;
    if (options.opus_vbr != "on") result["opus_vbr"] = options.opus_vbr;
    if (options.skip_threshold != 0) result["skip_threshold"] = options.skip_threshold;
    return result;
}

// Adapt Linux shutil.copy2/copystat/_copyxattr for these file destinations.
// See licenses/CPython-3.10.20.txt. Copy bytes before collecting source metadata,
// and copy xattrs before chmod can make the destination read-only.
void copy_preserving_metadata(const std::filesystem::path& source,
                              const std::filesystem::path& destination) {
    const auto fail = [&](const char* operation) {
        const auto error = errno;
        throw std::filesystem::filesystem_error(operation, source, destination,
                                                std::error_code(error, std::generic_category()));
    };
    struct stat original{}, target{};
    const bool source_exists = stat(source.c_str(), &original) == 0;
    const bool target_exists = stat(destination.c_str(), &target) == 0;
    if (source_exists && target_exists && original.st_dev == target.st_dev && original.st_ino == target.st_ino)
        throw std::invalid_argument("Source and destination are the same file");
    if ((source_exists && S_ISFIFO(original.st_mode)) || (target_exists && S_ISFIFO(target.st_mode)))
        throw std::invalid_argument("Cannot copy a named pipe");
    {
        FileDescriptor input(open(source.c_str(), O_RDONLY | O_CLOEXEC));
        if (input.get() < 0) fail("Cannot open copy source");
        FileDescriptor output(open(destination.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0666));
        if (output.get() < 0) fail("Cannot open copy destination");
        std::array<char, 64 * 1024> buffer{};
        for (;;) {
            ProcessInterrupt::check();
            const auto count = read(input.get(), buffer.data(), buffer.size());
            if (count < 0 && errno == EINTR) continue;
            if (count < 0) fail("Cannot read copy source");
            if (!count) break;
            write_descriptor(output.get(), buffer.data(), static_cast<std::size_t>(count));
        }
    }
    if (stat(source.c_str(), &original) != 0) fail("Cannot stat copy source");
    const timespec times[] = {original.st_atim, original.st_mtim};
    if (utimensat(AT_FDCWD, destination.c_str(), times, 0) != 0) fail("Cannot copy timestamps");

    const auto ignored_list_error = [](int error) {
        return error == ENOTSUP || error == ENODATA || error == EINVAL;
    };
    const auto ignored_attribute_error = [&](int error) {
        return error == EPERM || ignored_list_error(error);
    };
    std::vector<char> names;
    for (const std::size_t capacity : {std::size_t{256}, std::size_t{XATTR_LIST_MAX}}) {
        ProcessInterrupt::check();
        names.resize(capacity);
        const auto count = listxattr(source.c_str(), names.data(), names.size());
        if (count < 0 && errno == ERANGE && capacity != XATTR_LIST_MAX) continue;
        if (count < 0) {
            names.clear();
            if (ignored_list_error(errno)) break;
            fail("Cannot list source attributes");
        }
        names.resize(static_cast<std::size_t>(count));
        break;
    }
    for (auto first = names.begin(); first != names.end();) {
        ProcessInterrupt::check();
        const auto end = std::find(first, names.end(), '\0');
        if (end == names.end()) throw std::runtime_error("Malformed source attribute list");
        const std::string name(first, end);
        first = end + 1;
        for (const std::size_t capacity : {std::size_t{128}, std::size_t{XATTR_SIZE_MAX}}) {
            ProcessInterrupt::check();
            std::vector<char> value(capacity);
            const auto count = getxattr(source.c_str(), name.c_str(), value.data(), value.size());
            if (count < 0 && errno == ERANGE && capacity != XATTR_SIZE_MAX) continue;
            if (count < 0) {
                if (ignored_attribute_error(errno)) break;
                fail("Cannot read source attribute");
            }
            if (setxattr(destination.c_str(), name.c_str(), value.data(), static_cast<std::size_t>(count), 0) != 0 &&
                !ignored_attribute_error(errno)) fail("Cannot copy source attribute");
            break;
        }
    }
    if (chmod(destination.c_str(), original.st_mode & 07777) != 0) fail("Cannot copy permissions");
}

void copy_sidecars(const std::filesystem::path& source, const std::filesystem::path& stage) {
    for (const char* suffix : {".info.json", ".description", ".jpg", ".jpeg", ".png", ".webp"}) {
        auto sidecar = source;
        sidecar.replace_extension(suffix);
        if (std::filesystem::is_regular_file(sidecar) && !std::filesystem::is_symlink(sidecar)) {
            copy_preserving_metadata(sidecar, stage / sidecar.filename());
        }
    }
}

void link_or_check(const std::filesystem::path& source,
                   const std::filesystem::path& target,
                   const nlohmann::json& expected) {
    try {
        std::filesystem::create_hard_link(source, target);
    } catch (const std::filesystem::filesystem_error& error) {
        if (error.code() != std::errc::file_exists) throw;
        if (python_path_is_symlink(target) || !python_path_is_file(target) ||
            !json_python_equal(nlohmann::json(python_path_size(target)), expected.at("size")) ||
            !json_python_equal(nlohmann::json(sha256_file(target)), expected.at("sha256"))) {
            throw std::system_error(std::make_error_code(std::errc::file_exists),
                "Refusing to overwrite different output: " + target.string());
        }
    }
}

std::string publication_field_text(const nlohmann::json& value) {
    return value.is_string() ? value.get<std::string>() : python_json_repr(value);
}

void publish_flat(const std::filesystem::path& stage,
                  const std::filesystem::path& manifest,
                  nlohmann::json& metadata,
                  SharedLease* lease,
                  const std::string& lease_key) {
    auto stem = manifest.filename().string();
    if (stem.size() >= 8 && stem.compare(stem.size() - 8, 8, ".uf.json") == 0)
        stem.resize(stem.size() - 8);
    metadata["layout"] = "flat";
    metadata["archive_stem"] = python_filesystem_decode(stem);
    metadata["lease_key"] = python_filesystem_decode(lease_key);
    const auto& artifacts = metadata.at("artifacts");
    if (!artifacts.is_object()) json_compat_detail::missing_attribute(artifacts, "items");
    const bool rt = python_json_dict_get(metadata, "filename_style") == "rt";
    nlohmann::json names = nlohmann::json::object();
    std::set<std::string> distinct;
    for (auto it = artifacts.begin(); it != artifacts.end(); ++it) {
        std::string actual;
        if (it.key() == "video.bin") {
            if (rt) {
                const auto& video = metadata.at("video");
                const auto& config = metadata.at("pipeline");
                actual = stem + "_" + publication_field_text(video.at("width")) + "x" +
                    publication_field_text(video.at("height")) + "_qI" +
                    publication_field_text(config.at("qp_i")) + "_qP" +
                    publication_field_text(config.at("qp_p")) + ".bin";
            } else actual = stem + ".bin";
        } else if (it.key() == "audio.opus") actual = stem + ".opus";
        else if (it.key() == "source.info.json") actual = stem + (rt ? ".info.json" : ".uf.info.json");
        else actual = python_filesystem_encode(it.key());
        names[it.key()] = python_filesystem_decode(actual);
        distinct.insert(actual);
    }
    if (distinct.size() != names.size())
        throw std::invalid_argument("Sidecar names collide with encoded artifacts");
    metadata["files"] = names;
    const auto pending = manifest.parent_path() / ("." + manifest.filename().string() + ".pending");
    ProcessInterrupt::check();
    if (lease) lease->check();
    write_journal(pending, {{"stage", python_path_text(stage.filename())}, {"metadata", metadata}});
    const Archive publication{manifest, manifest.parent_path(), metadata};
    for (auto it = names.begin(); it != names.end(); ++it) {
        ProcessInterrupt::check();
        if (lease) lease->check();
        const auto target = artifact_path(publication, it.key());
        link_or_check(stage / python_path_from_text(it.key()), target, artifacts.at(it.key()));
    }
    fsync_path(manifest.parent_path(), true);
    const auto staged_manifest = stage / "flat-manifest.json";
    write_journal(staged_manifest, metadata);
    ProcessInterrupt::check();
    if (lease) lease->check();
    std::filesystem::create_hard_link(staged_manifest, manifest);
    fsync_path(manifest.parent_path(), true);
    // unlink(missing_ok=True), then sync the directory that recorded removal.
    if (unlink(pending.c_str()) != 0) {
        const int error = errno;
        if (error != ENOENT)
            throw std::system_error(error, std::generic_category(), "Cannot remove UF publication journal");
    }
    fsync_path(manifest.parent_path(), true);
}

std::optional<nlohmann::json> recover_flat(const std::filesystem::path& manifest,
                  const std::filesystem::path& pending,
                  const nlohmann::json& pipeline,
                  const std::filesystem::path& source,
                  const std::string& source_sha,
                  const std::optional<std::string>& remote_url,
                  const std::string& lease_key,
                  SharedLease* lease) {
    if (!python_path_exists(pending)) return std::nullopt;
    if (python_path_is_symlink(pending))
        throw std::invalid_argument("Unsafe publication journal");
    const auto transaction = read_python_json_file(
        pending, python_process_text_encoding(), true);
    auto metadata = transaction.at("metadata");
    if (!json_python_equal(metadata.at("pipeline"), pipeline))
        throw std::invalid_argument("Pending publication has different encoding settings");
    const auto& previous = metadata.at("source");
    const bool remote = remote_url && !remote_url->empty();
    if (remote ? !json_python_equal(python_json_dict_get(previous, "url"), nlohmann::json(*remote_url))
               : !json_python_equal(previous.at("path"), nlohmann::json(python_path_text(source))) ||
                 !json_python_equal(previous.at("sha256"), nlohmann::json(source_sha)))
        throw std::invalid_argument("Source differs from pending publication");
    const auto name = python_filesystem_encode(transaction.at("stage").get_ref<const std::string&>());
    const auto prefix = "." + manifest.filename().string() + ".stage-";
    if (name == "." || name.find('/') != std::string::npos || name.rfind(prefix, 0) != 0)
        throw std::invalid_argument("Unsafe pending stage path");
    const auto stage = manifest.parent_path() / name;
    if (python_path_is_symlink(stage) || !python_path_is_directory(stage))
        throw std::invalid_argument("Pending stage is missing; refusing to replace published payloads");
    const auto& artifacts = metadata.at("artifacts");
    if (!artifacts.is_object()) json_compat_detail::missing_attribute(artifacts, "items");
    for (auto it = artifacts.begin(); it != artifacts.end(); ++it) {
        ProcessInterrupt::check();
        if (it.key() == "." || it.key().find('/') != std::string::npos)
            throw std::invalid_argument("Unsafe staged artifact");
        const auto file = stage / python_path_from_text(it.key());
        if (python_path_is_symlink(file) || !python_path_is_file(file) ||
            !json_python_equal(nlohmann::json(python_path_size(file)), it.value().at("size")) ||
            !json_python_equal(nlohmann::json(sha256_file(file)), it.value().at("sha256")))
            throw std::invalid_argument("Pending staged artifact has changed");
    }
    publish_flat(stage, manifest, metadata, lease, lease_key);
    python_path_rmtree(stage);
    return metadata;
}

std::vector<std::string> download_command(
    const RemoteJob& remote, const std::string& format_id,
    const std::optional<std::filesystem::path>& cookies) {
    auto command = ytdlp_command(remote.ytdlp_binary, cookies);
    command.insert(command.end(), {"--no-playlist", "-f", python_filesystem_encode(format_id),
                                    "-o", "-", "--", python_filesystem_encode(remote.url)});
    return command;
}

void finish_remote_thumbnail(const EncodeOptions& options) {
    if (!options.remote || !options.remote->write_thumbnail) return;
    try {
        const auto image = store_remote_thumbnail(*options.remote);
        if (!image || options.remote->thumbnail_codec != "dcvc-intra") return;
        ImageEncodeOptions settings;
        settings.source = *image;
        settings.target = image->parent_path() /
            (image->filename().string() + "_qI" +
             std::to_string(options.remote->thumbnail_qp) + ".dcvci");
        settings.prepared_image = options.prepared_i;
        settings.checkpoint_image = options.checkpoint_i;
        settings.runtime = options.runtime;
        settings.device = options.device;
        settings.qp = options.remote->thumbnail_qp;
        settings.skip_threshold = options.skip_threshold;
        settings.shared_instance = options.shared_instance;
        settings.shared_lease_seconds = options.shared_lease_seconds;
        encode_intra_image(settings);
    } catch (const std::exception& error) {
        event("thumbnail_failed", {{"source", python_path_text(options.remote->manifest)}, {"error", error.what()}});
    }
}

}  // namespace

nlohmann::json encode_pipeline(const EncodeOptions& options) {
    const auto models = CodecModels::load(options.runtime,
        options.runtime == "int16" ? options.prepared_i : options.checkpoint_i,
        options.runtime == "int16" ? options.prepared_p : options.checkpoint_p,
        options.variant);
    return pipeline_json(options, models);
}

static std::filesystem::path encode_local_impl(const EncodeOptions& options, bool* resumed) {
    if (resumed) *resumed = false;
    const auto& remote = options.remote;
    if ((options.variant != "hts" && options.variant != "htl" && options.variant != "ld") ||
        options.qp_i < 0 || options.qp_i >= 64 || options.qp_p < 0 || options.qp_p >= 64 ||
        options.input_threads < 1 || options.prefetch_frames < 0 ||
        (options.hwaccel != "none" && options.hwaccel != "auto" && options.hwaccel != "jetson") ||
        (options.resolution && *options.resolution <= 0) ||
        (options.fps && (!std::isfinite(*options.fps) || *options.fps <= 0)) ||
        (options.max_frames && *options.max_frames == 0) ||
        options.reset_interval < -1 || options.intra_period == 0 ||
        options.intra_period < -1 ||
        (options.variant != "ld" && options.intra_period > 1 && options.intra_period % 8)) {
        throw std::invalid_argument("Invalid UF local encode settings");
    }
    const auto source = remote ? std::filesystem::path(remote->video_id)
                               : python_path_absolute(options.input_file);
    if (!remote && (std::filesystem::is_symlink(source) ||
                    !std::filesystem::is_regular_file(source))) {
        throw std::invalid_argument("UF source must be an existing regular file");
    }
    if (remote && options.max_frames) {
        throw std::invalid_argument("Streaming requires full videos; omit --max_frames");
    }
    if (options.audio != "opus" && options.audio != "none") {
        throw std::invalid_argument("UF audio must be opus or none");
    }
    auto models = CodecModels::load(options.runtime,
        options.runtime == "int16" ? options.prepared_i : options.checkpoint_i,
        options.runtime == "int16" ? options.prepared_p : options.checkpoint_p,
        options.variant);
    const auto pipeline = pipeline_json(options, models);
    const auto output_root = remote ? remote->manifest.parent_path()
                                    : std::filesystem::absolute(options.output_root);
    std::filesystem::create_directories(output_root);
    const auto stem = remote ? remote->manifest.stem().stem().string()
                             : source.stem().string();
    if (stem.empty() || stem == "." || stem == "..") throw std::invalid_argument("Invalid source name");
    const auto manifest = remote ? remote->manifest : output_root / (stem + ".uf.json");
    const auto lease_key = remote ? remote->video_id + ".uf"
                                  : source.filename().string() + ".uf";
    SharedWorkPool shared(WorkKind::video, options.shared_lease_seconds,
                          options.shared_instance);
    shared.ensure_pipeline(output_root, pipeline);
    auto lease = shared.try_acquire(output_root, lease_key);
    if (!lease) {
        event("claimed_by_peer", {{"source", python_path_text(source)}, {"archive", python_path_text(manifest)}});
        throw SharedWorkBusy("UF archive is claimed by a Python or C++ peer");
    }
    return with_shared_lease(*lease, [&]() -> std::filesystem::path {
    try {
    const auto pending = output_root / ("." + manifest.filename().string() + ".pending");
    const auto legacy = output_root / ((remote ? remote->video_id : source.filename().string()) + ".uf");
    if (std::filesystem::exists(manifest) || std::filesystem::exists(legacy)) {
        const auto existing = load_archive(
            std::filesystem::exists(manifest) ? manifest : legacy, true, &pipeline);
        if (remote && !json_python_equal(python_json_dict_get(existing.metadata.at("source"), "url"),
                                        nlohmann::json(remote->url)))
            throw std::invalid_argument("Existing archive has a different source URL");
        if (!remote && !json_python_equal(existing.metadata.at("source").at("sha256"),
                                         nlohmann::json(sha256_file(source))))
            throw std::invalid_argument("Original changed since the existing archive was produced");
        if (existing.manifest == manifest && std::filesystem::exists(pending)) {
            if (std::filesystem::is_symlink(pending)) {
                throw std::runtime_error("Unsafe UF publication journal");
            }
            const auto transaction = read_python_json_file(
                pending, python_process_text_encoding(), true);
            if (!json_python_equal(transaction.at("metadata"), existing.metadata)) {
                throw std::runtime_error("Pending UF journal differs from committed archive");
            }
            const auto stage_name = python_filesystem_encode(transaction.at("stage").get_ref<const std::string&>());
            const auto prefix = "." + manifest.filename().string() + ".stage-";
            if (stage_name.rfind(prefix, 0) != 0 ||
                std::filesystem::path(stage_name).filename() != stage_name) {
                throw std::runtime_error("Unsafe UF pending stage path");
            }
            std::filesystem::remove(pending);
            fsync_path(output_root, true);
            const auto stage = output_root / stage_name;
            if (!std::filesystem::is_symlink(stage) && std::filesystem::is_directory(stage)) {
                python_path_rmtree(stage);
            }
        }
        finish_remote_thumbnail(options);
        if (resumed) *resumed = true;
        event("already_encoded", {{"source", python_path_text(source)}, {"archive", python_path_text(existing.manifest)}});
        return existing.manifest;
    }
    const auto before = remote ? PythonPathIdentity{} : python_path_identity(source);
    const auto source_sha = remote ? std::string{} : sha256_file(source);
    if (const auto recovered = recover_flat(manifest, pending, pipeline, source, source_sha,
                     remote ? std::optional<std::string>(remote->url) : std::nullopt,
                     lease_key, lease.get())) {
        finish_remote_thumbnail(options);
        auto fields = recovered->at("video");
        fields["source"] = python_path_text(source);
        fields["archive"] = python_path_text(manifest);
        fields["recovered"] = true;
        fields["video_bytes"] = recovered->at("artifacts").at("video.bin").at("size");
        event("encode_completed", fields);
        return manifest;
    }
    const auto original = remote ? remote->media : probe_media(source);
    const auto [width, height] = scaled_dimensions(original.width, original.height,
        options.resolution ? std::optional<std::uint32_t>(*options.resolution) : std::nullopt);
    const double fps = options.fps.value_or(original.fps);
    if (!std::isfinite(fps) || fps <= 0) throw std::invalid_argument("UF FPS must be positive");
    event("encode_started", {{"source", python_path_text(source)}, {"device", options.device},
        {"input_threads", options.input_threads}, {"prefetch_frames", options.prefetch_frames},
        {"runtime", options.runtime}, {"variant", options.variant}, {"qi", options.qp_i}, {"qp", options.qp_p},
        {"fps", fps}, {"resolution", std::to_string(original.width) + "x" + std::to_string(original.height)
            + " -> " + std::to_string(width) + "x" + std::to_string(height)}});
    DirectoryStage memory("/dev/shm", "uf-encode-");
    torch::NoGradGuard no_grad;
    Codec codec(std::move(models), torch::Device(options.device), options.skip_threshold);
    CodecStatistics stats;
    {
        std::unique_ptr<SharedStreamState> near, policy;
        if (options.streaming) {
            if (!remote) throw std::invalid_argument("UF stream state requires remote input");
            near = std::make_unique<SharedStreamState>(options.streaming->state);
            policy = std::make_unique<SharedStreamState>(options.streaming->buffer);
            policy->claim();
        }
        try {
            const bool prepared = options.streaming && options.streaming->stream >= 0;
            const auto modes = options.hwaccel == "jetson" ? std::vector<std::string>{"jetson", "none"}
                                                           : std::vector<std::string>{options.hwaccel};
            // Python's per-job last=[0.0] reports the first packet and survives a
            // decoder fallback. Keep one timestamp across both attempts.
            std::chrono::steady_clock::time_point last{};
            for (const auto& mode : modes) {
                try {
                    // download_pipe belongs to each decoder attempt in the reference.
                    // Close a failed attempt before retrying with fresh source bytes
                    // and a private cookie copy. Prepared pipes retain their descriptor.
                    RemoteCookies cookies(remote && !prepared ? remote->cookies : std::nullopt);
                    with_remote_cookies(cookies, [&] {
                        std::unique_ptr<ChildPipe> download;
                        if (remote && !prepared) download = std::make_unique<ChildPipe>(
                            download_command(*remote, remote->video_format, cookies.path()),
                            std::nullopt, false, std::nullopt, true);
                        try {
                            {
                                const auto input = prepared ? std::optional<int>(options.streaming->stream)
                                    : (download ? std::optional<int>(download->output_fd()) : std::nullopt);
                                int capacity = policy ? policy->capacity() : options.prefetch_frames;
                                BufferedFrameReader frames(source, width, height, original, fps,
                                    options.input_threads, input, mode, capacity, policy != nullptr);
                                try {
                                    if (policy) frames.prime();
                                    const auto progress = [&](std::uint64_t count, double speed) {
                                        if (policy) {
                                            policy->observe(count, speed);
                                            const int updated = policy->capacity();
                                            if (updated != capacity) { frames.set_capacity(updated); capacity = updated; }
                                        }
                                        if (near && original.duration * fps > 0 &&
                                            original.duration * fps - count <= (policy ? policy->capacity() : 8)) {
                                            near->set_near_end();
                                        }
                                        const auto now = std::chrono::steady_clock::now();
                                        if (now - last >= std::chrono::seconds(2)) {
                                            event("encode_progress", {{"source", python_path_text(source.filename())}, {"frames", count}, {"fps", speed}});
                                            last = now;
                                            lease->check();
                                        }
                                    };
                                    stats = codec.encode([&] { return frames.read(); }, memory.path() / "video.bin",
                                                         width, height, options.qp_i, options.qp_p,
                                                         options.reset_interval, options.intra_period, options.max_frames, progress);
                                } catch (...) {
                                    frames.close_checked();
                                    throw;
                                }
                                frames.close_checked();
                            }
                            // Destroy the frame reader/buffer first, then perform the successful
                            // download context's drain and completion check before breaking.
                            if (prepared) drain_prepared_download(options.streaming->stream, options.streaming->errors, *near);
                            if (download) download->drain_download();
                        } catch (...) {
                            if (download) download->close_download_checked(false);
                            throw;
                        }
                    });
                    break;
                } catch (const DecoderError& error) {
                    if (mode != "jetson") throw;
                    std::filesystem::remove(memory.path() / "video.bin");
                    event("decoder_fallback", {{"source", python_path_text(source)}, {"error", error.what()},
                        {"message", "Retrying input with software decoding"}});
                }
            }
        } catch (...) {
            if (policy) policy->release();
            throw;
        }
        if (policy) policy->release();
        if (near) near->set_near_end();
    }
    const auto seconds = stats.encode_seconds;
    fsync_path(memory.path() / "video.bin");
    if (options.audio == "opus" && original.audio) {
        const auto existing_audio = output_root / (stem + ".opus");
        if (std::filesystem::is_regular_file(existing_audio) &&
            !std::filesystem::is_symlink(existing_audio)) {
            verify_opus(existing_audio);
            copy_preserving_metadata(existing_audio, memory.path() / "audio.opus");
            event("audio_reused", {{"source", python_path_text(existing_audio)}});
        } else {
            RemoteCookies cookies(remote ? remote->cookies : std::nullopt);
            with_remote_cookies(cookies, [&] {
                std::unique_ptr<ChildPipe> download;
                if (remote) {
                    if (!remote->audio_format) throw std::runtime_error("Missing remote audio format");
                    download = std::make_unique<ChildPipe>(
                        download_command(*remote, *remote->audio_format, cookies.path()),
                        std::nullopt, false, std::nullopt, true);
                }
                try {
                    encode_opus(source, memory.path() / "audio.opus", stats.frames / fps,
                                options.opus_channels, options.opus_bitrate,
                                options.opus_frame_ms == 20 ? nlohmann::json(20) : options.opus_frame_ms,
                                options.opus_complexity, options.opus_vbr,
                                download ? std::optional<int>(download->output_fd()) : std::nullopt);
                    if (download) download->drain_download();
                } catch (...) {
                    if (download) download->close_download_checked(false);
                    throw;
                }
            });
        }
        verify_opus(memory.path() / "audio.opus");
    }
    if (!remote && (!(python_path_identity(source) == before) || sha256_file(source) != source_sha))
        throw std::invalid_argument("Original changed during encoding; refusing to publish");
    if (remote) {
        const auto existing_info = output_root / (stem + ".info.json");
        if (std::filesystem::is_regular_file(existing_info) && !std::filesystem::is_symlink(existing_info)) {
            const auto info = read_python_json_file(
                existing_info, python_process_text_encoding(), true);
            if (!json_python_equal(python_json_dict_get(info, "id"), remote->public_metadata.at("id"))) {
                throw std::invalid_argument("Existing info.json belongs to another video");
            }
            copy_preserving_metadata(existing_info, memory.path() / "source.info.json");
        } else {
            write_journal(memory.path() / "source.info.json", remote->public_metadata);
        }
    } else copy_sidecars(source, memory.path());
    nlohmann::json artifacts = nlohmann::json::object();
    for (const auto& entry : std::filesystem::directory_iterator(memory.path())) {
        if (!entry.is_regular_file()) continue;
        artifacts[python_path_text(entry.path().filename())] = {
            {"size", entry.file_size()}, {"sha256", sha256_file(entry.path())}
        };
    }
    nlohmann::json source_metadata = {
        {"path", python_path_text(source)},
        {"relative_path", python_filesystem_decode(options.relative_path.value_or(source.filename().string()))},
        {"sha256", remote ? nlohmann::json(nullptr) : nlohmann::json(source_sha)},
        {"media", media_json(original)}
    };
    if (remote) {
        source_metadata["url"] = remote->url;
        source_metadata["kind"] = "youtube";
    } else {
        source_metadata["size"] = before.size;
        source_metadata["mtime_ns"] = python_json_integer(before.mtime_ns);
    }
    nlohmann::json metadata = {
        {"format", "dcvc-uf-archive"}, {"version", 1}, {"pipeline", pipeline},
        {"source", source_metadata},
        {"video", {{"frames", stats.frames}, {"packets", stats.packets},
                   {"width", width}, {"height", height}, {"fps", fps},
                   {"encode_seconds", seconds}}},
        {"validation", {{"mode", "artifact-hashes"}, {"decode_performed", false}}},
        {"artifacts", artifacts}, {"environment", runtime_environment(options.device)},
        {"filename_style", "rt"}, {"layout", "flat"}, {"archive_stem", python_filesystem_decode(stem)},
        {"lease_key", python_filesystem_decode(lease_key)}
    };
    event("publication_started", {{"source", python_path_text(source)},
        {"message", "Encoding complete; copying RAM output for atomic publication"}});
    DirectoryStage disk(output_root, "." + manifest.filename().string() + ".stage-");
    for (auto it = artifacts.begin(); it != artifacts.end(); ++it) {
        const auto name = python_path_from_text(it.key());
        const auto staged = disk.path() / name;
        copy_completed_artifact(memory.path() / name, staged, *lease);
    }
    fsync_path(disk.path(), true);
    ProcessInterrupt::check();
    lease->check();
    memory.remove();
    try {
        publish_flat(disk.path(), manifest, metadata, lease.get(), lease_key);
    } catch (...) {
        // Retain the stage while an interrupted publication still has a journal.
        // A failure to inspect the journal also retains it for later recovery.
        std::error_code error;
        if (std::filesystem::exists(pending, error) || error) disk.retain();
        throw;
    }
    disk.remove();
    fsync_path(output_root, true);
    finish_remote_thumbnail(options);
    event("encode_completed", {{"source", python_path_text(source)}, {"archive", python_path_text(manifest)},
        {"frames", stats.frames}, {"packets", stats.packets}, {"width", width}, {"height", height},
        {"fps", fps}, {"encode_seconds", seconds}, {"video_bytes", artifacts.at("video.bin").at("size")}});
    return manifest;
    } catch (const std::exception& error) {
        // Python reports body failures before finally releases the lease.
        // A release failure propagates without a second encode_failed event.
        event("encode_failed", {{"source", python_path_text(source)}, {"error", error.what()}});
        throw;
    }
    });
}

std::filesystem::path encode_local(const EncodeOptions& options, bool* resumed) {
    const auto source = options.remote ? std::filesystem::path(options.remote->video_id)
                                      : python_path_absolute(options.input_file);
    const auto root = options.remote ? options.remote->manifest.parent_path()
                                    : std::filesystem::absolute(options.output_root);
    const auto manifest = options.remote ? options.remote->manifest : root / (source.stem().string() + ".uf.json");
    std::filesystem::create_directories(root);
    JobEvents records(manifest.string(), root / ".uf-progress.jsonl");
    return encode_local_impl(options, resumed);
}

std::filesystem::path encode_remote(const EncodeOptions& options, bool* resumed) {
    if (!options.remote) throw std::invalid_argument("Missing remote video job");
    return encode_local(options, resumed);
}

std::vector<EncodeOptions> discover_local_jobs(
    const std::filesystem::path& base_root,
    const std::filesystem::path& output_root,
    const std::vector<std::string>& channel_ids,
    bool recursive,
    const EncodeOptions& settings) {
    const auto base = python_path_resolve(base_root);
    std::vector<std::filesystem::path> roots;
    if (channel_ids.empty()) roots.push_back(base);
    else {
        for (const auto& id : channel_ids) {
            const std::filesystem::path name(id);
            roots.push_back(python_path_normalize(base / name));
        }
    }
    for (const auto& root : roots) {
        if (!python_path_is_relative_to(python_path_resolve(root), base) ||
            !python_path_is_directory(root))
            throw std::invalid_argument("Invalid channel/input directory: " + root.string());
    }
    const std::map<std::string, int> rank = {{".mkv", 0}, {".mp4", 1},
        {".webm", 2}, {".mov", 3}, {".avi", 4}, {".m4v", 99}, {".ts", 99}};
    const auto lowercase_extension = [](const std::filesystem::path& path) {
        auto extension = path.extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(),
                       [](unsigned char c) { return std::tolower(c); });
        return extension;
    };
    std::vector<std::filesystem::path> candidates;
    const auto add_if_video = [&](const std::filesystem::path& path) {
        if (!python_path_is_file(path) || python_path_is_symlink(path)) return;
        const auto extension = lowercase_extension(path);
        if (!rank.count(extension)) return;
        const auto relative = python_path_relative_to(path, base);
        for (const auto& part : relative) {
            const auto text = part.string();
            if (text.empty() || text[0] == '.' ||
                (text.size() >= 3 && text.compare(text.size() - 3, 3, ".uf") == 0)) return;
        }
        candidates.push_back(path);
    };
    for (const auto& root : roots) {
        if (recursive) {
            if (!python_path_is_directory(root)) continue;
            for (const auto& entry : std::filesystem::recursive_directory_iterator(root,
                    std::filesystem::directory_options::skip_permission_denied)) {
                add_if_video(entry.path());
            }
        } else {
            for (const auto& entry : std::filesystem::directory_iterator(root)) {
                add_if_video(entry.path());
            }
        }
    }
    const auto output = python_path_resolve(output_root);
    std::sort(candidates.begin(), candidates.end(), [&](const auto& left, const auto& right) {
        const auto left_rank = rank.at(lowercase_extension(left));
        const auto right_rank = rank.at(lowercase_extension(right));
        return std::tuple{left_rank, python_path_text(left)} < std::tuple{right_rank, python_path_text(right)};
    });
    candidates.erase(std::unique(candidates.begin(), candidates.end(),
        [](const auto& left, const auto& right) { return left.native() == right.native(); }),
        candidates.end());
    std::map<std::pair<std::filesystem::path, std::string>, std::filesystem::path> selected;
    for (const auto& source : candidates) {
        const auto relative = python_path_relative_to(source, base);
        const auto [chosen, inserted] = selected.emplace(
            std::make_pair(relative.parent_path(), source.stem().string()), source);
        if (!inserted) {
            event("duplicate_source_retained", {{"source", python_path_text(source)},
                {"message", "selected " + python_path_text(chosen->second.filename()) +
                    "; retained duplicate " + python_path_text(source.filename())}});
        }
    }
    std::vector<std::filesystem::path> sources;
    for (const auto& [_, source] : selected) sources.push_back(source);
    std::sort(sources.begin(), sources.end(), python_path_less);
    std::vector<EncodeOptions> jobs;
    for (const auto& source : sources) {
        const auto relative = python_path_relative_to(source, base);
        auto job = settings;
        job.input_file = source;
        job.output_root = python_path_normalize(output / relative.parent_path());
        job.relative_path = relative.string();
        jobs.push_back(std::move(job));
    }
    return jobs;
}

}  // namespace uf
