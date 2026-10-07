#include "uf/image_workflow.hpp"

#include "uf/archive.hpp"
#include "uf/image_archive.hpp"
#include "uf/image_pixels.hpp"
#include "uf/integer_codec.hpp"
#include "uf/codec.hpp"
#include "uf/media.hpp"
#include "uf/prepared.hpp"
#include "uf/shared_work.hpp"
#include "uf/events.hpp"
#include "uf/process.hpp"
#include "uf/json_compat.hpp"
#include "uf/python_path.hpp"
#include "uf/python_json.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace uf {
namespace {

class ImageStage {
public:
    ImageStage(const std::filesystem::path& parent, std::string_view prefix, bool temporary = false)
        : directory_(python_temporary_mkdtemp(prefix, {}, parent)), temporary_(temporary) {}
    ~ImageStage() {
        if (!cleanup_attempted_ && !directory_.empty()) {
            try {
                if (temporary_) python_temporary_rmtree(directory_, false, false);
                else python_path_rmtree(directory_, true, false);
            } catch (...) {}
        }
    }
    ImageStage(const ImageStage&) = delete;
    ImageStage& operator=(const ImageStage&) = delete;
    const std::filesystem::path& path() const { return directory_; }
    void remove() {
        if (cleanup_attempted_ || directory_.empty()) return;
        cleanup_attempted_ = true;
        // Cleanup also runs while unwinding a latched foreground interruption.
        if (temporary_) python_temporary_rmtree(directory_, false, false);
        else python_path_rmtree(directory_, false, false);
        directory_.clear();
    }
private:
    std::filesystem::path directory_;
    bool cleanup_attempted_{};
    bool temporary_{};
};

void close_image_file(FileDescriptor& file, const std::filesystem::path& path) {
    const int owned = file.release();
    if (close(owned) != 0) {
        const int error = errno;
        throw std::system_error(error, std::generic_category(), "Cannot close UF image file: " + path.string());
    }
}

FileDescriptor open_image_file(const std::filesystem::path& path, int flags) {
    if (path.native().find('\0') != std::string::npos)
        throw std::invalid_argument("embedded null byte");
    int descriptor;
    for (;;) {
        ProcessInterrupt::check();
        descriptor = open(path.c_str(), flags | O_CLOEXEC, 0666);
        if (descriptor >= 0) break;
        const int error = errno;
        if (error == EINTR) continue;
        throw std::system_error(error, std::generic_category(), "Cannot open UF image file: " + path.string());
    }
    FileDescriptor file(descriptor);
    try {
        struct stat status{};
        const int result = fstat(file.get(), &status);
        const int error = errno;
        if (result < 0 && error == EBADF)
            throw std::system_error(error, std::generic_category(), "Cannot stat UF image file");
        if (result == 0 && S_ISDIR(status.st_mode))
            throw std::system_error(EISDIR, std::generic_category(), "Cannot open UF image directory as a file");
        return file;
    } catch (...) {
        close_image_file(file, path);
        throw;
    }
}

void sync_image_descriptor(int descriptor) {
    for (;;) {
        ProcessInterrupt::check();
        if (fsync(descriptor) == 0) return;
        const int error = errno;
        if (error == EINTR) continue;
        throw std::system_error(error, std::generic_category(), "Cannot sync UF image");
    }
}

void sync_file(const std::filesystem::path& input, bool directory = false) {
    const auto path = python_path_normalize(input);
    // Directory syncing uses open directly; FileIO's directory rejection
    // applies to the binary reader/writer helpers, not sync_directory().
    if (path.native().find('\0') != std::string::npos)
        throw std::invalid_argument("embedded null byte");
    int descriptor;
    for (;;) {
        ProcessInterrupt::check();
        descriptor = open(path.c_str(), O_RDONLY | O_CLOEXEC | (directory ? O_DIRECTORY : 0));
        if (descriptor >= 0) break;
        const int error = errno;
        if (error == EINTR) continue;
        throw std::system_error(error, std::generic_category(), "Cannot open UF image for sync");
    }
    FileDescriptor file(descriptor);
    try {
        sync_image_descriptor(file.get());
        close_image_file(file, path);
    } catch (...) {
        if (file.get() >= 0) close_image_file(file, path);
        throw;
    }
}

void write_image_bytes(int descriptor, const char* bytes, std::size_t size) {
    std::size_t offset = 0;
    while (offset < size) {
        ProcessInterrupt::check();
        const auto count = write(descriptor, bytes + offset, size - offset);
        if (count < 0) {
            const int error = errno;
            if (error == EINTR) continue;
            throw std::system_error(error, std::generic_category(), "Cannot write UF image container");
        }
        if (count == 0)
            throw std::system_error(EIO, std::generic_category(), "Cannot write UF image container");
        offset += static_cast<std::size_t>(count);
    }
}

void write_image_container(const std::filesystem::path& stage,
    const std::filesystem::path& payload, const std::string& header) {
    auto output = open_image_file(stage, O_WRONLY | O_CREAT | O_EXCL);
    try {
        auto input = open_image_file(payload, O_RDONLY);
        try {
            const std::array<char, 8> magic{'U','F','I','M','A','G','E',1};
            const auto length = static_cast<std::uint32_t>(header.size());
            const std::array<char, 4> size{
                static_cast<char>(length >> 24), static_cast<char>(length >> 16),
                static_cast<char>(length >> 8), static_cast<char>(length)};
            write_image_bytes(output.get(), magic.data(), magic.size());
            write_image_bytes(output.get(), size.data(), size.size());
            write_image_bytes(output.get(), header.data(), header.size());
            // Linux shutil.copyfileobj uses 64 KiB BufferedReader chunks.
            std::vector<char> buffer(64 * 1024);
            for (;;) {
                std::size_t received = 0;
                while (received < buffer.size()) {
                    ProcessInterrupt::check();
                    const auto count = read(input.get(), buffer.data() + received, buffer.size() - received);
                    if (count < 0) {
                        const int error = errno;
                        if (error == EINTR) continue;
                        throw std::system_error(error, std::generic_category(), "Cannot read UF image payload");
                    }
                    if (count == 0) break;
                    received += static_cast<std::size_t>(count);
                }
                if (received == 0) break;
                write_image_bytes(output.get(), buffer.data(), received);
            }
            sync_image_descriptor(output.get());
            close_image_file(input, payload);
        } catch (...) {
            // output, data with entries unwind data first, then output.
            if (input.get() >= 0) close_image_file(input, payload);
            throw;
        }
        close_image_file(output, stage);
    } catch (...) {
        if (output.get() >= 0) close_image_file(output, stage);
        throw;
    }
}

std::vector<std::uint8_t> yuv444_padded(const ImagePixels& image) {
    const auto width = image.width + image.width % 2;
    const auto height = image.height + image.height % 2;
    const auto area = static_cast<std::size_t>(width) * height;
    std::vector<std::uint8_t> result(area * 3);
    const auto byte = [](float value) {
        value = std::clamp(value, 0.0f, 255.0f);
        auto base = static_cast<unsigned>(std::floor(value));
        const float fraction = value - base;
        if (fraction > 0.5f || (fraction == 0.5f && base % 2)) ++base;
        return static_cast<std::uint8_t>(base);
    };
    for (std::uint32_t y = 0; y < height; ++y) {
        ProcessInterrupt::check();
        for (std::uint32_t x = 0; x < width; ++x) {
            const auto input = (static_cast<std::size_t>(std::min(y, image.height - 1)) * image.width +
                                std::min(x, image.width - 1)) * 3;
            const auto r = static_cast<float>(image.rgb[input]);
            const auto g = static_cast<float>(image.rgb[input + 1]);
            const auto b = static_cast<float>(image.rgb[input + 2]);
            // NumPy evaluates these as separate Float32 ufunc operations.
            // Materialize intermediates so a C++ optimizer cannot fuse them.
            const volatile float ry = 0.2126f * r;
            const volatile float gy = 0.7152f * g;
            const volatile float by = 0.0722f * b;
            const volatile float rg = ry + gy;
            const float luma = rg + by;
            const volatile float blue_difference = b - luma;
            const volatile float red_difference = r - luma;
            const volatile float u = blue_difference / 1.8556f;
            const volatile float v = red_difference / 1.5748f;
            const auto offset = static_cast<std::size_t>(y) * width + x;
            result[offset] = byte(luma);
            result[area + offset] = byte(u + 128.0f);
            result[2 * area + offset] = byte(v + 128.0f);
        }
    }
    return result;
}

}  // namespace

struct ImageEncoder::Impl {
    std::optional<CodecModels> models;
    std::unique_ptr<Codec> codec;
    std::filesystem::path model_path;
    std::string runtime, device;
    double skip_threshold{};

    const CodecModels& models_for(const ImageEncodeOptions& options) {
        const auto selected = options.runtime == "int16" ? options.prepared_image : options.checkpoint_image;
        if (selected.empty()) throw std::invalid_argument("UF intra image requires a model path");
        const auto path = std::filesystem::absolute(selected).lexically_normal();
        if (!models || runtime != options.runtime || model_path != path) {
            codec.reset();
            models.reset();
            models = CodecModels::load(options.runtime, path, std::nullopt, "hts");
            runtime = options.runtime;
            model_path = path;
        }
        return *models;
    }

    Codec& codec_for(const ImageEncodeOptions& options) {
        if (!models) throw std::logic_error("UF image codec requires loaded model data");
        if (codec && (device != options.device || skip_threshold != options.skip_threshold)) codec.reset();
        if (!codec) {
            // Retain identity metadata across images, but release prepared CPU
            // tensors after construction. A changed device/threshold reloads
            // them against the cached identity before creating a new codec.
            if (models->integer_image && models->integer_image->tensors.empty()) {
                const ExpectedModel expected{models->image_sha256, models->integer_image->identity};
                auto refreshed = CodecModels::load(models->runtime, model_path, std::nullopt,
                                                   "hts", expected);
                models->integer_image = std::move(refreshed.integer_image);
            }
            try {
                codec = std::make_unique<Codec>(*models, torch::Device(options.device), options.skip_threshold);
            } catch (...) {
                if (models->integer_image) models->integer_image->tensors.clear();
                throw;
            }
            if (models->integer_image) models->integer_image->tensors.clear();
            device = options.device;
            skip_threshold = options.skip_threshold;
        }
        return *codec;
    }
};

static std::filesystem::path encode_intra_image_impl(const ImageEncodeOptions& options,
    const CodecModels& models, const std::function<Codec&()>& get_codec, bool* resumed) {
    const auto source = python_path_absolute(options.source);
    const auto target = python_path_absolute(options.target);
    nlohmann::json pipeline = {
        {"codec", "dcvc-uf"}, {"runtime", options.runtime}, {"image_sha256", models.image_sha256},
        {"image_only", true}, {"variant", "image"}, {"video_sha256", nullptr},
        {"qp_i", options.qp}, {"qp_p", options.qp}, {"audio", "none"},
        {"max_frames", nullptr}, {"intra_period", 1}
    };
    if (models.integer_image) {
        pipeline["integer"] = {{"arithmetic", models.integer_image->arithmetic},
                               {"image", models.integer_image->identity},
                               {"video", std::string(64, '0')}};
    }
    if (options.skip_threshold) pipeline["skip_threshold"] = options.skip_threshold;
    std::filesystem::create_directories(target.parent_path());
    SharedWorkPool shared(WorkKind::image, options.shared_lease_seconds,
                          options.shared_instance);
    shared.ensure_pipeline(target.parent_path(), pipeline);
    auto lease = shared.try_acquire(target.parent_path(), target.filename().string());
    if (!lease) {
        event("claimed_by_peer", {{"source", python_path_text(source)}});
        throw SharedWorkBusy("UF intra image is claimed by a Python or C++ peer");
    }
    return with_shared_lease(*lease, [&]() -> std::filesystem::path {
    try {
    const auto before = python_path_identity(source);
    const auto source_hash = sha256_file(source);
    if (std::filesystem::exists(target)) {
        const auto existing = read_intra_image(target, true, false);
        if (existing.metadata.at("source").at("sha256") != source_hash ||
            !json_python_equal(existing.metadata.at("pipeline"), pipeline)) {
            throw std::invalid_argument("Existing intra image has different source or configuration");
        }
        event("already_encoded", {{"source", python_path_text(source)}, {"archive", python_path_text(target)}});
        if (resumed) *resumed = true;
        return target;
    }
    torch::NoGradGuard no_grad;
    auto& codec = get_codec();
    const auto pixels = read_image_rgb(source);
    const auto width = pixels.width + pixels.width % 2;
    const auto height = pixels.height + pixels.height % 2;
    event("encode_started", {{"source", python_path_text(source)},
        {"message", std::to_string(pixels.width) + "x" + std::to_string(pixels.height)
            + " -> " + std::to_string(pixels.width) + "x" + std::to_string(pixels.height) + " (intra image)"}});
    auto frame = yuv444_padded(pixels);
    ImageStage memory(memory_stage_root(), "uf-image-", true);
    try {
    const auto payload = memory.path() / "intra.bin";
    const auto container = memory.path() / target.filename();
    bool consumed = false;
    const auto stats = codec.encode([&]() -> std::optional<std::vector<std::uint8_t>> {
        if (consumed) return std::nullopt;
        consumed = true;
        return std::move(frame);
    }, payload, width, height, options.qp, options.qp, 32, 1, 1);
    const auto seconds = stats.encode_seconds;
    auto relative = source.filename();
    if (options.source_base) relative = python_path_relative_to(source, *options.source_base);
    nlohmann::json metadata = {
        {"format", "dcvc-uf-image"}, {"version", 1}, {"pipeline", pipeline},
        {"video", {{"frames", stats.frames}, {"packets", stats.packets},
                   {"width", width}, {"height", height}, {"fps", 1}, {"encode_seconds", seconds}}},
        {"validation", {{"mode", "artifact-hashes"}, {"decode_performed", false}}},
        {"image_width", pixels.width}, {"image_height", pixels.height},
        {"source", {{"path", python_path_text(source)}, {"relative_path", python_path_text(relative)},
                    {"sha256", source_hash}, {"size", before.size},
                    {"mtime_ns", python_json_integer(before.mtime_ns)}}},
        {"payload_sha256", sha256_file(payload)},
        {"payload_size", std::filesystem::file_size(payload)}
    };
    const auto header = python_sorted_json(metadata);
    if (header.size() > 1024u * 1024u) throw std::runtime_error("UF image header is too large");
    write_image_container(container, payload, header);
    const auto after = python_path_identity(source);
    if (!(after == before) ||
        sha256_file(source) != source_hash) {
        throw std::invalid_argument("Thumbnail changed during encoding");
    }
    lease->check();
    if (unlink(payload.c_str()) != 0) {
        const int error = errno;
        throw std::system_error(error, std::generic_category(), "Cannot remove UF image payload");
    }
    lease->check();
    ImageStage disk(target.parent_path(), "." + target.filename().native() + ".stage-");
    const auto disk_file = disk.path() / target.filename();
    copy_completed_archive_file(container, disk_file, [&] { lease->check(); });
    sync_file(disk.path(), true);
    lease->check();
    try {
        lease->check();
        std::filesystem::create_hard_link(disk_file, target);
        sync_file(target.parent_path(), true);
    } catch (...) {
        // Python's publication finally removes the disk stage on link/sync
        // failures too, and a cleanup failure can replace that active error.
        disk.remove();
        throw;
    }
    disk.remove();
    memory.remove();
    event("encode_completed", {{"source", python_path_text(source)}, {"archive", python_path_text(target)},
        {"frames", stats.frames}, {"packets", stats.packets}, {"width", width}, {"height", height},
        {"fps", 1}, {"encode_seconds", seconds}});
    return target;
    } catch (...) {
        // TemporaryDirectory exits before the enclosing encode-failed handler;
        // an explicit cleanup error may replace the original body exception.
        memory.remove();
        throw;
    }
    } catch (const std::exception& error) {
        event("encode_failed", {{"source", python_path_text(source)}, {"error", error.what()}});
        throw;
    }
    });
}

ImageEncoder::ImageEncoder() : impl_(std::make_unique<Impl>()) {}
ImageEncoder::~ImageEncoder() = default;

std::filesystem::path ImageEncoder::encode(const ImageEncodeOptions& options, bool* resumed) {
    if (resumed) *resumed = false;
    const auto target = python_path_absolute(options.target);
    std::filesystem::create_directories(target.parent_path());
    JobEvents records(target.string(), target.parent_path() / ".thumbnail-progress.jsonl");
    ProcessInterrupt::check();
    if (options.qp < 0 || options.qp >= 64 || !std::isfinite(options.skip_threshold) ||
        options.skip_threshold < 0 || options.skip_threshold > 16) {
        throw std::invalid_argument("Invalid UF intra image QP or skip threshold");
    }
    const auto& models = impl_->models_for(options);
    return encode_intra_image_impl(options, models, [&]() -> Codec& { return impl_->codec_for(options); }, resumed);
}

std::filesystem::path encode_intra_image(const ImageEncodeOptions& options, bool* resumed) {
    ImageEncoder encoder;
    return encoder.encode(options, resumed);
}

}  // namespace uf
