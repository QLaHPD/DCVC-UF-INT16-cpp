#pragma once

#include <filesystem>
#include <memory>
#include <optional>
#include <string>

namespace uf {

struct ImageEncodeOptions {
    std::filesystem::path source;
    std::filesystem::path target;
    std::filesystem::path prepared_image;
    std::filesystem::path checkpoint_image;
    std::string runtime{"fp16"};
    std::optional<std::filesystem::path> source_base;
    std::string device{"cuda:0"};
    int qp{45};
    double skip_threshold{};
    std::string shared_instance;
    double shared_lease_seconds{900};
};

// Sequential image passes retain model data and lazily create one codec.
// This context is owned by the image worker and is not shared between threads.
class ImageEncoder {
public:
    ImageEncoder();
    ~ImageEncoder();
    ImageEncoder(const ImageEncoder&) = delete;
    ImageEncoder& operator=(const ImageEncoder&) = delete;
    std::filesystem::path encode(const ImageEncodeOptions& options, bool* resumed = nullptr);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

std::filesystem::path encode_intra_image(const ImageEncodeOptions& options, bool* resumed = nullptr);

}  // namespace uf
