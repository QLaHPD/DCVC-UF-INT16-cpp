// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#pragma once

#include <torch/torch.h>
#include <nlohmann/json_fwd.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace uf::training {

// The loader supplies its worker's random stream. The function must draw an
// integer uniformly from [0, upper); PythonRandom::below has these semantics.
using RandomBelow = std::function<std::uint64_t(std::uint64_t upper)>;
// Image/frames followed by scalar int32 QP and scalar float32 lambda.
using DatasetSample = std::vector<torch::Tensor>;

class ImageFolder {
public:
    ImageFolder(std::filesystem::path root, std::int64_t patch_height,
                std::int64_t patch_width, std::int64_t qp_count,
                std::vector<double> lambdas);
    // Restore an epoch snapshot without rereading description.json.
    explicit ImageFolder(const nlohmann::json& worker_snapshot);
    nlohmann::json worker_snapshot() const;
    // Manifest names remain Python text through joining; only the complete
    // path is encoded as native filesystem bytes at image-open time.
    DatasetSample get(std::int64_t index, const RandomBelow& random) const;
    std::size_t size() const noexcept;
    std::pair<std::int64_t, std::int64_t> patch_size() const;
    void set_patch_size(std::int64_t width, std::int64_t height);

private:
    std::filesystem::path root_;
    std::vector<std::string> images_;
    std::int64_t qp_count_;
    std::vector<double> lambdas_;
    mutable std::mutex settings_mutex_;
    std::int64_t patch_height_, patch_width_;
};

class VideoFolder {
public:
    VideoFolder(std::filesystem::path root, std::int64_t patch_height,
                std::int64_t patch_width, std::int64_t qp_count,
                std::vector<double> lambdas, std::int64_t frame_count = 5,
                std::int64_t group_of_pictures = 1);
    explicit VideoFolder(const nlohmann::json& worker_snapshot);
    nlohmann::json worker_snapshot() const;
    // Root/sequence/frame join follows POSIX os.path.join before filesystem
    // encoding, including absolute later components and surrogateescape.
    DatasetSample get(std::int64_t index, const RandomBelow& random) const;
    std::size_t size() const noexcept;
    std::int64_t frame_count() const;
    std::pair<std::int64_t, std::int64_t> patch_size() const;
    void set_frame_count(std::int64_t count);
    void set_patch_size(std::int64_t width, std::int64_t height);

private:
    void load_description(const nlohmann::json& value);
    struct Sequence {
        std::string path;
        std::int64_t width, height, length;
    };
    std::filesystem::path root_;
    std::vector<Sequence> sequences_;
    std::vector<std::string> frames_;
    std::int64_t qp_count_;
    std::vector<double> lambdas_;
    mutable std::mutex settings_mutex_;
    std::int64_t patch_height_, patch_width_, frame_count_;
    std::int64_t group_of_pictures_;
};

// Default collate for the Python list of tensors: stack each list position,
// preserving integer QP/float lambda dtypes and frame group positions.
DatasetSample collate_samples(const std::vector<DatasetSample>& samples);

}  // namespace uf::training
