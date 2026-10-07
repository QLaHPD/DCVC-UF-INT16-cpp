// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#pragma once

#include "uf/evaluation_json.hpp"
#include <filesystem>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace uf::evaluation {

struct VideoEvaluationOptions {
    double skip_threshold = 0;
    std::filesystem::path model_i, model_p, config, stream_path = "out_bin", output;
    std::optional<std::filesystem::path> root_path;
    std::optional<std::vector<int>> qp_i, qp_p, cuda_indices;
    int rate_count = 4, workers = 1, reset_interval = 32, verbose = 0;
    std::int64_t force_frame_count = -1, force_intra_period = -1;
    std::string variant = "ld";
    bool force_intra = false, calc_ssim = false, check_existing = false;
    bool save_decoded = false, verbose_json = false;
};

struct EvaluationPoint {
    std::filesystem::path source, bin_folder, bin_path, reconstruction_path, json_path;
    std::string dataset, sequence, source_type;
    std::uint32_t width = 0, height = 0;
    std::uint64_t frame_count = 0;
    std::int64_t intra_period = -1;
    int rate_index = 0, qp_i = 0, qp_p = 0;
};

Json evaluation_settings_json(const VideoEvaluationOptions& options);
VideoEvaluationOptions evaluation_settings(const Json& value);
Json evaluation_point_json(const EvaluationPoint& point);
EvaluationPoint evaluation_point(const Json& value);

bool evaluation_cuda_compiled();
int evaluation_cuda_devices();
class EvaluationModels {
public:
    EvaluationModels(VideoEvaluationOptions options, const std::string& cuda_selector);
    ~EvaluationModels();
    Json run(const EvaluationPoint& point);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Submit while enumerating the configuration. Replies are collected in a
// background manager, so completed calls can suppress subsequent spawns.
// Destruction drains submitted calls when enumeration itself raises.
class EvaluationExecutor {
public:
    EvaluationExecutor(const VideoEvaluationOptions& options, int gpu_count);
    ~EvaluationExecutor();
    EvaluationExecutor(const EvaluationExecutor&) = delete;
    EvaluationExecutor& operator=(const EvaluationExecutor&) = delete;
    void submit(const EvaluationPoint& point);
    std::vector<Json> finish();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

std::vector<Json> evaluate_points(const VideoEvaluationOptions& options,
                                const std::vector<EvaluationPoint>& points, int gpu_count);
// Shared spawn/IPC scheduler. Initialization and point documents are interpreted
// by the selected model backend, with results retained in submission order.
// Per-point exceptions keep workers alive and drain submitted work before the
// first error in submission order is raised. Pool/transport failures cancel.
std::vector<Json> evaluate_documents(const std::vector<Json>& initializations,
                                    const std::vector<Json>& points);
// Own options so threshold normalization follows CUDA availability without
// changing the caller's settings. Command dispatch transfers its one-use value.
int evaluate_video(VideoEvaluationOptions options);

}  // namespace uf::evaluation
