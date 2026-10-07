// Copyright (c) Microsoft Corporation. Licensed under the MIT License.
#pragma once
#include "uf/dcvc_dc_media.hpp"
#include "uf/evaluation_metrics.hpp"
#include <memory>

namespace uf::dc {
using Json = uf::evaluation::Json;
using FrameQuality = uf::evaluation::FrameQuality;
using uf::evaluation::fspecial_gauss;
using uf::evaluation::calc_ssim;
using uf::evaluation::calc_msssim;
using uf::evaluation::calc_msssim_rgb;
torch::Tensor np_image_to_tensor(const torch::Tensor& image);
double PSNR(const torch::Tensor& first, const torch::Tensor& second);
double calc_psnr(const torch::Tensor& first, const torch::Tensor& second, double data_range = 255);
bool str2bool(const std::string& value);
std::string scale_list_to_str(const std::vector<double>& scales);
void create_folder(const std::filesystem::path& path, bool print_if_create = false);
void dump_json(const Json& value, std::ostream& output, int float_digits = -1, int indent = -1);
Json generate_log_json(std::int64_t frame_count, std::uint64_t frame_pixels, double test_time,
    const std::vector<int>& frame_types, const std::vector<double>& bits,
    const std::vector<FrameQuality>& quality, bool verbose = false);
class EvaluationModels {
public:
    // The shared process scheduler constructs this once per spawned worker.
    EvaluationModels(const Json& settings, int worker_index);
    ~EvaluationModels();
    Json run(const Json& point);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
int run_video_evaluation_command(int argc, char** argv);
}  // namespace uf::dc
