// Copyright (c) Microsoft Corporation. Licensed under the MIT License.
#pragma once
#include <torch/torch.h>
#include <cstdint>
#include <filesystem>
#include <string>

namespace uf::hem {
torch::Tensor read_image_to_torch(const std::filesystem::path& path);
void save_torch_image(const torch::Tensor& image, const std::filesystem::path& path);
double psnr(const torch::Tensor& first, const torch::Tensor& second);
class PNGReader {
public:
    PNGReader(std::filesystem::path folder, std::int64_t width, std::int64_t height);
    // CHW Float32 RGB in [0, 1]. An undefined tensor denotes EOF.
    torch::Tensor read_one_frame(const std::string& format = "rgb");
    void close();  // Reset the index; the reference retains its EOF flag.
private:
    std::filesystem::path folder_;
    std::int64_t width_, height_, index_ = 1;
    std::size_t padding_ = 1;
    bool eof_ = false;
};
int run_video_evaluation_command(int argc, char** argv);
int run_checkpoint_download_command(int argc, char** argv);
}  // namespace uf::hem
