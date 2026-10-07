// Copyright (c) Microsoft Corporation. Licensed under the MIT License.
#pragma once
#include <torch/torch.h>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>

namespace uf::dc {
using Yuv420 = std::pair<torch::Tensor, torch::Tensor>;
using YuvPlanes = std::array<torch::Tensor, 3>;
// Array helpers represent NumPy CHW arrays with CPU floating tensors. Zoom
// follows SciPy's constant boundary, grid_mode=false and spline prefilter.
Yuv420 rgb_to_ycbcr420(const torch::Tensor& rgb);
torch::Tensor ycbcr420_to_rgb(const torch::Tensor& y, const torch::Tensor& uv, int order = 1);
torch::Tensor ycbcr420_to_444(const torch::Tensor& y, const torch::Tensor& uv, int order = 1);
Yuv420 ycbcr444_to_420(const torch::Tensor& yuv);
torch::Tensor rgb_to_ycbcr(const torch::Tensor& rgb);
torch::Tensor ycbcr_to_rgb(const torch::Tensor& yuv);
// Tensor helpers retain the original device, autograd and unclipped values.
torch::Tensor rgb2ycbcr(const torch::Tensor& rgb);
torch::Tensor ycbcr2rgb(const torch::Tensor& ycbcr);
YuvPlanes yuv_444_to_420(const torch::Tensor& yuv, const std::string& mode = "avg_pool");
YuvPlanes yuv_444_to_420(const YuvPlanes& yuv, const std::string& mode = "avg_pool");
YuvPlanes yuv_420_to_444_tuple(const YuvPlanes& yuv, const std::string& mode = "bilinear");
torch::Tensor yuv_420_to_444(const YuvPlanes& yuv, const std::string& mode = "bilinear");

struct MediaFrame {
    torch::Tensor rgb, y, uv;
    // The requested representation is undefined at EOF.
    bool defined() const { return rgb.defined() || y.defined(); }
};
class VideoReader {
public:
    VideoReader(std::filesystem::path source, std::int64_t width, std::int64_t height);
    virtual ~VideoReader() = default;
    virtual MediaFrame read_one_frame(const std::string& format = "rgb") = 0;
    virtual void close() = 0;
    std::filesystem::path src_path;
    std::int64_t width, height;
    bool eof = false;
protected:
    static MediaFrame get_dst_format(MediaFrame frame, const std::string& format);
    static MediaFrame none_exist_frame(const std::string& format);
};
class PNGReader : public VideoReader {
public:
    PNGReader(std::filesystem::path source, std::int64_t width, std::int64_t height, std::int64_t start_num = 1);
    MediaFrame read_one_frame(const std::string& format = "rgb") override;
    void close() override;
    std::int64_t current_frame_index;
    unsigned padding;
};
class RGBReader : public VideoReader {
public:
    RGBReader(std::filesystem::path source, std::int64_t width, std::int64_t height,
              std::string src_format = "rgb", int bit_depth = 8);
    MediaFrame read_one_frame(const std::string& format = "420") override;
    void close() override;
    const std::string src_format;
    const int bit_depth;
private:
    std::ifstream file_;
    std::size_t rgb_size_;
    unsigned max_val_;
};
class YUVReader : public VideoReader {
public:
    YUVReader(std::filesystem::path source, std::int64_t width, std::int64_t height,
              std::string src_format = "420", std::int64_t skip_frame = 0);
    MediaFrame read_one_frame(const std::string& format = "420") override;
    void close() override;
    const std::string src_format;
private:
    std::ifstream file_;
    std::size_t y_size_, uv_size_;
};
class VideoWriter {
public:
    VideoWriter(std::filesystem::path destination, std::int64_t width, std::int64_t height);
    virtual ~VideoWriter() = default;
    virtual void write_one_frame(const MediaFrame& frame, const std::string& format = "rgb") = 0;
    virtual void close() = 0;
    std::filesystem::path dst_path;
    std::int64_t width, height;
};
class PNGWriter : public VideoWriter {
public:
    PNGWriter(std::filesystem::path destination, std::int64_t width, std::int64_t height);
    void write_one_frame(const MediaFrame& frame, const std::string& format = "rgb") override;
    void close() override;
    std::int64_t current_frame_index = 1;
    unsigned padding = 5;
};
class RGBWriter : public VideoWriter {
public:
    RGBWriter(std::filesystem::path destination, std::int64_t width, std::int64_t height,
              std::string dst_format = "rgb", int bit_depth = 8);
    void write_one_frame(const MediaFrame& frame, const std::string& format = "rgb") override;
    void close() override;
    const std::string dst_format;
    const int bit_depth;
private:
    std::ofstream file_;
    unsigned max_val_;
};
class YUVWriter : public VideoWriter {
public:
    YUVWriter(std::filesystem::path destination, std::int64_t width, std::int64_t height,
              std::string dst_format = "420");
    void write_one_frame(const MediaFrame& frame, const std::string& format = "420") override;
    void close() override;
    const std::string dst_format;
    bool eof = false;
private:
    std::ofstream file_;
};
std::uint64_t convert_one_seq_to_png(const std::filesystem::path& source, std::int64_t width,
                                   std::int64_t height, const std::filesystem::path& destination);
int run_data_to_png_command(int argc, char** argv);
int run_checkpoint_download_command(int argc, char** argv);
}  // namespace uf::dc
