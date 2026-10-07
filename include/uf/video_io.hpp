// Copyright (c) Microsoft Corporation. Licensed under the MIT License.
#pragma once
#include <torch/torch.h>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <utility>

namespace uf::video_io {
class PNGReader {
public:
    PNGReader(std::filesystem::path source, std::int64_t width, std::int64_t height, std::int64_t start_num = 1);
    torch::Tensor read_one_frame();  // CPU UInt8 CHW; undefined at EOF.
    void close();
    bool eof = false;
    std::filesystem::path src_path;
    std::int64_t width, height, current_frame_index;
    unsigned padding;
};
class YUV420Reader {
public:
    YUV420Reader(std::filesystem::path source, std::int64_t width, std::int64_t height, std::int64_t skip_frame = 0);
    std::pair<torch::Tensor, torch::Tensor> read_one_frame();
    void close();
    bool eof = false;
    std::filesystem::path src_path;
    std::int64_t y_width, y_height, uv_width, uv_height;
private:
    std::size_t y_size_, uv_size_;
    std::ifstream file_;
};
class PNGWriter {
public:
    PNGWriter(std::filesystem::path destination, std::int64_t width, std::int64_t height);
    void write_one_frame(const torch::Tensor& rgb);
    void close();
    std::filesystem::path dst_path;
    std::int64_t width, height, current_frame_index = 1;
    unsigned padding = 5;
};
class YUV420Writer {
public:
    YUV420Writer(std::filesystem::path destination, std::int64_t width, std::int64_t height);
    void write_one_frame(const torch::Tensor& y, const torch::Tensor& uv);
    void close();
    std::filesystem::path dst_path;
    std::int64_t width, height;
private:
    std::ofstream file_;
};
}  // namespace uf::video_io
