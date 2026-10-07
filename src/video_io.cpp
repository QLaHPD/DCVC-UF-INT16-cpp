// Copyright (c) Microsoft Corporation. Licensed under the MIT License.
#include "uf/video_io.hpp"
#include "uf/image_pixels.hpp"
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace uf::video_io {
namespace {
bool yuv_suffix(const std::filesystem::path& path) {
    const auto name = path.string();
    return name.size() >= 4 && name.compare(name.size() - 4, 4, ".yuv") == 0;
}
std::size_t area(std::int64_t width, std::int64_t height) {
    if (width < 0 || height < 0) throw std::invalid_argument("Negative raw frame dimensions");
    const auto maximum = std::min<std::uint64_t>(std::numeric_limits<std::size_t>::max(), std::numeric_limits<std::streamsize>::max());
    if (height && static_cast<std::uint64_t>(width) > maximum / static_cast<std::uint64_t>(height))
        throw std::overflow_error("Raw frame byte count exceeds native integers");
    return static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
}
std::string frame_name(std::int64_t index, unsigned padding) {
    auto number = std::to_string(index);
    if (number.size() < padding) number.insert(index < 0 ? 1 : 0, padding - number.size(), '0');
    return "im" + number + ".png";
}
void increment(std::int64_t& index) {
    if (index == std::numeric_limits<std::int64_t>::max()) throw std::overflow_error("PNG frame index exceeds int64");
    ++index;
}
std::vector<std::uint8_t> read(std::ifstream& file, std::size_t count) {
    if (!file.is_open()) throw std::runtime_error("Read from a closed raw video file");
    std::vector<std::uint8_t> bytes(count);
    if (count) {
        file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(count));
        if (file.bad()) throw std::runtime_error("Cannot read raw video file");
        bytes.resize(static_cast<std::size_t>(file.gcount()));
    }
    return bytes;
}
torch::Tensor byte_array(std::vector<std::uint8_t>& bytes, const std::vector<std::int64_t>& shape) {
    return torch::from_blob(bytes.data(), shape, torch::TensorOptions().device(torch::kCPU).dtype(torch::kUInt8)).clone();
}
void write(std::ofstream& file, const torch::Tensor& value) {
    if (!file.is_open()) throw std::runtime_error("Write to a closed raw video file");
    if (!value.defined() || !value.device().is_cpu()) throw std::invalid_argument("Raw video array requires a CPU tensor");
    const auto data = value.detach().contiguous();
    if (static_cast<std::uint64_t>(data.numel()) > static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max()) / data.element_size())
        throw std::overflow_error("Raw video array exceeds streamsize");
    const auto count = static_cast<std::streamsize>(data.numel()) * static_cast<std::streamsize>(data.element_size());
    if (count) file.write(static_cast<const char*>(data.data_ptr()), count);
    if (!file) throw std::runtime_error("Cannot write raw video file");
}
}
PNGReader::PNGReader(std::filesystem::path source, std::int64_t frame_width, std::int64_t frame_height, std::int64_t start)
    : src_path(std::move(source)), width(frame_width), height(frame_height), current_frame_index(start), padding(0) {
    bool short_name = false, long_name = false;
    for (const auto& entry : std::filesystem::directory_iterator(src_path)) {
        short_name |= entry.path().filename() == "im1.png";
        long_name |= entry.path().filename() == "im00001.png";
    }
    if (!short_name && !long_name) throw std::invalid_argument("unknown image naming convention; please specify");
    padding = short_name ? 1 : 5;
}
torch::Tensor PNGReader::read_one_frame() {
    if (eof) return {};
    const auto path = src_path / frame_name(current_frame_index, padding);
    if (!std::filesystem::exists(path)) { eof = true; return {}; }
    const auto image = read_image_rgb(path, false);
    if (image.width != width || image.height != height) throw std::invalid_argument("PNG dimensions differ from configured frame size");
    auto result = torch::from_blob(const_cast<std::uint8_t*>(image.rgb.data()),
        {static_cast<std::int64_t>(image.height), static_cast<std::int64_t>(image.width), 3},
        torch::TensorOptions().device(torch::kCPU).dtype(torch::kUInt8)).permute({2, 0, 1}).clone();
    increment(current_frame_index); return result;
}
void PNGReader::close() { current_frame_index = 1; }
YUV420Reader::YUV420Reader(std::filesystem::path source, std::int64_t width, std::int64_t height, std::int64_t skip)
    : src_path(std::move(source)), y_width(width), y_height(height), uv_width(width / 2), uv_height(height / 2),
      y_size_(area(width, height)), uv_size_(y_size_ / 2) {
    if (!yuv_suffix(src_path)) src_path = src_path.string() + ".yuv";
    file_.open(src_path, std::ios::binary);
    if (!file_) throw std::runtime_error("Cannot open raw YUV source: " + src_path.string());
    for (std::int64_t index = 0; !eof && index < skip; ++index) {
        const auto y = read(file_, y_size_), uv = read(file_, uv_size_);
        if (y.empty() || uv.empty()) eof = true;
    }
}
std::pair<torch::Tensor, torch::Tensor> YUV420Reader::read_one_frame() {
    if (eof) return {};
    auto y = read(file_, y_size_), uv = read(file_, uv_size_);
    if (y.empty() || uv.empty()) { eof = true; return {}; }
    if (y.size() != y_size_ || uv.size() != 2 * area(uv_width, uv_height))
        throw std::invalid_argument("Cannot reshape truncated or odd-dimension YUV frame");
    return {byte_array(y, {1, y_height, y_width}), byte_array(uv, {2, uv_height, uv_width})};
}
void YUV420Reader::close() {
    if (!file_.is_open()) return;
    // A prior EOF belongs to reading; only the close operation determines
    // whether explicit cleanup succeeds.
    file_.clear();
    file_.close();
    if (!file_) throw std::runtime_error("Cannot close raw YUV source");
}
PNGWriter::PNGWriter(std::filesystem::path destination, std::int64_t frame_width, std::int64_t frame_height)
    : dst_path(std::move(destination)), width(frame_width), height(frame_height) {
    std::filesystem::create_directories(dst_path);
}
void PNGWriter::write_one_frame(const torch::Tensor& rgb) {
    if (!rgb.device().is_cpu() || rgb.scalar_type() != torch::kUInt8 || rgb.dim() != 3 || rgb.size(0) != 3 ||
        rgb.size(1) <= 0 || rgb.size(2) <= 0 || rgb.size(1) > std::numeric_limits<std::uint32_t>::max() || rgb.size(2) > std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument("PNG writer requires a CPU UInt8 3xHxW array");
    const auto pixels = rgb.permute({1, 2, 0}).contiguous();
    ImagePixels image{static_cast<std::uint32_t>(rgb.size(2)), static_cast<std::uint32_t>(rgb.size(1)), {}};
    image.rgb.assign(pixels.data_ptr<std::uint8_t>(), pixels.data_ptr<std::uint8_t>() + pixels.numel());
    write_image_png(dst_path / frame_name(current_frame_index, padding), image);
    increment(current_frame_index);
}
void PNGWriter::close() { current_frame_index = 1; }
YUV420Writer::YUV420Writer(std::filesystem::path destination, std::int64_t frame_width, std::int64_t frame_height)
    : dst_path(std::move(destination)), width(frame_width), height(frame_height) {
    if (!yuv_suffix(dst_path)) dst_path = std::filesystem::path(dst_path.string() + "/out.yuv");
    file_.open(dst_path, std::ios::binary | std::ios::trunc);
    if (!file_) throw std::runtime_error("Cannot open raw YUV destination: " + dst_path.string());
}
void YUV420Writer::write_one_frame(const torch::Tensor& y, const torch::Tensor& uv) { write(file_, y); write(file_, uv); }
void YUV420Writer::close() {
    if (!file_.is_open()) return;
    file_.close(); if (!file_) throw std::runtime_error("Cannot close raw YUV destination");
}
}  // namespace uf::video_io
