// Copyright (c) Microsoft Corporation. Licensed under the MIT License.
// Native DCVC-DC video_reader.py, video_writer.py and test_data_to_png.py.
#include "uf/dcvc_dc_media.hpp"
#include "uf/image_pixels.hpp"
#include "uf/cli.hpp"
#include "uf/process.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace uf::dc {
namespace {
bool suffix(const std::filesystem::path& path, const std::string& extension) {
    const auto value = path.string();
    return value.size() >= extension.size() && value.compare(value.size() - extension.size(), extension.size(), extension) == 0;
}
std::size_t byte_count(std::int64_t width, std::int64_t height, unsigned multiplier = 1) {
    if (width < 0 || height < 0) throw std::invalid_argument("Negative DC raw frame dimensions");
    const auto bound = std::min<std::uint64_t>(std::numeric_limits<std::size_t>::max(),
        static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max()));
    if (height && static_cast<std::uint64_t>(width) > bound / static_cast<std::uint64_t>(height))
        throw std::overflow_error("DC raw frame size exceeds native integers");
    const auto area = static_cast<std::uint64_t>(width) * static_cast<std::uint64_t>(height);
    if (area > bound / multiplier) throw std::overflow_error("DC raw frame size exceeds native integers");
    return static_cast<std::size_t>(area * multiplier);
}
unsigned maximum(int depth) {
    if (depth < 8 || depth > 16) throw std::invalid_argument("DC RGB bit depth must be between eight and sixteen");
    return (1u << depth) - 1;
}
std::string frame_number(std::int64_t index, unsigned padding) {
    auto number = std::to_string(index);
    if (number.size() < padding) number.insert(index < 0 ? 1 : 0, padding - number.size(), '0');
    return number;
}
void next_index(std::int64_t& index) {
    if (index == std::numeric_limits<std::int64_t>::max()) throw std::overflow_error("DC PNG index exceeds native integers");
    ++index;
}
std::vector<std::uint8_t> read_bytes(std::ifstream& file, std::size_t count) {
    if (!file.is_open()) throw std::runtime_error("Read from a closed DC raw frame file");
    if (!count) return {};
    std::vector<std::uint8_t> bytes(count);
    file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(count));
    if (file.bad()) throw std::runtime_error("Cannot read DC raw frame file");
    bytes.resize(static_cast<std::size_t>(file.gcount()));
    return bytes;
}
torch::Tensor byte_plane(const std::vector<std::uint8_t>& bytes, const std::vector<std::int64_t>& shape, unsigned max) {
    return torch::from_blob(const_cast<std::uint8_t*>(bytes.data()), shape,
        torch::TensorOptions().dtype(torch::kUInt8)).to(torch::kFloat32) / max;
}
torch::Tensor output_rgb(const MediaFrame& frame, const std::string& format) {
    return format == "420" ? ycbcr420_to_rgb(frame.y, frame.uv, 1) : frame.rgb;
}
std::vector<std::uint8_t> quantized_bytes(const torch::Tensor& input, unsigned max, bool wide = false) {
    // Multiplication and ties-to-even rounding precede clipping and conversion,
    // retaining Float32 arithmetic for Float32 source arrays.
    const auto values = (input.detach().to(torch::kCPU) * max).round().clamp(0, max).to(torch::kFloat64).contiguous();
    const auto count = static_cast<std::size_t>(values.numel());
    if (count > std::numeric_limits<std::size_t>::max() / (wide ? 2 : 1))
        throw std::overflow_error("DC raw output size exceeds native integers");
    std::vector<std::uint8_t> bytes(count * (wide ? 2 : 1));
    const auto* numbers = values.data_ptr<double>();
    for (std::size_t index = 0; index < count; ++index) {
        const auto number = std::isnan(numbers[index]) ? 0u : static_cast<unsigned>(numbers[index]);
        if (wide) {
            const auto sample = static_cast<std::uint16_t>(number);
            std::memcpy(bytes.data() + 2 * index, &sample, sizeof(sample));
        } else bytes[index] = static_cast<std::uint8_t>(number);
    }
    return bytes;
}
void write_bytes(std::ofstream& file, const std::vector<std::uint8_t>& bytes) {
    if (!file.is_open()) throw std::runtime_error("Write to a closed DC raw frame file");
    if (bytes.empty()) return;
    if (bytes.size() > static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max()))
        throw std::overflow_error("DC raw output exceeds native file sizes");
    file.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!file) throw std::runtime_error("Cannot write DC raw frame file");
}
void close_output(std::ofstream& file) {
    if (!file.is_open()) return;
    file.close();
    if (!file) throw std::runtime_error("Cannot close DC raw frame output");
}
}  // namespace

VideoReader::VideoReader(std::filesystem::path source, std::int64_t width_, std::int64_t height_)
    : src_path(std::move(source)), width(width_), height(height_) {}
MediaFrame VideoReader::none_exist_frame(const std::string& format) {
    if (format != "rgb" && format != "420") throw std::invalid_argument("DC frame format must be rgb or 420");
    return {};
}
MediaFrame VideoReader::get_dst_format(MediaFrame frame, const std::string& format) {
    if (format == "rgb") {
        if (!frame.rgb.defined()) frame.rgb = ycbcr420_to_rgb(frame.y, frame.uv, 1);
        return {frame.rgb, {}, {}};
    }
    if (format != "420") throw std::invalid_argument("DC frame format must be rgb or 420");
    if (!frame.y.defined()) {
        const auto planes = rgb_to_ycbcr420(frame.rgb); frame.y = planes.first; frame.uv = planes.second;
    }
    return {{}, frame.y, frame.uv};
}
PNGReader::PNGReader(std::filesystem::path source, std::int64_t width_, std::int64_t height_, std::int64_t start_num)
    : VideoReader(std::move(source), width_, height_), current_frame_index(start_num) {
    bool short_names = false, long_names = false;
    for (const auto& entry : std::filesystem::directory_iterator(src_path)) {
        short_names |= entry.path().filename() == "im1.png";
        long_names |= entry.path().filename() == "im00001.png";
    }
    if (!short_names && !long_names) throw std::invalid_argument("unknown image naming convention; please specify");
    padding = short_names ? 1 : 5;
}
MediaFrame PNGReader::read_one_frame(const std::string& format) {
    if (eof) return none_exist_frame(format);
    const auto path = src_path / ("im" + frame_number(current_frame_index, padding) + ".png");
    if (!std::filesystem::exists(path)) { eof = true; return none_exist_frame(format); }
    const auto pixels = read_image_rgb(path, false);
    const auto rgb = torch::from_blob(const_cast<std::uint8_t*>(pixels.rgb.data()),
        {pixels.height, pixels.width, 3}, torch::TensorOptions().dtype(torch::kUInt8))
        .to(torch::kFloat32).permute({2, 0, 1}) / 255;
    if (pixels.height != height || pixels.width != width) throw std::invalid_argument("DC PNG dimensions differ from configuration");
    next_index(current_frame_index);
    return get_dst_format({rgb, {}, {}}, format);
}
void PNGReader::close() { current_frame_index = 1; }
RGBReader::RGBReader(std::filesystem::path source, std::int64_t width_, std::int64_t height_, std::string format, int depth)
    : VideoReader(std::move(source), width_, height_), src_format(std::move(format)), bit_depth(depth),
      rgb_size_(byte_count(width, height, depth > 8 ? 6 : 3)), max_val_(maximum(depth)) {
    if (!suffix(src_path, ".rgb")) src_path += ".rgb";
    file_.open(src_path, std::ios::binary);
    if (!file_) throw std::runtime_error("Cannot open DC RGB input: " + src_path.string());
}
MediaFrame RGBReader::read_one_frame(const std::string& format) {
    if (eof) return none_exist_frame(format);
    const auto bytes = read_bytes(file_, rgb_size_);
    if (bytes.empty()) { eof = true; return none_exist_frame(format); }
    if (bytes.size() != rgb_size_) throw std::runtime_error("Incomplete DC raw RGB frame");
    torch::Tensor rgb;
    if (bit_depth == 8) rgb = byte_plane(bytes, {3, height, width}, max_val_);
    else {
        rgb = torch::empty({3, height, width}, torch::TensorOptions().dtype(torch::kFloat32));
        auto* values = rgb.data_ptr<float>();
        for (std::int64_t index = 0; index < rgb.numel(); ++index) {
            std::uint16_t sample; std::memcpy(&sample, bytes.data() + 2 * index, sizeof(sample));
            values[index] = static_cast<float>(sample) / static_cast<float>(max_val_);
        }
    }
    return get_dst_format({rgb, {}, {}}, format);
}
void RGBReader::close() { if (file_.is_open()) file_.close(); }
YUVReader::YUVReader(std::filesystem::path source, std::int64_t width_, std::int64_t height_, std::string format, std::int64_t skip)
    : VideoReader(std::move(source), width_, height_), src_format(std::move(format)),
      y_size_(byte_count(width, height)), uv_size_(y_size_ / 2) {
    if (src_format != "420") throw std::invalid_argument("DC YUVReader requires 420 input");
    if (!suffix(src_path, ".yuv")) src_path += ".yuv";
    file_.open(src_path, std::ios::binary);
    if (!file_) throw std::runtime_error("Cannot open DC YUV input: " + src_path.string());
    for (std::int64_t skipped = 0; !eof && skipped < skip; ++skipped) {
        const auto y = read_bytes(file_, y_size_), uv = read_bytes(file_, uv_size_);
        if (y.empty() || uv.empty()) eof = true;
    }
}
MediaFrame YUVReader::read_one_frame(const std::string& format) {
    if (eof) return none_exist_frame(format);
    const auto y = read_bytes(file_, y_size_), uv = read_bytes(file_, uv_size_);
    if (y.empty() || uv.empty()) { eof = true; return none_exist_frame(format); }
    if (y.size() != y_size_ || uv.size() != uv_size_ || uv.size() != byte_count(width / 2, height / 2, 2))
        throw std::runtime_error("Incomplete DC YUV frame or invalid chroma shape");
    return get_dst_format({{}, byte_plane(y, {1, height, width}, 255), byte_plane(uv, {2, height / 2, width / 2}, 255)}, format);
}
void YUVReader::close() { if (file_.is_open()) file_.close(); }

VideoWriter::VideoWriter(std::filesystem::path destination, std::int64_t width_, std::int64_t height_)
    : dst_path(std::move(destination)), width(width_), height(height_) {}
PNGWriter::PNGWriter(std::filesystem::path destination, std::int64_t width_, std::int64_t height_)
    : VideoWriter(std::move(destination), width_, height_) { std::filesystem::create_directories(dst_path); }
void PNGWriter::write_one_frame(const MediaFrame& frame, const std::string& format) {
    const auto rgb = output_rgb(frame, format);
    if (!rgb.defined() || rgb.dim() != 3 || rgb.size(0) != 3 || rgb.size(1) <= 0 || rgb.size(2) <= 0 ||
        rgb.size(1) > std::numeric_limits<std::uint32_t>::max() || rgb.size(2) > std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument("Invalid DC PNG RGB shape");
    const auto values = quantized_bytes(rgb.permute({1, 2, 0}), 255);
    ImagePixels pixels{static_cast<std::uint32_t>(rgb.size(2)), static_cast<std::uint32_t>(rgb.size(1)), values};
    write_image_png(dst_path / ("im" + frame_number(current_frame_index, padding) + ".png"), pixels);
    next_index(current_frame_index);
}
void PNGWriter::close() { current_frame_index = 1; }
RGBWriter::RGBWriter(std::filesystem::path destination, std::int64_t width_, std::int64_t height_, std::string format, int depth)
    : VideoWriter(std::move(destination), width_, height_), dst_format(std::move(format)), bit_depth(depth), max_val_(maximum(depth)) {
    if (!suffix(dst_path, ".rgb")) dst_path += "/out.rgb";
    file_.open(dst_path, std::ios::binary | std::ios::trunc);
    if (!file_) throw std::runtime_error("Cannot open DC RGB output: " + dst_path.string());
}
void RGBWriter::write_one_frame(const MediaFrame& frame, const std::string& format) {
    write_bytes(file_, quantized_bytes(output_rgb(frame, format), max_val_, bit_depth > 8));
}
void RGBWriter::close() { close_output(file_); }
YUVWriter::YUVWriter(std::filesystem::path destination, std::int64_t width_, std::int64_t height_, std::string format)
    : VideoWriter(std::move(destination), width_, height_), dst_format(std::move(format)) {
    if (dst_format != "420") throw std::invalid_argument("DC YUVWriter requires 420 output");
    if (!suffix(dst_path, ".yuv")) dst_path += "/out.yuv";
    file_.open(dst_path, std::ios::binary | std::ios::trunc);
    if (!file_) throw std::runtime_error("Cannot open DC YUV output: " + dst_path.string());
}
void YUVWriter::write_one_frame(const MediaFrame& frame, const std::string& format) {
    const auto planes = format == "rgb" ? rgb_to_ycbcr420(frame.rgb) : Yuv420{frame.y, frame.uv};
    write_bytes(file_, quantized_bytes(planes.first, 255));
    write_bytes(file_, quantized_bytes(planes.second, 255));
}
void YUVWriter::close() { close_output(file_); }
std::uint64_t convert_one_seq_to_png(const std::filesystem::path& source, std::int64_t width,
                                   std::int64_t height, const std::filesystem::path& destination) {
    YUVReader reader(source, width, height, "420"); PNGWriter writer(destination, width, height);
    auto frame = reader.read_one_frame("rgb"); std::uint64_t processed = 0;
    while (!reader.eof) {
        ProcessInterrupt::check(); writer.write_one_frame(frame, "rgb");
        if (processed == std::numeric_limits<std::uint64_t>::max()) throw std::overflow_error("DC conversion frame count exceeds native integers");
        ++processed; frame = reader.read_one_frame("rgb");
    }
    std::cout << source.string() << ' ' << processed << '\n';
    return processed;
}
int run_data_to_png_command(int argc, char** argv) {
    const auto options = cli::parse(argc, argv); const ProcessInterrupt interruption;
    const auto source = options.count("__input") ? options.at("__input") : "source_yuv_path";
    const auto destination = options.count("__output") ? options.at("__output") : "destination_png_path";
    const auto width = options.count("__width") ? cli::signed_integer(options.at("__width"), "width") : 1920;
    const auto height = options.count("__height") ? cli::signed_integer(options.at("__height"), "height") : 1080;
    convert_one_seq_to_png(source, width, height, destination);
    return 0;
}
}  // namespace uf::dc
