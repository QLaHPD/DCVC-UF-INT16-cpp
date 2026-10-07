// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#include "uf/training_datasets.hpp"
#include "uf/image_pixels.hpp"
#include "uf/process.hpp"
#include "uf/python_json.hpp"
#include "uf/utf8.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace uf::training {
namespace {

void validate_patch(std::int64_t width, std::int64_t height) {
    if (width < 0 || height < 0) throw std::invalid_argument("Negative training patch dimensions");
}

void validate_qp(std::int64_t count, const std::vector<double>& lambdas) {
    if (count <= 0 || count > std::numeric_limits<std::int32_t>::max() ||
        static_cast<std::uint64_t>(count) > lambdas.size()) {
        throw std::invalid_argument("Training requires a lambda for every int32 QP");
    }
}

std::vector<std::uint64_t> snapshot_lambdas(const std::vector<double>& values) {
    static_assert(sizeof(double) == sizeof(std::uint64_t) && std::numeric_limits<double>::is_iec559,
                  "Worker lambda encoding requires IEEE binary64");
    std::vector<std::uint64_t> bits(values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        std::memcpy(&bits[index], &values[index], sizeof(double));
    }
    return bits;
}

std::vector<double> restore_lambdas(const nlohmann::json& state) {
    const auto bits = state.at("lambda_bits").get<std::vector<std::uint64_t>>();
    std::vector<double> values(bits.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        std::memcpy(&values[index], &bits[index], sizeof(double));
    }
    return values;
}

std::vector<std::uint8_t> snapshot_root(const std::filesystem::path& root) {
    // POSIX path bytes need not be UTF-8. JSON string serialization would
    // reject such a root even though the parent's native reader accepts it.
    const auto bytes = root.string();
    return {bytes.begin(), bytes.end()};
}

std::filesystem::path restore_root(const nlohmann::json& state) {
    const auto bytes = state.at("root_bytes").get<std::vector<std::uint8_t>>();
    return std::string(bytes.begin(), bytes.end());
}

std::filesystem::path dataset_path(const std::filesystem::path& root,
                                   std::initializer_list<std::string_view> parts) {
    // Manifest entries are Python text, not native filename bytes. Join before
    // encoding so an absolute later component can discard the entire prefix,
    // including text that cannot be encoded in the filesystem locale.
    auto text = python_filesystem_decode(root.native());
    for (const auto part : parts) {
        if (!part.empty() && part.front() == '/') text.assign(part.data(), part.size());
        else {
            if (!text.empty() && text.back() != '/') text.push_back('/');
            if (!part.empty()) text.append(part.data(), part.size());
        }
    }
    return std::filesystem::path(python_filesystem_encode(text));
}

nlohmann::json description(const std::filesystem::path& root) {
    return read_python_json_file(dataset_path(root, {"description.json"}),
                                 python_process_text_encoding(), false);
}

std::size_t list_index(std::int64_t index, std::size_t count) {
    if (count > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
        throw std::overflow_error("Training dataset list exceeds the signed index range");
    }
    const auto length = static_cast<std::int64_t>(count);
    if (index < 0) {
        if (index < -length) throw std::out_of_range("Training dataset index out of range");
        index += length;
    }
    if (index >= length) throw std::out_of_range("Training dataset index out of range");
    return static_cast<std::size_t>(index);
}

std::int64_t draw(const RandomBelow& random, std::int64_t upper) {
    if (!random || upper <= 0) throw std::invalid_argument("Invalid training random draw");
    const auto value = random(static_cast<std::uint64_t>(upper));
    if (value >= static_cast<std::uint64_t>(upper)) {
        throw std::runtime_error("Training random stream returned an out-of-range integer");
    }
    return static_cast<std::int64_t>(value);
}

struct Crop {
    std::int64_t height, width, pad_height, pad_width, y, x;
    bool flip;
};

// ImageFolder draws the crop first, while VideoFolder draws the flip first.
Crop crop(std::int64_t height, std::int64_t width, std::int64_t patch_height,
          std::int64_t patch_width, const RandomBelow& random, bool video) {
    if (height < 0 || width < 0) throw std::invalid_argument("Negative training image dimensions");
    Crop result{patch_height, patch_width, std::max<std::int64_t>(0, patch_height - height),
                std::max<std::int64_t>(0, patch_width - width), 0, 0, false};
    if (video) result.flip = draw(random, 2) == 0;  // random.choice([True, False])
    const auto y_range = std::max(height, patch_height) - patch_height;
    const auto x_range = std::max(width, patch_width) - patch_width;
    if (y_range == std::numeric_limits<std::int64_t>::max() ||
        x_range == std::numeric_limits<std::int64_t>::max()) {
        throw std::overflow_error("Training crop draw exceeds the signed index range");
    }
    result.y = draw(random, y_range + 1);
    result.x = draw(random, x_range + 1);
    if (!video) result.flip = draw(random, 2) == 0;
    return result;
}

std::int64_t crop_extent(std::uint32_t actual, std::int64_t padding,
                         std::int64_t start, std::int64_t requested) {
    if (padding > std::numeric_limits<std::int64_t>::max() - actual) {
        throw std::overflow_error("Training padded image dimension overflow");
    }
    return std::min(requested, std::max<std::int64_t>(0, actual + padding - start));
}

torch::Tensor crop_to_yuv(const ImagePixels& image, const Crop& transform) {
    // Padding for VideoFolder comes from the manifest. Do not substitute the
    // decoded dimensions: NumPy slicing truncates when the metadata differs.
    const auto height = crop_extent(image.height, transform.pad_height, transform.y, transform.height);
    const auto width = crop_extent(image.width, transform.pad_width, transform.x, transform.width);
    if (height % 2 || width % 2) {
        throw std::invalid_argument("rgb2ycbcr_np requires even training crop dimensions");
    }
    if (height && width > std::numeric_limits<std::int64_t>::max() / height / 3) {
        throw std::overflow_error("Training crop tensor size overflow");
    }
    const auto plane = height * width;
    if (static_cast<std::uint64_t>(plane) >
        std::numeric_limits<std::size_t>::max() / (3 * sizeof(float))) {
        throw std::overflow_error("Training crop tensor exceeds addressable memory");
    }
    auto output = torch::empty({3, height, width}, torch::TensorOptions().dtype(torch::kFloat32));
    auto* pixels = output.data_ptr<float>();
    const auto top = transform.pad_height / 2, left = transform.pad_width / 2;
    for (std::int64_t y = 0; y < height; ++y) {
        ProcessInterrupt::check();
        const auto input_y = transform.y + y - top;
        for (std::int64_t x = 0; x < width; ++x) {
            const auto padded_x = transform.x + x - left;
            std::uint8_t red = 0, green = 0, blue = 0;
            if (input_y >= 0 && input_y < image.height && padded_x >= 0 && padded_x < image.width) {
                const auto input_x = transform.flip ? image.width - 1 - padded_x : padded_x;
                const auto offset = (static_cast<std::size_t>(input_y) * image.width + input_x) * 3;
                red = image.rgb[offset];
                green = image.rgb[offset + 1];
                blue = image.rgb[offset + 2];
            }
            // NumPy materializes each float32 ufunc result. Volatile intermediates
            // prevent fused expressions or extended-precision contractions here.
            volatile float r = static_cast<float>(red) / 255.0f;
            volatile float g = static_cast<float>(green) / 255.0f;
            volatile float b = static_cast<float>(blue) / 255.0f;
            volatile float yr = 0.2126f * r, yg = 0.7152f * g, yb = 0.0722f * b;
            volatile float yrg = yr + yg;
            volatile float luma = yrg + yb;
            volatile float cb_difference = b - luma, cr_difference = r - luma;
            volatile float cb_half = 0.5f * cb_difference, cr_half = 0.5f * cr_difference;
            volatile float cb_scaled = cb_half / static_cast<float>(1.0 - 0.0722);
            volatile float cr_scaled = cr_half / static_cast<float>(1.0 - 0.2126);
            volatile float cb = cb_scaled + 0.5f, cr = cr_scaled + 0.5f;
            const auto output_index = y * width + x;
            pixels[output_index] = std::clamp(static_cast<float>(luma), 0.0f, 1.0f) - 0.5f;
            pixels[plane + output_index] = std::clamp(static_cast<float>(cb), 0.0f, 1.0f) - 0.5f;
            pixels[2 * plane + output_index] = std::clamp(static_cast<float>(cr), 0.0f, 1.0f) - 0.5f;
        }
    }
    return output;
}

void append_qp(DatasetSample& sample, std::int64_t count, const std::vector<double>& lambdas,
               const RandomBelow& random) {
    const auto qp = draw(random, count);
    sample.push_back(torch::tensor(static_cast<std::int32_t>(qp),
                                  torch::TensorOptions().dtype(torch::kInt32)));
    sample.push_back(torch::tensor(static_cast<float>(lambdas.at(static_cast<std::size_t>(qp))),
                                  torch::TensorOptions().dtype(torch::kFloat32)));
}

}  // namespace

ImageFolder::ImageFolder(std::filesystem::path root, std::int64_t patch_height,
                         std::int64_t patch_width, std::int64_t qp_count,
                         std::vector<double> lambdas)
    : root_(std::move(root)), qp_count_(qp_count), lambdas_(std::move(lambdas)),
      patch_height_(patch_height), patch_width_(patch_width) {
    validate_patch(patch_width_, patch_height_);
    validate_qp(qp_count_, lambdas_);
    images_ = description(root_).get<std::vector<std::string>>();
}

ImageFolder::ImageFolder(const nlohmann::json& state)
    : root_(restore_root(state)),
      images_(state.at("description").get<std::vector<std::string>>()),
      qp_count_(state.at("qp_count").get<std::int64_t>()),
      lambdas_(restore_lambdas(state)),
      patch_height_(state.at("patch_height").get<std::int64_t>()),
      patch_width_(state.at("patch_width").get<std::int64_t>()) {
    if (state.at("kind") != "image") throw std::invalid_argument("Invalid image worker snapshot");
    validate_patch(patch_width_, patch_height_);
    validate_qp(qp_count_, lambdas_);
}

nlohmann::json ImageFolder::worker_snapshot() const {
    const std::lock_guard<std::mutex> lock(settings_mutex_);
    return {{"kind", "image"}, {"root_bytes", snapshot_root(root_)}, {"description", images_},
            {"qp_count", qp_count_}, {"lambda_bits", snapshot_lambdas(lambdas_)},
            {"patch_height", patch_height_}, {"patch_width", patch_width_}};
}

DatasetSample ImageFolder::get(std::int64_t index, const RandomBelow& random) const {
    ProcessInterrupt::check();
    const auto image = read_image_rgb(
        dataset_path(root_, {images_.at(list_index(index, images_.size()))}), false);
    const auto patch = patch_size();
    const auto transform = crop(image.height, image.width, patch.second, patch.first, random, false);
    DatasetSample output{crop_to_yuv(image, transform)};
    append_qp(output, qp_count_, lambdas_, random);
    return output;
}

std::size_t ImageFolder::size() const noexcept { return images_.size(); }

std::pair<std::int64_t, std::int64_t> ImageFolder::patch_size() const {
    const std::lock_guard<std::mutex> lock(settings_mutex_);
    return {patch_width_, patch_height_};
}

void ImageFolder::set_patch_size(std::int64_t width, std::int64_t height) {
    validate_patch(width, height);
    const std::lock_guard<std::mutex> lock(settings_mutex_);
    patch_width_ = width;
    patch_height_ = height;
}

VideoFolder::VideoFolder(std::filesystem::path root, std::int64_t patch_height,
                         std::int64_t patch_width, std::int64_t qp_count,
                         std::vector<double> lambdas, std::int64_t frame_count,
                         std::int64_t group_of_pictures)
    : root_(std::move(root)), qp_count_(qp_count), lambdas_(std::move(lambdas)),
      patch_height_(patch_height), patch_width_(patch_width), frame_count_(frame_count),
      group_of_pictures_(group_of_pictures) {
    validate_patch(patch_width_, patch_height_);
    validate_qp(qp_count_, lambdas_);
    if (frame_count_ <= 0 || group_of_pictures_ <= 0) {
        throw std::invalid_argument("Training frame count and picture group must be positive");
    }
    load_description(description(root_));
}

VideoFolder::VideoFolder(const nlohmann::json& state)
    : root_(restore_root(state)),
      qp_count_(state.at("qp_count").get<std::int64_t>()),
      lambdas_(restore_lambdas(state)),
      patch_height_(state.at("patch_height").get<std::int64_t>()),
      patch_width_(state.at("patch_width").get<std::int64_t>()),
      frame_count_(state.at("frame_count").get<std::int64_t>()),
      group_of_pictures_(state.at("group_of_pictures").get<std::int64_t>()) {
    if (state.at("kind") != "video") throw std::invalid_argument("Invalid video worker snapshot");
    validate_patch(patch_width_, patch_height_);
    validate_qp(qp_count_, lambdas_);
    if (frame_count_ <= 0 || group_of_pictures_ <= 0) {
        throw std::invalid_argument("Training frame count and picture group must be positive");
    }
    load_description(state.at("description"));
}

void VideoFolder::load_description(const nlohmann::json& manifest) {
    frames_ = manifest.at("frames").get<std::vector<std::string>>();
    const auto& sequences = manifest.at("seqs");
    if (!sequences.is_array()) throw std::invalid_argument("Training seqs must be an array");
    for (const auto& sequence : sequences) {
        Sequence entry{sequence.at("path").get<std::string>(),
                       python_json_index_int64(sequence.at("width")),
                       python_json_index_int64(sequence.at("height")),
                       python_json_index_int64(sequence.at("seq_length"))};
        if (entry.length < 0) throw std::invalid_argument("Negative training sequence length");
        sequences_.push_back(std::move(entry));
    }
}

nlohmann::json VideoFolder::worker_snapshot() const {
    const std::lock_guard<std::mutex> lock(settings_mutex_);
    auto sequences = nlohmann::json::array();
    for (const auto& sequence : sequences_) {
        sequences.push_back({{"path", sequence.path}, {"width", sequence.width},
                             {"height", sequence.height}, {"seq_length", sequence.length}});
    }
    return {{"kind", "video"}, {"root_bytes", snapshot_root(root_)},
            {"description", {{"frames", frames_}, {"seqs", std::move(sequences)}}},
            {"qp_count", qp_count_}, {"lambda_bits", snapshot_lambdas(lambdas_)},
            {"patch_height", patch_height_}, {"patch_width", patch_width_},
            {"frame_count", frame_count_}, {"group_of_pictures", group_of_pictures_}};
}

DatasetSample VideoFolder::get(std::int64_t index, const RandomBelow& random) const {
    ProcessInterrupt::check();
    const auto& sequence = sequences_.at(list_index(index, sequences_.size()));
    std::int64_t count, patch_height, patch_width;
    {
        const std::lock_guard<std::mutex> lock(settings_mutex_);
        count = frame_count_;
        patch_height = patch_height_;
        patch_width = patch_width_;
    }
    std::vector<std::int64_t> indexes;
    if (static_cast<std::uint64_t>(count) > indexes.max_size()) {
        throw std::overflow_error("Training frame count exceeds addressable memory");
    }
    indexes.reserve(static_cast<std::size_t>(count));
    if (count < sequence.length) {
        // The reference excludes the last possible window.
        const auto start = draw(random, sequence.length - count);
        for (std::int64_t offset = 0; offset < count; ++offset) indexes.push_back(start + offset);
    } else {
        bool increasing = true;
        std::int64_t frame = 0;
        while (static_cast<std::int64_t>(indexes.size()) < count) {
            ProcessInterrupt::check();
            indexes.push_back(frame);
            if (increasing) {
                if (frame == sequence.length - 1) { --frame; increasing = false; }
                else ++frame;
            } else {
                if (frame == 0) { ++frame; increasing = true; }
                else --frame;
            }
        }
    }
    const auto transform = crop(sequence.height, sequence.width, patch_height, patch_width, random, true);
    DatasetSample output;
    std::vector<torch::Tensor> group;
    for (const auto frame : indexes) {
        ProcessInterrupt::check();
        const auto& filename = frames_.at(list_index(frame, frames_.size()));
        auto image = crop_to_yuv(read_image_rgb(
            dataset_path(root_, {sequence.path, filename}), false), transform);
        if (output.empty()) output.push_back(std::move(image));
        else {
            group.push_back(std::move(image));
            if (static_cast<std::int64_t>(group.size()) == group_of_pictures_) {
                output.push_back(torch::cat(group, 0));
                group.clear();
            }
        }
    }
    // A partial final group is dropped, as in VideoFolder.__getitem__.
    append_qp(output, qp_count_, lambdas_, random);
    return output;
}

std::size_t VideoFolder::size() const noexcept { return sequences_.size(); }

std::int64_t VideoFolder::frame_count() const {
    const std::lock_guard<std::mutex> lock(settings_mutex_);
    return frame_count_;
}

std::pair<std::int64_t, std::int64_t> VideoFolder::patch_size() const {
    const std::lock_guard<std::mutex> lock(settings_mutex_);
    return {patch_width_, patch_height_};
}

void VideoFolder::set_frame_count(std::int64_t count) {
    if (count <= 0) throw std::invalid_argument("Training frame count must be positive");
    const std::lock_guard<std::mutex> lock(settings_mutex_);
    frame_count_ = count;
}

void VideoFolder::set_patch_size(std::int64_t width, std::int64_t height) {
    validate_patch(width, height);
    const std::lock_guard<std::mutex> lock(settings_mutex_);
    patch_width_ = width;
    patch_height_ = height;
}

DatasetSample collate_samples(const std::vector<DatasetSample>& samples) {
    if (samples.empty()) throw std::invalid_argument("Cannot collate an empty training batch");
    const auto fields = samples.front().size();
    for (const auto& sample : samples) {
        if (sample.size() != fields) throw std::invalid_argument("Training samples have different list lengths");
    }
    DatasetSample output;
    output.reserve(fields);
    std::vector<torch::Tensor> tensors;
    tensors.reserve(samples.size());
    for (std::size_t field = 0; field < fields; ++field) {
        ProcessInterrupt::check();
        tensors.clear();
        for (const auto& sample : samples) tensors.push_back(sample[field]);
        output.push_back(torch::stack(tensors, 0));
    }
    return output;
}

}  // namespace uf::training
