// Copyright (c) Microsoft Corporation. Licensed under the MIT License.
#include "uf/dcvc_dc_evaluation.hpp"
#include "uf/process.hpp"
#include "dcvc_dc_numpy.hpp"
#include <nlohmann/json.hpp>
#include <array>
#include <cmath>
#include <limits>
#include <iomanip>
#include <iostream>
#include <locale>
#include <sstream>
#include <stdexcept>

namespace uf::dc {
torch::Tensor np_image_to_tensor(const torch::Tensor& image) {
    if (!image.defined() || !image.device().is_cpu()) throw std::invalid_argument("Expected a defined DC CPU image array");
    return image.to(torch::kFloat32).unsqueeze(0);
}
double PSNR(const torch::Tensor& first, const torch::Tensor& second) {
    const auto mse = (first - second).pow(2).mean();
    return (20 * torch::log10(1 / torch::sqrt(mse))).item<double>();
}
double calc_psnr(const torch::Tensor& first, const torch::Tensor& second, double range) {
    const torch::NoGradGuard guard; ProcessInterrupt::check();
    const auto options = torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat64);
    const auto error = first.detach().to(options) - second.detach().to(options);
    const auto squared = error.square().contiguous();
    const auto mse = squared.numel() ? (0.0 + detail::numpy_pairwise_sum(squared.data_ptr<double>(), squared.numel())) / squared.numel()
                                    : std::numeric_limits<double>::quiet_NaN();
    // DC has no UF 99.9 cap. NaN also follows the source's false comparison.
    return mse > 1e-10 ? 10 * std::log10(range * range / mse) : 999.9;
}
bool str2bool(const std::string& value) {
    auto normalized = value;
    for (auto& character : normalized) if (character >= 'A' && character <= 'Z') character += 'a' - 'A';
    return normalized == "yes" || normalized == "y" || normalized == "true" || normalized == "t" || normalized == "1";
}
std::string scale_list_to_str(const std::vector<double>& scales) {
    std::ostringstream result; result.imbue(std::locale::classic()); result << std::fixed << std::setprecision(2);
    for (const auto scale : scales) result << scale << ' ';
    return result.str();
}
void create_folder(const std::filesystem::path& path, bool print) {
    if (!std::filesystem::exists(path)) {
        std::filesystem::create_directories(path);
        if (print) std::cout << "created folder: " << path.string() << '\n';
    }
}
namespace {
std::uint64_t product(std::uint64_t left, std::uint64_t right) {
    if (left && right > std::numeric_limits<std::uint64_t>::max() / left)
        throw std::overflow_error("DC evaluation pixel count exceeds native integers");
    return left * right;
}
struct Totals { std::int64_t count = 0; double bits = 0; std::array<double, 4> psnr{}, ssim{}; };
bool has_float(const Json& value) {
    if (value.is_number_float()) return true;
    if (value.is_array() || value.is_object()) for (const auto& child : value) if (has_float(child)) return true;
    return false;
}
void metric_averages(Json& result, const std::string& group, const Totals& totals, bool yuv) {
    result[group + "psnr"] = totals.count ? Json(totals.psnr[0] / totals.count) : Json(0);
    result[group + "msssim"] = totals.count ? Json(totals.ssim[0] / totals.count) : Json(0);
    if (!yuv) return;
    const std::array<const char*, 3> channels{{"y", "u", "v"}};
    for (std::size_t channel = 0; channel < 3; ++channel)
        result[group + "psnr_" + channels[channel]] = totals.count ? Json(totals.psnr[channel + 1] / totals.count) : Json(0);
    for (std::size_t channel = 0; channel < 3; ++channel)
        result[group + "msssim_" + channels[channel]] = totals.count ? Json(totals.ssim[channel + 1] / totals.count) : Json(0);
}
}  // namespace
void dump_json(const Json& value, std::ostream& output, int digits, int indent) {
    // common.py replaces json's float formatter with format(o, '.%df').
    // That formatter emits lowercase nan/inf and errors on a negative precision.
    if (digits < 0 && has_float(value)) throw std::invalid_argument("DC JSON float precision cannot be negative");
    std::ostringstream serialized; uf::evaluation::write_json(value, serialized, digits < 0 ? -1 : digits, indent);
    const auto text = serialized.str(); bool quoted = false, escaped = false;
    for (std::size_t index = 0; index < text.size();) {
        const auto character = text[index];
        if (quoted) {
            output.put(character); ++index;
            if (escaped) escaped = false;
            else if (character == '\\') escaped = true;
            else if (character == '"') quoted = false;
        } else if (character == '"') { quoted = true; output.put(character); ++index; }
        else if (text.compare(index, 3, "NaN") == 0) { output << "nan"; index += 3; }
        else if (text.compare(index, 8, "Infinity") == 0) { output << "inf"; index += 8; }
        else {
            output.put(character); ++index;
            if (indent < 0 && (character == ',' || character == ':')) output.put(' ');
        }
    }
    if (!output) throw std::runtime_error("Cannot write DC evaluation JSON");
}
Json generate_log_json(std::int64_t count, std::uint64_t pixels, double time,
    const std::vector<int>& types, const std::vector<double>& bits,
    const std::vector<FrameQuality>& qualities, bool verbose) {
    if (count <= 0 || !pixels || static_cast<std::uint64_t>(count) > types.size() ||
        static_cast<std::uint64_t>(count) > bits.size() || static_cast<std::uint64_t>(count) > qualities.size())
        throw std::invalid_argument("DC evaluation requires frame logs and nonzero pixels");
    const bool yuv = qualities[0].psnr.size() == 4;
    Totals intra, inter;
    for (std::int64_t index = 0; index < count; ++index) {
        const auto& quality = qualities[index];
        const auto channels = yuv ? 4u : 1u;
        if (quality.psnr.size() != channels || quality.msssim.size() != channels)
            throw std::invalid_argument("DC quality logs have inconsistent channel counts");
        auto& totals = types[index] == 0 ? intra : inter;
        ++totals.count; totals.bits += bits[index];
        for (std::size_t channel = 0; channel < channels; ++channel) {
            totals.psnr[channel] += quality.psnr[channel]; totals.ssim[channel] += quality.msssim[channel];
        }
    }
    if (!intra.count) throw std::invalid_argument("DC evaluation has no I frames");
    Json result{{"frame_pixel_num", pixels}, {"i_frame_num", intra.count}, {"p_frame_num", inter.count},
        {"ave_i_frame_bpp", intra.bits / intra.count / pixels}};
    metric_averages(result, "ave_i_frame_", intra, yuv);
    if (verbose) {
        for (const auto* name : {"frame_bpp", "frame_psnr", "frame_msssim", "frame_type"}) result[name] = Json::array();
        for (std::int64_t index = 0; index < count; ++index) {
            result["frame_bpp"].push_back(bits[index] / pixels); result["frame_psnr"].push_back(qualities[index].psnr[0]);
            result["frame_msssim"].push_back(qualities[index].msssim[0]); result["frame_type"].push_back(types[index]);
        }
        if (yuv) {
            const std::array<const char*, 3> channels{{"y", "u", "v"}};
            for (const auto* metric : {"psnr", "msssim"})
                for (std::size_t channel = 0; channel < 3; ++channel) {
                    const auto field = std::string("frame_") + metric + "_" + channels[channel]; result[field] = Json::array();
                    for (std::int64_t index = 0; index < count; ++index)
                        result[field].push_back(std::string(metric) == "psnr" ? qualities[index].psnr[channel + 1] : qualities[index].msssim[channel + 1]);
                }
        }
    }
    result["test_time"] = time;
    result["ave_p_frame_bpp"] = inter.count ? Json(inter.bits / product(inter.count, pixels)) : Json(0);
    metric_averages(result, "ave_p_frame_", inter, yuv);
    Totals all; all.count = count; all.bits = intra.bits + inter.bits;
    for (std::size_t channel = 0; channel < 4; ++channel) {
        all.psnr[channel] = intra.psnr[channel] + inter.psnr[channel]; all.ssim[channel] = intra.ssim[channel] + inter.ssim[channel];
    }
    result["ave_all_frame_bpp"] = all.bits / product(count, pixels);
    metric_averages(result, "ave_all_frame_", all, yuv);
    return result;
}
}  // namespace uf::dc
