#include "uf/integer_model.hpp"
#include "uf/int16_ops.hpp"
#include <ATen/TensorIndexing.h>
#include <ATen/core/grad_mode.h>
#include <ATen/ops/from_blob.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <utility>

namespace uf {
namespace {

std::vector<std::int32_t> int32_values(const torch::Tensor& value) {
    if (value.scalar_type() != torch::kInt32) {
        throw std::runtime_error("Prepared UF CDF is not INT32");
    }
    const auto tensor = value.to(torch::kCPU).contiguous();
    if (tensor.numel() == 0) return {};
    const auto* begin = tensor.data_ptr<std::int32_t>();
    return {begin, begin + tensor.numel()};
}

template <typename T>
torch::Tensor vector_tensor(std::vector<T>&& values,
                            const std::vector<std::int64_t>& shape,
                            torch::ScalarType dtype, const torch::Device& device) {
    const auto options = torch::TensorOptions().dtype(dtype);
    if (values.empty()) return torch::empty(shape, options.device(device));
    auto owner = std::make_shared<std::vector<T>>(std::move(values));
    auto* data = owner->data();
    // Views retain this storage owner on CPU. Device transfer is synchronous;
    // the CPU storage remains alive until that copy has returned.
    return torch::from_blob(data, shape,
        [owner = std::move(owner)](void*) mutable { owner.reset(); }, options).to(device);
}

std::vector<std::int8_t> int8_values(const torch::Tensor& value) {
    const auto tensor = value.to(torch::kCPU).contiguous();
    if (tensor.numel() == 0) return {};
    const auto* begin = tensor.data_ptr<std::int8_t>();
    return {begin, begin + tensor.numel()};
}

std::vector<std::uint8_t> uint8_values(const torch::Tensor& value) {
    const auto tensor = value.to(torch::kCPU).contiguous();
    if (tensor.numel() == 0) return {};
    const auto* begin = tensor.data_ptr<std::uint8_t>();
    return {begin, begin + tensor.numel()};
}

bool is_image(const std::string& variant) { return variant == "image"; }

}  // namespace

torch::Tensor pad_right_bottom(const torch::Tensor& value, std::int64_t multiple) {
    auto out = value;
    const auto height = out.size(-2), width = out.size(-1);
    if (multiple == 0) throw std::invalid_argument("Integer modulo by zero in UF padding");
    const auto padding = [multiple](std::int64_t extent) {
        // Tensor extents are nonnegative, so -extent cannot overflow. Unlike
        // multiple-extent%multiple this also handles a negative Int64 factor.
        const auto remainder = (-extent) % multiple;
        return remainder < 0 && multiple > 0 ? remainder + multiple : remainder;
    };
    const auto bottom = padding(height), right = padding(width);
    using at::indexing::None;
    using at::indexing::Slice;
    if (bottom) {
        out = torch::cat({out, out.index({Slice(), Slice(), Slice(-1, None), Slice()})
                                   .expand({-1, -1, bottom, -1})}, 2);
    }
    if (right) {
        out = torch::cat({out, out.index({Slice(), Slice(), Slice(), Slice(-1, None)})
                                   .expand({-1, -1, -1, right})}, 3);
    }
    return out;
}

IntegerModel::IntegerModel(const PreparedModel& prepared, torch::Device device,
                           bool reconstruct_on_encode)
    : variant_(prepared.variant), device_(std::move(device)), layers_(prepared, device_),
      reconstruct_on_encode_(reconstruct_on_encode), z_channels_(128) {
    if (prepared.variant != "image" && prepared.variant != "hts" &&
        prepared.variant != "htl" && prepared.variant != "ld") {
        throw std::invalid_argument("Unsupported UF integer model variant");
    }
    for (int index = 0; index != 2; ++index) {
        const std::string table = index == 0 ? "z" : "y";
        const auto cdf = int32_values(prepared.tensor("tables." + table + "_cdf"));
        const auto lengths = int32_values(prepared.tensor("tables." + table + "_lengths"));
        encoder_.set_cdf(index, cdf, lengths);
        decoder_.set_cdf(index, cdf, lengths);
    }
    clear_dpb();
}

void IntegerModel::clear_dpb() {
    reference_ = torch::Tensor{};
    memory_ = torch::Tensor{};
    context_ = torch::Tensor{};
}

void IntegerModel::add_ref_feature_from_frame(const torch::Tensor& frame,
                                             bool apply_feature_adaptor) {
    // The maintained INT16 Python implementation accepts but ignores this
    // common FP16/INT16 interface argument.
    (void)apply_feature_adaptor;
    clear_dpb();
    reference_ = uf::pixel_unshuffle(frame, 8);
}

void IntegerModel::set_skip_threshold(double threshold) {
    // Python checks int(float(value) * FEATURE_SCALE + 0.5), rather than the
    // unrounded input. int() truncates toward zero, so small negative values
    // may produce zero and values above 32767/512 may still produce 32767.
    const volatile double scaled = threshold * int16::feature_scale;
    const auto rounded = std::trunc(scaled + 0.5);
    if (!std::isfinite(rounded) || rounded < 0 || rounded > 32767) {
        throw std::invalid_argument("UF skip threshold is outside the INT16 feature range");
    }
    skip_threshold_ = static_cast<int>(rounded);
}

void IntegerModel::update_context() {
    if (!reference_.defined()) throw std::runtime_error("UF P-frame has no reference");
    memory_ = memory_.defined()
        ? layers_.run("feature_adaptor_m", {memory_, reference_})
        : layers_.run("feature_adaptor_i", {reference_});
    context_ = layers_.run("feature_extractor", {memory_});
}

void IntegerModel::set_positions(const torch::Tensor& scales) {
    std::vector<std::int64_t> shape(scales.sizes().begin(), scales.sizes().end());
    if (shape == position_shape_) return;
    if (shape.size() != 4 ||
        shape[1] % (variant_ == "ld" ? 2 : 4)) {
        throw std::invalid_argument("Unsupported UF spatial prior shape");
    }
    const auto channels = shape[1], height = shape[2], width = shape[3];
    const int parts = variant_ == "ld" ? 2 : 4;
    std::vector<std::vector<std::int64_t>> flat(parts);
    const std::array<std::array<int, 4>, 4> four = {{{0, 1, 2, 3},
                                                     {3, 2, 1, 0},
                                                     {2, 3, 0, 1},
                                                     {1, 0, 3, 2}}};
    // Python flattens masks containing every batch in NCHW order. A running
    // index avoids intermediate dimension products and parity sums; no walk
    // is needed when any extent is zero.
    if (scales.numel() != 0) {
        std::int64_t index = 0;
        for (std::int64_t batch = 0; batch < shape[0]; ++batch) {
            for (std::int64_t c = 0; c < channels; ++c) {
                const auto group = c / (channels / parts);
                for (std::int64_t y = 0; y < height; ++y) {
                    for (std::int64_t x = 0; x < width; ++x, ++index) {
                        const int micro = static_cast<int>((y & 1) * 2 + (x & 1));
                        int part = 0;
                        if (parts == 2) {
                            part = static_cast<int>((group ^ (y & 1) ^ (x & 1)) & 1);
                        } else {
                            for (; part < 4 && four[part][group] != micro; ++part) {}
                        }
                        flat[part].push_back(index);
                    }
                }
            }
        }
    }
    std::vector<torch::Tensor> positions;
    positions.reserve(parts);
    for (auto& values : flat) {
        const auto count = static_cast<std::int64_t>(values.size());
        positions.push_back(vector_tensor(std::move(values), {count}, torch::kInt64, device_));
    }
    // _mask_positions builds the tuple before replacing its previous cache.
    positions_ = std::move(positions);
    position_shape_ = std::move(shape);
}

torch::Tensor IntegerModel::params(const torch::Tensor& z, int qp,
                                    std::int64_t height, std::int64_t width) {
    if (is_image(variant_)) {
        const auto hyper = layers_.run("hyper_dec", {int16::from_symbols(z)});
        return layers_.run("y_prior_fusion", {hyper})
            .slice(2, 0, height).slice(3, 0, width);
    }
    const auto hyper = layers_.run("hyper_decoder", {int16::from_symbols(z)})
                           .slice(2, 0, height).slice(3, 0, width);
    const auto quant = layers_.parameter("q_feature", qp);
    if (variant_ == "ld") {
        const auto temporal = layers_.run("temporal_prior_encoder", {memory_});
        return layers_.run("y_prior_fusion", {hyper, temporal, quant});
    }
    const auto temporal = layers_.run("temporal_prior_encoder", {memory_, quant});
    return layers_.run("y_prior_fusion", {hyper, temporal});
}

std::pair<torch::Tensor, std::vector<std::vector<std::int16_t>>>
IntegerModel::prior(const torch::Tensor& y, const torch::Tensor& parameters,
                     int qp, bool encode) {
    const bool image = is_image(variant_);
    const auto pieces = parameters.chunk(image ? 2 : 3, 1);
    auto scales = pieces[image ? 0 : 1];
    auto means = pieces[image ? 1 : 2];
    const auto q_dec = image ? layers_.parameter("q_scale_y_dec", qp)
                             : pieces[0].clamp_min(int16::feature_scale / 2);
    const auto q_enc = image ? layers_.parameter("q_scale_y_enc", qp)
                             : int16::reciprocal(q_dec);
    const auto encoded_y = encode ? int16::multiply(y, q_enc) : torch::Tensor{};
    set_positions(scales);
    const auto reduction = variant_ == "ld" ? torch::Tensor{}
        : layers_.run("y_spatial_prior_reduction", {parameters});
    auto restored = torch::zeros(scales.sizes(), scales.options().dtype(torch::kInt16));
    std::vector<std::vector<std::int16_t>> pending;
    for (std::size_t part = 0; part < positions_.size(); ++part) {
        if (part) {
            if (variant_ == "ld") {
                means = layers_.run("y_spatial_prior", {restored, parameters});
            } else {
                const auto adaptor = "y_spatial_prior_adaptor_" + std::to_string(part);
                const auto adapted = variant_ == "hts"
                    ? layers_.run(adaptor, {restored, reduction})
                    : layers_.run(adaptor, {torch::cat({restored, reduction}, 1)});
                const auto next = layers_.run("y_spatial_prior", {adapted});
                if (variant_ == "hts") means = next;
                else {
                    const auto split = next.chunk(2, 1);
                    scales = split[0];
                    means = split[1];
                }
            }
        }
        const auto& position = positions_[part];
        const auto chosen_scales = scales.reshape({-1}).index_select(0, position);
        const auto chosen_means = means.reshape({-1}).index_select(0, position).to(torch::kInt64);
        const auto use_mask = skip_threshold_ != 0;
        torch::Tensor indexes, symbols, coded_symbols;
        if (use_mask) {
            const auto active = chosen_scales > skip_threshold_;
            const bool has_active = active.any().item<bool>();
            indexes = has_active
                ? int16::lookup(chosen_scales.index({active}), layers_.value("tables.scale_index"))
                : torch::empty({0}, torch::TensorOptions().dtype(torch::kUInt8).device(device_));
            symbols = torch::zeros({position.numel()},
                                   torch::TensorOptions().dtype(torch::kInt8).device(device_));
            if (encode) {
                const auto residual = encoded_y.reshape({-1}).index_select(0, position)
                                          .to(torch::kInt64) - chosen_means;
                const auto calculated = int16::divide_round(residual.index({active}), int16::feature_scale)
                                            .clamp(-128, 127).to(torch::kInt8);
                symbols.index_put_({active}, calculated);
                coded_symbols = symbols.index({active});
            } else if (has_active) {
                auto decoded = decoder_.decode_y(uint8_values(indexes));
                const auto count = static_cast<std::int64_t>(decoded.size());
                const auto values = vector_tensor(std::move(decoded), {count}, torch::kInt8, device_);
                symbols.index_put_({active}, values);
            }
        } else {
            indexes = int16::lookup(chosen_scales, layers_.value("tables.scale_index"));
            if (encode) {
                const auto residual = encoded_y.reshape({-1}).index_select(0, position)
                                          .to(torch::kInt64) - chosen_means;
                symbols = int16::divide_round(residual, int16::feature_scale)
                              .clamp(-128, 127).to(torch::kInt8);
                coded_symbols = symbols;
            } else {
                auto decoded = decoder_.decode_y(uint8_values(indexes));
                const auto count = static_cast<std::int64_t>(decoded.size());
                symbols = vector_tensor(std::move(decoded), {count}, torch::kInt8, device_);
            }
        }
        if (encode) {
            const auto symbol_bytes = int8_values(coded_symbols);
            const auto index_bytes = uint8_values(indexes);
            std::vector<std::int16_t> packed(symbol_bytes.size());
            for (std::size_t i = 0; i < packed.size(); ++i) {
                // Both operands have byte bounds, so this signed value fits
                // Int16 without an implementation-defined unsigned narrowing.
                packed[i] = static_cast<std::int16_t>(
                    static_cast<std::int32_t>(symbol_bytes[i]) * 256 + index_bytes[i]);
            }
            pending.push_back(std::move(packed));
        }
        restored.reshape({-1}).index_copy_(0, position,
            int16::saturate(symbols.to(torch::kInt64) * int16::feature_scale + chosen_means));
    }
    return {int16::multiply(restored, q_dec), std::move(pending)};
}

std::vector<torch::Tensor> IntegerModel::reconstruct(const torch::Tensor& y,
                                                       int qp, bool reset,
                                                       bool output_pixels) {
    if (is_image(variant_)) {
        return {layers_.run("dec", {y, layers_.parameter("q_scale_dec", qp)})};
    }
    const auto feature = layers_.run("decoder", {y, context_, layers_.parameter("q_decoder", qp)});
    auto frames = output_pixels ? layers_.reconstruct(feature) : std::vector<torch::Tensor>{};
    reference_ = feature;
    if (reset) {
        const auto reset_feature = layers_.reconstruct(feature, true).at(0);
        clear_dpb();
        reference_ = reset_feature;
    }
    return frames;
}

CompressedFrame IntegerModel::compress(const torch::Tensor& frames, int qp, bool reset) {
    at::NoGradGuard no_grad;
    if (qp < 0 || qp >= 64) throw std::invalid_argument("UF QP must be between 0 and 63");
    const bool image = is_image(variant_);
    const auto x = pad_right_bottom(frames, 16);
    if (!image) update_context();
    const auto y = image
        ? layers_.run("enc", {x, layers_.parameter("q_scale_enc", qp)})
        : layers_.run("encoder", {x, context_, layers_.parameter("q_encoder", qp)});
    const auto z = int16::symbols(layers_.run(image ? "hyper_enc" : "hyper_encoder",
                                             {pad_right_bottom(y, 4)}));
    const auto parameters = params(z, qp, y.size(2), y.size(3));
    auto [y_hat, parts] = prior(y, parameters, qp, true);
    encoder_.reset();
    for (auto it = parts.rbegin(); it != parts.rend(); ++it) {
        if (!it->empty()) encoder_.encode_y(*it);
    }
    const auto z_nhwc = z.permute({0, 2, 3, 1}).contiguous();
    encoder_.encode_z(int8_values(z_nhwc), qp * z_channels_, z_channels_);
    auto payload = encoder_.flush();
    auto reconstructed = reconstruct(y_hat, qp, reset, reconstruct_on_encode_);
    return {std::move(payload), std::move(reconstructed)};
}

std::vector<torch::Tensor> IntegerModel::decompress(
    const std::vector<std::uint8_t>& payload, std::uint32_t width,
    std::uint32_t height, int qp, int entropy_parallel, bool reset) {
    at::NoGradGuard no_grad;
    if (entropy_parallel != 1 || qp < 0 || qp >= 64 || !width || !height) {
        throw std::invalid_argument("Unsupported UF integer packet parameters");
    }
    if (!is_image(variant_)) update_context();
    const auto y_height = (static_cast<std::int64_t>(height) + 15) / 16;
    const auto y_width = (static_cast<std::int64_t>(width) + 15) / 16;
    const auto z_height = (y_height + 3) / 4;
    const auto z_width = (y_width + 3) / 4;
    decoder_.set_stream(payload);
    auto symbols = decoder_.decode_z(z_height * z_width * z_channels_,
                                            qp * z_channels_, z_channels_);
    // Python reshapes and permutes the CPU decoder result before transferring
    // it. Preserve that view layout instead of forcing an NCHW copy here.
    const auto z = vector_tensor(std::move(symbols), {1, z_height, z_width, z_channels_},
                                torch::kInt8, torch::kCPU)
                       .permute({0, 3, 1, 2}).to(device_);
    const auto parameters = params(z, qp, y_height, y_width);
    const auto prior_result = prior({}, parameters, qp, false);
    return reconstruct(prior_result.first, qp, reset, true);
}

}  // namespace uf
