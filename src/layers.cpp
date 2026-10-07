#include "uf/layers.hpp"
#include "uf/int16_ops.hpp"
#include <ATen/TensorIndexing.h>
#include <ATen/ops/pixel_shuffle.h>
#include <ATen/ops/pixel_unshuffle.h>

#include <limits>
#include <stdexcept>

namespace uf {
namespace {

torch::Tensor cat(const std::vector<torch::Tensor>& values) {
    // FeatureAdaptorM passes its entire argument tuple to torch.cat, including
    // a single input; let the same operator own count/shape validation.
    return torch::cat(values, 1);
}

bool begins(const std::string& value, const std::string& prefix) {
    return value.rfind(prefix, 0) == 0;
}

}  // namespace

torch::Tensor pixel_shuffle(const torch::Tensor& value, std::int64_t factor) {
    // Python Layers calls F.pixel_shuffle, including its shape validation,
    // device dispatch and layout/allocation policy. Use the same operator.
    return at::pixel_shuffle(value, factor);
}

torch::Tensor pixel_unshuffle(const torch::Tensor& value, std::int64_t factor) {
    return at::pixel_unshuffle(value, factor);
}

Layers::Layers(const PreparedModel& prepared, torch::Device device) : variant_(prepared.variant) {
    for (const auto& [name, value] : prepared.tensors) {
        if (!begins(name, "tables.z_") && !begins(name, "tables.y_")) {
            values_.emplace(name, value.to(device));
        }
    }
}

const torch::Tensor& Layers::tensor(const std::string& name) const {
    const auto it = values_.find(name);
    if (it == values_.end()) throw std::runtime_error("Missing prepared UF tensor: " + name);
    return it->second;
}

const torch::Tensor& Layers::value(const std::string& name) const {
    return tensor(name);
}

bool Layers::has(const std::string& name) const {
    return values_.count(name) != 0;
}

bool Layers::has_child(const std::string& name) const {
    const auto prefix = name + ".";
    const auto it = values_.lower_bound(prefix);
    return it != values_.end() && begins(it->first, prefix);
}

torch::Tensor Layers::parameter(const std::string& name, std::int64_t qp) const {
    const auto& value = tensor(name);
    using at::indexing::None;
    using at::indexing::Slice;
    // Python slices clamp even negative/out-of-range QPs, and insert None
    // after two axes without discarding any trailing parameter dimensions.
    // At Int64 max, both positive endpoints already clamp to the row count.
    const auto end = qp == std::numeric_limits<std::int64_t>::max() ? qp : qp + 1;
    return value.index({Slice(qp, end), Slice(), None, None});
}

torch::Tensor Layers::run_conv(const std::string& name, const torch::Tensor& input) const {
    const auto& weight = tensor(name + ".weight");
    const auto bias = has(name + ".bias") ? tensor(name + ".bias") : torch::Tensor{};
    const auto padding = weight.size(2) == 3 ? 1 : 0;
    const auto stride = name == "encoder.down" || name == "enc.enc_2.6" ? 2 : 1;
    const auto groups = weight.size(1) == 1 && weight.size(0) == input.size(1)
                            ? input.size(1) : 1;
    return int16::conv2d(input, weight, bias, stride, stride, padding, padding, groups);
}

bool Layers::residual_shortcut(const std::string& name) const {
    // Only ResidualBlockUpsample/WithStride2 sets DepthConvBlock.shortcut.
    // HT-S and LD hyper blocks and the HT-S temporal block disable it.
    if (variant_ == "image" || variant_ == "htl") return true;
    return false;
}

torch::Tensor Layers::run_depth(const std::string& name, const torch::Tensor& input,
                                bool shortcut) const {
    auto x = has(name + ".adaptor.weight") ? run_conv(name + ".adaptor", input) : input;
    auto dc = run_conv(name + ".dc.0", x);
    dc = int16::lookup(dc, tensor("tables.wsilu"));
    dc = run_conv(name + ".dc.2", dc);
    dc = run_conv(name + ".dc.3", dc);
    auto out = int16::add({dc, x});
    auto ffn = run_conv(name + ".ffn.0", out);
    ffn = int16::wsilu4(ffn, tensor("tables.wsilu"));
    ffn = run_conv(name + ".ffn.2", ffn);
    out = int16::add({ffn, out});
    return shortcut ? int16::add({out, x}) : out;
}

torch::Tensor Layers::run_simple(const std::string& name, const torch::Tensor& input,
                                  bool shortcut) const {
    if (has(name + ".weight")) return run_conv(name, input);
    if (has(name + ".dc.0.weight")) return run_depth(name, input, shortcut);
    if (has(name + ".up.conv.0.weight") && has(name + ".conv.dc.0.weight")) {
        const auto up = uf::pixel_shuffle(run_conv(name + ".up.conv.0", input), 2);
        return run_depth(name + ".conv", up, residual_shortcut(name));
    }
    if (has(name + ".down.weight") && has(name + ".conv.dc.0.weight")) {
        const auto down = run_conv(name + ".down", uf::pixel_unshuffle(input, 2));
        return run_depth(name + ".conv", down, residual_shortcut(name));
    }
    if (has(name + ".conv.0.weight") && !has(name + ".conv.1.weight")) {
        // SubpelConv2x contains Conv2d followed by PixelShuffle(2).
        return uf::pixel_shuffle(run_conv(name + ".conv.0", input), 2);
    }
    if (has_child(name + ".0")) {
        auto value = input;
        for (int index = 0; has_child(name + "." + std::to_string(index)); ++index) {
            value = run_simple(name + "." + std::to_string(index), value);
        }
        return value;
    }
    throw std::runtime_error("Unsupported UF module path: " + name);
}

torch::Tensor Layers::run(const std::string& module,
                          const std::vector<torch::Tensor>& args) const {
    if (args.empty() || args[0].scalar_type() != torch::kInt16) {
        throw std::invalid_argument("UF integer layer requires an INT16 input");
    }
    const auto& x = args[0];
    if (module == "enc" && variant_ == "image") {
        if (args.size() != 2) throw std::invalid_argument("Image encoder needs its QP parameter");
        const auto first = run_simple("enc.enc_1", uf::pixel_unshuffle(x, 8));
        return run_simple("enc.enc_2", int16::multiply(first, args[1]));
    }
    if (module == "dec" && variant_ == "image") {
        if (args.size() != 2) throw std::invalid_argument("Image decoder needs its QP parameter");
        const auto first = run_simple("dec.dec_1", x);
        return uf::pixel_shuffle(run_simple("dec.dec_2", int16::multiply(first, args[1])), 8);
    }
    if (module == "encoder") {
        if (args.size() != 3) throw std::invalid_argument("Video encoder needs context and QP");
        auto out = run_simple("encoder.conv1", cat({uf::pixel_unshuffle(x, 8), args[1]}));
        if (has_child("encoder.conv2")) out = run_simple("encoder.conv2", out);
        return run_conv("encoder.down", int16::multiply(out, args[2]));
    }
    if (module == "decoder") {
        if (args.size() != 3) throw std::invalid_argument("Video decoder needs context and QP");
        const auto up = run_simple("decoder.up", x);
        auto out = run_simple("decoder.conv1", cat({up, args[1]}));
        if (has("decoder.conv2.weight")) out = run_conv("decoder.conv2", out);
        return int16::multiply(out, args[2]);
    }
    if (module == "feature_adaptor_m") {
        return run_simple(module + ".conv", cat(args));
    }
    if (module == "temporal_prior_encoder") {
        return run_simple(module + ".conv", args.size() == 2 ? int16::multiply(x, args[1]) : x);
    }
    if (module == "y_prior_fusion" && variant_ != "image" && args.size() > 1) {
        const auto other = args.size() == 3 ? int16::multiply(args[1], args[2]) : args.at(1);
        return run_simple(module + ".conv", cat({x, other}));
    }
    if (begins(module, "y_spatial_prior_adaptor_") && args.size() == 2) {
        return run_depth(module, cat(args), false);
    }
    if (module == "y_spatial_prior" && args.size() == 2) {
        return run_simple(module + ".conv", cat(args));
    }
    if (module == "hyper_enc" || module == "hyper_dec" ||
        module == "hyper_encoder" || module == "hyper_decoder" ||
        module == "feature_adaptor_i" || module == "feature_extractor" ||
        module == "y_spatial_prior" || module == "y_prior_fusion") {
        return run_simple(module + ".conv", args.size() > 1 ? cat(args) : x);
    }
    return run_simple(module, x);
}

std::vector<torch::Tensor> Layers::reconstruct(const torch::Tensor& feature,
                                                bool for_reset) const {
    if (variant_ == "image") {
        throw std::invalid_argument("Image model uses the dec module for reconstruction");
    }
    if (variant_ == "ld") {
        const auto output = run_conv("recon_head.head",
                                     run_simple("recon_head.conv", feature));
        return {for_reset ? output : uf::pixel_shuffle(output, 8)};
    }
    if (variant_ == "hts") {
        std::vector<torch::Tensor> frames;
        torch::Tensor shared;
        const auto first = for_reset ? 7 : 0;
        for (int index = first; index < 8; ++index) {
            if (index % 2 == 0 || for_reset) {
                shared = run_simple("recon_head.conv1." + std::to_string(index / 2), feature);
            }
            auto frame = run_simple("recon_head.conv2." + std::to_string(index), shared);
            frames.push_back(for_reset ? frame : uf::pixel_shuffle(frame, 8));
        }
        return frames;
    }
    std::vector<torch::Tensor> frames;
    const auto first = for_reset ? 7 : 0;
    for (int index = first; index < 8; ++index) {
        auto frame = run_simple("recon_head.conv." + std::to_string(index), feature);
        frames.push_back(for_reset ? frame : uf::pixel_shuffle(frame, 8));
    }
    return frames;
}

}  // namespace uf
