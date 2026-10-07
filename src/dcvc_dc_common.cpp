// Copyright (c) Microsoft Corporation. Licensed under the MIT License.
#include "uf/dcvc_dc.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>

namespace uf::dc {
namespace {
constexpr std::array<std::array<int, 4>, 4> order{{{{0, 1, 2, 3}}, {{3, 2, 1, 0}},
                                                 {{2, 3, 0, 1}}, {{1, 0, 3, 2}}}};
FourParts four(const std::vector<torch::Tensor>& parts, std::size_t offset = 0) {
    return {parts.at(offset), parts.at(offset + 1), parts.at(offset + 2), parts.at(offset + 3)};
}
torch::Tensor sum_four(const FourParts& parts) {
    return ((parts[0] + parts[1]) + parts[2]) + parts[3];
}
}  // namespace
CompressionModelImpl::CompressionModelImpl(std::string selected, std::int64_t z,
    std::optional<std::int64_t> mv_z, bool threaded, int stream_parts)
    : y_distribution(std::move(selected)), z_channel(z), mv_z_channel(mv_z), ec_thread(threaded),
      stream_part(stream_parts), gaussian_encoder(y_distribution) {
    bit_estimator_z = register_module("bit_estimator_z", BitEstimator(z_channel));
    if (mv_z_channel) bit_estimator_z_mv = register_module("bit_estimator_z_mv", BitEstimator(*mv_z_channel));
}
torch::Tensor CompressionModelImpl::forward(const torch::Tensor&) {
    throw std::logic_error("DC CompressionModel has no forward implementation");
}
torch::Tensor CompressionModelImpl::quant(const torch::Tensor& input) { return input.round(); }
torch::Tensor CompressionModelImpl::get_curr_q(const torch::Tensor& scale, const torch::Tensor& basic,
                                             std::optional<std::int64_t> index) {
    if (!scale.defined()) throw std::runtime_error("DC fine quality scales have not been prepared by state loading");
    const auto selected = index ? scale[*index] : scale.unsqueeze(0);
    return basic * selected;
}
torch::Tensor CompressionModelImpl::probs_to_bits(const torch::Tensor& probabilities) {
    return (-1.0 * (probabilities + 1e-5).log() / std::log(2.0)).clamp_min(0);
}
torch::Tensor CompressionModelImpl::get_y_gaussian_bits(const torch::Tensor& y, const torch::Tensor& sigma) {
    const auto scales = sigma.clamp(1e-5, 1e10);
    return probs_to_bits(normal_cdf(y + 0.5, scales) - normal_cdf(y - 0.5, scales));
}
torch::Tensor CompressionModelImpl::get_y_laplace_bits(const torch::Tensor& y, const torch::Tensor& sigma) {
    const auto scales = sigma.clamp(1e-5, 1e10);
    if (!(scales > 0).all().item<bool>() || !(y == y).all().item<bool>())
        throw std::invalid_argument("Invalid DC Laplace distribution parameter/sample");
    return probs_to_bits(dcvc2021::laplace_cdf(y + 0.5, scales) - dcvc2021::laplace_cdf(y - 0.5, scales));
}
torch::Tensor CompressionModelImpl::get_z_bits(const torch::Tensor& z, BitEstimator& estimator) {
    return probs_to_bits(estimator->get_cdf(z + 0.5) - estimator->get_cdf(z - 0.5));
}
void CompressionModelImpl::update(bool force) {
    entropy_coder = std::make_shared<EntropyCoder>(ec_thread, stream_part);
    gaussian_encoder.update(force, entropy_coder);
    bit_estimator_z->update(force, entropy_coder);
    if (bit_estimator_z_mv) bit_estimator_z_mv->update(force, entropy_coder);
}
SliceShape CompressionModelImpl::get_to_y_slice_shape(std::int64_t height, std::int64_t width) {
    const auto shape = dcvc2021::downsampled_shape(height, width, 4);
    if (shape.first > std::numeric_limits<std::int64_t>::max() / 4 ||
        shape.second > std::numeric_limits<std::int64_t>::max() / 4)
        throw std::overflow_error("DC latent padded size exceeds int64");
    return {0, width - shape.second * 4, 0, height - shape.first * 4};
}
std::pair<torch::Tensor, SliceShape> CompressionModelImpl::pad_for_y(const torch::Tensor& y) {
    const auto slice = get_to_y_slice_shape(y.size(2), y.size(3));
    return {torch::nn::functional::pad(y,
        torch::nn::functional::PadFuncOptions({-slice[0], -slice[1], -slice[2], -slice[3]}).mode(torch::kReplicate)), slice};
}
torch::Tensor CompressionModelImpl::slice_to_y(const torch::Tensor& params, const SliceShape& slice) {
    return torch::nn::functional::pad(params, torch::nn::functional::PadFuncOptions(
        std::vector<std::int64_t>(slice.begin(), slice.end())));
}
std::array<torch::Tensor, 3> CompressionModelImpl::separate_prior(const torch::Tensor& params) {
    const auto parts = params.chunk(3, 1);
    return {parts.at(0), parts.at(1), parts.at(2)};
}
MaskedResult CompressionModelImpl::process_with_mask(const torch::Tensor& y, const torch::Tensor& scales,
                                                    const torch::Tensor& means, const torch::Tensor& mask) {
    const auto scale_hat = scales * mask, mean_hat = means * mask;
    const auto residual = (y - mean_hat) * mask, rounded = quant(residual);
    return {residual, rounded, rounded + mean_hat, scale_hat};
}
FourParts CompressionModelImpl::get_mask_four_parts(std::int64_t height, std::int64_t width,
                                                  torch::ScalarType dtype, torch::Device device) {
    const auto key = std::to_string(width) + "x" + std::to_string(height);
    if (const auto found = masks.find(key); found != masks.end()) return found->second;
    FourParts result;
    for (int index = 0; index < 4; ++index) {
        auto micro = torch::zeros({2, 2}, torch::TensorOptions().dtype(dtype).device(device));
        micro.index_put_({index / 2, index % 2}, 1);
        result[index] = micro.repeat({(height + 1) / 2, (width + 1) / 2})
            .slice(0, 0, height).slice(1, 0, width).unsqueeze(0).unsqueeze(0);
    }
    masks[key] = result;
    return result;
}
torch::Tensor CompressionModelImpl::combine_four_parts(const std::array<FourParts, 4>& parts) {
    return torch::cat({sum_four(parts[0]), sum_four(parts[1]), sum_four(parts[2]), sum_four(parts[3])}, 1);
}
struct CompressionModelImpl::Stages {
    std::array<FourParts, 4> residual, rounded, recovered, scales;
    FourParts stream_symbols, stream_scales;
    torch::Tensor step;
};
CompressionModelImpl::Stages CompressionModelImpl::prior_stages(const torch::Tensor& input,
    const torch::Tensor& common, const PriorAdaptors& adaptors, torch::nn::Sequential& spatial) {
    const auto params = separate_prior(common);
    const auto masks = get_mask_four_parts(input.size(2), input.size(3), input.scalar_type(), input.device());
    Stages stages; stages.step = params[0].clamp_min(0.5);
    const auto y = four((input / stages.step).chunk(4, 1));
    auto scales = four(params[1].chunk(4, 1)), means = four(params[2].chunk(4, 1));
    torch::Tensor recovered;
    for (int stage = 0; stage < 4; ++stage) {
        if (stage) {
            const auto prediction = spatial->forward(adaptors[stage - 1].ptr()->forward(
                torch::cat({recovered, common}, 1))).chunk(8, 1);
            scales = four(prediction); means = four(prediction, 4);
        }
        FourParts current, symbols, stream_scales;
        for (int channel = 0; channel < 4; ++channel) {
            const auto mask = order[stage][channel];
            const auto value = process_with_mask(y[channel], scales[channel], means[channel], masks[mask]);
            stages.residual[channel][mask] = value.y_res; stages.rounded[channel][mask] = value.y_q;
            stages.recovered[channel][mask] = value.y_hat; stages.scales[channel][mask] = value.scales_hat;
            current[channel] = value.y_hat; symbols[channel] = value.y_q; stream_scales[channel] = value.scales_hat;
        }
        const auto current_step = torch::cat({current[0], current[1], current[2], current[3]}, 1);
        recovered = stage ? recovered + current_step : current_step;
        stages.stream_symbols[stage] = sum_four(symbols);
        stages.stream_scales[stage] = sum_four(stream_scales);
    }
    return stages;
}
PriorResult CompressionModelImpl::forward_four_part_prior(const torch::Tensor& y, const torch::Tensor& params,
    const PriorAdaptors& adaptors, torch::nn::Sequential& spatial) {
    const auto stages = prior_stages(y, params, adaptors, spatial);
    return {combine_four_parts(stages.residual), combine_four_parts(stages.rounded),
            combine_four_parts(stages.recovered) * stages.step, combine_four_parts(stages.scales)};
}
PriorStream CompressionModelImpl::compress_four_part_prior(const torch::Tensor& y, const torch::Tensor& params,
    const PriorAdaptors& adaptors, torch::nn::Sequential& spatial) {
    const auto stages = prior_stages(y, params, adaptors, spatial);
    return {stages.stream_symbols, stages.stream_scales, combine_four_parts(stages.recovered) * stages.step};
}
torch::Tensor CompressionModelImpl::decompress_four_part_prior(const torch::Tensor& common,
    const PriorAdaptors& adaptors, torch::nn::Sequential& spatial) {
    const auto params = separate_prior(common);
    const auto masks = get_mask_four_parts(params[2].size(2), params[2].size(3), params[2].scalar_type(), params[2].device());
    const auto step = params[0].clamp_min(0.5);
    auto scales = four(params[1].chunk(4, 1)), means = four(params[2].chunk(4, 1));
    torch::Tensor recovered;
    for (int stage = 0; stage < 4; ++stage) {
        if (stage) {
            const auto prediction = spatial->forward(adaptors[stage - 1].ptr()->forward(
                torch::cat({recovered, common}, 1))).chunk(8, 1);
            scales = four(prediction); means = four(prediction, 4);
        }
        FourParts scale_parts;
        for (int channel = 0; channel < 4; ++channel) scale_parts[channel] = scales[channel] * masks[order[stage][channel]];
        const auto symbols = gaussian_encoder.decode_stream(sum_four(scale_parts), params[2].scalar_type(), params[2].device());
        FourParts current;
        for (int channel = 0; channel < 4; ++channel)
            current[channel] = (symbols + means[channel]) * masks[order[stage][channel]];
        const auto current_step = torch::cat({current[0], current[1], current[2], current[3]}, 1);
        recovered = stage ? recovered + current_step : current_step;
    }
    return recovered * step;
}
void CompressionModelImpl::load_state_dict(const StateDict& state, bool strict) {
    StateDict targets;
    for (const auto& item : named_parameters(true)) targets.emplace(item.key(), item.value());
    for (const auto& item : named_buffers(true)) targets.emplace(item.key(), item.value());
    for (const auto& item : targets) {
        const auto found = state.find(item.first);
        if (found == state.end()) { if (strict) throw std::runtime_error("Missing DC checkpoint entry: " + item.first); }
        else if (found->second.sizes() != item.second.sizes()) throw std::runtime_error("DC checkpoint shape differs: " + item.first);
    }
    if (strict) for (const auto& item : state) if (!targets.count(item.first))
        throw std::runtime_error("Unexpected DC checkpoint entry: " + item.first);
    torch::NoGradGuard no_grad;
    for (auto& item : targets) if (const auto found = state.find(item.first); found != state.end()) item.second.copy_(found->second);
}
void CompressionModelImpl::load_checkpoint(const std::filesystem::path& path, bool strict) {
    load_state_dict(get_state_dict(path), strict);
}
}  // namespace uf::dc
