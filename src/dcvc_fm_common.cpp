// Copyright (c) Microsoft Corporation. Licensed under the MIT License.
#include "uf/dcvc_fm.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>

namespace uf::fm {
namespace {
constexpr std::array<std::array<int, 4>, 4> order{{{{0, 1, 2, 3}}, {{3, 2, 1, 0}},
                                                 {{2, 3, 0, 1}}, {{1, 0, 3, 2}}}};
std::vector<std::int64_t> quality_indexes(const QIndex& index) {
    if (const auto scalar = std::get_if<std::int64_t>(&index)) return {*scalar};
    if (const auto list = std::get_if<std::vector<std::int64_t>>(&index)) return *list;
    throw std::invalid_argument("FM quality index cannot be None");
}
torch::Tensor paired_sum(const FourParts& parts) { return (parts[0] + parts[1]) + (parts[2] + parts[3]); }
}
CompressionModelImpl::CompressionModelImpl(std::string selected, std::int64_t z,
    std::optional<std::int64_t> mv_z, bool threaded, int stream_parts)
    : y_distribution(std::move(selected)), z_channel(z), mv_z_channel(mv_z), ec_thread(threaded),
      stream_part(stream_parts), gaussian_encoder(y_distribution) {
    bit_estimator_z = register_module("bit_estimator_z", BitEstimator(mv_z_channel ? 1 : 64, z_channel));
    if (mv_z_channel) bit_estimator_z_mv = register_module("bit_estimator_z_mv", BitEstimator(1, *mv_z_channel));
}
torch::Tensor CompressionModelImpl::quant(const torch::Tensor& input) { return input.round(); }
torch::Tensor CompressionModelImpl::get_one_q_scale(const torch::Tensor& scale, std::int64_t index) {
    const auto minimum = scale.slice(0, 0, 1), maximum = scale.slice(0, 1, 2);
    const auto step = (maximum.log() - minimum.log()) / (get_qp_num() - 1);
    return (minimum.log() + step * index).exp();
}
torch::Tensor CompressionModelImpl::get_curr_q(const torch::Tensor& scale, const QIndex& index) {
    if (const auto scalar = std::get_if<std::int64_t>(&index)) return get_one_q_scale(scale, *scalar);
    std::vector<torch::Tensor> values;
    for (const auto i : quality_indexes(index)) values.push_back(get_one_q_scale(scale, i));
    return torch::cat(values, 0);
}
torch::Tensor CompressionModelImpl::get_index_tensor(const QIndex& index, torch::Device device) {
    const auto values = quality_indexes(index);
    auto cpu = torch::empty({static_cast<std::int64_t>(values.size())}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (values[i] < std::numeric_limits<std::int32_t>::min() || values[i] > std::numeric_limits<std::int32_t>::max())
            throw std::overflow_error("FM quality index exceeds int32");
        cpu.data_ptr<std::int32_t>()[i] = static_cast<std::int32_t>(values[i]);
    }
    return cpu.to(device);
}
torch::Tensor CompressionModelImpl::probs_to_bits(const torch::Tensor& probabilities) {
    return ((probabilities + 1e-5).log() * (-1.0 / std::log(2.0))).clamp_min(0);
}
torch::Tensor CompressionModelImpl::get_y_gaussian_bits(const torch::Tensor& y, const torch::Tensor& sigma) {
    const auto scales = sigma.clamp(1e-5, 1e10);
    return probs_to_bits((normal_cdf(y + 0.5, scales) - normal_cdf(y - 0.5, scales)).to(torch::kFloat32));
}
torch::Tensor CompressionModelImpl::get_y_laplace_bits(const torch::Tensor& y, const torch::Tensor& sigma) {
    const auto scales = sigma.clamp(1e-5, 1e10);
    if (!(scales > 0).all().item<bool>() || !(y == y).all().item<bool>())
        throw std::invalid_argument("Invalid FM Laplace distribution parameter/sample");
    return probs_to_bits((dcvc2021::laplace_cdf(y + 0.5, scales) - dcvc2021::laplace_cdf(y - 0.5, scales)).to(torch::kFloat32));
}
torch::Tensor CompressionModelImpl::get_z_bits(const torch::Tensor& z, BitEstimator& estimator, const torch::Tensor& index) {
    return probs_to_bits((estimator->get_cdf(z + 0.5, index) - estimator->get_cdf(z - 0.5, index)).to(torch::kFloat32));
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
        throw std::overflow_error("FM latent padded size exceeds int64");
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
std::array<torch::Tensor, 4> CompressionModelImpl::separate_prior(const torch::Tensor& params, bool is_video) {
    if (is_video) {
        const auto parts = params.chunk(3, 1);
        const auto step = parts.at(0).clamp_min(0.5);
        return {1.0 / step, step, parts.at(1), parts.at(2)};
    }
    const auto q = (params.slice(1, 0, 2).sigmoid() * 1.5 + 0.5).chunk(2, 1);
    const auto prior = params.slice(1, 2).chunk(2, 1);
    return {q.at(0), q.at(1), prior.at(0), prior.at(1)};
}
std::array<torch::Tensor, 2> CompressionModelImpl::get_mask(std::int64_t height, std::int64_t width,
                                                        torch::ScalarType dtype, torch::Device device) {
    const auto key = std::to_string(width) + "x" + std::to_string(height);
    if (!masks.count(key)) {
        auto micro = torch::zeros({2, 2}, torch::TensorOptions().dtype(dtype).device(device));
        micro.index_put_({0, 0}, 1); micro.index_put_({1, 1}, 1);
        const auto mask = micro.repeat({(height + 1) / 2, (width + 1) / 2})
            .slice(0, 0, height).slice(1, 0, width).unsqueeze(0).unsqueeze(0);
        masks[key] = {mask, torch::ones_like(mask) - mask};
    }
    const auto& found = masks.at(key);
    return {found.at(0), found.at(1)};
}
FourParts CompressionModelImpl::get_one_channel_four_parts_mask(std::int64_t height, std::int64_t width,
                                                               torch::ScalarType dtype, torch::Device device) {
    FourParts result;
    for (int i = 0; i < 4; ++i) {
        auto micro = torch::zeros({2, 2}, torch::TensorOptions().dtype(dtype).device(device));
        micro.index_put_({i / 2, i % 2}, 1);
        result[i] = micro.repeat({(height + 1) / 2, (width + 1) / 2})
            .slice(0, 0, height).slice(1, 0, width).unsqueeze(0).unsqueeze(0);
    }
    return result;
}
FourParts CompressionModelImpl::get_mask_four_parts(std::int64_t batch, std::int64_t channels,
    std::int64_t height, std::int64_t width, torch::ScalarType dtype, torch::Device device) {
    const auto key = std::to_string(batch) + "_" + std::to_string(channels) + "x" + std::to_string(width) + "x" + std::to_string(height);
    torch::NoGradGuard no_grad;
    if (!masks.count(key)) {
        if (channels % 4 != 0) throw std::invalid_argument("FM four-part prior requires channels divisible by four");
        const auto m = torch::ones({batch, channels / 4, height, width}, torch::TensorOptions().dtype(dtype).device(device));
        const auto micro = get_one_channel_four_parts_mask(height, width, dtype, device);
        std::vector<torch::Tensor> result;
        for (int stage = 0; stage < 4; ++stage)
            result.push_back(torch::cat({m * micro[order[stage][0]], m * micro[order[stage][1]],
                                         m * micro[order[stage][2]], m * micro[order[stage][3]]}, 1));
        masks[key] = std::move(result);
    }
    const auto& found = masks.at(key);
    return {found.at(0), found.at(1), found.at(2), found.at(3)};
}
MaskedResult CompressionModelImpl::process_with_mask(const torch::Tensor& y, const torch::Tensor& scales,
                                                    const torch::Tensor& means, const torch::Tensor& mask) {
    const auto scales_hat = scales * mask, means_hat = means * mask;
    const auto residual = (y - means_hat) * mask, rounded = quant(residual);
    return {residual, rounded, rounded + means_hat, scales_hat};
}
torch::Tensor CompressionModelImpl::combine_four_parts(const std::array<FourParts, 4>& parts) {
    std::vector<torch::Tensor> result;
    for (const auto& channel : parts) result.push_back(((channel[0] + channel[1]) + channel[2]) + channel[3]);
    return torch::cat(result, 1);
}
torch::Tensor CompressionModelImpl::combine_for_writing(const torch::Tensor& input) {
    const auto parts = input.chunk(4, 1);
    return (parts.at(0) + parts.at(1)) + (parts.at(2) + parts.at(3));
}
struct CompressionModelImpl::Stages {
    FourParts residual, rounded, scales, stream_symbols, stream_scales;
    torch::Tensor recovered;
};
CompressionModelImpl::Stages CompressionModelImpl::prior_stages(const torch::Tensor& input,
    const torch::Tensor& common, const PriorAdaptors& adaptors, torch::nn::Sequential& spatial,
    torch::nn::Conv2d reduction) {
    const auto params = separate_prior(common, !reduction);
    const auto common_params = reduction ? reduction(common) : common;
    const auto stage_masks = get_mask_four_parts(input.size(0), input.size(1), input.size(2), input.size(3), input.scalar_type(), input.device());
    const auto y = input * params[0];
    auto scales = params[2], means = params[3];
    Stages result;
    for (int stage = 0; stage < 4; ++stage) {
        if (stage) {
            const auto prediction = spatial->forward(adaptors[stage - 1].ptr()->forward(torch::cat({result.recovered, common_params}, 1))).chunk(2, 1);
            scales = prediction.at(0); means = prediction.at(1);
        }
        const auto value = process_with_mask(y, scales, means, stage_masks[stage]);
        result.residual[stage] = value.y_res; result.rounded[stage] = value.y_q; result.scales[stage] = value.scales_hat;
        result.recovered = stage ? result.recovered + value.y_hat : value.y_hat;
        result.stream_symbols[stage] = combine_for_writing(value.y_q);
        result.stream_scales[stage] = combine_for_writing(value.scales_hat);
    }
    result.recovered = result.recovered * params[1];
    return result;
}
PriorResult CompressionModelImpl::forward_four_part_prior(const torch::Tensor& y, const torch::Tensor& params,
    const PriorAdaptors& adaptors, torch::nn::Sequential& spatial, torch::nn::Conv2d reduction) {
    const auto stages = prior_stages(y, params, adaptors, spatial, reduction);
    return {paired_sum(stages.residual), paired_sum(stages.rounded), stages.recovered, paired_sum(stages.scales)};
}
PriorStream CompressionModelImpl::compress_four_part_prior(const torch::Tensor& y, const torch::Tensor& params,
    const PriorAdaptors& adaptors, torch::nn::Sequential& spatial, torch::nn::Conv2d reduction) {
    const auto stages = prior_stages(y, params, adaptors, spatial, reduction);
    return {stages.stream_symbols, stages.stream_scales, stages.recovered};
}
torch::Tensor CompressionModelImpl::decompress_four_part_prior(const torch::Tensor& common,
    const PriorAdaptors& adaptors, torch::nn::Sequential& spatial, torch::nn::Conv2d reduction) {
    const auto params = separate_prior(common, !reduction);
    const auto common_params = reduction ? reduction(common) : common;
    auto scales = params[2], means = params[3];
    const auto dtype = means.scalar_type(); const auto device = means.device();
    const auto stage_masks = get_mask_four_parts(means.size(0), means.size(1), means.size(2), means.size(3), dtype, device);
    torch::Tensor recovered;
    for (int stage = 0; stage < 4; ++stage) {
        if (stage) {
            const auto prediction = spatial->forward(adaptors[stage - 1].ptr()->forward(torch::cat({recovered, common_params}, 1))).chunk(2, 1);
            scales = prediction.at(0); means = prediction.at(1);
        }
        const auto symbols = gaussian_encoder.decode_stream(combine_for_writing(scales * stage_masks[stage]), dtype, device);
        const auto current = (torch::cat({symbols, symbols, symbols, symbols}, 1) + means) * stage_masks[stage];
        recovered = stage ? recovered + current : current;
    }
    return recovered * params[1];
}
void CompressionModelImpl::load_state_dict(const StateDict& state, bool strict) {
    StateDict targets;
    for (const auto& item : named_parameters(true)) targets.emplace(item.key(), item.value());
    for (const auto& item : named_buffers(true)) targets.emplace(item.key(), item.value());
    for (const auto& item : targets) {
        const auto found = state.find(item.first);
        if (found == state.end()) { if (strict) throw std::runtime_error("Missing FM checkpoint entry: " + item.first); }
        else if (found->second.sizes() != item.second.sizes()) throw std::runtime_error("FM checkpoint shape differs: " + item.first);
    }
    if (strict) for (const auto& item : state) if (!targets.count(item.first))
        throw std::runtime_error("Unexpected FM checkpoint entry: " + item.first);
    torch::NoGradGuard no_grad;
    for (auto& item : targets) if (const auto found = state.find(item.first); found != state.end()) item.second.copy_(found->second);
}
void CompressionModelImpl::load_checkpoint(const std::filesystem::path& path, bool strict) {
    load_state_dict(get_state_dict(path), strict);
}
}  // namespace uf::fm
