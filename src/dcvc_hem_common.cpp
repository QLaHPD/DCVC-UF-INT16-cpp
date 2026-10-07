// Copyright (c) Microsoft Corporation. Licensed under the MIT License.
#include "dcvc_hem_internal.hpp"
#include "uf/torch_pickle.hpp"
#include <cmath>
#include <functional>
#include <stdexcept>
#include <unordered_set>

namespace uf::hem {
CompressionModelImpl::CompressionModelImpl(std::string selected, std::int64_t z, std::optional<std::int64_t> mv_z)
    : y_distribution(std::move(selected)), z_channel(z), mv_z_channel(mv_z), gaussian_encoder(y_distribution) {
    bit_estimator_z = register_module("bit_estimator_z", BitEstimator(z_channel));
    if (mv_z_channel) bit_estimator_z_mv = register_module("bit_estimator_z_mv", BitEstimator(*mv_z_channel));
    mse = register_module("mse", torch::nn::MSELoss(torch::nn::MSELossOptions().reduction(torch::kNone)));
    ssim = register_module("ssim", MSSSIM(11, false));
}
torch::Tensor CompressionModelImpl::forward(const torch::Tensor&) {
    throw std::logic_error("HEM CompressionModel has no forward implementation");
}
void CompressionModelImpl::initialize_weights() {
    torch::NoGradGuard no_grad;
    std::unordered_set<torch::nn::Module*> visited;
    std::function<void(torch::nn::Module&)> visit = [&](torch::nn::Module& module) {
        if (!visited.insert(&module).second) return;
        if (auto* conv = dynamic_cast<torch::nn::Conv2dImpl*>(&module)) {
            torch::nn::init::xavier_normal_(conv->weight, std::sqrt(2.0));
            if (conv->bias.defined()) conv->bias.fill_(0.01);
        } else if (auto* conv = dynamic_cast<torch::nn::ConvTranspose2dImpl*>(&module)) {
            torch::nn::init::xavier_normal_(conv->weight, std::sqrt(2.0));
            if (conv->bias.defined()) conv->bias.fill_(0.01);
        }
        for (const auto& child : module.named_children()) visit(*child.value());
    };
    visit(*this);  // Safe during construction, before shared_from_this exists.
}
torch::Tensor CompressionModelImpl::quant(const torch::Tensor& input, bool force_detach) const {
    if (is_training() || force_detach) return input + (input.round() - input).clone().detach();
    return input.round();
}
torch::Tensor CompressionModelImpl::add_noise(const torch::Tensor& input) const {
    auto noise = torch::zeros_like(input).uniform_(-0.5, 0.5).clone().detach(); return input + noise;
}
torch::Tensor CompressionModelImpl::probs_to_bits(const torch::Tensor& probabilities) {
    return lower_bound(-1.0 * (probabilities + 1e-5).log() / std::log(2.0), 0);
}
torch::Tensor CompressionModelImpl::get_y_gaussian_bits(const torch::Tensor& y, const torch::Tensor& sigma) const {
    const auto scales = sigma.clamp(0.11, 1e10);
    return probs_to_bits(normal_cdf(y + 0.5, scales) - normal_cdf(y - 0.5, scales));
}
torch::Tensor CompressionModelImpl::get_y_laplace_bits(const torch::Tensor& y, const torch::Tensor& sigma) const {
    const auto scales = sigma.clamp(1e-5, 1e10);
    if (!(scales > 0).all().item<bool>() || !(y == y).all().item<bool>())
        throw std::invalid_argument("Invalid HEM Laplace distribution parameter/sample");
    return probs_to_bits(dcvc2021::laplace_cdf(y + 0.5, scales) - dcvc2021::laplace_cdf(y - 0.5, scales));
}
torch::Tensor CompressionModelImpl::get_z_bits(const torch::Tensor& z, BitEstimator& estimator) const {
    return probs_to_bits(estimator->get_cdf(z + 0.5) - estimator->get_cdf(z - 0.5));
}
void CompressionModelImpl::update(bool force) {
    entropy_coder = std::make_shared<EntropyCoder>();
    gaussian_encoder.update(force, entropy_coder); bit_estimator_z->update(force, entropy_coder);
    if (bit_estimator_z_mv) bit_estimator_z_mv->update(force, entropy_coder);
}
std::pair<torch::Tensor, torch::Tensor> CompressionModelImpl::get_mask(std::int64_t height, std::int64_t width, torch::Device device) {
    const auto micro = torch::tensor({{1, 0}, {0, 1}}, torch::TensorOptions().device(device).dtype(torch::kFloat32));
    const auto first = micro.repeat({height / 2, width / 2}).unsqueeze(0).unsqueeze(0);
    return {first, torch::ones_like(first) - first};
}
MaskedResult CompressionModelImpl::process_with_mask(const torch::Tensor& y, const torch::Tensor& scales,
                                                    const torch::Tensor& means, const torch::Tensor& mask) const {
    const auto scaled = scales * mask, shifted = means * mask;
    const auto residual = (y - shifted) * mask, rounded = quant(residual);
    return {residual, rounded, rounded + shifted, scaled};
}
struct CompressionModelImpl::DualStages {
    MaskedResult a0, b1, a1, b0;
    torch::Tensor step;
};
CompressionModelImpl::DualStages CompressionModelImpl::dual_stages(const torch::Tensor& input,
    const torch::Tensor& means, const torch::Tensor& scales, const torch::Tensor& quant_step,
    torch::nn::Sequential& spatial_prior) {
    if (input.dim() != 4) throw std::invalid_argument("HEM dual prior requires a four dimensional latent");
    const auto masks = get_mask(input.size(2), input.size(3), input.device());
    const auto step = lower_bound(quant_step, 0.5);
    const auto y = (input / step).chunk(2, 1), s = scales.chunk(2, 1), m = means.chunk(2, 1);
    const auto a0 = process_with_mask(y.at(0), s.at(0), m.at(0), masks.first);
    const auto b1 = process_with_mask(y.at(1), s.at(1), m.at(1), masks.second);
    const auto params = spatial_prior->forward(torch::cat({a0.y_hat, b1.y_hat, means, scales, step}, 1)).chunk(4, 1);
    const auto a1 = process_with_mask(y.at(0), params.at(0), params.at(1), masks.second);
    const auto b0 = process_with_mask(y.at(1), params.at(2), params.at(3), masks.first);
    return {a0, b1, a1, b0, step};
}
DualPriorResult CompressionModelImpl::forward_dual_prior(const torch::Tensor& y, const torch::Tensor& means,
    const torch::Tensor& scales, const torch::Tensor& step, torch::nn::Sequential& spatial_prior) {
    const auto v = dual_stages(y, means, scales, step, spatial_prior);
    return {torch::cat({v.a0.y_res + v.a1.y_res, v.b1.y_res + v.b0.y_res}, 1),
        torch::cat({v.a0.y_q + v.a1.y_q, v.b1.y_q + v.b0.y_q}, 1),
        torch::cat({v.a0.y_hat + v.a1.y_hat, v.b1.y_hat + v.b0.y_hat}, 1) * v.step,
        torch::cat({v.a0.scales_hat + v.a1.scales_hat, v.b1.scales_hat + v.b0.scales_hat}, 1)};
}
DualPriorStream CompressionModelImpl::compress_dual_prior(const torch::Tensor& y, const torch::Tensor& means,
    const torch::Tensor& scales, const torch::Tensor& step, torch::nn::Sequential& spatial_prior) {
    const auto v = dual_stages(y, means, scales, step, spatial_prior);
    return {v.a0.y_q + v.b1.y_q, v.a1.y_q + v.b0.y_q,
        v.a0.scales_hat + v.b1.scales_hat, v.a1.scales_hat + v.b0.scales_hat,
        torch::cat({v.a0.y_hat + v.a1.y_hat, v.b1.y_hat + v.b0.y_hat}, 1) * v.step};
}
torch::Tensor CompressionModelImpl::decompress_dual_prior(const torch::Tensor& means,
    const torch::Tensor& scales, const torch::Tensor& quant_step, torch::nn::Sequential& spatial_prior) {
    const auto masks = get_mask(means.size(2), means.size(3), means.device());
    const auto step = quant_step.clamp_min(0.5);
    const auto s = scales.chunk(2, 1), m = means.chunk(2, 1);
    const auto first_scales = s.at(0) * masks.first + s.at(1) * masks.second;
    const auto first = gaussian_encoder.decode_stream(first_scales).to(means.device());
    const auto a0 = (first + m.at(0)) * masks.first, b1 = (first + m.at(1)) * masks.second;
    const auto params = spatial_prior->forward(torch::cat({a0, b1, means, scales, step}, 1)).chunk(4, 1);
    const auto second_scales = params.at(0) * masks.second + params.at(2) * masks.first;
    const auto second = gaussian_encoder.decode_stream(second_scales).to(means.device());
    const auto a1 = (second + params.at(1)) * masks.second, b0 = (second + params.at(3)) * masks.first;
    return torch::cat({a0 + a1, b1 + b0}, 1) * step;
}
void CompressionModelImpl::load_state_dict(const StateDict& state, bool strict) {
    StateDict targets;
    for (const auto& item : named_parameters(true)) targets.emplace(item.key(), item.value());
    for (const auto& item : named_buffers(true)) targets.emplace(item.key(), item.value());
    for (const auto& item : targets) {
        const auto found = state.find(item.first);
        if (found == state.end()) { if (strict) throw std::runtime_error("Missing HEM checkpoint entry: " + item.first); }
        else if (found->second.sizes() != item.second.sizes()) throw std::runtime_error("HEM checkpoint shape differs: " + item.first);
    }
    if (strict) for (const auto& item : state) if (!targets.count(item.first)) throw std::runtime_error("Unexpected HEM checkpoint entry: " + item.first);
    torch::NoGradGuard no_grad;
    for (auto& item : targets) if (const auto found = state.find(item.first); found != state.end()) item.second.copy_(found->second);
}
void CompressionModelImpl::load_checkpoint(const std::filesystem::path& path, bool strict) { load_state_dict(get_state_dict(path), strict); }
}  // namespace uf::hem
