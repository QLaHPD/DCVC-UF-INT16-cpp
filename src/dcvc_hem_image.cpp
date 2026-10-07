// Copyright (c) Microsoft Corporation. Licensed under the MIT License.
#include "dcvc_hem_internal.hpp"
#include <stdexcept>

namespace uf::hem {
IntraNoARImpl::IntraNoARImpl(std::int64_t n, std::int64_t anchors)
    : CompressionModelImpl("gaussian", n), N(n), anchor_num(anchors) {
    const auto transforms = get_enc_dec_models(3, 16, N);
    enc = register_module("enc", transforms.first); dec = register_module("dec", transforms.second);
    refine = register_module("refine", torch::nn::Sequential());
    refine->push_back(UNet(16, 16)); refine->push_back(conv3x3(16, 3));
    const auto hyper = get_hyper_enc_dec_models(N, N);
    hyper_enc = register_module("hyper_enc", hyper.first); hyper_dec = register_module("hyper_dec", hyper.second);
    y_prior_fusion = register_module("y_prior_fusion", detail::prior(N * 2, N * 3, N * 3));
    y_spatial_prior = register_module("y_spatial_prior", detail::prior(N * 4, N * 3, N * 2));
    q_basic = register_parameter("q_basic", torch::ones({1, N, 1, 1}));
    q_scale = register_parameter("q_scale", torch::ones({anchor_num, 1, 1, 1}));
    initialize_weights();
}
torch::Tensor IntraNoARImpl::get_curr_q(const torch::Tensor& scale) const {
    if (!scale.defined()) throw std::invalid_argument("HEM IntraNoAR requires a quality scale");
    return lower_bound(q_basic, 0.5) * scale;
}
ImageResult IntraNoARImpl::forward(const torch::Tensor& input, const torch::Tensor& scale) {
    const auto current_q = get_curr_q(scale);
    const auto y = enc->forward(input) / current_q, z = hyper_enc->forward(y), z_hat = quant(z);
    const auto params = y_prior_fusion->forward(hyper_dec->forward(z_hat)).chunk(3, 1);
    const auto latent = forward_dual_prior(y, params.at(2), params.at(1), params.at(0), y_spatial_prior);
    const auto y_hat = latent.y_hat * current_q;
    const auto x_hat = refine->forward(dec->forward(y_hat));
    torch::Tensor y_for_bit, z_for_bit;
    if (is_training()) { y_for_bit = add_noise(latent.y_res); z_for_bit = add_noise(z); }
    else { y_for_bit = latent.y_q; z_for_bit = z_hat; }
    const auto bits_y = get_y_gaussian_bits(y_for_bit, latent.scales_hat), bits_z = get_z_bits(z_for_bit, bit_estimator_z);
    const auto distortion = mse(input, x_hat), quality = ssim(input, x_hat);
    const auto pixels = input.size(2) * input.size(3);
    const auto bpp_y = bits_y.sum({1, 2, 3}) / pixels, bpp_z = bits_z.sum({1, 2, 3}) / pixels;
    return {x_hat, distortion.sum({1, 2, 3}) / pixels, quality, bpp_y + bpp_z, bpp_y, bpp_z,
            ((bpp_y + bpp_z).sum() * pixels).item<double>()};
}
torch::Tensor IntraNoARImpl::get_q_scales_from_ckpt(const std::filesystem::path& path) { return get_state_dict(path).at("q_scale").reshape({-1}); }
ImageResult IntraNoARImpl::encode_decode(const torch::Tensor& input, const torch::Tensor& scale,
    std::optional<std::filesystem::path> output, std::optional<std::int64_t> width, std::optional<std::int64_t> height) {
    if (!output) return forward(input, scale);
    if (!width || !height) throw std::invalid_argument("HEM I-frame stream coding requires original dimensions");
    const auto rounded = get_rounded_q(scale);
    const auto stream = compress(input, detail::scalar(rounded.scale));
    write_i(*output, {detail::dimension(*height), detail::dimension(*width), rounded.index, stream});
    if (!std::filesystem::is_regular_file(*output)) throw std::runtime_error("HEM I-frame output is not a regular file");
    const auto bits = static_cast<double>(std::filesystem::file_size(*output)) * 8;
    const auto decoded = read_i(*output);
    ImageResult result; result.bit = bits;
    result.x_hat = decompress(decoded.bit_stream, decoded.height, decoded.width, detail::scalar(decoded.q_index / 100.0)); return result;
}
std::string IntraNoARImpl::compress(const torch::Tensor& input, const torch::Tensor& scale) {
    const auto current_q = get_curr_q(scale), y = enc->forward(input) / current_q, z = hyper_enc->forward(y), z_hat = z.round();
    const auto params = y_prior_fusion->forward(hyper_dec->forward(z_hat)).chunk(3, 1);
    const auto latent = compress_dual_prior(y, params.at(2), params.at(1), params.at(0), y_spatial_prior);
    const auto y_hat = latent.y_hat * current_q; (void)y_hat;
    if (!entropy_coder) throw std::runtime_error("HEM image model must be updated before coding");
    entropy_coder->reset_encoder(); bit_estimator_z->encode(z_hat);
    gaussian_encoder.encode(latent.y_q_w_0, latent.scales_w_0); gaussian_encoder.encode(latent.y_q_w_1, latent.scales_w_1);
    return entropy_coder->flush_encoder();
}
torch::Tensor IntraNoARImpl::decompress(const std::string& stream, std::int64_t height, std::int64_t width, const torch::Tensor& scale) {
    const auto current_q = get_curr_q(scale);
    if (!entropy_coder) throw std::runtime_error("HEM image model must be updated before coding");
    entropy_coder->set_stream(stream);
    const auto z_hat = bit_estimator_z->decode_stream(dcvc2021::downsampled_shape(height, width, 64)).to(parameters().at(0).device());
    const auto params = y_prior_fusion->forward(hyper_dec->forward(z_hat)).chunk(3, 1);
    const auto y_hat = decompress_dual_prior(params.at(2), params.at(1), params.at(0), y_spatial_prior) * current_q;
    return refine->forward(dec->forward(y_hat)).clamp_(0, 1);
}
}  // namespace uf::hem
