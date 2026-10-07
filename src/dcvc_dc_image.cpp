// Copyright (c) Microsoft Corporation. Licensed under the MIT License.
// Fine-scale interpolation follows NumPy's BSD-3-Clause linspace order;
// its retained notice is in licenses/NumPy.txt.
#include "dcvc_dc_internal.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>

namespace uf::dc {
namespace detail {
torch::nn::LeakyReLU leaky() { return torch::nn::LeakyReLU(); }
double endpoint_log(const torch::Tensor& endpoint) {
    if (!endpoint.device().is_cpu()) throw std::runtime_error("DC fine q-scale interpolation requires CPU parameters");
    if (endpoint.scalar_type() == torch::kFloat32) return std::log(endpoint.item<float>());
    if (endpoint.scalar_type() == torch::kFloat16)
        return static_cast<double>(static_cast<c10::Half>(std::log(endpoint.item<float>())));
    if (endpoint.scalar_type() == torch::kFloat64) return std::log(endpoint.item<double>());
    throw std::invalid_argument("DC fine q-scale interpolation requires Float16, Float32 or Float64");
}
torch::Tensor fine_scales(const torch::Tensor& scale) {
    const auto first = endpoint_log(scale.index({0, 0, 0, 0}));
    const auto last = endpoint_log(scale.index({3, 0, 0, 0}));
    const auto delta = last - first;
    const auto step = delta / 63;
    auto result = torch::empty({64}, torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat64));
    for (int index = 0; index < 64; ++index) {
        const auto value = (step == 0 ? (static_cast<double>(index) / 63) * delta : static_cast<double>(index) * step) + first;
        result.data_ptr<double>()[index] = std::exp(index == 63 ? last : value);
    }
    return result;  // Plain Float64 CPU attribute, matching the NumPy fine table.
}
std::uint32_t dimension(std::int64_t value) {
    if (value < 0 || value > std::numeric_limits<std::uint32_t>::max()) throw std::out_of_range("DC dimension exceeds uint32");
    return static_cast<std::uint32_t>(value);
}
std::uint8_t quality_index(std::int64_t index) {
    if (index < 0 || index > 63) throw std::out_of_range("DC file quality index must be between 0 and 63");
    return static_cast<std::uint8_t>(index);
}
}  // namespace detail
using detail::dimension;
using detail::fine_scales;
using detail::leaky;
using detail::quality_index;
IntraEncoderImpl::IntraEncoderImpl(std::int64_t n, bool inplace) {
    enc_1 = register_module("enc_1", torch::nn::Sequential());
    enc_1->push_back(ResidualBlockWithStride(3, 128, 2, inplace)); enc_1->push_back(DepthConvBlock2(128, 128, 3, 1, 0.01, 0.1, inplace));
    enc_2 = register_module("enc_2", torch::nn::Sequential());
    enc_2->push_back(ResidualBlockWithStride(128, 192, 2, inplace)); enc_2->push_back(DepthConvBlock2(192, 192, 3, 1, 0.01, 0.1, inplace));
    enc_2->push_back(ResidualBlockWithStride(192, n, 2, inplace)); enc_2->push_back(DepthConvBlock2(n, n, 3, 1, 0.01, 0.1, inplace));
    enc_2->push_back(conv3x3(n, n, 2));
}
torch::Tensor IntraEncoderImpl::forward(const torch::Tensor& input, const torch::Tensor& step) {
    return enc_2->forward(enc_1->forward(input) * step);
}
IntraDecoderImpl::IntraDecoderImpl(std::int64_t n, bool inplace) {
    dec_1 = register_module("dec_1", torch::nn::Sequential());
    dec_1->push_back(DepthConvBlock2(n, n, 3, 1, 0.01, 0.1, inplace)); dec_1->push_back(ResidualBlockUpsample(n, n, 2, inplace));
    dec_1->push_back(DepthConvBlock2(n, n, 3, 1, 0.01, 0.1, inplace)); dec_1->push_back(ResidualBlockUpsample(n, 192, 2, inplace));
    dec_1->push_back(DepthConvBlock2(192, 192, 3, 1, 0.01, 0.1, inplace)); dec_1->push_back(ResidualBlockUpsample(192, 128, 2, inplace));
    dec_2 = register_module("dec_2", torch::nn::Sequential());
    dec_2->push_back(DepthConvBlock2(128, 128, 3, 1, 0.01, 0.1, inplace)); dec_2->push_back(ResidualBlockUpsample(128, 16, 2, inplace));
}
torch::Tensor IntraDecoderImpl::forward(const torch::Tensor& input, const torch::Tensor& step) {
    return dec_2->forward(dec_1->forward(input) * step);
}
IntraNoARImpl::IntraNoARImpl(std::int64_t n, std::int64_t anchors, bool threaded, int parts, bool inplace)
    : CompressionModelImpl("gaussian", n, std::nullopt, threaded, parts), N(n), anchor_num(anchors) {
    enc = register_module("enc", IntraEncoder(N, inplace));
    hyper_enc = register_module("hyper_enc", torch::nn::Sequential());
    hyper_enc->push_back(DepthConvBlock2(N, N, 3, 1, 0.01, 0.1, inplace));
    hyper_enc->push_back(conv3x3(N, N, 2)); hyper_enc->push_back(leaky()); hyper_enc->push_back(conv3x3(N, N, 2));
    hyper_dec = register_module("hyper_dec", torch::nn::Sequential());
    hyper_dec->push_back(ResidualBlockUpsample(N, N, 2, inplace)); hyper_dec->push_back(ResidualBlockUpsample(N, N, 2, inplace));
    hyper_dec->push_back(DepthConvBlock2(N, N));  // The source omits its inplace argument here.
    y_prior_fusion = register_module("y_prior_fusion", torch::nn::Sequential());
    y_prior_fusion->push_back(DepthConvBlock2(N, N * 2, 3, 1, 0.01, 0.1, inplace));
    y_prior_fusion->push_back(DepthConvBlock2(N * 2, N * 3, 3, 1, 0.01, 0.1, inplace));
    y_spatial_prior_adaptor_1 = register_module("y_spatial_prior_adaptor_1", conv1x1(N * 4, N * 3));
    y_spatial_prior_adaptor_2 = register_module("y_spatial_prior_adaptor_2", conv1x1(N * 4, N * 3));
    y_spatial_prior_adaptor_3 = register_module("y_spatial_prior_adaptor_3", conv1x1(N * 4, N * 3));
    y_spatial_prior = register_module("y_spatial_prior", torch::nn::Sequential());
    y_spatial_prior->push_back(DepthConvBlock2(N * 3, N * 3, 3, 1, 0.01, 0.1, inplace));
    y_spatial_prior->push_back(DepthConvBlock2(N * 3, N * 2, 3, 1, 0.01, 0.1, inplace));
    y_spatial_prior->push_back(DepthConvBlock2(N * 2, N * 2, 3, 1, 0.01, 0.1, inplace));
    dec = register_module("dec", IntraDecoder(N, inplace));
    refine = register_module("refine", torch::nn::Sequential());
    refine->push_back(UNet2(16, 16, inplace)); refine->push_back(conv3x3(16, 3));
    q_basic_enc = register_parameter("q_basic_enc", torch::ones({1, 128, 1, 1}));
    q_scale_enc = register_parameter("q_scale_enc", torch::ones({anchor_num, 1, 1, 1}));
    q_basic_dec = register_parameter("q_basic_dec", torch::ones({1, 128, 1, 1}));
    q_scale_dec = register_parameter("q_scale_dec", torch::ones({anchor_num, 1, 1, 1}));
}
PriorAdaptors IntraNoARImpl::adaptors() const {
    return {y_spatial_prior_adaptor_1, y_spatial_prior_adaptor_2, y_spatial_prior_adaptor_3};
}
std::pair<torch::Tensor, torch::Tensor> IntraNoARImpl::get_q_for_inference(bool in_checkpoint, std::optional<std::int64_t> index) {
    const auto enc_scale = in_checkpoint ? q_scale_enc.select(3, 0).select(2, 0).select(1, 0) : q_scale_enc_fine;
    const auto dec_scale = in_checkpoint ? q_scale_dec.select(3, 0).select(2, 0).select(1, 0) : q_scale_dec_fine;
    return {get_curr_q(enc_scale, q_basic_enc, index), get_curr_q(dec_scale, q_basic_dec, index)};
}
ImageResult IntraNoARImpl::forward(const torch::Tensor& input, bool in_checkpoint, std::optional<std::int64_t> index) {
    const auto q = get_q_for_inference(in_checkpoint, index);
    const auto y = enc(input, q.first);
    const auto padded = pad_for_y(y);
    const auto z = hyper_enc->forward(padded.first), z_hat = quant(z);
    const auto params = slice_to_y(y_prior_fusion->forward(hyper_dec->forward(z_hat)), padded.second);
    const auto latent = forward_four_part_prior(y, params, adaptors(), y_spatial_prior);
    const auto x_hat = refine->forward(dec(latent.y_hat, q.second));
    const auto bits_y = get_y_gaussian_bits(latent.y_q, latent.scales_hat);
    const auto bits_z = get_z_bits(z_hat, bit_estimator_z);
    const auto pixels = input.size(2) * input.size(3);
    const auto bpp_y = bits_y.sum({1, 2, 3}) / pixels, bpp_z = bits_z.sum({1, 2, 3}) / pixels;
    const auto bpp = bpp_y + bpp_z;
    return {x_hat, bpp.sum() * pixels, bpp, bpp_y, bpp_z};
}
std::pair<torch::Tensor, torch::Tensor> IntraNoARImpl::get_q_scales_from_ckpt(const std::filesystem::path& path) {
    const auto state = get_state_dict(path);
    return {state.at("q_scale_enc").reshape({-1}), state.at("q_scale_dec").reshape({-1})};
}
void IntraNoARImpl::load_state_dict(const StateDict& state, bool strict) {
    CompressionModelImpl::load_state_dict(state, strict);
    torch::NoGradGuard no_grad;
    q_scale_enc_fine = fine_scales(q_scale_enc);
    q_scale_dec_fine = fine_scales(q_scale_dec);
}
ImageFrame IntraNoARImpl::encode_decode(const torch::Tensor& input, bool in_checkpoint, std::int64_t index,
    std::optional<std::filesystem::path> output, std::optional<std::int64_t> width, std::optional<std::int64_t> height) {
    if (!output) {
        const auto result = forward(input, in_checkpoint, index);
        return {result.bit.item<double>(), result.x_hat};
    }
    if (!height || !width) throw std::invalid_argument("DC I-frame file coding requires original dimensions");
    const auto compressed = compress(input, in_checkpoint, index);
    write_i(*output, {dimension(*height), dimension(*width), in_checkpoint, quality_index(index), compressed.bit_stream});
    if (!std::filesystem::is_regular_file(*output)) throw std::runtime_error("DC I-frame output is not a regular file");
    const auto bit = static_cast<double>(std::filesystem::file_size(*output)) * 8;
    const auto decoded = read_i(*output);
    return {bit, decompress(decoded.bit_stream, decoded.height, decoded.width, decoded.q_in_ckpt, decoded.q_index)};
}
ImageStream IntraNoARImpl::compress(const torch::Tensor& input, bool in_checkpoint, std::int64_t index) {
    const auto q = get_q_for_inference(in_checkpoint, index);
    const auto y = enc(input, q.first);
    const auto padded = pad_for_y(y);
    const auto z_hat = hyper_enc->forward(padded.first).round();
    const auto params = slice_to_y(y_prior_fusion->forward(hyper_dec->forward(z_hat)), padded.second);
    const auto latent = compress_four_part_prior(y, params, adaptors(), y_spatial_prior);
    if (!entropy_coder) throw std::runtime_error("DC image model must be updated before coding");
    entropy_coder->reset(); bit_estimator_z->encode(z_hat);
    for (int stage = 0; stage < 4; ++stage) gaussian_encoder.encode(latent.y_q[stage], latent.scales[stage]);
    entropy_coder->flush();
    const auto x_hat = refine->forward(dec(latent.y_hat, q.second)).clamp_(0, 1);
    return {entropy_coder->get_encoded_stream(), x_hat};
}
torch::Tensor IntraNoARImpl::decompress(const std::string& stream, std::int64_t height, std::int64_t width,
                                      bool in_checkpoint, std::int64_t index) {
    const auto parameter = parameters().at(0);
    const auto q = get_q_for_inference(in_checkpoint, index);
    if (!entropy_coder) throw std::runtime_error("DC image model must be updated before coding");
    entropy_coder->set_stream(stream);
    const auto z_size = dcvc2021::downsampled_shape(height, width, 64);
    const auto y_size = dcvc2021::downsampled_shape(height, width, 16);
    const auto slice = get_to_y_slice_shape(y_size.first, y_size.second);
    const auto z_hat = bit_estimator_z->decode_stream(z_size, parameter.scalar_type(), parameter.device());
    const auto params = slice_to_y(y_prior_fusion->forward(hyper_dec->forward(z_hat)), slice);
    const auto y_hat = decompress_four_part_prior(params, adaptors(), y_spatial_prior);
    return refine->forward(dec(y_hat, q.second)).clamp_(0, 1);
}
}  // namespace uf::dc
