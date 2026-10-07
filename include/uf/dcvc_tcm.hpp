// Copyright (c) Microsoft Corporation. Licensed under the MIT License.
// DCVC-family/DCVC-TCM video model, entropy and stream interfaces.
#pragma once
#include "uf/dcvc2021.hpp"
#include <array>
#include <memory>

namespace uf::tcm {
using CdfTables = dcvc2021::CdfTables;
using GDN = dcvc2021::GDN;
using Bitparm = dcvc2021::Bitparm;
using Contexts = std::array<torch::Tensor, 3>;

// These groups share one encoder/decoder state, in submission order.
class EntropyCoder {
public:
    explicit EntropyCoder(int precision = 16);
    void reset_encoder();
    void encode_with_indexes(const std::vector<std::int32_t>& symbols,
                             const std::vector<std::int32_t>& indexes, const CdfTables& tables);
    std::string flush_encoder();
    void set_stream(const std::string& stream);
    void set_decoder_cdf(CdfTables tables);
    std::vector<std::int32_t> decode_stream(const std::vector<std::int32_t>& indexes, const CdfTables& tables);
    std::vector<std::int32_t> decode_stream_only_indexes(const std::vector<std::int32_t>& indexes);
private:
    std::vector<std::int32_t> symbols_, indexes_;
    CdfTables tables_;
    dcvc2021::RansDecoder decoder_;
    std::optional<CdfTables> decoder_tables_;
};

struct BitEstimatorImpl : torch::nn::Module {
    explicit BitEstimatorImpl(std::int64_t channels);
    torch::Tensor forward(const torch::Tensor& input);
    void update(bool force = false, std::shared_ptr<EntropyCoder> coder = {});
    void encode(const torch::Tensor& input);
    torch::Tensor decode_stream(std::pair<std::int64_t, std::int64_t> size);
    static torch::Tensor build_indexes(const std::vector<std::int64_t>& size);
    Bitparm f1{nullptr}, f2{nullptr}, f3{nullptr}, f4{nullptr};
private:
    std::int64_t channels_;
    std::shared_ptr<EntropyCoder> coder_;
    std::optional<CdfTables> tables_;
};
TORCH_MODULE(BitEstimator);

// The default and the PMF stage use Laplace, including the Python
// distribution="gaussian" quirk (only its center search uses Normal).
class GaussianEncoder {
public:
    explicit GaussianEncoder(std::string distribution = "laplace");
    void update(bool force = false, std::shared_ptr<EntropyCoder> coder = {});
    torch::Tensor build_indexes(const torch::Tensor& scales) const;
    void encode(const torch::Tensor& input, const torch::Tensor& scales);
    torch::Tensor decode_stream(const torch::Tensor& scales);
    void set_decoder_cdf();
    void encode_with_indexes(const std::vector<std::int32_t>& symbols, const std::vector<std::int32_t>& indexes);
    std::vector<std::int32_t> decode_stream_only_indexes(const std::vector<std::int32_t>& indexes);
private:
    std::string distribution_;
    torch::Tensor scale_table_;
    double log_min_, log_step_;
    std::shared_ptr<EntropyCoder> coder_;
    std::optional<CdfTables> tables_;
};

torch::Tensor flow_warp(const torch::Tensor& image, const torch::Tensor& flow);
torch::Tensor bilinear_up(const torch::Tensor& input);
torch::Tensor bilinear_down(const torch::Tensor& input);
torch::Tensor pad(const torch::Tensor& input, std::int64_t multiple = 64);
torch::Tensor crop(const torch::Tensor& input, std::int64_t height, std::int64_t width);
void write_p(const std::filesystem::path& path, const std::string& stream);
std::string read_p(const std::filesystem::path& path);

struct ResBlockImpl : torch::nn::Module {
    ResBlockImpl(std::int64_t channels, double slope = 0.01, bool start_from_relu = true,
                 bool end_with_relu = false, bool bottleneck = false);
    torch::Tensor forward(const torch::Tensor& input);
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    torch::nn::LeakyReLU leaky_relu{nullptr};
private:
    bool start_, end_;
};
TORCH_MODULE(ResBlock);
struct MEBasicImpl : torch::nn::Module {
    MEBasicImpl();
    torch::Tensor forward(const torch::Tensor& input);
    torch::nn::ReLU relu{nullptr};
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr}, conv4{nullptr}, conv5{nullptr};
};
TORCH_MODULE(MEBasic);
struct SpynetImpl : torch::nn::Module {
    SpynetImpl();
    torch::Tensor forward(const torch::Tensor& current, const torch::Tensor& reference);
    torch::nn::ModuleList moduleBasic{nullptr};
};
TORCH_MODULE(Spynet);
struct FeatureExtractorImpl : torch::nn::Module {
    explicit FeatureExtractorImpl(std::int64_t channels = 64);
    Contexts forward(const torch::Tensor& input);
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
    ResBlock res_block1{nullptr}, res_block2{nullptr}, res_block3{nullptr};
};
TORCH_MODULE(FeatureExtractor);
struct MultiScaleContextFusionImpl : torch::nn::Module {
    MultiScaleContextFusionImpl(std::int64_t in = 64, std::int64_t out = 64);
    Contexts forward(const Contexts& contexts);
    torch::nn::Sequential conv3_up{nullptr}, conv2_up{nullptr};
    torch::nn::Conv2d conv3_out{nullptr}, conv2_out{nullptr}, conv1_out{nullptr};
    ResBlock res_block3_up{nullptr}, res_block3_out{nullptr}, res_block2_up{nullptr},
             res_block2_out{nullptr}, res_block1_out{nullptr};
};
TORCH_MODULE(MultiScaleContextFusion);
struct ContextualEncoderImpl : torch::nn::Module {
    ContextualEncoderImpl(std::int64_t n = 64, std::int64_t m = 96);
    torch::Tensor forward(const torch::Tensor& input, const Contexts& contexts);
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr}, conv4{nullptr};
    GDN gdn1{nullptr}, gdn2{nullptr}, gdn3{nullptr};
    ResBlock res1{nullptr}, res2{nullptr};
};
TORCH_MODULE(ContextualEncoder);
struct ContextualDecoderImpl : torch::nn::Module {
    ContextualDecoderImpl(std::int64_t n = 64, std::int64_t m = 96);
    torch::Tensor forward(const torch::Tensor& input, const Contexts& contexts);
    torch::nn::Sequential up1{nullptr}, up2{nullptr}, up3{nullptr}, up4{nullptr};
    GDN gdn1{nullptr}, gdn2{nullptr}, gdn3{nullptr};
    ResBlock res1{nullptr}, res2{nullptr};
};
TORCH_MODULE(ContextualDecoder);
struct TemporalPriorEncoderImpl : torch::nn::Module {
    TemporalPriorEncoderImpl(std::int64_t n = 64, std::int64_t m = 96);
    torch::Tensor forward(const Contexts& contexts);
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr}, conv4{nullptr};
    GDN gdn1{nullptr}, gdn2{nullptr}, gdn3{nullptr};
};
TORCH_MODULE(TemporalPriorEncoder);
struct ReconGenerationImpl : torch::nn::Module {
    ReconGenerationImpl(std::int64_t context = 64, std::int64_t residual = 32, std::int64_t channels = 64);
    std::pair<torch::Tensor, torch::Tensor> forward(const torch::Tensor& first, const torch::Tensor& second);
    torch::nn::Sequential feature_conv{nullptr};
    torch::nn::Conv2d recon_conv{nullptr};
};
TORCH_MODULE(ReconGeneration);

// Differentiable pinned pytorch-msssim 0.2.0 RGB loss.
struct MSSSIMImpl : torch::nn::Module {
    explicit MSSSIMImpl(std::int64_t window_size = 11, bool size_average = true);
    torch::Tensor forward(const torch::Tensor& input, const torch::Tensor& reference);
private:
    torch::Tensor window_;  // A Python attribute, intentionally not a buffer.
    bool size_average_;
};
TORCH_MODULE(MSSSIM);

struct ForwardResult {
    torch::Tensor bpp_mv_y, bpp_mv_z, bpp_y, bpp_z, bpp, me_mse, mse_loss, me_ms_ssim, recon_ms_ssim;
    torch::Tensor recon_image, feature, warp_frame, temporal_params, y, z, mv_y, mv_z, mv;
    torch::Tensor bit_y, bit_z, bit_mv_y, bit_mv_z, bit;
};
struct CompressedLatents { std::uint64_t bit{}; std::string string; };
struct DecodedFrame { torch::Tensor x_hat, feature; };
struct FrameResult {
    torch::Tensor x_hat, feature;
    double bit_y{}, bit_z{}, bit_mv_y{}, bit_mv_z{}, bit{}, decoding_time{};
};
struct DMCImpl : torch::nn::Module {
    static constexpr std::int64_t channel_mv = 128, channel_N = 64, channel_M = 96;
    explicit DMCImpl(std::int64_t window_size = 11);
    ForwardResult forward_one_frame(const torch::Tensor& input, const torch::Tensor& reference,
                                    const torch::Tensor& reference_feature = {});
    ForwardResult forward(const torch::Tensor& input, const torch::Tensor& reference,
                          const torch::Tensor& reference_feature = {});
    CompressedLatents compress(const torch::Tensor& y, const torch::Tensor& z, const torch::Tensor& mv_y,
                               const torch::Tensor& mv_z, const torch::Tensor& temporal_params);
    DecodedFrame decompress(const torch::Tensor& reference, const torch::Tensor& reference_feature,
                            const std::string& stream, std::int64_t height, std::int64_t width);
    FrameResult encode_decode(const torch::Tensor& input, const torch::Tensor& reference,
                             const torch::Tensor& reference_feature = {},
                             std::optional<std::filesystem::path> output = std::nullopt,
                             std::int64_t width = 0, std::int64_t height = 0);
    void update(bool force = false);
    void load_dict(const std::map<std::string, torch::Tensor>& state, bool strict = true);
    void load_checkpoint(const std::filesystem::path& path, bool strict = true);
    static torch::Tensor quant(const torch::Tensor& input);
    static std::pair<torch::Tensor, torch::Tensor> get_y_bits_probs(const torch::Tensor& y, const torch::Tensor& scales);
    static std::pair<torch::Tensor, torch::Tensor> get_z_bits_probs(const torch::Tensor& z, BitEstimator& estimator);

    Spynet optic_flow{nullptr};
    torch::nn::Sequential mv_encoder{nullptr}, mv_prior_encoder{nullptr}, mv_prior_decoder{nullptr}, mv_decoder{nullptr};
    torch::nn::Conv2d feature_adaptor_I{nullptr}, feature_adaptor_P{nullptr};
    FeatureExtractor feature_extractor{nullptr};
    MultiScaleContextFusion context_fusion_net{nullptr};
    ContextualEncoder contextual_encoder{nullptr};
    torch::nn::Sequential contextual_hyper_prior_encoder{nullptr}, contextual_hyper_prior_decoder{nullptr};
    TemporalPriorEncoder temporal_prior_encoder{nullptr};
    torch::nn::Sequential contextual_entropy_parameter{nullptr};
    ContextualDecoder contextual_decoder{nullptr};
    ReconGeneration recon_generation_net{nullptr};
    std::shared_ptr<EntropyCoder> entropy_coder;
    BitEstimator bit_estimator_z{nullptr}, bit_estimator_z_mv{nullptr};
    GaussianEncoder gaussian_encoder;
    MSSSIM ms_ssim_loss{nullptr};
private:
    Contexts multi_scale_feature_extractor(const torch::Tensor& reference, const torch::Tensor& feature);
    std::pair<Contexts, torch::Tensor> motion_compensation(const torch::Tensor& reference,
                                                        const torch::Tensor& feature, const torch::Tensor& motion);
    void load_normalized_dict(const std::map<std::string, torch::Tensor>& state, bool strict);
};
TORCH_MODULE(DMC);
}  // namespace uf::tcm
