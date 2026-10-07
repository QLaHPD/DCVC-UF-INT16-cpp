// Copyright (c) Microsoft Corporation. Licensed under the MIT License.
// Native DCVC-family/DCVC-HEM model, entropy and stream interfaces.
#pragma once
#include "uf/dcvc2021.hpp"
#include "uf/dcvc_tcm.hpp"
#include <array>
#include <memory>
#include <tuple>

namespace uf::hem {
using CdfTables = dcvc2021::CdfTables;
using Contexts = std::array<torch::Tensor, 3>;
using StateDict = std::map<std::string, torch::Tensor>;
using MSSSIM = tcm::MSSSIM;
torch::Tensor lower_bound(const torch::Tensor& input, double bound);
torch::Tensor normal_cdf(const torch::Tensor& input, const torch::Tensor& scales);
torch::Tensor flow_warp(const torch::Tensor& input, const torch::Tensor& flow);
torch::Tensor bilinear_up(const torch::Tensor& input);
torch::Tensor bilinear_down(const torch::Tensor& input);
torch::Tensor pad(const torch::Tensor& input, std::int64_t multiple = 64);
torch::Tensor crop(const torch::Tensor& input, std::int64_t height, std::int64_t width);
torch::nn::Conv2d conv3x3(std::int64_t in, std::int64_t out, std::int64_t stride = 1);
torch::nn::Conv2d conv1x1(std::int64_t in, std::int64_t out, std::int64_t stride = 1);
torch::nn::Sequential subpel_conv3x3(std::int64_t in, std::int64_t out, std::int64_t factor = 1);
torch::nn::Sequential subpel_conv1x1(std::int64_t in, std::int64_t out, std::int64_t factor = 1);

class CdfHelper {
public:
    void set_cdf(const torch::Tensor& offsets, const torch::Tensor& cdf, const torch::Tensor& sizes);
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> get_cdf_info() const;
    CdfTables tables() const;
    torch::Tensor offset, quantized, lengths;  // Plain CPU attributes, not module buffers.
};
class EntropyCoder {
public:
    static torch::Tensor pmf_to_quantized_cdf(const torch::Tensor& pmf, int precision = 16);
    static torch::Tensor pmf_to_cdf(const torch::Tensor& pmf, const torch::Tensor& tails,
                                  const torch::Tensor& lengths, std::int64_t maximum);
    void reset_encoder();
    void encode_with_indexes(const std::vector<std::int32_t>& symbols,
                             const std::vector<std::int32_t>& indexes, const CdfTables& tables);
    std::string flush_encoder();
    void set_stream(const std::string& stream);
    torch::Tensor decode_stream(const std::vector<std::int32_t>& indexes, const CdfTables& tables);
private:
    std::vector<std::int32_t> symbols_, indexes_;
    CdfTables tables_;
    dcvc2021::RansDecoder decoder_;
};
struct BitparmImpl : torch::nn::Module {
    explicit BitparmImpl(std::int64_t channels, bool final = false);
    torch::Tensor forward(const torch::Tensor& input);
    torch::Tensor h, b, a;
private:
    bool final_;
};
TORCH_MODULE(Bitparm);
struct BitEstimatorImpl : torch::nn::Module {
    explicit BitEstimatorImpl(std::int64_t channels);
    torch::Tensor forward(const torch::Tensor& input);
    torch::Tensor get_logits_cdf(const torch::Tensor& input);
    torch::Tensor get_cdf(const torch::Tensor& input);
    void update(bool force = false, std::shared_ptr<EntropyCoder> coder = {});
    static torch::Tensor build_indexes(const std::vector<std::int64_t>& size);
    void encode(const torch::Tensor& input);
    torch::Tensor decode_stream(std::pair<std::int64_t, std::int64_t> size);
    Bitparm f1{nullptr}, f2{nullptr}, f3{nullptr}, f4{nullptr};
    std::shared_ptr<EntropyCoder> entropy_coder;
    std::shared_ptr<CdfHelper> cdf_helper;
    const std::int64_t channel;
};
TORCH_MODULE(BitEstimator);
class GaussianEncoder {
public:
    explicit GaussianEncoder(std::string distribution = "laplace");
    static torch::Tensor get_scale_table(double minimum, double maximum, std::int64_t levels);
    void update(bool force = false, std::shared_ptr<EntropyCoder> coder = {});
    torch::Tensor build_indexes(const torch::Tensor& scales) const;
    void encode(const torch::Tensor& input, const torch::Tensor& scales);
    torch::Tensor decode_stream(const torch::Tensor& scales);
    void set_decoder_cdf();  // The source's EntropyCoder has no such method.
    void encode_with_indexes(const std::vector<std::int32_t>& symbols, const std::vector<std::int32_t>& indexes);
    const std::string distribution;
    torch::Tensor scale_table;
    std::shared_ptr<EntropyCoder> entropy_coder;
    std::shared_ptr<CdfHelper> cdf_helper;
private:
    double log_min_, log_step_;
};

struct ResidualBlockImpl : torch::nn::Module {
    ResidualBlockImpl(std::int64_t in, std::int64_t out, double slope = 0.01);
    torch::Tensor forward(const torch::Tensor& input);
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    torch::nn::LeakyReLU leaky_relu{nullptr};
};
TORCH_MODULE(ResidualBlock);
struct ResidualBlockWithStrideImpl : torch::nn::Module {
    ResidualBlockWithStrideImpl(std::int64_t in, std::int64_t out, std::int64_t stride = 2);
    torch::Tensor forward(const torch::Tensor& input);
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, downsample{nullptr};
    torch::nn::LeakyReLU leaky_relu{nullptr}, leaky_relu2{nullptr};
};
TORCH_MODULE(ResidualBlockWithStride);
struct ResidualBlockUpsampleImpl : torch::nn::Module {
    ResidualBlockUpsampleImpl(std::int64_t in, std::int64_t out, std::int64_t factor = 2);
    torch::Tensor forward(const torch::Tensor& input);
    torch::nn::Sequential subpel_conv{nullptr}, upsample{nullptr};
    torch::nn::Conv2d conv{nullptr};
    torch::nn::LeakyReLU leaky_relu{nullptr}, leaky_relu2{nullptr};
};
TORCH_MODULE(ResidualBlockUpsample);
struct ResBlockImpl : torch::nn::Module {
    ResBlockImpl(std::int64_t channels, double slope = 0.01, bool start = true, bool end = false, bool bottleneck = false);
    torch::Tensor forward(const torch::Tensor& input);
    torch::nn::AnyModule relu;
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
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
struct SELayerImpl : torch::nn::Module {
    explicit SELayerImpl(std::int64_t channels, std::int64_t reduction = 16);
    torch::Tensor forward(const torch::Tensor& input);
    torch::nn::Sequential fc{nullptr};
};
TORCH_MODULE(SELayer);
struct ConvBlockResidualImpl : torch::nn::Module {
    ConvBlockResidualImpl(std::int64_t in, std::int64_t out, bool se_layer = true);
    torch::Tensor forward(const torch::Tensor& input);
    torch::nn::Sequential conv{nullptr};
    torch::nn::Conv2d up_dim{nullptr};
};
TORCH_MODULE(ConvBlockResidual);
struct UNetImpl : torch::nn::Module {
    UNetImpl(std::int64_t in = 64, std::int64_t out = 64);
    torch::Tensor forward(const torch::Tensor& input);
    torch::nn::MaxPool2d max_pool{nullptr};
    ConvBlockResidual conv1{nullptr}, conv2{nullptr}, conv3{nullptr}, up_conv3{nullptr}, up_conv2{nullptr};
    torch::nn::Sequential context_refine{nullptr}, up3{nullptr}, up2{nullptr};
};
TORCH_MODULE(UNet);
std::pair<torch::nn::Sequential, torch::nn::Sequential> get_enc_dec_models(
    std::int64_t input, std::int64_t output, std::int64_t channels);
std::pair<torch::nn::Sequential, torch::nn::Sequential> get_hyper_enc_dec_models(std::int64_t y, std::int64_t z);

struct DualPriorResult { torch::Tensor y_res, y_q, y_hat, scales_hat; };
struct DualPriorStream { torch::Tensor y_q_w_0, y_q_w_1, scales_w_0, scales_w_1, y_hat; };
struct MaskedResult { torch::Tensor y_res, y_q, y_hat, scales_hat; };
struct CompressionModelImpl : torch::nn::Module {
    CompressionModelImpl(std::string distribution, std::int64_t z_channels,
                         std::optional<std::int64_t> mv_z_channels = std::nullopt);
    torch::Tensor forward(const torch::Tensor& input);  // Base nn.Module forward is unimplemented.
    void initialize_weights();
    torch::Tensor quant(const torch::Tensor& input, bool force_detach = false) const;
    torch::Tensor add_noise(const torch::Tensor& input) const;
    static torch::Tensor probs_to_bits(const torch::Tensor& probabilities);
    torch::Tensor get_y_gaussian_bits(const torch::Tensor& y, const torch::Tensor& sigma) const;
    torch::Tensor get_y_laplace_bits(const torch::Tensor& y, const torch::Tensor& sigma) const;
    torch::Tensor get_z_bits(const torch::Tensor& z, BitEstimator& estimator) const;
    void update(bool force = false);
    static std::pair<torch::Tensor, torch::Tensor> get_mask(std::int64_t height, std::int64_t width, torch::Device device);
    MaskedResult process_with_mask(const torch::Tensor& y, const torch::Tensor& scales,
                                  const torch::Tensor& means, const torch::Tensor& mask) const;
    DualPriorResult forward_dual_prior(const torch::Tensor& y, const torch::Tensor& means,
                                      const torch::Tensor& scales, const torch::Tensor& quant_step,
                                      torch::nn::Sequential& spatial_prior);
    DualPriorStream compress_dual_prior(const torch::Tensor& y, const torch::Tensor& means,
                                       const torch::Tensor& scales, const torch::Tensor& quant_step,
                                       torch::nn::Sequential& spatial_prior);
    torch::Tensor decompress_dual_prior(const torch::Tensor& means, const torch::Tensor& scales,
                                       const torch::Tensor& quant_step, torch::nn::Sequential& spatial_prior);
    void load_state_dict(const StateDict& state, bool strict = true);
    void load_checkpoint(const std::filesystem::path& path, bool strict = true);
    const std::string y_distribution;
    const std::int64_t z_channel;
    const std::optional<std::int64_t> mv_z_channel;
    std::shared_ptr<EntropyCoder> entropy_coder;
    BitEstimator bit_estimator_z{nullptr}, bit_estimator_z_mv{nullptr};
    GaussianEncoder gaussian_encoder;
    torch::nn::MSELoss mse{nullptr};
    MSSSIM ssim{nullptr};
private:
    struct DualStages;
    DualStages dual_stages(const torch::Tensor& y, const torch::Tensor& means, const torch::Tensor& scales,
                          const torch::Tensor& quant_step, torch::nn::Sequential& spatial_prior);
};
TORCH_MODULE(CompressionModel);

struct RoundedQ { double scale; std::uint16_t index; };
RoundedQ get_rounded_q(double scale);
RoundedQ get_rounded_q(const torch::Tensor& scale);
StateDict get_state_dict(const std::filesystem::path& path);
struct IFrameStream { std::uint32_t height{}, width{}; std::uint16_t q_index{}; std::string bit_stream; };
struct PFrameStream { std::uint16_t mv_y_q_index{}, y_q_index{}; std::string bit_stream; };
void write_i(const std::filesystem::path& path, const IFrameStream& stream);
IFrameStream read_i(const std::filesystem::path& path);
void write_p(const std::filesystem::path& path, const PFrameStream& stream);
PFrameStream read_p(const std::filesystem::path& path);

struct ImageResult { torch::Tensor x_hat, mse, ssim, bpp, bpp_y, bpp_z; double bit{}; };
struct IntraNoARImpl : CompressionModelImpl {
    explicit IntraNoARImpl(std::int64_t n = 192, std::int64_t anchors = 4);
    torch::Tensor get_curr_q(const torch::Tensor& scale) const;
    ImageResult forward(const torch::Tensor& input, const torch::Tensor& scale = {});
    static torch::Tensor get_q_scales_from_ckpt(const std::filesystem::path& path);
    ImageResult encode_decode(const torch::Tensor& input, const torch::Tensor& scale,
                             std::optional<std::filesystem::path> output = std::nullopt,
                             std::optional<std::int64_t> width = std::nullopt,
                             std::optional<std::int64_t> height = std::nullopt);
    std::string compress(const torch::Tensor& input, const torch::Tensor& scale);
    torch::Tensor decompress(const std::string& stream, std::int64_t height, std::int64_t width, const torch::Tensor& scale);
    torch::nn::Sequential enc{nullptr}, dec{nullptr}, refine{nullptr}, hyper_enc{nullptr}, hyper_dec{nullptr},
                          y_prior_fusion{nullptr}, y_spatial_prior{nullptr};
    torch::Tensor q_basic, q_scale;
    const std::int64_t N, anchor_num;
};
TORCH_MODULE(IntraNoAR);

struct Dpb { torch::Tensor ref_frame, ref_feature, ref_y, ref_mv_y; };
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
    ResBlock res1{nullptr}, res2{nullptr};
};
TORCH_MODULE(ContextualEncoder);
struct ContextualDecoderImpl : torch::nn::Module {
    ContextualDecoderImpl(std::int64_t n = 64, std::int64_t m = 96);
    torch::Tensor forward(const torch::Tensor& input, const Contexts& contexts);
    torch::nn::Sequential up1{nullptr}, up2{nullptr}, up3{nullptr}, up4{nullptr};
    ResBlock res1{nullptr}, res2{nullptr};
};
TORCH_MODULE(ContextualDecoder);
struct ReconGenerationImpl : torch::nn::Module {
    ReconGenerationImpl(std::int64_t context = 64, std::int64_t residual = 32, std::int64_t channels = 64);
    std::pair<torch::Tensor, torch::Tensor> forward(const torch::Tensor& first, const torch::Tensor& second);
    torch::nn::Conv2d first_conv{nullptr}, recon_conv{nullptr};
    UNet unet_1{nullptr}, unet_2{nullptr};
};
TORCH_MODULE(ReconGeneration);
struct ForwardResult {
    torch::Tensor bpp_mv_y, bpp_mv_z, bpp_y, bpp_z, bpp, me_mse, mse, ssim;
    Dpb dpb;
    torch::Tensor bit, bit_y, bit_z, bit_mv_y, bit_mv_z;
};
struct CompressedFrame { Dpb dbp; std::string bit_stream; };  // Python compress returns the key "dbp".
struct FrameResult {
    Dpb dpb;
    double bit{}, decoding_time{};
    std::optional<double> bit_y, bit_z, bit_mv_y, bit_mv_z;  // Absent in stream mode.
};
struct DMCImpl : CompressionModelImpl {
    static constexpr std::int64_t channel_mv = 64, channel_N = 64, channel_M = 96;
    explicit DMCImpl(std::int64_t anchors = 4);
    Contexts multi_scale_feature_extractor(const Dpb& reference);
    std::pair<Contexts, torch::Tensor> motion_compensation(const Dpb& reference, const torch::Tensor& motion);
    static std::pair<torch::Tensor, torch::Tensor> get_q_scales_from_ckpt(const std::filesystem::path& path);
    torch::Tensor get_curr_mv_y_q(const torch::Tensor& scale) const;
    torch::Tensor get_curr_y_q(const torch::Tensor& scale) const;
    ForwardResult forward_one_frame(const torch::Tensor& input, const Dpb& reference,
                                   const torch::Tensor& mv_scale = {}, const torch::Tensor& y_scale = {});
    ForwardResult forward(const torch::Tensor& input, const Dpb& reference,
                          const torch::Tensor& mv_scale = {}, const torch::Tensor& y_scale = {});
    CompressedFrame compress(const torch::Tensor& input, const Dpb& reference,
                             const torch::Tensor& mv_scale, const torch::Tensor& y_scale);
    Dpb decompress(const Dpb& reference, const std::string& stream, std::int64_t height, std::int64_t width,
                   const torch::Tensor& mv_scale, const torch::Tensor& y_scale);
    FrameResult encode_decode(const torch::Tensor& input, const Dpb& reference,
                              std::optional<std::filesystem::path> output = std::nullopt,
                              std::optional<std::int64_t> width = std::nullopt,
                              std::optional<std::int64_t> height = std::nullopt,
                              const torch::Tensor& mv_scale = {}, const torch::Tensor& y_scale = {});
    const std::string DMC_version = "1.19";
    const std::int64_t anchor_num;
    Spynet optic_flow{nullptr};
    torch::nn::Sequential mv_encoder{nullptr}, mv_decoder{nullptr}, mv_hyper_prior_encoder{nullptr},
                          mv_hyper_prior_decoder{nullptr}, mv_y_prior_fusion{nullptr}, mv_y_spatial_prior{nullptr};
    torch::nn::Conv2d feature_adaptor_I{nullptr}, feature_adaptor_P{nullptr};
    FeatureExtractor feature_extractor{nullptr};
    MultiScaleContextFusion context_fusion_net{nullptr};
    ContextualEncoder contextual_encoder{nullptr};
    torch::nn::Sequential contextual_hyper_prior_encoder{nullptr}, contextual_hyper_prior_decoder{nullptr},
                          temporal_prior_encoder{nullptr}, y_prior_fusion{nullptr}, y_spatial_prior{nullptr};
    ContextualDecoder contextual_decoder{nullptr};
    ReconGeneration recon_generation_net{nullptr};
    torch::Tensor mv_y_q_basic, mv_y_q_scale, y_q_basic, y_q_scale;
private:
    struct Analysis;
    Analysis analyze(const torch::Tensor& input, const Dpb& reference, const torch::Tensor& mv_scale,
                     const torch::Tensor& y_scale, bool stream);
};
TORCH_MODULE(DMC);
}  // namespace uf::hem
