// Copyright (c) Microsoft Corporation. Licensed under the MIT License.
// Native DCVC-family/DCVC-DC interfaces; embedded layers/streams also adapt
// InterDigital Apache-2.0 source. Retained notices are in licenses/.
#pragma once
#include "uf/dcvc2021.hpp"
#include <array>
#include <memory>
#include <tuple>

namespace uf::dc {
using CdfTables = dcvc2021::CdfTables;
using StateDict = std::map<std::string, torch::Tensor>;
using FourParts = std::array<torch::Tensor, 4>;
using SliceShape = std::array<std::int64_t, 4>;
torch::Tensor normal_cdf(const torch::Tensor& input, const torch::Tensor& scales);
torch::Tensor flow_warp(const torch::Tensor& feature, const torch::Tensor& flow);
torch::Tensor bilinear_up(const torch::Tensor& input);
torch::Tensor bilinear_down(const torch::Tensor& input);
torch::nn::Conv2d conv3x3(std::int64_t in, std::int64_t out, std::int64_t stride = 1);
torch::nn::Conv2d conv1x1(std::int64_t in, std::int64_t out, std::int64_t stride = 1);
torch::nn::Sequential subpel_conv3x3(std::int64_t in, std::int64_t out, std::int64_t factor = 1);
torch::nn::Sequential subpel_conv1x1(std::int64_t in, std::int64_t out, std::int64_t factor = 1);

struct ResidualBlockWithStrideImpl : torch::nn::Module {
    ResidualBlockWithStrideImpl(std::int64_t in, std::int64_t out, std::int64_t stride = 2, bool inplace = false);
    torch::Tensor forward(const torch::Tensor& input);
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, downsample{nullptr};
    torch::nn::LeakyReLU leaky_relu{nullptr}, leaky_relu2{nullptr};
};
TORCH_MODULE(ResidualBlockWithStride);
struct ResidualBlockUpsampleImpl : torch::nn::Module {
    ResidualBlockUpsampleImpl(std::int64_t in, std::int64_t out, std::int64_t factor = 2, bool inplace = false);
    torch::Tensor forward(const torch::Tensor& input);
    torch::nn::Sequential subpel_conv{nullptr}, upsample{nullptr};
    torch::nn::Conv2d conv{nullptr};
    torch::nn::LeakyReLU leaky_relu{nullptr}, leaky_relu2{nullptr};
};
TORCH_MODULE(ResidualBlockUpsample);
struct ResidualBlockImpl : torch::nn::Module {
    ResidualBlockImpl(std::int64_t in, std::int64_t out, double slope = 0.01, bool inplace = false);
    torch::Tensor forward(const torch::Tensor& input);
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, adaptor{nullptr};
    torch::nn::LeakyReLU leaky_relu{nullptr};
};
TORCH_MODULE(ResidualBlock);
struct DepthConvImpl : torch::nn::Module {
    DepthConvImpl(std::int64_t in, std::int64_t out, std::int64_t kernel = 3, std::int64_t stride = 1,
                  double slope = 0.01, bool inplace = false);
    torch::Tensor forward(const torch::Tensor& input);
    torch::nn::Sequential conv1{nullptr};
    torch::nn::Conv2d depth_conv{nullptr}, conv2{nullptr}, adaptor{nullptr};
};
TORCH_MODULE(DepthConv);
struct ConvFFNImpl : torch::nn::Module {
    ConvFFNImpl(std::int64_t channels, double slope = 0.1, bool inplace = false);
    torch::Tensor forward(const torch::Tensor& input);
    torch::nn::Sequential conv{nullptr};
};
TORCH_MODULE(ConvFFN);
struct ConvFFN2Impl : torch::nn::Module {
    ConvFFN2Impl(std::int64_t channels, double slope = 0.1, bool inplace = false);
    torch::Tensor forward(const torch::Tensor& input);
    torch::nn::Conv2d conv{nullptr}, conv_out{nullptr};
    torch::nn::LeakyReLU relu{nullptr};
};
TORCH_MODULE(ConvFFN2);
struct DepthConvBlockImpl : torch::nn::Module {
    DepthConvBlockImpl(std::int64_t in, std::int64_t out, std::int64_t kernel = 3, std::int64_t stride = 1,
                       double depth_slope = 0.01, double ffn_slope = 0.1, bool inplace = false, bool gated = false);
    torch::Tensor forward(const torch::Tensor& input);
    torch::nn::Sequential block{nullptr};
};
TORCH_MODULE(DepthConvBlock);
struct DepthConvBlock2Impl : DepthConvBlockImpl {
    DepthConvBlock2Impl(std::int64_t in, std::int64_t out, std::int64_t kernel = 3, std::int64_t stride = 1,
                        double depth_slope = 0.01, double ffn_slope = 0.1, bool inplace = false);
};
TORCH_MODULE(DepthConvBlock2);
struct ResBlockImpl : torch::nn::Module {
    ResBlockImpl(std::int64_t channels, double slope = 0.01, bool end = false, bool bottleneck = false, bool inplace = false);
    torch::Tensor forward(const torch::Tensor& input);
    torch::nn::LeakyReLU first_layer{nullptr}, relu{nullptr};
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    torch::nn::AnyModule last_layer;
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
struct UNetImpl : torch::nn::Module {
    UNetImpl(std::int64_t in = 64, std::int64_t out = 64, bool inplace = false, bool gated = false);
    torch::Tensor forward(const torch::Tensor& input);
    torch::nn::MaxPool2d max_pool{nullptr};
    DepthConvBlock conv1{nullptr}, conv2{nullptr}, conv3{nullptr}, up_conv3{nullptr}, up_conv2{nullptr};
    torch::nn::Sequential context_refine{nullptr}, up3{nullptr}, up2{nullptr};
};
TORCH_MODULE(UNet);
struct UNet2Impl : UNetImpl {
    UNet2Impl(std::int64_t in = 64, std::int64_t out = 64, bool inplace = false);
};
TORCH_MODULE(UNet2);
std::pair<torch::nn::Sequential, torch::nn::Sequential> get_hyper_enc_dec_models(
    std::int64_t y, std::int64_t z, bool reduced = false, bool inplace = false);

class EntropyCoder {
public:
    explicit EntropyCoder(bool threaded = false, int stream_parts = 1);
    ~EntropyCoder();
    EntropyCoder(const EntropyCoder&) = delete;
    EntropyCoder& operator=(const EntropyCoder&) = delete;
    static torch::Tensor pmf_to_quantized_cdf(const torch::Tensor& pmf, int precision = 16);
    static torch::Tensor pmf_to_cdf(const torch::Tensor& pmf, const torch::Tensor& tails,
                                  const torch::Tensor& lengths, std::int64_t maximum);
    void reset();
    void encode_with_indexes(const torch::Tensor& symbols, const torch::Tensor& indexes,
                             const torch::Tensor& cdf, const torch::Tensor& lengths, const torch::Tensor& offsets);
    void flush();
    std::string get_encoded_stream();
    void set_stream(const std::string& stream);
    torch::Tensor decode_stream(const torch::Tensor& indexes, const torch::Tensor& cdf,
                                const torch::Tensor& lengths, const torch::Tensor& offsets);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
class AEHelper {
public:
    void set_entropy_coder(std::shared_ptr<EntropyCoder> coder);
    void set_cdf_info(const torch::Tensor& cdf, const torch::Tensor& lengths, const torch::Tensor& offsets);
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> get_cdf_info() const;
    std::shared_ptr<EntropyCoder> entropy_coder;
    torch::Tensor quantized_cdf, cdf_length, offset;  // Plain CPU attributes.
};
struct BitparmImpl : torch::nn::Module {
    BitparmImpl(std::int64_t channels, bool final = false);
    torch::Tensor forward(const torch::Tensor& input);
    torch::Tensor h, b, a;
private:
    bool final_;
};
TORCH_MODULE(Bitparm);
struct BitEstimatorImpl : AEHelper, torch::nn::Module {
    explicit BitEstimatorImpl(std::int64_t channels);
    torch::Tensor forward(const torch::Tensor& input);
    torch::Tensor get_logits_cdf(const torch::Tensor& input);
    torch::Tensor get_cdf(const torch::Tensor& input);
    void update(bool force = false, std::shared_ptr<EntropyCoder> coder = {});
    static torch::Tensor build_indexes(const std::vector<std::int64_t>& size);
    void encode(const torch::Tensor& input);
    torch::Tensor decode_stream(std::pair<std::int64_t, std::int64_t> size, torch::ScalarType dtype, torch::Device device);
    Bitparm f1{nullptr}, f2{nullptr}, f3{nullptr}, f4{nullptr};
    const std::int64_t channel;
};
TORCH_MODULE(BitEstimator);
class GaussianEncoder : public AEHelper {
public:
    explicit GaussianEncoder(std::string distribution = "laplace");
    static torch::Tensor get_scale_table(double minimum, double maximum, std::int64_t levels);
    void update(bool force = false, std::shared_ptr<EntropyCoder> coder = {});
    torch::Tensor build_indexes(const torch::Tensor& scales) const;
    void encode(const torch::Tensor& input, const torch::Tensor& scales);
    torch::Tensor decode_stream(const torch::Tensor& scales, torch::ScalarType dtype, torch::Device device);
    const std::string distribution;
    torch::Tensor scale_table;
private:
    double log_min_, log_step_;
};

struct MaskedResult { torch::Tensor y_res, y_q, y_hat, scales_hat; };
struct PriorResult { torch::Tensor y_res, y_q, y_hat, scales_hat; };
struct PriorStream { FourParts y_q, scales; torch::Tensor y_hat; };
using PriorAdaptors = std::array<torch::nn::Conv2d, 3>;
struct CompressionModelImpl : torch::nn::Module {
    CompressionModelImpl(std::string distribution, std::int64_t z_channels,
                         std::optional<std::int64_t> mv_z_channels = std::nullopt,
                         bool threaded = false, int stream_parts = 1);
    torch::Tensor forward(const torch::Tensor& input);
    static torch::Tensor quant(const torch::Tensor& input);
    static torch::Tensor get_curr_q(const torch::Tensor& scale, const torch::Tensor& basic,
                                   std::optional<std::int64_t> index = std::nullopt);
    static torch::Tensor probs_to_bits(const torch::Tensor& probabilities);
    static torch::Tensor get_y_gaussian_bits(const torch::Tensor& y, const torch::Tensor& scales);
    static torch::Tensor get_y_laplace_bits(const torch::Tensor& y, const torch::Tensor& scales);
    static torch::Tensor get_z_bits(const torch::Tensor& z, BitEstimator& estimator);
    void update(bool force = false);
    static std::pair<torch::Tensor, SliceShape> pad_for_y(const torch::Tensor& y);
    static SliceShape get_to_y_slice_shape(std::int64_t height, std::int64_t width);
    static torch::Tensor slice_to_y(const torch::Tensor& params, const SliceShape& slice);
    static std::array<torch::Tensor, 3> separate_prior(const torch::Tensor& params);
    static MaskedResult process_with_mask(const torch::Tensor& y, const torch::Tensor& scales,
                                         const torch::Tensor& means, const torch::Tensor& mask);
    FourParts get_mask_four_parts(std::int64_t height, std::int64_t width, torch::ScalarType dtype, torch::Device device);
    static torch::Tensor combine_four_parts(const std::array<FourParts, 4>& parts);
    PriorResult forward_four_part_prior(const torch::Tensor& y, const torch::Tensor& params,
                                       const PriorAdaptors& adaptors, torch::nn::Sequential& spatial);
    PriorStream compress_four_part_prior(const torch::Tensor& y, const torch::Tensor& params,
                                        const PriorAdaptors& adaptors, torch::nn::Sequential& spatial);
    torch::Tensor decompress_four_part_prior(const torch::Tensor& params, const PriorAdaptors& adaptors,
                                            torch::nn::Sequential& spatial);
    virtual void load_state_dict(const StateDict& state, bool strict = true);
    void load_checkpoint(const std::filesystem::path& path, bool strict = true);
    const std::string y_distribution;
    const std::int64_t z_channel;
    const std::optional<std::int64_t> mv_z_channel;
    const bool ec_thread;
    const int stream_part;
    std::shared_ptr<EntropyCoder> entropy_coder;
    BitEstimator bit_estimator_z{nullptr}, bit_estimator_z_mv{nullptr};
    GaussianEncoder gaussian_encoder;
    std::map<std::string, FourParts> masks;  // Source cache key excludes dtype/device.
private:
    struct Stages;
    Stages prior_stages(const torch::Tensor& y, const torch::Tensor& params,
                        const PriorAdaptors& adaptors, torch::nn::Sequential& spatial);
};
TORCH_MODULE(CompressionModel);

StateDict get_state_dict(const std::filesystem::path& path);
torch::Tensor pad(const torch::Tensor& input, std::int64_t multiple = 64);
torch::Tensor crop(const torch::Tensor& input, std::int64_t height, std::int64_t width);
struct IFrameStream { std::uint32_t height{}, width{}; bool q_in_ckpt{}; std::uint8_t q_index{}; std::string bit_stream; };
struct PFrameStream { bool q_in_ckpt{}; std::uint8_t q_index{}, frame_idx{}; std::string bit_stream; };
void write_i(const std::filesystem::path& path, const IFrameStream& stream);
IFrameStream read_i(const std::filesystem::path& path);
void write_p(const std::filesystem::path& path, const PFrameStream& stream);
PFrameStream read_p(const std::filesystem::path& path);

struct IntraEncoderImpl : torch::nn::Module {
    IntraEncoderImpl(std::int64_t n, bool inplace = false);
    torch::Tensor forward(const torch::Tensor& input, const torch::Tensor& step);
    torch::nn::Sequential enc_1{nullptr}, enc_2{nullptr};
};
TORCH_MODULE(IntraEncoder);
struct IntraDecoderImpl : torch::nn::Module {
    IntraDecoderImpl(std::int64_t n, bool inplace = false);
    torch::Tensor forward(const torch::Tensor& input, const torch::Tensor& step);
    torch::nn::Sequential dec_1{nullptr}, dec_2{nullptr};
};
TORCH_MODULE(IntraDecoder);
struct ImageResult { torch::Tensor x_hat, bit, bpp, bpp_y, bpp_z; };
struct ImageStream { std::string bit_stream; torch::Tensor x_hat; };
struct ImageFrame { double bit{}; torch::Tensor x_hat; };
struct IntraNoARImpl : CompressionModelImpl {
    IntraNoARImpl(std::int64_t n = 256, std::int64_t anchors = 4, bool threaded = false,
                 int stream_parts = 1, bool inplace = false);
    std::pair<torch::Tensor, torch::Tensor> get_q_for_inference(bool in_checkpoint, std::optional<std::int64_t> index);
    ImageResult forward(const torch::Tensor& input, bool in_checkpoint = false, std::optional<std::int64_t> index = std::nullopt);
    static std::pair<torch::Tensor, torch::Tensor> get_q_scales_from_ckpt(const std::filesystem::path& path);
    void load_state_dict(const StateDict& state, bool strict = true) override;
    ImageFrame encode_decode(const torch::Tensor& input, bool in_checkpoint, std::int64_t index,
                            std::optional<std::filesystem::path> output = std::nullopt,
                            std::optional<std::int64_t> width = std::nullopt,
                            std::optional<std::int64_t> height = std::nullopt);
    ImageStream compress(const torch::Tensor& input, bool in_checkpoint, std::int64_t index);
    torch::Tensor decompress(const std::string& stream, std::int64_t height, std::int64_t width,
                             bool in_checkpoint, std::int64_t index);
    IntraEncoder enc{nullptr};
    IntraDecoder dec{nullptr};
    torch::nn::Sequential hyper_enc{nullptr}, hyper_dec{nullptr}, y_prior_fusion{nullptr},
                          y_spatial_prior{nullptr}, refine{nullptr};
    torch::nn::Conv2d y_spatial_prior_adaptor_1{nullptr}, y_spatial_prior_adaptor_2{nullptr}, y_spatial_prior_adaptor_3{nullptr};
    torch::Tensor q_basic_enc, q_scale_enc, q_scale_enc_fine, q_basic_dec, q_scale_dec, q_scale_dec_fine;
    const std::int64_t N, anchor_num;
private:
    PriorAdaptors adaptors() const;
};
TORCH_MODULE(IntraNoAR);

using Contexts = std::array<torch::Tensor, 3>;
struct Dpb { torch::Tensor ref_frame, ref_feature, ref_mv_feature, ref_y, ref_mv_y; };
struct OffsetDiversityImpl : torch::nn::Module {
    OffsetDiversityImpl(std::int64_t in = 48, std::int64_t auxiliary = 53, std::int64_t offsets = 2,
                        std::int64_t groups = 16, double maximum = 40, bool inplace = false);
    torch::Tensor forward(const torch::Tensor& input, const torch::Tensor& auxiliary, const torch::Tensor& flow);
    torch::nn::Sequential conv_offset{nullptr};
    torch::nn::Conv2d fusion{nullptr};
    const std::int64_t in_channel, offset_num, group_num;
    const double max_residue_magnitude;
};
TORCH_MODULE(OffsetDiversity);
struct FeatureExtractorImpl : torch::nn::Module {
    explicit FeatureExtractorImpl(bool inplace = false);
    Contexts forward(const torch::Tensor& input);
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
    ResBlock res_block1{nullptr}, res_block2{nullptr}, res_block3{nullptr};
};
TORCH_MODULE(FeatureExtractor);
struct MultiScaleContextFusionImpl : torch::nn::Module {
    explicit MultiScaleContextFusionImpl(bool inplace = false);
    Contexts forward(const Contexts& contexts);
    torch::nn::Sequential conv3_up{nullptr}, conv2_up{nullptr};
    torch::nn::Conv2d conv3_out{nullptr}, conv2_out{nullptr}, conv1_out{nullptr};
    ResBlock res_block3_up{nullptr}, res_block3_out{nullptr}, res_block2_up{nullptr}, res_block2_out{nullptr}, res_block1_out{nullptr};
};
TORCH_MODULE(MultiScaleContextFusion);
struct MvEncImpl : torch::nn::Module {
    MvEncImpl(std::int64_t in, std::int64_t channels, bool inplace = false);
    torch::Tensor forward(const torch::Tensor& input, const torch::Tensor& context, const torch::Tensor& step);
    torch::nn::Sequential enc_1{nullptr}, enc_3{nullptr};
    ResidualBlockWithStride enc_2{nullptr};
    DepthConvBlock adaptor_0{nullptr}, adaptor_1{nullptr};
};
TORCH_MODULE(MvEnc);
struct MvDecImpl : torch::nn::Module {
    MvDecImpl(std::int64_t out, std::int64_t channels, bool inplace = false);
    std::pair<torch::Tensor, torch::Tensor> forward(const torch::Tensor& input, const torch::Tensor& step);
    torch::nn::Sequential dec_1{nullptr}, dec_3{nullptr};
    ResidualBlockUpsample dec_2{nullptr};
};
TORCH_MODULE(MvDec);
struct ContextualEncoderImpl : torch::nn::Module {
    explicit ContextualEncoderImpl(bool inplace = false);
    torch::Tensor forward(const torch::Tensor& input, const Contexts& contexts, const torch::Tensor& step);
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr}, conv4{nullptr};
    ResBlock res1{nullptr}, res2{nullptr};
};
TORCH_MODULE(ContextualEncoder);
struct ContextualDecoderImpl : torch::nn::Module {
    explicit ContextualDecoderImpl(bool inplace = false);
    torch::Tensor forward(const torch::Tensor& input, const Contexts& contexts, const torch::Tensor& step);
    torch::nn::Sequential up1{nullptr}, up2{nullptr}, up3{nullptr}, up4{nullptr};
    ResBlock res1{nullptr}, res2{nullptr};
};
TORCH_MODULE(ContextualDecoder);
struct ReconGenerationImpl : torch::nn::Module {
    ReconGenerationImpl(std::int64_t context = 48, std::int64_t residual = 32, bool inplace = false);
    std::pair<torch::Tensor, torch::Tensor> forward(const torch::Tensor& first, const torch::Tensor& second);
    torch::nn::Conv2d first_conv{nullptr}, recon_conv{nullptr};
    UNet unet_1{nullptr}, unet_2{nullptr};
};
TORCH_MODULE(ReconGeneration);
struct ForwardResult {
    torch::Tensor bpp_mv_y, bpp_mv_z, bpp_y, bpp_z, bpp;
    Dpb dpb;
    torch::Tensor bit, bit_y, bit_z, bit_mv_y, bit_mv_z;
};
struct CompressedFrame { Dpb dpb; std::string bit_stream; };
struct FrameResult { Dpb dpb; double bit{}, encoding_time{}, decoding_time{}; };
struct DMCImpl : CompressionModelImpl {
    DMCImpl(std::int64_t anchors = 4, bool threaded = false, int stream_parts = 1, bool inplace = false);
    void load_state_dict(const StateDict& state, bool strict = true) override;
    Contexts multi_scale_feature_extractor(const Dpb& reference, std::int64_t index);
    std::pair<Contexts, torch::Tensor> motion_compensation(const Dpb& reference, const torch::Tensor& motion, std::int64_t index);
    static std::array<torch::Tensor, 4> get_q_scales_from_ckpt(const std::filesystem::path& path);
    torch::Tensor mv_prior_param_decoder(const torch::Tensor& z, const Dpb& reference, const SliceShape& slice);
    torch::Tensor res_prior_param_decoder(const torch::Tensor& z, const Dpb& reference, const torch::Tensor& context, const SliceShape& slice);
    std::pair<torch::Tensor, torch::Tensor> get_recon_and_feature(const torch::Tensor& y, const Contexts& contexts, const torch::Tensor& step);
    torch::Tensor motion_estimation_and_mv_encoding(const torch::Tensor& input, const Dpb& reference, const torch::Tensor& step);
    std::array<torch::Tensor, 4> get_q_for_inference(bool in_checkpoint, std::optional<std::int64_t> index);
    CompressedFrame compress(const torch::Tensor& input, const Dpb& reference, bool in_checkpoint,
                             std::int64_t index, std::int64_t frame_index);
    Dpb decompress(const Dpb& reference, const std::string& stream, std::int64_t height, std::int64_t width,
                   bool in_checkpoint, std::int64_t index, std::int64_t frame_index);
    FrameResult encode_decode(const torch::Tensor& input, const Dpb& reference, bool in_checkpoint, std::int64_t index,
                              std::optional<std::filesystem::path> output = std::nullopt,
                              std::optional<std::int64_t> width = std::nullopt,
                              std::optional<std::int64_t> height = std::nullopt, std::int64_t frame_index = 0);
    ForwardResult forward_one_frame(const torch::Tensor& input, const Dpb& reference, bool in_checkpoint = false,
                                    std::optional<std::int64_t> index = std::nullopt, std::int64_t frame_index = 0);
    Spynet optic_flow{nullptr};
    OffsetDiversity align{nullptr};
    MvEnc mv_encoder{nullptr};
    MvDec mv_decoder{nullptr};
    torch::nn::Sequential mv_hyper_prior_encoder{nullptr}, mv_hyper_prior_decoder{nullptr},
                          mv_y_prior_fusion{nullptr}, mv_y_spatial_prior{nullptr};
    DepthConvBlock mv_y_prior_fusion_adaptor_0{nullptr}, mv_y_prior_fusion_adaptor_1{nullptr};
    torch::nn::Conv2d mv_y_spatial_prior_adaptor_1{nullptr}, mv_y_spatial_prior_adaptor_2{nullptr}, mv_y_spatial_prior_adaptor_3{nullptr};
    torch::nn::Conv2d feature_adaptor_I{nullptr};
    torch::nn::ModuleList feature_adaptor{nullptr};
    FeatureExtractor feature_extractor{nullptr};
    MultiScaleContextFusion context_fusion_net{nullptr};
    ContextualEncoder contextual_encoder{nullptr};
    ContextualDecoder contextual_decoder{nullptr};
    ReconGeneration recon_generation_net{nullptr};
    torch::nn::Sequential contextual_hyper_prior_encoder{nullptr}, contextual_hyper_prior_decoder{nullptr},
                          temporal_prior_encoder{nullptr}, y_prior_fusion{nullptr}, y_spatial_prior{nullptr};
    DepthConvBlock y_prior_fusion_adaptor_0{nullptr}, y_prior_fusion_adaptor_1{nullptr};
    torch::nn::Conv2d y_spatial_prior_adaptor_1{nullptr}, y_spatial_prior_adaptor_2{nullptr}, y_spatial_prior_adaptor_3{nullptr};
    torch::Tensor mv_y_q_basic_enc, mv_y_q_scale_enc, mv_y_q_scale_enc_fine, mv_y_q_basic_dec, mv_y_q_scale_dec, mv_y_q_scale_dec_fine;
    torch::Tensor y_q_basic_enc, y_q_scale_enc, y_q_scale_enc_fine, y_q_basic_dec, y_q_scale_dec, y_q_scale_dec_fine;
private:
    struct Analysis;
    Analysis analyze(const torch::Tensor& input, const Dpb& reference, const std::array<torch::Tensor, 4>& q,
                     std::int64_t frame_index, bool stream);
    PriorAdaptors motion_adaptors() const;
    PriorAdaptors image_adaptors() const;
};
TORCH_MODULE(DMC);
}  // namespace uf::dc
