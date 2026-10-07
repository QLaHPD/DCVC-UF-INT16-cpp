// Microsoft Corporation, MIT. Layers also adapt InterDigital Apache-2.0
// source. Retained notices are in licenses/.
#pragma once
#include "uf/dcvc2021.hpp"
#include <array>
#include <memory>
#include <variant>

namespace uf::fm {
using CdfTables = dcvc2021::CdfTables;
using StateDict = std::map<std::string, torch::Tensor>;
using FourParts = std::array<torch::Tensor, 4>;
using SliceShape = std::array<std::int64_t, 4>;
// monostate preserves the Python None default: using it as an index fails.
using QIndex = std::variant<std::monostate, std::int64_t, std::vector<std::int64_t>>;
torch::Tensor normal_cdf(const torch::Tensor& input, const torch::Tensor& scales);
torch::nn::Conv2d conv3x3(std::int64_t in, std::int64_t out, std::int64_t stride = 1);
torch::nn::Conv2d conv1x1(std::int64_t in, std::int64_t out, std::int64_t stride = 1);
torch::nn::Sequential subpel_conv3x3(std::int64_t in, std::int64_t out, std::int64_t factor = 1);
torch::nn::Sequential subpel_conv1x1(std::int64_t in, std::int64_t out, std::int64_t factor = 1);
struct ResidualBlockWithStride2Impl : torch::nn::Module {
    ResidualBlockWithStride2Impl(std::int64_t in, std::int64_t out, bool inplace = false);
    torch::Tensor forward(const torch::Tensor& input);
    torch::nn::Conv2d down{nullptr};
    torch::nn::Sequential conv{nullptr};
};
TORCH_MODULE(ResidualBlockWithStride2);
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
    DepthConvImpl(std::int64_t in, std::int64_t out, double slope = 0.01, bool inplace = false);
    torch::Tensor forward(const torch::Tensor& input);
    torch::nn::Sequential conv1{nullptr};
    torch::nn::Conv2d depth_conv{nullptr}, conv2{nullptr}, adaptor{nullptr};
};
TORCH_MODULE(DepthConv);
struct DepthConv2Impl : torch::nn::Module {
    DepthConv2Impl(std::int64_t in, std::int64_t out, double slope = 0.01, bool inplace = false);
    torch::Tensor forward(const torch::Tensor& input);
    torch::nn::Sequential conv1{nullptr};
    torch::nn::Conv2d conv2{nullptr}, out_conv{nullptr}, adaptor{nullptr};
};
TORCH_MODULE(DepthConv2);
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
struct ConvFFN3Impl : torch::nn::Module {
    ConvFFN3Impl(std::int64_t channels, bool inplace = false);
    torch::Tensor forward(const torch::Tensor& input);
    torch::nn::Conv2d conv{nullptr}, conv_out{nullptr};
    torch::nn::LeakyReLU relu1{nullptr}, relu2{nullptr};
};
TORCH_MODULE(ConvFFN3);
struct DepthConvBlockImpl : torch::nn::Module {
    DepthConvBlockImpl(std::int64_t in, std::int64_t out, double depth_slope = 0.01,
                       double ffn_slope = 0.1, bool inplace = false, int variant = 1);
    torch::Tensor forward(const torch::Tensor& input);
    torch::nn::Sequential block{nullptr};
};
TORCH_MODULE(DepthConvBlock);
struct DepthConvBlock2Impl : DepthConvBlockImpl {
    DepthConvBlock2Impl(std::int64_t in, std::int64_t out, double depth_slope = 0.01, double ffn_slope = 0.1, bool inplace = false);
};
TORCH_MODULE(DepthConvBlock2);
struct DepthConvBlock3Impl : DepthConvBlockImpl {
    DepthConvBlock3Impl(std::int64_t in, std::int64_t out, double depth_slope = 0.01, double ffn_slope = 0.1, bool inplace = false);
};
TORCH_MODULE(DepthConvBlock3);
struct DepthConvBlock4Impl : DepthConvBlockImpl {
    DepthConvBlock4Impl(std::int64_t in, std::int64_t out, double depth_slope = 0.01, bool inplace = false);
};
TORCH_MODULE(DepthConvBlock4);
struct ResBlockImpl : torch::nn::Module {
    ResBlockImpl(std::int64_t channels, double slope = 0.01, bool end = false, bool bottleneck = false, bool inplace = false);
    torch::Tensor forward(const torch::Tensor& input);
    torch::nn::LeakyReLU first_layer{nullptr}, relu{nullptr};
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    torch::nn::AnyModule last_layer;
};
TORCH_MODULE(ResBlock);
struct MEBasicImpl : torch::nn::Module {
    explicit MEBasicImpl(int complexity = 0);
    torch::Tensor forward(const torch::Tensor& input);
    torch::nn::ReLU relu{nullptr};
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr}, conv4{nullptr}, conv5{nullptr};
    const bool by_pass;
};
TORCH_MODULE(MEBasic);
struct ME_SpynetImpl : torch::nn::Module {
    ME_SpynetImpl();
    torch::Tensor forward(const torch::Tensor& current, const torch::Tensor& reference);
    MEBasic me_8x{nullptr}, me_4x{nullptr}, me_2x{nullptr}, me_1x{nullptr};
};
TORCH_MODULE(ME_Spynet);
struct UNetImpl : torch::nn::Module {
    UNetImpl(std::int64_t in = 64, std::int64_t out = 64, bool inplace = false, bool second = false);
    torch::Tensor forward(const torch::Tensor& input);
    torch::nn::Conv2d down1{nullptr}, down2{nullptr};
    torch::nn::MaxPool2d max_pool{nullptr};
    DepthConvBlock conv1{nullptr}, conv2{nullptr}, conv3{nullptr}, up_conv3{nullptr}, up_conv2{nullptr};
    torch::nn::Sequential context_refine{nullptr}, up3{nullptr}, up2{nullptr};
};
TORCH_MODULE(UNet);
struct UNet2Impl : UNetImpl {
    UNet2Impl(std::int64_t in = 64, std::int64_t out = 64, bool inplace = false);
};
TORCH_MODULE(UNet2);
torch::Tensor bilinear_up(const torch::Tensor& input);
torch::Tensor bilinear_down(const torch::Tensor& input);
void set_force_recalculate_grid(bool value);
torch::Tensor torch_warp(const torch::Tensor& feature, const torch::Tensor& flow);
torch::Tensor flow_warp(const torch::Tensor& feature, const torch::Tensor& flow);
torch::Tensor block_mc_func(const torch::Tensor& feature, const torch::Tensor& flow);

class EntropyCoder {
public:
    explicit EntropyCoder(bool threaded = false, int stream_parts = 1);
    ~EntropyCoder();
    EntropyCoder(const EntropyCoder&) = delete;
    EntropyCoder& operator=(const EntropyCoder&) = delete;
    static torch::Tensor pmf_to_quantized_cdf(const torch::Tensor& pmf, int precision = 16);
    static torch::Tensor pmf_to_cdf(const torch::Tensor& pmf, const torch::Tensor& tails,
                                  const torch::Tensor& lengths, std::int64_t maximum);
    int add_cdf(const torch::Tensor& cdf, const torch::Tensor& lengths, const torch::Tensor& offsets);
    void empty_cdf_buffer();
    void reset();
    void encode_with_indexes(const torch::Tensor& symbols, const torch::Tensor& indexes, int group);
    // Native counterpart of the flattened NumPy entry points.
    void encode_with_indexes_np(const std::vector<std::int16_t>& symbols, const std::vector<std::int16_t>& indexes, int group);
    void flush();
    std::string get_encoded_stream();
    void set_stream(const std::string& stream);
    torch::Tensor decode_stream(const torch::Tensor& indexes, int group);
    std::vector<std::int16_t> decode_stream_np(const std::vector<std::int16_t>& indexes, int group);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
class AEHelper {
public:
    void set_cdf_info(const torch::Tensor& cdf, const torch::Tensor& lengths, const torch::Tensor& offsets);
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> get_cdf_info() const;
    std::shared_ptr<EntropyCoder> entropy_coder;
    std::optional<int> cdf_group_index;
    torch::Tensor quantized_cdf, cdf_length, offset;  // Plain CPU attributes.
};
struct BitparmImpl : torch::nn::Module {
    BitparmImpl(std::int64_t qp_num, std::int64_t channels, bool final = false);
    torch::Tensor forward(const torch::Tensor& input, const torch::Tensor& index);
    torch::Tensor h, b, a;
private:
    bool final_;
};
TORCH_MODULE(Bitparm);
struct BitEstimatorImpl : AEHelper, torch::nn::Module {
    BitEstimatorImpl(std::int64_t qp_num, std::int64_t channels);
    torch::Tensor forward(const torch::Tensor& input, const torch::Tensor& index);
    torch::Tensor get_logits_cdf(const torch::Tensor& input, const torch::Tensor& index);
    torch::Tensor get_cdf(const torch::Tensor& input, const torch::Tensor& index);
    void update(bool force = false, std::shared_ptr<EntropyCoder> coder = {});
    torch::Tensor build_indexes(const std::vector<std::int64_t>& size, std::int64_t qp) const;
    std::vector<std::int16_t> build_indexes_np(const std::vector<std::int64_t>& size, std::int64_t qp) const;
    void encode(const torch::Tensor& input, std::int64_t qp);
    torch::Tensor decode_stream(std::pair<std::int64_t, std::int64_t> size, torch::ScalarType dtype, torch::Device device, std::int64_t qp);
    Bitparm f1{nullptr}, f2{nullptr}, f3{nullptr}, f4{nullptr};
    const std::int64_t qp_num, channel;
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
using PriorResult = MaskedResult;
struct PriorStream { FourParts y_q, scales; torch::Tensor y_hat; };
using PriorAdaptors = std::array<DepthConvBlock2, 3>;
struct CompressionModelImpl : torch::nn::Module {
    CompressionModelImpl(std::string distribution, std::int64_t z_channels,
                         std::optional<std::int64_t> mv_z_channels = std::nullopt,
                         bool threaded = false, int stream_parts = 1);
    static torch::Tensor quant(const torch::Tensor& input);
    static std::int64_t get_qp_num() { return 64; }
    static torch::Tensor get_one_q_scale(const torch::Tensor& scale, std::int64_t index);
    static torch::Tensor get_curr_q(const torch::Tensor& scale, const QIndex& index);
    static torch::Tensor get_index_tensor(const QIndex& index, torch::Device device);
    static torch::Tensor probs_to_bits(const torch::Tensor& probabilities);
    static torch::Tensor get_y_gaussian_bits(const torch::Tensor& y, const torch::Tensor& scales);
    static torch::Tensor get_y_laplace_bits(const torch::Tensor& y, const torch::Tensor& scales);
    static torch::Tensor get_z_bits(const torch::Tensor& z, BitEstimator& estimator, const torch::Tensor& index);
    void update(bool force = false);
    static std::pair<torch::Tensor, SliceShape> pad_for_y(const torch::Tensor& y);
    static SliceShape get_to_y_slice_shape(std::int64_t height, std::int64_t width);
    static torch::Tensor slice_to_y(const torch::Tensor& params, const SliceShape& slice);
    static std::array<torch::Tensor, 4> separate_prior(const torch::Tensor& params, bool is_video = false);
    std::array<torch::Tensor, 2> get_mask(std::int64_t height, std::int64_t width, torch::ScalarType dtype, torch::Device device);
    static FourParts get_one_channel_four_parts_mask(std::int64_t height, std::int64_t width, torch::ScalarType dtype, torch::Device device);
    FourParts get_mask_four_parts(std::int64_t batch, std::int64_t channels, std::int64_t height, std::int64_t width,
                                 torch::ScalarType dtype, torch::Device device);
    static MaskedResult process_with_mask(const torch::Tensor& y, const torch::Tensor& scales,
                                         const torch::Tensor& means, const torch::Tensor& mask);
    static torch::Tensor combine_four_parts(const std::array<FourParts, 4>& parts);
    static torch::Tensor combine_for_writing(const torch::Tensor& input);
    PriorResult forward_four_part_prior(const torch::Tensor& y, const torch::Tensor& params,
                                       const PriorAdaptors& adaptors, torch::nn::Sequential& spatial,
                                       torch::nn::Conv2d reduction = nullptr);
    PriorStream compress_four_part_prior(const torch::Tensor& y, const torch::Tensor& params,
                                        const PriorAdaptors& adaptors, torch::nn::Sequential& spatial,
                                        torch::nn::Conv2d reduction = nullptr);
    torch::Tensor decompress_four_part_prior(const torch::Tensor& params, const PriorAdaptors& adaptors,
                                            torch::nn::Sequential& spatial, torch::nn::Conv2d reduction = nullptr);
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
    std::map<std::string, std::vector<torch::Tensor>> masks;  // Source key excludes dtype/device.
private:
    struct Stages;
    Stages prior_stages(const torch::Tensor& y, const torch::Tensor& params, const PriorAdaptors& adaptors,
                        torch::nn::Sequential& spatial, torch::nn::Conv2d reduction);
};
TORCH_MODULE(CompressionModel);
StateDict get_state_dict(const std::filesystem::path& path);

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
struct SPS;
struct ImageResult { torch::Tensor x_hat, bit; };
struct ImageStream { std::string bit_stream; torch::Tensor x_hat; };
struct ImageFrame { double bit{}; torch::Tensor x_hat; };
struct DMCIImpl : CompressionModelImpl {
    DMCIImpl(std::int64_t n = 256, std::int64_t z_channels = 128, bool threaded = false, int stream_parts = 1, bool inplace = false);
    ImageResult forward_one_frame(const torch::Tensor& input, const QIndex& index = std::monostate{});
    ImageFrame encode(const torch::Tensor& input, const QIndex& index, int sps_id = 0, std::ostream* output = nullptr);
    ImageStream compress(const torch::Tensor& input, std::int64_t index);
    torch::Tensor decompress(const std::string& stream, const SPS& sps);
    IntraEncoder enc{nullptr};
    IntraDecoder dec{nullptr};
    torch::nn::Sequential hyper_enc{nullptr}, hyper_dec{nullptr}, y_prior_fusion{nullptr}, y_spatial_prior{nullptr}, refine{nullptr};
    torch::nn::Conv2d y_spatial_prior_reduction{nullptr};
    DepthConvBlock2 y_spatial_prior_adaptor_1{nullptr}, y_spatial_prior_adaptor_2{nullptr}, y_spatial_prior_adaptor_3{nullptr};
    torch::Tensor q_scale_enc, q_scale_dec;
private:
    PriorAdaptors adaptors() const;
};
TORCH_MODULE(DMCI);
}  // namespace uf::fm
