// Copyright (c) Microsoft Corporation. Licensed under the MIT License.
// Native interfaces for DCVC-family/DCVC (the original 2021 codec).
#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace uf::dcvc2021 {

struct CdfTables {
    std::vector<std::vector<std::int32_t>> rows;
    std::vector<std::int32_t> offsets;
    void validate() const;
};

// This format uses 64-bit state, 32-bit little-endian words and 4-bit
// bypasses. It is a different format from UF's byte-aligned entropy coder.
std::vector<std::int32_t> quantized_cdf(const std::vector<float>& pmf);
std::string rans_encode(const std::vector<std::int32_t>& symbols,
                        const std::vector<std::int32_t>& indexes, const CdfTables& tables);
class RansDecoder {
public:
    void set_stream(std::string encoded);
    std::vector<std::int32_t> decode(const std::vector<std::int32_t>& indexes,
                                     const CdfTables& tables);
private:
    std::uint32_t word();
    std::uint32_t bypass();
    void renormalize();
    std::string stream_;
    std::size_t position_{};
    std::uint64_t state_{};
    bool initialized_{};
};

torch::Tensor laplace_cdf(const torch::Tensor& values, const torch::Tensor& scales);
torch::Tensor flow_warp(const torch::Tensor& image, const torch::Tensor& flow);
torch::Tensor pad(const torch::Tensor& input, std::int64_t multiple = 64);
torch::Tensor crop(const torch::Tensor& input, std::int64_t height, std::int64_t width);
std::pair<std::int64_t, std::int64_t> downsampled_shape(
    std::int64_t height, std::int64_t width, std::int64_t divisor);

struct GDNImpl : torch::nn::Module {
    explicit GDNImpl(std::int64_t channels, bool inverse = false, double beta_min = 1e-6,
                     double gamma_init = 0.1, double reparam_offset = 1.0 / 262144.0);
    torch::Tensor forward(torch::Tensor input);
    torch::Tensor beta, gamma;
private:
    bool inverse_;
    double pedestal_, beta_bound_, gamma_bound_;
};
TORCH_MODULE(GDN);

struct ResBlockImpl : torch::nn::Module {
    ResBlockImpl(std::int64_t input_channels, std::int64_t output_channels,
                 std::int64_t kernel = 3, std::int64_t stride = 1);
    torch::Tensor forward(const torch::Tensor& input);
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, adapt_conv{nullptr};
    torch::nn::ReLU relu1{nullptr}, relu2{nullptr};
};
TORCH_MODULE(ResBlock);

struct LeakyResBlockImpl : torch::nn::Module {
    explicit LeakyResBlockImpl(std::int64_t channels);
    torch::Tensor forward(const torch::Tensor& input);
    torch::nn::Sequential conv{nullptr};
};
TORCH_MODULE(LeakyResBlock);

// Adapted from InterDigital's Apache-2.0 MaskedConv2d implementation.
// Weight and bias remain at the module root for Python state_dict names.
struct MaskedConv2dImpl : torch::nn::Conv2dImpl {
    explicit MaskedConv2dImpl(std::int64_t channels);
    torch::Tensor forward(const torch::Tensor& input);
    torch::Tensor mask;
};
TORCH_MODULE(MaskedConv2d);

struct MEBasicImpl : torch::nn::Module {
    MEBasicImpl();
    torch::Tensor forward(const torch::Tensor& input);
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr}, conv4{nullptr}, conv5{nullptr};
    torch::nn::ReLU relu1{nullptr}, relu2{nullptr}, relu3{nullptr}, relu4{nullptr};
};
TORCH_MODULE(MEBasic);

struct SpynetImpl : torch::nn::Module {
    SpynetImpl();
    torch::Tensor forward(const torch::Tensor& current, const torch::Tensor& reference);
    torch::nn::ModuleList moduleBasic{nullptr};
};
TORCH_MODULE(Spynet);

struct BitparmImpl : torch::nn::Module {
    BitparmImpl(std::int64_t channels, bool final = false);
    torch::Tensor forward(const torch::Tensor& input);
    torch::Tensor h, b, a;
private:
    bool final_;
};
TORCH_MODULE(Bitparm);

struct BitEstimatorImpl : torch::nn::Module {
    explicit BitEstimatorImpl(std::int64_t channels);
    torch::Tensor forward(const torch::Tensor& input);
    void update(bool force = false);
    void invalidate();
    std::string compress(const torch::Tensor& input) const;
    torch::Tensor decompress(const std::string& encoded, std::int64_t height,
                             std::int64_t width) const;
    Bitparm f1{nullptr}, f2{nullptr}, f3{nullptr}, f4{nullptr};
private:
    std::int64_t channels_;
    std::optional<CdfTables> tables_;
};
TORCH_MODULE(BitEstimator);

// GaussianEncoder in this Python family actually uses a Laplace distribution.
class LaplaceEncoder {
public:
    LaplaceEncoder();
    void update(bool force = false);
    torch::Tensor build_indexes(const torch::Tensor& scales) const;
    std::string compress(const torch::Tensor& symbols, const torch::Tensor& scales) const;
    torch::Tensor decompress(const std::string& encoded, const torch::Tensor& scales) const;
    void set_stream(const std::string& encoded);
    torch::Tensor decode_stream(const torch::Tensor& scales);
private:
    torch::Tensor scale_table_;
    std::optional<CdfTables> tables_;
    RansDecoder decoder_;
};

struct ForwardResult {
    torch::Tensor bpp_mv_y, bpp_mv_z, bpp_y, bpp_z, bpp, recon_image, context;
};
struct CompressedFrame {
    std::string mv_y_string, mv_z_string, y_string, z_string;
    torch::Tensor recon_image;
    double bpp_mv_y{}, bpp_mv_z{}, bpp_y{}, bpp_z{}, bpp{};
};
struct PFrameStream {
    std::uint32_t height{}, width{};
    std::string mv_y_string, mv_z_string, y_string, z_string;
};
struct IFrameStream {
    std::uint32_t height{}, width{};
    std::string y_string, z_string;
};

void write_p(const std::filesystem::path& output, const PFrameStream& stream);
PFrameStream read_p(const std::filesystem::path& input);
void write_i(const std::filesystem::path& output, const IFrameStream& stream);
IFrameStream read_i(const std::filesystem::path& input);

struct VideoModelImpl : torch::nn::Module {
    static constexpr std::int64_t out_channel_mv = 128, out_channel_N = 64, out_channel_M = 96;
    VideoModelImpl();
    ForwardResult forward(const torch::Tensor& reference, const torch::Tensor& input);
    CompressedFrame compress(const torch::Tensor& reference, const torch::Tensor& input);
    torch::Tensor decompress(const torch::Tensor& reference, const PFrameStream& encoded);
    CompressedFrame encode(const torch::Tensor& reference, const torch::Tensor& input,
                           const std::filesystem::path& output);
    torch::Tensor decode(const torch::Tensor& reference, const std::filesystem::path& input);
    CompressedFrame encode_decode(const torch::Tensor& reference, const torch::Tensor& input,
                                  const std::filesystem::path& output);
    void update(bool force = false);
    void load_dict(const std::map<std::string, torch::Tensor>& state);
    void load_checkpoint(const std::filesystem::path& checkpoint);

    BitEstimator bitEstimator_z{nullptr}, bitEstimator_z_mv{nullptr};
    torch::nn::Sequential feature_extract{nullptr}, context_refine{nullptr};
    torch::nn::Sequential mvEncoder{nullptr}, mvDecoder_part1{nullptr}, mvDecoder_part2{nullptr};
    torch::nn::Sequential contextualEncoder{nullptr}, contextualDecoder_part1{nullptr};
    torch::nn::Sequential contextualDecoder_part2{nullptr}, priorEncoder{nullptr}, priorDecoder{nullptr};
    torch::nn::Sequential mvpriorEncoder{nullptr}, mvpriorDecoder{nullptr};
    torch::nn::Sequential entropy_parameters{nullptr}, entropy_parameters_mv{nullptr};
    MaskedConv2d auto_regressive{nullptr}, auto_regressive_mv{nullptr};
    torch::nn::Sequential temporalPriorEncoder{nullptr};
    Spynet opticFlow{nullptr};
    LaplaceEncoder gaussian_encoder;
private:
    void load_normalized_dict(const std::map<std::string, torch::Tensor>& state);
    torch::Tensor motion_compensation(const torch::Tensor& reference, const torch::Tensor& motion);
    torch::Tensor refine_motion(const torch::Tensor& reference, const torch::Tensor& motion);
    std::pair<std::string, torch::Tensor> compress_ar(
        const torch::Tensor& latent, MaskedConv2d& prediction, const torch::Tensor& params,
        torch::nn::Sequential& entropy);
    torch::Tensor decompress_ar(const std::string& encoded, std::int64_t channels,
        std::int64_t height, std::int64_t width, MaskedConv2d& prediction,
        const torch::Tensor& params, torch::nn::Sequential& entropy);
};
TORCH_MODULE(VideoModel);

}  // namespace uf::dcvc2021
