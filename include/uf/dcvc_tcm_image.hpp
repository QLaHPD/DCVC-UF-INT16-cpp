// Copyright 2020 InterDigital Communications, Inc., Apache-2.0.
// DCVC-TCM's embedded IntraNoAR image model.
#pragma once
#include "uf/dcvc2021_image.hpp"

namespace uf::tcm::image {
using ForwardResult = dcvc2021::image::ForwardResult;
using Compressed = dcvc2021::image::Compressed;
using GDN = dcvc2021::image::GDN;
using LowerBound = dcvc2021::image::LowerBound;
using NonNegativeParametrizer = dcvc2021::image::NonNegativeParametrizer;
using ResidualBlockWithStride = dcvc2021::image::ResidualBlockWithStride;
using ResidualBlockUpsample = dcvc2021::image::ResidualBlockUpsample;
struct GDN1Impl : dcvc2021::image::GDNImpl {
    explicit GDN1Impl(std::int64_t channels, bool inverse = false, double beta_min = 1e-6, double gamma_init = 0.1);
    torch::Tensor forward(const torch::Tensor& input);
private:
    bool inverse_;
};
TORCH_MODULE(GDN1);
struct ResidualBlockImpl : torch::nn::Module {
    ResidualBlockImpl(std::int64_t in, std::int64_t out, double slope = 0.01);
    torch::Tensor forward(const torch::Tensor& input);
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    torch::nn::LeakyReLU leaky_relu{nullptr};
};
TORCH_MODULE(ResidualBlock);
struct ResidualBlockType2Impl : torch::nn::Module {
    ResidualBlockType2Impl(std::int64_t in, std::int64_t out);
    torch::Tensor forward(const torch::Tensor& input);
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    torch::nn::ReLU relu{nullptr};
};
TORCH_MODULE(ResidualBlockType2);
torch::nn::Conv2d conv3x3(std::int64_t in, std::int64_t out, std::int64_t stride = 1);
torch::nn::Conv2d conv1x1(std::int64_t in, std::int64_t out, std::int64_t stride = 1);
torch::nn::Sequential subpel_conv3x3(std::int64_t in, std::int64_t out, std::int64_t factor = 1);
torch::Tensor d_quant(const torch::Tensor& input, const torch::Tensor& means);

struct EntropyBottleneckImpl : dcvc2021::image::EntropyBottleneckImpl {
    explicit EntropyBottleneckImpl(std::int64_t channels);
    void update(bool force = false);
    std::vector<std::string> compress(const torch::Tensor& input) const;
    torch::Tensor decompress(const std::vector<std::string>& strings,
                             std::pair<std::int64_t, std::int64_t> shape) const;
private:
    bool coder_initialized_ = false;
};
TORCH_MODULE(EntropyBottleneck);

// The scale table is an ordinary Python attribute; it does not move with
// the module and is absent from state_dict. Indexing uses log/step + 1.
struct GaussianConditionalImpl : dcvc2021::image::EntropyModelImpl {
    explicit GaussianConditionalImpl(double scale_bound = 0.11, double tail_mass = 1e-9,
                                     double likelihood_bound = 1e-9);
    std::pair<torch::Tensor, torch::Tensor> forward(const torch::Tensor& input, const torch::Tensor& scales,
                                                  const torch::Tensor& means = {});
    torch::Tensor likelihood(const torch::Tensor& input, const torch::Tensor& scales,
                             const torch::Tensor& means = {});
    torch::Tensor build_indexes(const torch::Tensor& scales) const;
    void update();
    std::vector<std::string> compress(const torch::Tensor& input, const torch::Tensor& indexes,
                                     const torch::Tensor& means = {}) const;
    torch::Tensor decompress(const std::vector<std::string>& strings, const torch::Tensor& indexes,
                             const torch::Tensor& means = {}) const;
    torch::Tensor scale_table, scale_bound;
    dcvc2021::image::LowerBound lower_bound_scale{nullptr};
private:
    double tail_mass_, log_min_, log_step_;
    bool coder_initialized_ = false;
};
TORCH_MODULE(GaussianConditional);

struct CompressionModelImpl : torch::nn::Module {
    explicit CompressionModelImpl(std::int64_t channels, bool init_weights = true);
    void to(torch::Device device, torch::Dtype dtype, bool non_blocking = false) override;
    void to(torch::Dtype dtype, bool non_blocking = false) override;
    void to(torch::Device device, bool non_blocking = false) override;
    virtual ForwardResult forward(const torch::Tensor& input);
    virtual void update(bool force = false);
    std::vector<torch::Tensor> parameters() const;
    std::vector<torch::Tensor> aux_parameters() const;
    torch::Tensor aux_loss() const;
    EntropyBottleneck entropy_bottleneck{nullptr};
};
TORCH_MODULE(CompressionModel);

struct EncodedResult { double bit{}; torch::Tensor x_hat; };
struct IntraNoARImpl : CompressionModelImpl {
    explicit IntraNoARImpl(std::int64_t n, bool init_weights = true);
    ForwardResult forward(const torch::Tensor& input) override;
    void update(bool force = false) override;
    std::pair<torch::Tensor, torch::Tensor> get_y_z(const torch::Tensor& input);
    std::pair<torch::Tensor, torch::Tensor> get_recon_wo_stream(const torch::Tensor& y, const torch::Tensor& z);
    Compressed compress(const torch::Tensor& input = {}, const torch::Tensor& y = {}, const torch::Tensor& z = {});
    torch::Tensor decompress(const std::vector<std::vector<std::string>>& strings,
                             std::pair<std::int64_t, std::int64_t> shape);
    torch::Tensor decompress(const Compressed& compressed);
    EncodedResult encode_decode(const torch::Tensor& input,
                               std::optional<std::filesystem::path> output = std::nullopt,
                               std::optional<std::int64_t> width = std::nullopt,
                               std::optional<std::int64_t> height = std::nullopt);
    void load_state_dict(const std::map<std::string, torch::Tensor>& state);
    torch::nn::Sequential g_a{nullptr}, h_a{nullptr}, h_s{nullptr}, g_s{nullptr};
    GaussianConditional gaussian_conditional{nullptr};
    const std::int64_t N;
};
TORCH_MODULE(IntraNoAR);

// The mutable overload removes a legacy scale_table entry from the caller,
// following IntraNoAR.from_state_dict. The const overload copies the map.
IntraNoAR from_state_dict(std::map<std::string, torch::Tensor>& state);
IntraNoAR from_state_dict(const std::map<std::string, torch::Tensor>& state);
IntraNoAR from_checkpoint(const std::filesystem::path& input);
}  // namespace uf::tcm::image
