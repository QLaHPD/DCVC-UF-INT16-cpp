// Copyright 2020 InterDigital Communications, Inc.
// Licensed under Apache-2.0; see licenses/Apache-2.0.txt.
// Original DCVC family's embedded CompressAI image codecs.
#pragma once
#include "uf/dcvc2021.hpp"

namespace uf::dcvc2021::image {

struct LowerBoundImpl : torch::nn::Module {
    explicit LowerBoundImpl(double value);
    torch::Tensor forward(const torch::Tensor& input);
    torch::Tensor bound;
};
TORCH_MODULE(LowerBound);

struct NonNegativeParametrizerImpl : torch::nn::Module {
    explicit NonNegativeParametrizerImpl(double minimum = 0, double offset = 1.0 / 262144.0);
    torch::Tensor init(const torch::Tensor& input);
    torch::Tensor forward(const torch::Tensor& input);
    torch::Tensor pedestal;
    LowerBound lower_bound{nullptr};
};
TORCH_MODULE(NonNegativeParametrizer);

struct GDNImpl : torch::nn::Module {
    explicit GDNImpl(std::int64_t channels, bool inverse = false,
                     double beta_min = 1e-6, double gamma_init = 0.1);
    torch::Tensor forward(const torch::Tensor& input);
    NonNegativeParametrizer beta_reparam{nullptr}, gamma_reparam{nullptr};
    torch::Tensor beta, gamma;
private:
    bool inverse_;
};
TORCH_MODULE(GDN);

struct ResidualBlockImpl : torch::nn::Module {
    ResidualBlockImpl(std::int64_t in, std::int64_t out);
    torch::Tensor forward(const torch::Tensor& input);
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    torch::nn::LeakyReLU leaky_relu{nullptr};
};
TORCH_MODULE(ResidualBlock);

struct ResidualBlockWithStrideImpl : torch::nn::Module {
    ResidualBlockWithStrideImpl(std::int64_t in, std::int64_t out, std::int64_t stride = 2);
    torch::Tensor forward(const torch::Tensor& input);
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, downsample{nullptr};
    torch::nn::LeakyReLU leaky_relu{nullptr};
    GDN gdn{nullptr};
};
TORCH_MODULE(ResidualBlockWithStride);

struct ResidualBlockUpsampleImpl : torch::nn::Module {
    ResidualBlockUpsampleImpl(std::int64_t in, std::int64_t out, std::int64_t factor = 2);
    torch::Tensor forward(const torch::Tensor& input);
    torch::nn::Sequential subpel_conv{nullptr}, upsample{nullptr};
    torch::nn::Conv2d conv{nullptr};
    torch::nn::LeakyReLU leaky_relu{nullptr};
    GDN igdn{nullptr};
};
TORCH_MODULE(ResidualBlockUpsample);

torch::nn::Conv2d conv(std::int64_t in, std::int64_t out, std::int64_t kernel = 5, std::int64_t stride = 2);
torch::nn::ConvTranspose2d deconv(std::int64_t in, std::int64_t out, std::int64_t kernel = 5, std::int64_t stride = 2);
torch::nn::Sequential subpel(std::int64_t in, std::int64_t out, std::int64_t factor = 2);
torch::Tensor default_scale_table();

struct EntropyModelImpl : torch::nn::Module {
    explicit EntropyModelImpl(double likelihood_bound = 1e-9);
    void to(torch::Device device, torch::Dtype dtype, bool non_blocking = false) override;
    void to(torch::Dtype dtype, bool non_blocking = false) override;
    void to(torch::Device device, bool non_blocking = false) override;
    static torch::Tensor quantize(const torch::Tensor& input, const std::string& mode,
                                  const torch::Tensor& means = {});
    static torch::Tensor dequantize(const torch::Tensor& input, const torch::Tensor& means = {});
    std::vector<std::string> compress(const torch::Tensor& input, const torch::Tensor& indexes,
                                      const torch::Tensor& means = {}) const;
    torch::Tensor decompress(const std::vector<std::string>& strings, const torch::Tensor& indexes,
                              const torch::Tensor& means = {}) const;
    CdfTables tables() const;
    torch::Tensor offset, quantized, lengths;
    LowerBound likelihood_lower_bound{nullptr};
protected:
    void set_cdf(const torch::Tensor& pmf, const torch::Tensor& tails,
                 const torch::Tensor& pmf_lengths, const torch::Tensor& offsets);
};

struct EntropyBottleneckImpl : EntropyModelImpl {
    explicit EntropyBottleneckImpl(std::int64_t channels, double tail_mass = 1e-9,
                                  double init_scale = 10, std::vector<std::int64_t> filters = {3, 3, 3, 3},
                                  double likelihood_bound = 1e-9);
    torch::Tensor medians() const;
    torch::Tensor logits_cumulative(const torch::Tensor& input, bool stop_gradient);
    torch::Tensor likelihood(const torch::Tensor& input);
    std::pair<torch::Tensor, torch::Tensor> forward(const torch::Tensor& input);
    void update(bool force = false);
    std::vector<std::string> compress(const torch::Tensor& input) const;
    torch::Tensor decompress(const std::vector<std::string>& strings,
                              std::pair<std::int64_t, std::int64_t> size) const;
    torch::nn::ParameterList biases{nullptr}, factors{nullptr}, matrices{nullptr};
    torch::Tensor quantiles, target;
private:
    std::int64_t channels_;
};
TORCH_MODULE(EntropyBottleneck);

struct GaussianConditionalImpl : EntropyModelImpl {
    explicit GaussianConditionalImpl(torch::Tensor scale_table = {}, double scale_bound = 0.11,
                                     double tail_mass = 1e-9, double likelihood_bound = 1e-9);
    std::pair<torch::Tensor, torch::Tensor> forward(const torch::Tensor& input, const torch::Tensor& scales,
                                                  const torch::Tensor& means = {});
    torch::Tensor likelihood(const torch::Tensor& input, const torch::Tensor& scales,
                              const torch::Tensor& means = {});
    torch::Tensor build_indexes(const torch::Tensor& scales);
    void update_scale_table(const torch::Tensor& values, bool force = false);
    void update();
    torch::Tensor scale_table, scale_bound;
    LowerBound lower_bound_scale{nullptr};
private:
    double tail_mass_;
};
TORCH_MODULE(GaussianConditional);

enum class Kind { factorized, scale_hyperprior, mean_scale_hyperprior, joint, cheng2020_anchor };
Kind architecture_kind(const std::string& name);
struct ForwardResult { torch::Tensor x_hat; std::map<std::string, torch::Tensor> likelihoods; };
struct Compressed {
    std::vector<std::vector<std::string>> strings;
    std::pair<std::int64_t, std::int64_t> shape;
};
struct EncodedResult { double bpp{}; torch::Tensor x_hat; };

struct ModelImpl : torch::nn::Module {
    ModelImpl(Kind selected, std::int64_t n, std::int64_t m, bool init_weights = true);
    void to(torch::Device device, torch::Dtype dtype, bool non_blocking = false) override;
    void to(torch::Dtype dtype, bool non_blocking = false) override;
    void to(torch::Device device, bool non_blocking = false) override;
    ForwardResult forward(const torch::Tensor& input);
    Compressed compress(const torch::Tensor& input);
    torch::Tensor decompress(const Compressed& compressed);
    std::uint64_t encode(const torch::Tensor& input, const std::filesystem::path& output);
    torch::Tensor decode(const std::filesystem::path& input);
    EncodedResult encode_decode(const torch::Tensor& input, const std::filesystem::path& output);
    void update(const torch::Tensor& scales = {}, bool force = false);
    void load_state_dict(const std::map<std::string, torch::Tensor>& state);
    // The Python CompressionModel excludes the entire entropy bottleneck
    // from its primary parameter iterator, and puts it in aux_parameters.
    std::vector<torch::Tensor> parameters() const;
    std::vector<torch::Tensor> aux_parameters() const;
    torch::Tensor aux_loss() const;
    torch::nn::Sequential g_a{nullptr}, g_s{nullptr}, h_a{nullptr}, h_s{nullptr}, entropy_parameters{nullptr};
    EntropyBottleneck entropy_bottleneck{nullptr};
    GaussianConditional gaussian_conditional{nullptr};
    uf::dcvc2021::MaskedConv2d context_prediction{nullptr};
    const Kind kind;
    const std::int64_t N, M;
};
TORCH_MODULE(Model);

Model from_state_dict(const std::map<std::string, torch::Tensor>& state, const std::string& architecture);
Model from_checkpoint(const std::filesystem::path& input, const std::string& architecture);

}  // namespace uf::dcvc2021::image
