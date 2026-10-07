// Copyright 2020 InterDigital Communications, Inc., Apache-2.0.
// Native adaptation of the original DCVC priors.py, waseda.py and zoo/image.py.
#include "uf/dcvc2021_image.hpp"
#include "uf/process.hpp"
#include "uf/torch_pickle.hpp"
#include "dcvc2021_image_internal.hpp"

#include <algorithm>
#include <iostream>
#include <limits>
#include <mutex>
#include <stdexcept>

namespace uf::dcvc2021::image {
namespace {
bool autoregressive(Kind kind) { return kind == Kind::joint || kind == Kind::cheng2020_anchor; }
torch::nn::LeakyReLU leaky() { return torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().inplace(true)); }
torch::nn::ReLU relu() { return torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)); }
torch::nn::Sequential analysis(std::int64_t n, std::int64_t m) {
    torch::nn::Sequential result;
    for (int i = 0; i < 4; ++i) {
        result->push_back(conv(i ? n : 3, i == 3 ? m : n));
        if (i != 3) result->push_back(GDN(n));
    }
    return result;
}
torch::nn::Sequential synthesis(std::int64_t n, std::int64_t m) {
    torch::nn::Sequential result;
    for (int i = 0; i < 4; ++i) {
        result->push_back(deconv(i ? n : m, i == 3 ? 3 : n));
        if (i != 3) result->push_back(GDN(n, true));
    }
    return result;
}
torch::nn::Sequential hyper_analysis(std::int64_t n, std::int64_t m, bool use_relu) {
    torch::nn::Sequential result;
    result->push_back(conv(m, n, 3, 1));
    if (use_relu) result->push_back(relu()); else result->push_back(leaky());
    result->push_back(conv(n, n));
    if (use_relu) result->push_back(relu()); else result->push_back(leaky());
    result->push_back(conv(n, n));
    return result;
}
torch::nn::Sequential scale_synthesis(std::int64_t n, std::int64_t m) {
    torch::nn::Sequential result;
    result->push_back(deconv(n, n));
    result->push_back(relu());
    result->push_back(deconv(n, n));
    result->push_back(relu());
    result->push_back(conv(n, m, 3, 1));
    result->push_back(relu());
    return result;
}
torch::nn::Sequential mean_synthesis(std::int64_t n, std::int64_t m, bool joint) {
    torch::nn::Sequential result;
    result->push_back(deconv(n, m));
    // The local joint model has no activation after its first deconvolution.
    if (!joint) result->push_back(leaky());
    result->push_back(deconv(m, m * 3 / 2));
    result->push_back(leaky());
    result->push_back(conv(m * 3 / 2, m * 2, 3, 1));
    return result;
}
std::vector<std::int32_t> integers(const torch::Tensor& input) {
    const auto cpu = input.to(torch::kCPU, torch::kInt32).contiguous().view({-1});
    if (!cpu.numel()) return {};
    return {cpu.data_ptr<std::int32_t>(), cpu.data_ptr<std::int32_t>() + cpu.numel()};
}
torch::Tensor floating_symbols(const std::vector<std::int32_t>& values) {
    const std::vector<std::int64_t> shape{1, static_cast<std::int64_t>(values.size()), 1, 1};
    auto integers = torch::empty(shape, torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt32));
    std::copy(values.begin(), values.end(), integers.data_ptr<std::int32_t>());
    auto result = torch::empty(shape, torch::TensorOptions().device(torch::kCPU));
    result.copy_(integers);
    return result;
}
void warn_gpu(const ModelImpl& model, bool decoding) {
    if (model.parameters().at(0).device().is_cpu()) return;
    static std::once_flag encoder_warning, decoder_warning;
    auto& flag = decoding ? decoder_warning : encoder_warning;
    std::call_once(flag, [] {
        std::cerr << "Inference on GPU is not recommended for the autoregressive models "
                     "(the entropy coder is run sequentially on CPU).\n";
    });
}
std::vector<std::int32_t> index_list(const torch::Tensor& indices, std::int64_t batch) {
    const auto reduced = indices.select(0, batch).squeeze().to(torch::kInt32);
    // The reference passes squeeze().tolist() directly to a list binding.
    if (reduced.dim() != 1) throw std::invalid_argument("Autoregressive indexes must form a channel list");
    return integers(reduced);
}
void append(std::vector<std::int32_t>& destination, const std::vector<std::int32_t>& values) {
    destination.insert(destination.end(), values.begin(), values.end());
}
std::uint32_t dimension(std::int64_t value) {
    if (value < 0 || value > std::numeric_limits<std::uint32_t>::max()) throw std::out_of_range("I-frame dimension exceeds uint32");
    return static_cast<std::uint32_t>(value);
}
}  // namespace

Kind architecture_kind(const std::string& name) {
    if (name == "bmshj2018-factorized") return Kind::factorized;
    if (name == "bmshj2018-hyperprior") return Kind::scale_hyperprior;
    if (name == "mbt2018-mean") return Kind::mean_scale_hyperprior;
    if (name == "mbt2018") return Kind::joint;
    if (name == "cheng2020-anchor") return Kind::cheng2020_anchor;
    throw std::invalid_argument("Unknown original DCVC image architecture: " + name);
}

ModelImpl::ModelImpl(Kind selected, std::int64_t n, std::int64_t m, bool init_weights)
    : kind(selected), N(n), M(m) {
    if (kind == Kind::cheng2020_anchor && n != m) throw std::invalid_argument("Cheng2020Anchor requires M=N");
    entropy_bottleneck = register_module("entropy_bottleneck", EntropyBottleneck(kind == Kind::factorized ? M : N));
    // CompressionModel._initialize_weights runs before any convolutions
    // exist. Both flag values therefore leave the later default initializers.
    (void)init_weights;
    g_a = register_module("g_a", analysis(N, M));
    g_s = register_module("g_s", synthesis(N, M));
    if (kind == Kind::factorized) return;
    const bool joint = autoregressive(kind);
    h_a = register_module("h_a", hyper_analysis(N, M, !joint));
    h_s = register_module("h_s", joint ? mean_synthesis(N, M, true) : scale_synthesis(N, M));
    if (joint) {
        entropy_parameters = register_module("entropy_parameters", torch::nn::Sequential());
        entropy_parameters->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(M * 12 / 3, M * 10 / 3, 1)));
        entropy_parameters->push_back(leaky());
        entropy_parameters->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(M * 10 / 3, M * 8 / 3, 1)));
        entropy_parameters->push_back(leaky());
        entropy_parameters->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(M * 8 / 3, M * 6 / 3, 1)));
        context_prediction = register_module("context_prediction", uf::dcvc2021::MaskedConv2d(M));
    }
    gaussian_conditional = register_module("gaussian_conditional", GaussianConditional());
    if (kind == Kind::mean_scale_hyperprior) {
        // Preserve the inherited constructors' discarded initializers too.
        h_a = replace_module("h_a", hyper_analysis(N, M, false));
        h_s = replace_module("h_s", mean_synthesis(N, M, false));
    }
    if (kind == Kind::cheng2020_anchor) {
        torch::nn::Sequential encoder;
        for (int i = 0; i < 3; ++i) {
            encoder->push_back(ResidualBlockWithStride(i ? N : 3, N, 2));
            encoder->push_back(ResidualBlock(N, N));
        }
        encoder->push_back(conv(N, N, 3, 2));
        g_a = replace_module("g_a", encoder);
        torch::nn::Sequential hyper_encoder;
        hyper_encoder->push_back(conv(N, N, 3, 1)); hyper_encoder->push_back(leaky());
        hyper_encoder->push_back(conv(N, N, 3, 1)); hyper_encoder->push_back(leaky());
        hyper_encoder->push_back(conv(N, N, 3, 2)); hyper_encoder->push_back(leaky());
        hyper_encoder->push_back(conv(N, N, 3, 1)); hyper_encoder->push_back(leaky());
        hyper_encoder->push_back(conv(N, N, 3, 2));
        h_a = replace_module("h_a", hyper_encoder);
        torch::nn::Sequential hyper_decoder;
        hyper_decoder->push_back(conv(N, N, 3, 1)); hyper_decoder->push_back(leaky());
        hyper_decoder->push_back(subpel(N, N)); hyper_decoder->push_back(leaky());
        hyper_decoder->push_back(conv(N, N * 3 / 2, 3, 1)); hyper_decoder->push_back(leaky());
        hyper_decoder->push_back(subpel(N * 3 / 2, N * 3 / 2)); hyper_decoder->push_back(leaky());
        hyper_decoder->push_back(conv(N * 3 / 2, N * 2, 3, 1));
        h_s = replace_module("h_s", hyper_decoder);
        torch::nn::Sequential decoder;
        for (int i = 0; i < 3; ++i) {
            decoder->push_back(ResidualBlock(N, N));
            decoder->push_back(ResidualBlockUpsample(N, N));
        }
        decoder->push_back(ResidualBlock(N, N)); decoder->push_back(subpel(N, 3));
        g_s = replace_module("g_s", decoder);
    }
}

void ModelImpl::to(torch::Device device, torch::Dtype dtype, bool non_blocking) {
    detail::move_module(*this, device, dtype, non_blocking);
}
void ModelImpl::to(torch::Dtype dtype, bool non_blocking) {
    detail::move_module(*this, std::nullopt, dtype, non_blocking);
}
void ModelImpl::to(torch::Device device, bool non_blocking) {
    detail::move_module(*this, device, std::nullopt, non_blocking);
}

ForwardResult ModelImpl::forward(const torch::Tensor& input) {
    const auto y = g_a->forward(input);
    if (kind == Kind::factorized) {
        const auto bottleneck = entropy_bottleneck(y);
        return {g_s->forward(bottleneck.first), {{"y", bottleneck.second}}};
    }
    const auto z = h_a->forward(kind == Kind::scale_hyperprior ? y.abs() : y);
    const auto bottleneck = entropy_bottleneck(z);
    const auto params = h_s->forward(bottleneck.first);
    torch::Tensor y_hat, scales, means, probability;
    if (autoregressive(kind)) {
        y_hat = EntropyModelImpl::quantize(y, is_training() ? "noise" : "dequantize");
        const auto gaussian = entropy_parameters->forward(torch::cat({params, context_prediction(y_hat)}, 1)).chunk(2, 1);
        scales = gaussian.at(0); means = gaussian.at(1);
        probability = gaussian_conditional(y, scales, means).second;
    } else {
        if (kind == Kind::mean_scale_hyperprior) {
            const auto gaussian = params.chunk(2, 1);
            scales = gaussian.at(0); means = gaussian.at(1);
        } else scales = params;
        const auto conditional = gaussian_conditional(y, scales, means);
        y_hat = conditional.first; probability = conditional.second;
    }
    return {g_s->forward(y_hat), {{"y", probability}, {"z", bottleneck.second}}};
}

Compressed ModelImpl::compress(const torch::Tensor& input) {
    if (autoregressive(kind)) warn_gpu(*this, false);
    const auto y = g_a->forward(input);
    if (kind == Kind::factorized) return {{entropy_bottleneck->compress(y)}, {y.size(2), y.size(3)}};
    const auto z = h_a->forward(kind == Kind::scale_hyperprior ? y.abs() : y);
    const auto z_strings = entropy_bottleneck->compress(z);
    const auto z_hat = entropy_bottleneck->decompress(z_strings, {z.size(2), z.size(3)});
    const auto params = h_s->forward(z_hat);
    if (!autoregressive(kind)) {
        torch::Tensor scales = params, means;
        if (kind == Kind::mean_scale_hyperprior) {
            const auto gaussian = params.chunk(2, 1); scales = gaussian.at(0); means = gaussian.at(1);
        }
        const auto indices = gaussian_conditional->build_indexes(scales);
        return {{gaussian_conditional->compress(y, indices, means), z_strings}, {z.size(2), z.size(3)}};
    }
    const auto height = z_hat.size(2) * 4, width = z_hat.size(3) * 4;
    auto y_hat = torch::nn::functional::pad(y, torch::nn::functional::PadFuncOptions({2, 2, 2, 2}));
    const auto cdf = gaussian_conditional->tables();
    std::vector<std::string> y_strings;
    for (std::int64_t i = 0; i < y.size(0); ++i) {
        std::vector<std::int32_t> symbols, indices;
        for (std::int64_t h = 0; h < height; ++h) {
            ProcessInterrupt::check();
            for (std::int64_t w = 0; w < width; ++w) {
                const auto tile = y_hat.slice(0, i, i + 1).slice(2, h, h + 5).slice(3, w, w + 5);
                const auto context = torch::conv2d(tile, context_prediction->weight, context_prediction->bias);
                const auto p = params.slice(0, i, i + 1).slice(2, h, h + 1).slice(3, w, w + 1);
                const auto gaussian = entropy_parameters->forward(torch::cat({p, context}, 1)).chunk(2, 1);
                const auto index = gaussian_conditional->build_indexes(gaussian.at(0));
                const auto q = (tile - gaussian.at(1)).round();
                // Retain the source's batch indexing (its entropy bottleneck
                // decode already rejects a broadcast median for batch > 1).
                y_hat.select(0, i).select(1, h + 2).select(1, w + 2).copy_(
                    (q + gaussian.at(1)).select(0, i).select(1, 2).select(1, 2));
                append(symbols, integers(q.select(0, i).select(1, 2).select(1, 2)));
                append(indices, index_list(index, i));
            }
        }
        y_strings.push_back(rans_encode(symbols, indices, cdf));
    }
    return {{y_strings, z_strings}, {z.size(2), z.size(3)}};
}

torch::Tensor ModelImpl::decompress(const Compressed& compressed) {
    if (kind == Kind::factorized) {
        if (compressed.strings.size() != 1) throw std::invalid_argument("FactorizedPrior requires one string group");
        return g_s->forward(entropy_bottleneck->decompress(compressed.strings[0], compressed.shape));
    }
    if (compressed.strings.size() != 2) throw std::invalid_argument("Hyperprior requires two string groups");
    if (autoregressive(kind)) warn_gpu(*this, true);
    const auto z_hat = entropy_bottleneck->decompress(compressed.strings[1], compressed.shape);
    const auto params = h_s->forward(z_hat);
    if (!autoregressive(kind)) {
        torch::Tensor scales = params, means;
        if (kind == Kind::mean_scale_hyperprior) {
            const auto gaussian = params.chunk(2, 1); scales = gaussian.at(0); means = gaussian.at(1);
        }
        const auto index = gaussian_conditional->build_indexes(scales);
        auto y_hat = gaussian_conditional->decompress(compressed.strings[0], index, means);
        if (kind == Kind::scale_hyperprior) y_hat = y_hat.to(z_hat.device());
        return g_s->forward(y_hat).clamp_(0, 1);
    }
    const auto height = z_hat.size(2) * 4, width = z_hat.size(3) * 4;
    auto y_hat = torch::zeros({z_hat.size(0), M, height + 4, width + 4}, torch::TensorOptions().device(z_hat.device()));
    const auto cdf = gaussian_conditional->tables();
    RansDecoder decoder;
    for (std::size_t i = 0; i < compressed.strings[0].size(); ++i) {
        decoder.set_stream(compressed.strings[0][i]);
        for (std::int64_t h = 0; h < height; ++h) {
            ProcessInterrupt::check();
            for (std::int64_t w = 0; w < width; ++w) {
                const auto tile = y_hat.slice(0, i, i + 1).slice(2, h, h + 5).slice(3, w, w + 5);
                const auto context = torch::conv2d(tile, context_prediction->weight, context_prediction->bias);
                const auto p = params.slice(0, i, i + 1).slice(2, h, h + 1).slice(3, w, w + 1);
                const auto gaussian = entropy_parameters->forward(torch::cat({p, context}, 1)).chunk(2, 1);
                const auto index = gaussian_conditional->build_indexes(gaussian.at(0));
                const auto rv = EntropyModelImpl::dequantize(floating_symbols(decoder.decode(index_list(index, i), cdf)), gaussian.at(1));
                y_hat.select(0, i).slice(1, h + 2, h + 3).slice(2, w + 2, w + 3).copy_(rv.select(0, 0));
            }
        }
    }
    return g_s->forward(y_hat.slice(2, 2, height + 2).slice(3, 2, width + 2)).clamp_(0, 1);
}

void ModelImpl::update(const torch::Tensor& scales, bool force) {
    if (gaussian_conditional) gaussian_conditional->update_scale_table(scales.defined() ? scales : default_scale_table(), force);
    entropy_bottleneck->update(force);
}
std::uint64_t ModelImpl::encode(const torch::Tensor& input, const std::filesystem::path& output) {
    if (kind == Kind::factorized) throw std::logic_error("The reference FactorizedPrior has no file encode helper");
    const auto compressed = compress(input);
    const auto& y = compressed.strings.at(0).at(0), &z = compressed.strings.at(1).at(0);
    write_i(output, {dimension(input.size(2)), dimension(input.size(3)), y, z});
    return std::uint64_t(y.size()) + z.size();
}
torch::Tensor ModelImpl::decode(const std::filesystem::path& input) {
    if (kind == Kind::factorized) throw std::logic_error("The reference FactorizedPrior has no file decode helper");
    const auto encoded = read_i(input);
    return decompress({{{encoded.y_string}, {encoded.z_string}}, downsampled_shape(encoded.height, encoded.width, 64)});
}
EncodedResult ModelImpl::encode_decode(const torch::Tensor& input, const std::filesystem::path& output) {
    const auto bits = encode(input, output) * 8;
    const auto bpp = static_cast<double>(bits) / (input.size(2) * input.size(3));
    return {bpp, decode(output)};
}

std::vector<torch::Tensor> ModelImpl::parameters() const {
    std::vector<torch::Tensor> result;
    for (const auto& child : named_children()) {
        if (dynamic_cast<EntropyBottleneckImpl*>(child.value().get())) continue;
        const auto values = child.value()->parameters();
        result.insert(result.end(), values.begin(), values.end());
    }
    return result;
}
std::vector<torch::Tensor> ModelImpl::aux_parameters() const {
    std::vector<torch::Tensor> result;
    for (const auto& child : named_children()) {
        if (!dynamic_cast<EntropyBottleneckImpl*>(child.value().get())) continue;
        const auto values = child.value()->parameters();
        result.insert(result.end(), values.begin(), values.end());
    }
    return result;
}
torch::Tensor ModelImpl::aux_loss() const {
    throw std::logic_error("The reference EntropyBottleneck does not implement loss()");
}
void ModelImpl::load_state_dict(const std::map<std::string, torch::Tensor>& state) {
    const auto resize_empty = [&](const std::string& prefix, EntropyModelImpl& entropy, bool gaussian) {
        for (auto& item : entropy.named_buffers(false)) {
            if (item.key() != "_quantized_cdf" && item.key() != "_offset" && item.key() != "_cdf_length" &&
                !(gaussian && item.key() == "scale_table")) continue;
            const auto& value = state.at(prefix + "." + item.key());
            if (!item.value().numel()) item.value().resize_(value.sizes());
        }
    };
    resize_empty("entropy_bottleneck", *entropy_bottleneck, false);
    if (gaussian_conditional) resize_empty("gaussian_conditional", *gaussian_conditional, true);
    std::map<std::string, torch::Tensor> targets;
    for (const auto& item : named_parameters(true)) targets.emplace(item.key(), item.value());
    for (const auto& item : named_buffers(true)) targets.emplace(item.key(), item.value());
    for (const auto& item : targets) {
        const auto found = state.find(item.first);
        if (found == state.end()) throw std::runtime_error("Image checkpoint is missing: " + item.first);
        if (found->second.sizes() != item.second.sizes()) throw std::runtime_error("Image checkpoint shape differs: " + item.first);
    }
    for (const auto& item : state) if (!targets.count(item.first)) throw std::runtime_error("Unexpected image checkpoint parameter: " + item.first);
    torch::NoGradGuard no_grad;
    for (auto& item : targets) item.second.copy_(state.at(item.first));
}
Model from_state_dict(const std::map<std::string, torch::Tensor>& state, const std::string& architecture) {
    const auto kind = architecture_kind(architecture);
    const auto n = state.at(kind == Kind::cheng2020_anchor ? "g_a.0.conv1.weight" : "g_a.0.weight").size(0);
    const auto m = kind == Kind::cheng2020_anchor ? n : state.at("g_a.6.weight").size(0);
    Model result(kind, n, m);
    result->load_state_dict(state);
    return result;
}
Model from_checkpoint(const std::filesystem::path& input, const std::string& architecture) {
    const auto value = load_torch_pickle_cpu(input);
    if (!value.isGenericDict()) throw std::runtime_error("Image checkpoint must be a raw state dictionary");
    std::map<std::string, torch::Tensor> state;
    for (const auto& entry : value.toGenericDict()) {
        if (!entry.key().isString() || !entry.value().isTensor()) throw std::runtime_error("Invalid image checkpoint entry");
        state[entry.key().toStringRef()] = entry.value().toTensor();
    }
    return from_state_dict(state, architecture);
}
}  // namespace uf::dcvc2021::image
