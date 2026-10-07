// Copyright 2020 InterDigital Communications, Inc., Apache-2.0.
// Native adaptation of DCVC-TCM/src/models/priors.py and buffer utilities.
#include "uf/dcvc_tcm_image.hpp"
#include "uf/torch_pickle.hpp"
#include "dcvc2021_image_internal.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>

namespace uf::tcm::image {
namespace {
namespace shared = dcvc2021::image;
torch::nn::LeakyReLU leaky() {
    return torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().inplace(true));
}
std::uint32_t dimension(std::int64_t value) {
    if (value < 0 || value > std::numeric_limits<std::uint32_t>::max())
        throw std::out_of_range("TCM I-frame dimension exceeds uint32");
    return static_cast<std::uint32_t>(value);
}
}  // namespace

CompressionModelImpl::CompressionModelImpl(std::int64_t channels, bool init_weights) {
    entropy_bottleneck = register_module("entropy_bottleneck", EntropyBottleneck(channels));
    // The Python initializer runs now, before any convolution exists.
    (void)init_weights;
}
void CompressionModelImpl::to(torch::Device device, torch::Dtype dtype, bool non_blocking) {
    shared::detail::move_module(*this, device, dtype, non_blocking);
}
void CompressionModelImpl::to(torch::Dtype dtype, bool non_blocking) {
    shared::detail::move_module(*this, std::nullopt, dtype, non_blocking);
}
void CompressionModelImpl::to(torch::Device device, bool non_blocking) {
    shared::detail::move_module(*this, device, std::nullopt, non_blocking);
}
ForwardResult CompressionModelImpl::forward(const torch::Tensor&) {
    throw std::logic_error("TCM CompressionModel.forward is not implemented in the reference");
}
void CompressionModelImpl::update(bool force) { entropy_bottleneck->update(force); }
std::vector<torch::Tensor> CompressionModelImpl::parameters() const {
    std::vector<torch::Tensor> result;
    for (const auto& child : named_children()) {
        if (dynamic_cast<EntropyBottleneckImpl*>(child.value().get())) continue;
        const auto values = child.value()->parameters();
        result.insert(result.end(), values.begin(), values.end());
    }
    return result;
}
std::vector<torch::Tensor> CompressionModelImpl::aux_parameters() const {
    return entropy_bottleneck->parameters();
}
torch::Tensor CompressionModelImpl::aux_loss() const {
    throw std::logic_error("The TCM reference EntropyBottleneck does not implement loss()");
}

IntraNoARImpl::IntraNoARImpl(std::int64_t n, bool init_weights) : CompressionModelImpl(n, init_weights), N(n) {
    g_a = register_module("g_a", torch::nn::Sequential());
    for (int i = 0; i < 3; ++i) {
        g_a->push_back(shared::ResidualBlockWithStride(i ? N : 3, N, 2));
        g_a->push_back(ResidualBlock(N, N));
    }
    g_a->push_back(shared::conv(N, N, 3, 2));
    h_a = register_module("h_a", torch::nn::Sequential());
    for (int i = 0; i < 5; ++i) {
        h_a->push_back(shared::conv(N, N, 3, i == 2 || i == 4 ? 2 : 1));
        if (i != 4) h_a->push_back(leaky());
    }
    h_s = register_module("h_s", torch::nn::Sequential());
    h_s->push_back(shared::conv(N, N, 3, 1)); h_s->push_back(leaky());
    h_s->push_back(shared::subpel(N, N, 2)); h_s->push_back(leaky());
    h_s->push_back(shared::conv(N, N * 3 / 2, 3, 1)); h_s->push_back(leaky());
    h_s->push_back(shared::subpel(N * 3 / 2, N * 3 / 2, 2)); h_s->push_back(leaky());
    h_s->push_back(shared::conv(N * 3 / 2, N * 2, 3, 1));
    g_s = register_module("g_s", torch::nn::Sequential());
    for (int i = 0; i < 3; ++i) {
        g_s->push_back(ResidualBlock(N, N));
        g_s->push_back(shared::ResidualBlockUpsample(N, N, 2));
    }
    g_s->push_back(ResidualBlock(N, N));
    g_s->push_back(shared::subpel(N, 3, 2));
    gaussian_conditional = register_module("gaussian_conditional", GaussianConditional());
}
std::pair<torch::Tensor, torch::Tensor> IntraNoARImpl::get_y_z(const torch::Tensor& input) {
    const auto y = g_a->forward(input);
    return {y, h_a->forward(y)};
}
ForwardResult IntraNoARImpl::forward(const torch::Tensor& input) {
    const auto latents = get_y_z(input);
    const auto bottleneck = entropy_bottleneck(latents.second);
    const auto params = h_s->forward(bottleneck.first).chunk(2, 1);
    const auto gaussian = gaussian_conditional(latents.first, params.at(0), params.at(1));
    return {g_s->forward(gaussian.first), {{"y", gaussian.second}, {"z", bottleneck.second}}};
}
void IntraNoARImpl::update(bool force) {
    gaussian_conditional->update();
    CompressionModelImpl::update(force);
}
std::pair<torch::Tensor, torch::Tensor> IntraNoARImpl::get_recon_wo_stream(
    const torch::Tensor& y, const torch::Tensor& z) {
    const auto bottleneck = entropy_bottleneck(z);
    const auto params = h_s->forward(bottleneck.first).chunk(2, 1);
    const auto gaussian = gaussian_conditional(y, params.at(0), params.at(1));
    const auto reconstruction = g_s->forward(gaussian.first);
    const auto bits = (gaussian.second.log().sum() + bottleneck.second.log().sum()) / -std::log(2.0);
    return {reconstruction, bits};
}
Compressed IntraNoARImpl::compress(const torch::Tensor& input, const torch::Tensor& supplied_y,
                                  const torch::Tensor& supplied_z) {
    torch::Tensor y = supplied_y, z = supplied_z;
    if (input.defined()) {
        if (y.defined() || z.defined()) throw std::invalid_argument("Supply either the image or both TCM I-frame latents");
        const auto latents = get_y_z(input); y = latents.first; z = latents.second;
    } else if (!y.defined() || !z.defined()) {
        throw std::invalid_argument("TCM I-frame compression requires both y and z");
    }
    const auto z_strings = entropy_bottleneck->compress(z);
    const std::pair<std::int64_t, std::int64_t> shape{z.size(-2), z.size(-1)};
    const auto z_hat = entropy_bottleneck->decompress(z_strings, shape);
    const auto params = h_s->forward(z_hat).chunk(2, 1);
    const auto indexes = gaussian_conditional->build_indexes(params.at(0));
    return {{gaussian_conditional->compress(y, indexes, params.at(1)), z_strings}, shape};
}
torch::Tensor IntraNoARImpl::decompress(const std::vector<std::vector<std::string>>& strings,
                                      std::pair<std::int64_t, std::int64_t> shape) {
    if (strings.size() != 2) throw std::invalid_argument("TCM I-frame requires y and z string groups");
    const auto z_hat = entropy_bottleneck->decompress(strings.at(1), shape);
    const auto params = h_s->forward(z_hat).chunk(2, 1);
    const auto indexes = gaussian_conditional->build_indexes(params.at(0));
    const auto y_hat = gaussian_conditional->decompress(strings.at(0), indexes, params.at(1));
    return g_s->forward(y_hat).clamp_(0, 1);
}
torch::Tensor IntraNoARImpl::decompress(const Compressed& compressed) {
    return decompress(compressed.strings, compressed.shape);
}
EncodedResult IntraNoARImpl::encode_decode(const torch::Tensor& input,
    std::optional<std::filesystem::path> output, std::optional<std::int64_t> width,
    std::optional<std::int64_t> height) {
    const auto latents = get_y_z(input);
    if (!output) {
        const auto result = get_recon_wo_stream(latents.first, latents.second);
        return {result.second.item<double>(), result.first};
    }
    if (!width || !height) throw std::invalid_argument("TCM I-frame stream coding requires original dimensions");
    const auto compressed = compress({}, latents.first, latents.second);
    dcvc2021::write_i(*output, {dimension(*height), dimension(*width),
        compressed.strings.at(0).at(0), compressed.strings.at(1).at(0)});
    if (!std::filesystem::is_regular_file(*output)) throw std::runtime_error("TCM I-frame output is not a regular file");
    const auto bits = static_cast<double>(std::filesystem::file_size(*output)) * 8;
    const auto stream = dcvc2021::read_i(*output);
    return {bits, decompress({{stream.y_string}, {stream.z_string}},
        dcvc2021::downsampled_shape(stream.height, stream.width, 64))};
}
void IntraNoARImpl::load_state_dict(const std::map<std::string, torch::Tensor>& state) {
    const auto resize_empty = [&](const std::string& prefix, dcvc2021::image::EntropyModelImpl& model) {
        for (const auto& name : {"_quantized_cdf", "_offset", "_cdf_length"}) {
            const auto value = state.at(prefix + "." + name);
            auto buffer = model.named_buffers(false)[name];
            if (!buffer.numel()) buffer.resize_(value.sizes());
        }
    };
    resize_empty("entropy_bottleneck", *entropy_bottleneck);
    resize_empty("gaussian_conditional", *gaussian_conditional);
    std::map<std::string, torch::Tensor> targets;
    for (const auto& item : named_parameters(true)) targets.emplace(item.key(), item.value());
    for (const auto& item : named_buffers(true)) targets.emplace(item.key(), item.value());
    for (const auto& item : targets) {
        const auto found = state.find(item.first);
        if (found == state.end()) throw std::runtime_error("TCM I-frame checkpoint is missing: " + item.first);
        if (found->second.sizes() != item.second.sizes()) throw std::runtime_error("TCM I-frame checkpoint shape differs: " + item.first);
    }
    for (const auto& item : state) if (!targets.count(item.first))
        throw std::runtime_error("Unexpected TCM I-frame checkpoint entry: " + item.first);
    torch::NoGradGuard no_grad;
    for (auto& item : targets) item.second.copy_(state.at(item.first));
}
IntraNoAR from_state_dict(std::map<std::string, torch::Tensor>& state) {
    IntraNoAR result(state.at("g_s.0.conv1.weight").size(0));
    state.erase("gaussian_conditional.scale_table");
    result->load_state_dict(state);
    return result;
}
IntraNoAR from_state_dict(const std::map<std::string, torch::Tensor>& state) {
    auto copy = state;
    return from_state_dict(copy);
}
IntraNoAR from_checkpoint(const std::filesystem::path& input) {
    auto value = load_torch_pickle_cpu(input);
    if (!value.isGenericDict()) throw std::runtime_error("TCM I-frame checkpoint must be a dictionary");
    const auto dictionary = value.toGenericDict();
    if (const auto wrapped = dictionary.find("state_dict"); wrapped != dictionary.end()) value = wrapped->value();
    if (!value.isGenericDict()) throw std::runtime_error("TCM I-frame state_dict must be a dictionary");
    std::map<std::string, torch::Tensor> state;
    for (const auto& entry : value.toGenericDict()) {
        if (!entry.key().isString() || !entry.value().isTensor()) throw std::runtime_error("Invalid TCM I-frame checkpoint entry");
        state[entry.key().toStringRef()] = entry.value().toTensor();
    }
    return from_state_dict(state);
}
}  // namespace uf::tcm::image
