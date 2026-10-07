// Entropy CDF quantization follows Microsoft's DCVC-UF py_rans/py_rans.cpp
// (MIT license); model and table arithmetic follows src/int16/prepared.py.
#include "uf/prepare.hpp"
#include "uf/archive.hpp"
#include "uf/int16_ops.hpp"
#include "uf/model_schema.hpp"
#include "uf/torch_pickle.hpp"
#include "uf/process.hpp"
#include "uf/python_path.hpp"

#include <c10/core/DefaultDtype.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace uf {
namespace {

using State = std::map<std::string, torch::Tensor>;

std::string source_hash_prefix(const std::string& text) {
    // expected_source[:16] counts Python characters. SHA-256 normally supplies
    // ASCII; retain the text slicing boundary for self-contained artifact data.
    std::size_t end = 0;
    for (int characters = 0; characters < 16 && end < text.size(); ++characters) {
        ++end;
        while (end < text.size() &&
               (static_cast<unsigned char>(text[end]) & 0xc0u) == 0x80u) ++end;
    }
    return text.substr(0, end);
}

std::optional<std::filesystem::path> existing_prepared_path(
    const std::filesystem::path& checkpoint, std::optional<std::filesystem::path> path,
    const std::optional<std::string>& expected_source) {
    if (path) *path = python_path_normalize(*path);
    if (!path && expected_source) {
        const auto source = python_path_normalize(checkpoint);
        const auto name = python_path_name_text(python_path_text(source));
        const auto candidate = python_path_normalize(source.parent_path() /
            python_path_from_text(name + "." + int16::arithmetic_id + "." +
                                  source_hash_prefix(*expected_source) + ".int16.pt"));
        if (python_path_exists(candidate)) path = candidate;
    }
    return path;
}

bool ends(const std::string& value, const std::string& suffix) {
    return value.size() >= suffix.size() &&
           value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

State checkpoint_state(const std::filesystem::path& path) {
    auto value = load_torch_pickle_cpu(path);
    if (!value.isGenericDict()) throw std::runtime_error("UF checkpoint is not a dictionary");
    auto dictionary = value.toGenericDict();
    for (const auto& wrapper : {"state_dict", "net"}) {
        std::optional<c10::impl::GenericDict> nested;
        for (const auto& entry : dictionary) {
            if (entry.key().isString() && entry.key().toStringRef() == wrapper) {
                if (!entry.value().isGenericDict()) {
                    throw std::runtime_error("UF checkpoint wrapper is not a dictionary");
                }
                nested = entry.value().toGenericDict();
                break;
            }
        }
        if (nested) dictionary = *nested;
    }
    State result;
    std::vector<std::string> keys;
    for (const auto& entry : dictionary) {
        if (!entry.key().isString() || !entry.value().isTensor()) {
            throw std::runtime_error("UF checkpoint state contains a non-tensor entry");
        }
        const auto name = entry.key().toStringRef();
        keys.push_back(name);
        result[name] = entry.value().toTensor();
    }
    // consume_prefix_in_state_dict_if_present visits the original key list.
    // A prefixed entry replaces an existing unprefixed name in that pass.
    for (const auto& name : keys) if (name.rfind("module.", 0) == 0) {
        const auto tensor = result.at(name);
        result.erase(name); result[name.substr(7)] = tensor;
    }
    return result;
}

const torch::Tensor& required(const State& state, const std::string& name) {
    const auto it = state.find(name);
    if (it == state.end()) throw std::runtime_error("UF checkpoint is missing parameter: " + name);
    return it->second;
}

torch::Tensor quantize(const torch::Tensor& value, std::int64_t scale) {
    if (!value.is_floating_point()) {
        throw std::runtime_error("UF checkpoint parameter is non-finite or non-floating");
    }
    // Prepared weights/tables use the common Python operation sequence. Keep
    // this wrapper's floating-parameter policy and packed storage convention.
    return int16::quantize_parameter(value, static_cast<double>(scale)).contiguous();
}

std::vector<std::int32_t> quantized_cdf(const std::vector<float>& pmf) {
    constexpr std::uint32_t maximum = 1u << 16u;
    std::vector<std::uint32_t> cdf(pmf.size() + 1);
    for (std::size_t i = 0; i < pmf.size(); ++i) {
        // The UF extension multiplies Float32 p, then adds the Double literal
        // 0.5. A Float32 half addition can round a value just below a tie up.
        const auto frequency = pmf[i] * maximum + 0.5;
        // The reference extension's out-of-range float-to-unsigned conversion
        // is undefined. Reject that boundary without rejecting unrelated float
        // checkpoint parameters before the operations that actually use them.
        if (!std::isfinite(frequency) || frequency <= -1.0 ||
            frequency >= static_cast<double>(std::numeric_limits<std::uint32_t>::max()) + 1.0) {
            throw std::runtime_error("UF entropy frequency cannot be converted to UInt32");
        }
        cdf[i + 1] = static_cast<std::uint32_t>(frequency);
    }
    const auto total = std::accumulate(cdf.begin(), cdf.end(), std::uint32_t{});
    if (!total) throw std::runtime_error("UF entropy PMF is empty");
    for (auto& value : cdf) value = static_cast<std::uint32_t>(
        static_cast<std::uint64_t>(maximum) * value / total);
    std::partial_sum(cdf.begin(), cdf.end(), cdf.begin());
    cdf.back() = maximum;
    for (std::size_t i = 0; i + 1 < cdf.size(); ++i) {
        if (cdf[i] + 1 <= cdf[i + 1]) continue;
        std::uint32_t best_frequency = std::numeric_limits<std::uint32_t>::max();
        std::size_t donor = cdf.size();
        for (std::size_t j = 0; j + 1 < cdf.size(); ++j) {
            const auto frequency = cdf[j + 1] - cdf[j];
            if (frequency >= 2 && frequency < best_frequency) {
                donor = j;
                best_frequency = frequency;
            }
        }
        if (donor == cdf.size() || donor == i) throw std::runtime_error("UF CDF repair failed");
        if (donor < i) {
            for (std::size_t j = donor + 1; j <= i; ++j) --cdf[j];
        } else {
            for (std::size_t j = i + 1; j <= donor; ++j) ++cdf[j];
        }
    }
    return {cdf.begin(), cdf.end()};
}

std::pair<torch::Tensor, torch::Tensor> make_cdf(const torch::Tensor& pmf_values,
                                                 const torch::Tensor& tail_values,
                                                 const torch::Tensor& lengths) {
    const auto pmf = pmf_values.to(torch::kCPU).to(torch::kFloat32).contiguous();
    const auto tail = tail_values.to(torch::kCPU).to(torch::kFloat32).contiguous();
    const auto pmf_length = lengths.to(torch::kCPU).to(torch::kInt32).contiguous();
    if (pmf.dim() != 2 || tail.numel() != pmf.size(0) || pmf_length.numel() != pmf.size(0)) {
        throw std::invalid_argument("UF entropy PMF shapes do not match");
    }
    const auto rows = pmf.size(0), max_length = pmf.size(1);
    const auto width = max_length + 2;
    std::vector<std::int32_t> cdf(rows * width);
    auto cdf_lengths = torch::empty({rows}, torch::TensorOptions().dtype(torch::kInt32));
    const auto* probabilities = pmf.data_ptr<float>();
    const auto* tails = tail.data_ptr<float>();
    const auto* sizes = pmf_length.data_ptr<std::int32_t>();
    auto* output_lengths = cdf_lengths.data_ptr<std::int32_t>();
    for (std::int64_t row = 0; row < rows; ++row) {
        ProcessInterrupt::check();
        const auto length = sizes[row];
        if (length < 1 || length > max_length || length % 2 == 0) {
            throw std::runtime_error("UF entropy PMF length is invalid");
        }
        std::vector<float> reordered(length + 1);
        const auto center = length / 2;
        reordered[0] = probabilities[row * max_length + center];
        for (int i = 1; i <= center; ++i) {
            reordered[2 * i - 1] = probabilities[row * max_length + center + i];
            reordered[2 * i] = probabilities[row * max_length + center - i];
        }
        reordered[length] = tails[row];
        const auto values = quantized_cdf(reordered);
        std::copy(values.begin(), values.end(), cdf.begin() + row * width);
        output_lengths[row] = length + 2;
    }
    auto result = torch::from_blob(cdf.data(), {rows, width},
                                   torch::TensorOptions().dtype(torch::kInt32)).clone();
    return {result, cdf_lengths};
}

torch::Tensor normal_cdf(const torch::Tensor& x, const torch::Tensor& scale) {
    // Normal(0., scales) represents loc in the scale tensor's dtype. Its
    // subtraction precedes multiplication and can promote Float32 samples.
    return 0.5 * (1.0 + torch::erf(((x - torch::zeros_like(scale)) *
                                   scale.reciprocal()) / std::sqrt(2.0)));
}

std::pair<torch::Tensor, torch::Tensor> gaussian_cdf() {
    constexpr int levels = 128;
    constexpr int maximum = 8;
    const auto options = torch::TensorOptions()
        .dtype(c10::get_default_dtype_as_scalartype()).device(torch::kCPU);
    const auto scales = torch::exp(torch::linspace(std::log(0.11), std::log(16.0), levels, options));
    const auto zeros = torch::zeros_like(scales);
    auto range = zeros + maximum;
    for (int i = maximum; i > 1; --i) {
        const auto samples = zeros + i;
        range = torch::where(normal_cdf(samples, scales) > 0.999, i, range);
    }
    range = range.to(torch::kInt32);
    const auto lengths = (2 * range + 1).contiguous();
    // Python forms integer offsets before its explicit samples.float(). The
    // subsequent CDF arithmetic promotes against the default-dtype scales.
    const auto samples = (torch::arange(2 * maximum + 1,
        torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU)).unsqueeze(0) -
        range.unsqueeze(1)).to(torch::kFloat32);
    const auto scaled = scales.unsqueeze(1);
    const auto upper = normal_cdf(samples + 0.5, scaled);
    const auto lower = normal_cdf(samples - 0.5, scaled);
    return make_cdf(upper - lower, 2 * lower.slice(1, 0, 1), lengths);
}

torch::Tensor bit_forward(torch::Tensor x, const State& state) {
    const auto h = required(state, "bit_estimator_z.h").to(torch::kCPU);
    const auto b = required(state, "bit_estimator_z.b").to(torch::kCPU);
    const auto a = required(state, "bit_estimator_z.a").to(torch::kCPU);
    for (int i = 0; i < 4; ++i) {
        const auto h_i = h.select(2, i).unsqueeze(2).unsqueeze(3);
        const auto b_i = b.select(2, i).unsqueeze(2).unsqueeze(3);
        x = x * torch::softplus(h_i) + b_i;
        if (i != 3) {
            const auto a_i = a.select(2, i).unsqueeze(2).unsqueeze(3);
            x = x + torch::tanh(x) * torch::tanh(a_i);
        }
    }
    return torch::sigmoid(x);
}

std::pair<torch::Tensor, torch::Tensor> bit_estimator_cdf(const State& state) {
    constexpr int maximum = 8;
    const auto& h = required(state, "bit_estimator_z.h");
    if (h.dim() != 3 || h.size(0) != 64 || h.size(1) != 128 || h.size(2) != 4) {
        throw std::runtime_error("UF bit estimator shape is invalid");
    }
    const auto zeros = torch::zeros({64, 128, 1, 1}, torch::TensorOptions()
        .dtype(c10::get_default_dtype_as_scalartype()).device(torch::kCPU));
    auto range = zeros + maximum;
    for (int i = maximum; i > 1; --i) {
        const auto negative = bit_forward(zeros - i, state);
        const auto positive = bit_forward(zeros + i, state);
        const auto eligible = torch::logical_and(negative < 0.001, positive > 0.999);
        range = torch::where(eligible, i, range);
    }
    range = range.to(torch::kInt32);
    const auto lengths = (range * 2 + 1).reshape({-1});
    const auto samples = torch::arange(2 * maximum + 1,
        torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU))
        .view({1, 1, 1, 2 * maximum + 1}) - range;
    const auto lower = bit_forward(samples - 0.5, state);
    const auto upper = bit_forward(samples + 0.5, state);
    const auto pmf = (upper - lower).reshape({-1, 2 * maximum + 1});
    const auto tail = (lower.slice(3, 0, 1) +
                       (1.0 - bit_forward(range.to(torch::kFloat32), state))).reshape({-1, 1});
    return make_cdf(pmf, tail, lengths);
}

}  // namespace

CheckpointState load_float_checkpoint(const std::filesystem::path& checkpoint,
                                      bool add_entropy_tables,
                                      const std::optional<std::string>& variant) {
    const torch::NoGradGuard no_grad;
    auto state = checkpoint_state(checkpoint);
    if (variant) validate_checkpoint_state(state, *variant);
    const auto options = torch::TensorOptions()
        .dtype(c10::get_default_dtype_as_scalartype()).device(torch::kCPU);
    for (auto& [name, tensor] : state) {
        ProcessInterrupt::check();
        // load_state_dict copies into fresh real module parameters rather than
        // adopting checkpoint dtype/storage. copy_ also owns complex-to-real
        // conversion, as it does in Python's module loader.
        auto parameter = torch::empty(tensor.sizes(), options);
        parameter.copy_(tensor);
        tensor = std::move(parameter);
    }
    if (add_entropy_tables) {
        // CompressionModel.update updates Gaussian before BitEstimator.
        auto [y_cdf, y_lengths] = gaussian_cdf();
        auto [z_cdf, z_lengths] = bit_estimator_cdf(state);
        state["bit_estimator_z.quantized_cdf"] = std::move(z_cdf);
        state["bit_estimator_z.cdf_length"] = std::move(z_lengths);
        state["gaussian_encoder.quantized_cdf"] = std::move(y_cdf);
        state["gaussian_encoder.cdf_length"] = std::move(y_lengths);
    }
    return state;
}

PreparationResult prepare_checkpoint(const std::filesystem::path& checkpoint,
                                     const std::string& variant,
                                     std::optional<std::filesystem::path> output) {
    const torch::NoGradGuard no_grad;
    ProcessInterrupt::check();
    // Python resolves symlinks before choosing the source name and default
    // cache directory; the link's filename is not a distinct preparation key.
    const auto source = python_path_resolve(checkpoint);
    const auto source_sha = sha256_file(source);
    const auto target = output && !output->empty() ? python_path_normalize(*output) :
        source.parent_path() / (source.filename().string() + "." + int16::arithmetic_id + "." +
                               source_sha.substr(0, 16) + ".int16.pt");
    if (python_path_exists(target)) {
        const auto prepared = load_prepared(target, source_sha, variant);
        return {target, variant, prepared.identity};
    }
    if (variant != "image" && variant != "hts" && variant != "htl" && variant != "ld") {
        throw std::invalid_argument("Unknown UF model variant");
    }
    // Python updates entropy tables immediately after load_state_dict, before
    // quantizing convolution/QP parameters. Share that order with FP16 loading.
    auto state = load_float_checkpoint(source, true, variant);
    PreparedModel model;
    model.format = "dcvc-uf-prepared-v1";
    model.arithmetic = int16::arithmetic_id;
    model.variant = variant;
    model.source_sha256 = source_sha;
    for (const auto& [name, value] : state) {
        ProcessInterrupt::check();
        if (ends(name, ".weight")) model.tensors[name] = quantize(value, int16::weight_scale);
        else if (ends(name, ".bias") || name.rfind("q_", 0) == 0) {
            model.tensors[name] = quantize(value, int16::feature_scale);
        }
    }
    if (model.tensors.empty()) throw std::runtime_error("UF checkpoint has no convolution parameters");
    const auto values = torch::arange(-32768, 32768,
        torch::TensorOptions().dtype(torch::kFloat64)) / int16::feature_scale;
    model.tensors["tables.wsilu"] = quantize(values * torch::sigmoid(4 * values),
                                               int16::feature_scale);
    const auto scales = values.clamp(0.11, 16.0);
    const auto scalar_options = torch::TensorOptions().dtype(torch::kFloat64);
    const auto scale_min = torch::log(torch::tensor(0.11, scalar_options));
    const auto scale_max = torch::log(torch::tensor(16.0, scalar_options));
    const auto indexes = ((scales.log() - scale_min) *
        (127.0 / (scale_max - scale_min))).floor();
    model.tensors["tables.scale_index"] = indexes.clamp(0, 127).to(torch::kUInt8);
    model.tensors["tables.z_cdf"] = required(state, "bit_estimator_z.quantized_cdf");
    model.tensors["tables.z_lengths"] = required(state, "bit_estimator_z.cdf_length");
    model.tensors["tables.y_cdf"] = required(state, "gaussian_encoder.quantized_cdf");
    model.tensors["tables.y_lengths"] = required(state, "gaussian_encoder.cdf_length");
    state.clear();
    model.identity = prepared_identity(model);
    if (sha256_file(source) != source_sha) {
        throw std::runtime_error("UF checkpoint changed during preparation");
    }
    save_prepared(model, target);
    // Python validates the published artifact again, including after a racing
    // cache writer. Release preparation tensors before that independent read.
    model.tensors.clear();
    const auto published = load_prepared(target, source_sha, variant);
    return {target, variant, published.identity};
}

std::filesystem::path prepared_artifact_path(
    const std::filesystem::path& checkpoint, const std::string& variant,
    std::optional<std::filesystem::path> path,
    const std::optional<std::string>& expected_source) {
    path = existing_prepared_path(checkpoint, std::move(path), expected_source);
    if (path && python_path_exists(*path)) return *path;
    return prepare_checkpoint(checkpoint, variant, std::move(path)).path;
}

ResolvedPreparedModel resolve_prepared(
    const std::filesystem::path& checkpoint, const std::string& variant,
    std::optional<std::filesystem::path> path,
    const std::optional<std::string>& expected_identity,
    const std::optional<std::string>& expected_source) {
    path = existing_prepared_path(checkpoint, std::move(path), expected_source);
    if (path && python_path_exists(*path))
        return {*path, load_prepared(*path, expected_source, variant, expected_identity)};
    const auto prepared = prepare_checkpoint(checkpoint, variant, path);
    auto data = load_prepared(prepared.path);
    if (expected_identity && data.identity != *expected_identity)
        throw std::runtime_error("Prepared identity mismatch; copy the prepared file used by the encoder");
    if (expected_source && data.source_sha256 != *expected_source)
        throw std::runtime_error("Prepared checkpoint identity mismatch");
    return {prepared.path, std::move(data)};
}

}  // namespace uf
