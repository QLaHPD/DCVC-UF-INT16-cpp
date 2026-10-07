// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#pragma once

#include "uf/training_layers.hpp"
#include "uf/prepare.hpp"

#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace uf::training {

struct BitEstimatorImpl : torch::nn::Module {
    BitEstimatorImpl(std::int64_t qp_count, std::int64_t channels);
    // Scalar QPs use Python's [qp:qp+1] slice, including empty/clamped
    // negative or out-of-range slices; tensor QPs use index_select.
    // Optional dtype conversion preserves device and gradient connections.
    std::array<torch::Tensor, 3> get_hba(const torch::Tensor& qp,
        std::optional<torch::ScalarType> dtype = std::nullopt) const;
    std::array<torch::Tensor, 3> get_hba(std::int64_t qp,
        std::optional<torch::ScalarType> dtype = std::nullopt) const;
    torch::Tensor forward(const torch::Tensor& input, const torch::Tensor& qp);
    // A distinct name retains ModuleHolder/AnyModule's tensor forward contract.
    torch::Tensor forward_scalar(const torch::Tensor& input, std::int64_t qp);
    torch::Tensor probability(const torch::Tensor& input, const torch::Tensor& qp);
    torch::Tensor probability(const torch::Tensor& input, std::int64_t qp);
    torch::Tensor h, b, a;
};
TORCH_MODULE(BitEstimator);

struct PriorResult {
    torch::Tensor residual, quantized, reconstruction, scales;
};

struct RateDistortion {
    torch::Tensor reconstruction, mse, bpp, bits_y, bits_z;
};

struct Loss {
    torch::Tensor losses, mean;
};

using LossInfo = std::map<std::string, std::string>;

struct TrainingForward {
    RateDistortion result;
    Loss loss;
    std::optional<LossInfo> info;
};

Loss rate_distortion_loss(const RateDistortion& result, const torch::Tensor& lambdas);
LossInfo loss_info(const RateDistortion& result, const Loss& loss, std::int64_t pixels);

// All names and tensor shapes follow the Python nn.Module state dictionary.
// Detached views on their original devices; the checkpoint writer maps shared
// storage to CPU. Later parameter updates remain visible through these views.
CheckpointState export_training_state(const torch::nn::Module& module);
// Default Python Module loading: local parameters/buffers then children;
// matching tensors remain copied if another key/shape/copy error is raised.
void load_training_state(torch::nn::Module& module, const CheckpointState& state, bool strict = true);

class CompressionTrainingImpl : public torch::nn::Module {
public:
    using Transform = std::function<torch::Tensor(const torch::Tensor&)>;
    using Combine = std::function<torch::Tensor(const torch::Tensor&, const torch::Tensor&)>;
    explicit CompressionTrainingImpl(std::int64_t z_channels);
    static constexpr std::int64_t qp_count = 64;

    void initialize_weights();
    static torch::Tensor add_noise(const torch::Tensor& input);
    // Undefined scales represent Python None and bypass index selection.
    // Defined scales follow index_select(0, qp)[:, :, None, None], retaining
    // trailing axes and the indexing operator's rank validation/autograd.
    static torch::Tensor select_q(const torch::Tensor& scales, const torch::Tensor& qp);
    static torch::Tensor probabilities_to_bits(const torch::Tensor& probabilities);
    static PriorResult process_mask(const torch::Tensor& y, const torch::Tensor& scales,
                                    const torch::Tensor& means, const torch::Tensor& mask);
    torch::Tensor mse(const torch::Tensor& input, const torch::Tensor& reconstruction);
    torch::Tensor y_bits(const torch::Tensor& residual, const torch::Tensor& scales);
    torch::Tensor z_bits(const torch::Tensor& residual, const torch::Tensor& qp);
    // Python caches by B,C,H,W and partition count, retaining the first
    // requested device for that shape. Prior calls use common_params.device.
    std::vector<torch::Tensor> masks(const torch::Tensor& latent, int parts);
    std::vector<torch::Tensor> masks(const torch::Tensor& latent, int parts,
                                    const torch::Device& device);

    PriorResult prior_two(const torch::Tensor& y, const torch::Tensor& parameters,
                          const Combine& spatial_prior);
    PriorResult prior_four(const torch::Tensor& y, const torch::Tensor& q_encoder,
        const torch::Tensor& q_decoder, const torch::Tensor& parameters, const Transform& reduction,
        const std::array<Combine, 3>& adaptors, const Transform& spatial_prior,
        bool spatial_prior_has_scales = false);

    BitEstimator bit_estimator_z{nullptr};
private:
    using MaskKey = std::tuple<std::int64_t, std::int64_t, std::int64_t, std::int64_t, int>;
    std::map<MaskKey, std::vector<torch::Tensor>> masks_;
};

}  // namespace uf::training
