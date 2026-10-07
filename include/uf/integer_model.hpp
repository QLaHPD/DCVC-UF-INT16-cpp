#pragma once

#include "uf/entropy.hpp"
#include "uf/layers.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace uf {

struct CompressedFrame {
    std::vector<std::uint8_t> payload;
    std::vector<torch::Tensor> frames;
};

class IntegerModel {
public:
    // Own variant metadata, layer tensor handles and copied entropy tables;
    // the caller may release the PreparedModel container after construction.
    IntegerModel(const PreparedModel& prepared, torch::Device device = torch::kCPU,
                 bool reconstruct_on_encode = true);

    void clear_dpb();
    // Match Python's common codec interface. INT16 accepts this flag but
    // always clears state and unshuffles the reference; adaptation is deferred
    // until the next P-frame context update for either flag value.
    void add_ref_feature_from_frame(const torch::Tensor& frame,
                                    bool apply_feature_adaptor = true);
    // Convert through int(threshold * 512 + 0.5), then check [0, 32767].
    void set_skip_threshold(double threshold);
    // Read-only state views for the standalone determinism report. Undefined
    // tensors represent Python None; these accessors allocate no new storage.
    const torch::Tensor& reference() const noexcept { return reference_; }
    const torch::Tensor& memory() const noexcept { return memory_; }
    const torch::Tensor& context() const noexcept { return context_; }
    // Direct model calls disable reverse-mode gradients for their scope,
    // restoring the caller's mode on success or exception, as Python no_grad.
    // Nonzero skip thresholds round only active residuals and assign their
    // symbols through the same Boolean mask used for scale lookup/decoding.
    CompressedFrame compress(const torch::Tensor& frames, int qp, bool reset = false);
    std::vector<torch::Tensor> decompress(const std::vector<std::uint8_t>& payload,
                                          std::uint32_t width, std::uint32_t height,
                                          int qp, int entropy_parallel,
                                          bool reset = false);

private:
    std::string variant_;
    torch::Device device_;
    Layers layers_;
    EntropyEncoder encoder_;
    EntropyDecoder decoder_;
    bool reconstruct_on_encode_;
    int z_channels_;
    int skip_threshold_{};
    torch::Tensor reference_;
    torch::Tensor memory_;
    torch::Tensor context_;
    std::vector<std::int64_t> position_shape_;
    // One shape's ordered flat mask positions across all NCHW batches.
    std::vector<torch::Tensor> positions_;

    void update_context();
    torch::Tensor params(const torch::Tensor& z, int qp, std::int64_t height,
                          std::int64_t width);
    std::vector<torch::Tensor> reconstruct(const torch::Tensor& y, int qp, bool reset,
                                            bool output_pixels);
    void set_positions(const torch::Tensor& scales);
    std::pair<torch::Tensor, std::vector<std::vector<std::int16_t>>>
    prior(const torch::Tensor& y, const torch::Tensor& parameters, int qp, bool encode);
};

// Use last-two-axis extents and Python (-extent)%multiple, then its explicit
// four-axis replication slices only when needed; no-padding retains the input.
torch::Tensor pad_right_bottom(const torch::Tensor& value, std::int64_t multiple);

}  // namespace uf
