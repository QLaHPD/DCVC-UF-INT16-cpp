#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace uf {

// Canonical strides matter even for 1x1 weights, which LibTorch considers
// contiguous in either format. This primitive transfers/reformats a tensor;
// it does not read a checkpoint or construct a model.
torch::Tensor fp16_parameter_to_device(const torch::Tensor& value, const torch::Device& device);

struct Fp16Packet {
    std::vector<std::uint8_t> payload;
    torch::Tensor reconstruction;
    std::uint8_t entropy_parallel{};
};

class Fp16Model {
public:
    // Threshold follows Python's raw update/proxy Float32 argument, including
    // nonfinite values. Archive CLI bounds and benchmark normalization are
    // applied by their callers.
    Fp16Model(const std::filesystem::path& checkpoint, std::string variant,
              torch::Device device, double skip_threshold = 0,
              const std::optional<std::string>& expected_sha256 = std::nullopt);
    ~Fp16Model();
    Fp16Model(const Fp16Model&) = delete;
    Fp16Model& operator=(const Fp16Model&) = delete;

    Fp16Packet compress(const torch::Tensor& input, int qp,
                        int padding_bottom, int padding_right,
                        bool reset_feature_memory = false);
    std::vector<torch::Tensor> decompress(
        const std::vector<std::uint8_t>& payload,
        std::uint32_t width, std::uint32_t height,
        int qp, int entropy_parallel, bool reset_feature_memory = false);
    void add_reference(const torch::Tensor& frame, bool apply_adaptor = true);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

bool fp16_compiled();
int fp16_compute_capability();
bool fp16_available();

}  // namespace uf
