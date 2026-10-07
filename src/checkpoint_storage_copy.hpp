// Tensor/storage copying follows installed PyTorch _tensor.py/storage.py.
// Retained BSD notice: licenses/PyTorch.txt.
#pragma once

#include "uf/process.hpp"

#include <ATen/ops/empty.h>
#include <c10/core/Storage.h>
#include <torch/types.h>

#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace uf::checkpoint_detail {

// One copy per untyped storage, retaining each tensor's offset/size/stride.
// The optional destination is fixed for a traversal; otherwise keep devices.
// Callers provide no-grad scope and tensor identity/grad memoization.
class StorageCopies {
public:
    explicit StorageCopies(std::optional<c10::Device> destination = std::nullopt)
        : destination_(std::move(destination)) {}

    torch::Tensor copy(const torch::Tensor& tensor, bool retain_math_bits = false) {
        ProcessInterrupt::check();
        if (tensor.layout() != torch::kStrided || tensor.device().is_meta()) {
            throw std::invalid_argument("Checkpoint storage copy requires a strided tensor with storage");
        }
        const auto source = tensor.storage();
        const auto identity = source.unsafeGetStorageImpl();
        auto found = copies_.find(identity);
        const auto device = destination_.value_or(tensor.device());
        if (found == copies_.end()) {
            if (source.nbytes() > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
                throw std::overflow_error("Checkpoint storage exceeds the native tensor size interface");
            }
            auto bytes = at::empty({0}, torch::TensorOptions().dtype(torch::kUInt8).device(tensor.device()))
                .set_(source, 0, {static_cast<std::int64_t>(source.nbytes())}, {1});
            // Copy all storage bytes, including elements outside this view.
            found = copies_.emplace(identity, bytes.to(device, torch::kUInt8, false, true)).first;
        }
        // clone keeps a quantized tensor's quantizer; replace its storage with
        // the shared copy. Plain UF tensors need only an empty typed view.
        auto output = tensor.is_quantized() ? tensor.detach().to(device).clone()
            : at::empty({0}, tensor.options().device(device).requires_grad(false));
        output.set_(found->second.storage(), tensor.storage_offset(), tensor.sizes(), tensor.strides());
        output._set_conj(retain_math_bits && tensor.is_conj());
        output._set_neg(retain_math_bits && tensor.is_neg());
        return output;
    }

private:
    std::optional<c10::Device> destination_;
    std::unordered_map<c10::StorageImpl*, torch::Tensor> copies_;
};

}  // namespace uf::checkpoint_detail
