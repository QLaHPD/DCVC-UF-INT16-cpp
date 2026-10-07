#pragma once

#include <torch/torch.h>

#include <filesystem>
#include <map>
#include <optional>
#include <string>

namespace uf {

struct PreparedModel {
    std::string format;
    std::string arithmetic;
    std::string variant;
    std::string source_sha256;
    std::string identity;
    std::map<std::string, torch::Tensor> tensors;

    const torch::Tensor& tensor(const std::string& name) const;
};

PreparedModel load_prepared(
    const std::filesystem::path& path,
    const std::optional<std::string>& expected_source_sha256 = std::nullopt,
    const std::optional<std::string>& expected_variant = std::nullopt,
    const std::optional<std::string>& expected_identity = std::nullopt);

std::string prepared_identity(const PreparedModel& model);
void save_prepared(const PreparedModel& model, const std::filesystem::path& path);

}  // namespace uf
