#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace uf {

struct InventoryItem {
    std::filesystem::path manifest;
    std::filesystem::path source;
    std::string kind{"video"};
    bool eligible{};
    bool remote{};
    bool exists{};
    bool metadata{};
};

struct ChannelInventory {
    std::filesystem::path channel;
    std::vector<InventoryItem> archives;
    std::uint64_t originals{};
    std::uint64_t bytes{};
    std::uint64_t streamed{};
    std::uint64_t absent{};
    std::uint64_t ineligible{};
    std::vector<std::string> blockers;
};

struct InventoryError {
    std::filesystem::path manifest;
    std::string message;
};

struct InventoryResult {
    std::vector<ChannelInventory> channels;
    std::vector<InventoryError> errors;
};

InventoryResult inventory_channels(
    const std::filesystem::path& output_root,
    const std::vector<std::string>& channel_ids = {},
    std::optional<std::filesystem::path> base_root = std::nullopt,
    bool allow_missing_metadata = false,
    const std::optional<std::set<std::filesystem::path>>& sources = std::nullopt);

}  // namespace uf
