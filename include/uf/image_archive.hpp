#pragma once

#include <cstdint>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <vector>

namespace uf {

struct IntraImageArchive {
    std::filesystem::path path;
    nlohmann::json metadata;
    // Python's gallery cache uses repr(pipeline), including dictionary order
    // and scalar types. Keep this ordered subtree for the lazy viewer key.
    nlohmann::ordered_json recorded_pipeline;
    std::uint64_t payload_offset{};
    std::vector<std::uint8_t> payload;
};

// Verification always hashes/counts chunks. Callers that reopen the payload
// can avoid retaining its bytes; disabled verification leaves payload empty.
IntraImageArchive read_intra_image(const std::filesystem::path& path,
                                    bool verify_payload = true,
                                    bool retain_payload = true);

}  // namespace uf
