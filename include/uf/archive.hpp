#pragma once

#include "uf/bitstream.hpp"

#include <filesystem>
#include <functional>
#include <nlohmann/json.hpp>
#include <string>

namespace uf {

struct Archive {
    std::filesystem::path manifest;
    std::filesystem::path root;
    nlohmann::json metadata;
};

std::filesystem::path manifest_path(const std::filesystem::path& input);
std::filesystem::path artifact_path(const Archive& archive, const std::string& logical_name);
std::string sha256_file(const std::filesystem::path& path);
// storage.memory_stage_root: read the complete mount table as Python text,
// require a /dev/shm tmpfs entry and return that RAM staging path.
std::filesystem::path memory_stage_root();
// completed_disk_stage file copy: exclusive create, filled 1 MiB reads,
// callback before open/after nonempty chunks, sync and writer/reader close.
// The caller creates/syncs the stage directory and handles stage cleanup.
void copy_completed_archive_file(const std::filesystem::path& source,
    const std::filesystem::path& target, const std::function<void()>& check = {});
// A non-null expected pipeline is compared before completion/artifact checks,
// matching load_bundle(path, expected). JSON null represents Python None.
Archive load_archive(const std::filesystem::path& input, bool verify_artifacts = true,
                     const nlohmann::json* expected = nullptr);
// Convert represented integral dimensions without truncation or wrapping.
// Streaming decode compares every SPS with these metadata dimensions.
Sps archive_video_dimensions(const nlohmann::json& video);
ScanResult scan_archive_bitstream(const Archive& archive);

}  // namespace uf
