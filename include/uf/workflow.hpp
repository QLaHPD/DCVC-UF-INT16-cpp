#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

#include "uf/remote.hpp"

namespace uf {

struct StreamInputs {
    int stream{-1}, errors{-1}, state{-1}, buffer{-1};
};

struct EncodeOptions {
    std::filesystem::path input_file;
    std::filesystem::path output_root;
    std::optional<std::string> relative_path;
    std::filesystem::path prepared_i;
    std::filesystem::path prepared_p;
    std::filesystem::path checkpoint_i;
    std::filesystem::path checkpoint_p;
    std::string runtime{"fp16"};
    std::string variant{"hts"};
    std::string device{"cuda:0"};
    int qp_i{36}, qp_p{30};
    int reset_interval{32}, intra_period{-1};
    std::optional<int> resolution;
    std::optional<double> fps;
    std::optional<std::uint64_t> max_frames;
    double skip_threshold{};
    std::string audio{"opus"};
    std::string opus_channels{"mono"};
    std::string opus_bitrate{"6k"};
    // argparse retains integer default 60, but explicit values are floats.
    nlohmann::json opus_frame_ms = 60;
    int opus_complexity{10};
    std::string opus_vbr{"on"};
    int input_threads{1};
    int prefetch_frames{};
    std::string hwaccel{"none"};
    int workers{1};
    std::vector<std::string> devices;
    std::string shared_instance;
    double shared_lease_seconds{900};
    std::optional<RemoteJob> remote;
    std::optional<StreamInputs> streaming;
};

// Archive encode. Payloads are staged in /dev/shm, copied into a hidden
// destination stage, then hard-linked under final names with the manifest last.
nlohmann::json encode_pipeline(const EncodeOptions& options);
std::filesystem::path encode_local(const EncodeOptions& options, bool* resumed = nullptr);
std::filesystem::path encode_remote(const EncodeOptions& options, bool* resumed = nullptr);

struct EncodeRun;

std::vector<EncodeOptions> discover_local_jobs(
    const std::filesystem::path& base_root,
    const std::filesystem::path& output_root,
    const std::vector<std::string>& channel_ids,
    bool recursive, const EncodeOptions& settings);

std::vector<std::filesystem::path> encode_folder(
    const std::filesystem::path& base_root,
    const std::filesystem::path& output_root,
    const std::vector<std::string>& channel_ids,
    bool recursive,
    const EncodeOptions& settings, EncodeRun* report = nullptr);

}  // namespace uf
