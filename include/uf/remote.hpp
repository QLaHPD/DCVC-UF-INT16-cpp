#pragma once

#include "uf/media.hpp"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace uf {

struct RemoteOptions {
    // Native argv bytes. remote_source_urls decodes these to Python text.
    std::vector<std::string> source_urls;
    std::vector<std::string> youtube_channels;
    std::vector<std::string> twitch_channels;
    std::filesystem::path output_root;
    std::string ytdlp_binary{"yt-dlp"};
    std::optional<std::filesystem::path> cookies;
    int max_height{480};
    std::optional<int> max_videos;
    bool write_thumbnail{true};
    std::string thumbnail_codec{"keep"};
    int thumbnail_qp{45};
    std::optional<nlohmann::json> pipeline;
};

struct RemoteJob {
    std::string video_id;
    std::string channel_id;
    // Provider/JSON Python text (UTF-8/WTF-8). Convert at POSIX argv boundaries.
    std::string url;
    std::string video_format;
    std::optional<std::string> audio_format;
    MediaInfo media;
    nlohmann::json public_metadata;
    std::filesystem::path manifest;
    std::string ytdlp_binary;  // Native POSIX executable spelling.
    std::optional<std::filesystem::path> cookies;
    bool write_thumbnail{true};
    std::string thumbnail_codec{"keep"};
    int thumbnail_qp{45};
};

struct RemoteDiscovery {
    std::vector<RemoteJob> jobs;
    int resumed{};
    int failed{};
    int skipped_live{};
};

std::vector<std::string> remote_source_urls(const RemoteOptions& options);
// Resolve a POSIX executable name or explicit path without launching it.
// Call before preparing models/artifacts, as maintained run_stream() does.
std::string resolve_ytdlp_binary(const std::string& binary);
RemoteDiscovery discover_remote_jobs(
    const RemoteOptions& options,
    const std::function<void(RemoteJob&&)>& consume = {},
    const std::function<bool()>& cancelled = {});

struct RemoteAheadItem {
    std::optional<RemoteJob> job;
    bool finished{};
};

// One queued job plus one metadata request may be ahead of the consumer.
class RemoteAheadJobs {
public:
    explicit RemoteAheadJobs(RemoteOptions options);
    ~RemoteAheadJobs();
    RemoteAheadJobs(const RemoteAheadJobs&) = delete;
    RemoteAheadJobs& operator=(const RemoteAheadJobs&) = delete;
    RemoteAheadItem poll();
    RemoteDiscovery finish();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
std::vector<std::string> ytdlp_command(
    const std::string& binary, const std::optional<std::filesystem::path>& cookies);
std::optional<std::filesystem::path> store_remote_thumbnail(const RemoteJob& job);

// yt-dlp is given a private cookie copy. Normal contexts use checked close;
// the destructor supplies fallback removal only when no explicit close ran.
class RemoteCookies {
public:
    explicit RemoteCookies(const std::optional<std::filesystem::path>& source);
    ~RemoteCookies();
    RemoteCookies(const RemoteCookies&) = delete;
    RemoteCookies& operator=(const RemoteCookies&) = delete;
    const std::optional<std::filesystem::path>& path() const { return path_; }
    void close();
private:
    std::optional<std::filesystem::path> path_;
    bool cleanup_attempted_{};
};

// cookie_copy's try/finally: unlink failure replaces an active body error,
// and explicit cleanup is not retried by the destructor.
template<class Operation>
auto with_remote_cookies(RemoteCookies& cookies, Operation&& operation)
    -> std::invoke_result_t<Operation> {
    using Result = std::invoke_result_t<Operation>;
    if constexpr (std::is_void_v<Result>) {
        try { std::forward<Operation>(operation)(); }
        catch (...) { cookies.close(); throw; }
        cookies.close();
    } else {
        Result result = [&]() -> Result {
            try { return std::forward<Operation>(operation)(); }
            catch (...) { cookies.close(); throw; }
        }();
        cookies.close();
        return std::forward<Result>(result);
    }
}

}  // namespace uf
