#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "uf/remote.hpp"
#include "uf/media.hpp"
#include "uf/image_pixels.hpp"
#include "uf/process.hpp"
#include "uf/archive.hpp"
#include "uf/events.hpp"
#include "uf/json_compat.hpp"
#include "uf/python_json.hpp"
#include "uf/python_path.hpp"
#include "uf/python_numeric.hpp"
#include "uf/utf8.hpp"

#include <curl/curl.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <condition_variable>
#include <deque>
#include <exception>
#include <iomanip>
#include <iostream>
#include <array>
#include <arpa/inet.h>
#include <cerrno>
#include <fcntl.h>
#include <limits>
#include <locale>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <pwd.h>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <tuple>
#include <thread>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace uf {
namespace {

std::string safe_id(const nlohmann::json& value, bool channel) {
    static const std::regex video_pattern("[A-Za-z0-9_-]{11}");
    static const std::regex channel_pattern("UC[A-Za-z0-9_-]{22}");
    if (!value.is_string() || !std::regex_match(value.get_ref<const std::string&>(),
                                               channel ? channel_pattern : video_pattern))
        throw std::invalid_argument("Invalid YouTube " + std::string(channel ? "channel" : "video") +
                                    " ID: " + python_json_repr(value));
    return value.get<std::string>();
}

std::string safe_component(const nlohmann::json& value) {
    static const std::regex pattern("[A-Za-z0-9_-]{1,128}");
    if (!value.is_string() || !std::regex_match(value.get_ref<const std::string&>(), pattern))
        throw std::invalid_argument("Invalid channel/video identifier: " + python_json_repr(value));
    return value.get<std::string>();
}

std::string string(const nlohmann::json& item, const char* key) {
    if (!item.is_object() || !item.contains(key) || !item.at(key).is_string()) return {};
    return item.at(key).get<std::string>();
}

const nlohmann::json& truth_field(const nlohmann::json& value, const char* key) {
    static const nlohmann::json zero(0);
    const auto& selected = python_json_dict_get(value, key);
    return json_truthy(selected) ? selected : zero;
}

template<std::size_t Count>
bool key_less(const std::array<const nlohmann::json*, Count>& first,
              const std::array<const nlohmann::json*, Count>& second) {
    for (std::size_t index = 0; index < Count; ++index) {
        if (first[index] == second[index] || json_python_equal(*first[index], *second[index])) continue;
        return json_python_less(*first[index], *second[index]);
    }
    return false;
}

template<std::size_t Count>
bool key_equal(const std::array<const nlohmann::json*, Count>& first,
               const std::array<const nlohmann::json*, Count>& second) {
    for (std::size_t index = 0; index < Count; ++index) {
        if (first[index] != second[index] &&
            !json_python_equal(*first[index], *second[index])) return false;
    }
    return true;
}

template<class Visit>
void visit_formats(const nlohmann::json& info, const Visit& visit) {
    static const nlohmann::json empty = nlohmann::json::array();
    const auto& formats = python_json_dict_get(info, "formats", empty);
    if (formats.is_array()) {
        for (const auto& format : formats) visit(format);
    } else if (formats.is_object() || formats.is_string()) {
        if (json_truthy(formats)) throw PythonJsonAttributeError("'str' object has no attribute 'get'");
    } else {
        throw PythonJsonTypeError("'" + std::string(json_compat_detail::python_type_name(formats)) +
                                  "' object is not iterable");
    }
}

bool video_format(const nlohmann::json& format) {
    if (!json_truthy(python_json_dict_get(format, "url"))) return false;
    const auto& codec = python_json_dict_get(format, "vcodec");
    return !codec.is_null() && codec != "none" &&
        json_truthy(python_json_dict_get(format, "height")) &&
        !json_truthy(python_json_dict_get(format, "has_drm"));
}

bool audio_format(const nlohmann::json& format) {
    const auto& codec = python_json_dict_get(format, "acodec");
    return !codec.is_null() && codec != "none" &&
        json_truthy(python_json_dict_get(format, "url")) &&
        !json_truthy(python_json_dict_get(format, "has_drm"));
}

std::pair<nlohmann::json, std::optional<nlohmann::json>> select_formats(
    const nlohmann::json& info, int max_height) {
    // Keep references to the provider's format objects, as Python's filtered
    // lists do. Copies would discard container/field identity during ranking.
    std::vector<const nlohmann::json*> videos;
    visit_formats(info, [&](const auto& format) {
        if (video_format(format)) videos.push_back(&format);
    });
    if (videos.empty()) throw std::invalid_argument("No downloadable video format");
    if (max_height) {
        const nlohmann::json maximum(max_height);
        std::vector<const nlohmann::json*> bounded;
        for (const auto& format : videos) {
            const auto& height = format->at("height");
            if (json_python_equal(height, maximum) || json_python_less(height, maximum))
                bounded.push_back(format);
        }
        if (bounded.empty()) {
            const auto minimum = std::min_element(videos.begin(), videos.end(),
                [](const auto& a, const auto& b) {
                    return json_python_less(a->at("height"), b->at("height"));
                });
            for (const auto& format : videos)
                if (json_python_equal(format->at("height"), (*minimum)->at("height"))) bounded.push_back(format);
        }
        videos = std::move(bounded);
    }
    if (videos.empty()) throw std::invalid_argument("max() arg is an empty sequence");
    const auto geometry_key = [](const auto& format) {
        return std::array<const nlohmann::json*, 2>{{
            &truth_field(format, "height"), &truth_field(format, "width")}};
    };
    const auto best_geometry = std::max_element(videos.begin(), videos.end(),
        [&](const auto* a, const auto* b) { return key_less(geometry_key(*a), geometry_key(*b)); });
    const auto geometry = geometry_key(**best_geometry);
    std::vector<const nlohmann::json*> best_formats;
    for (const auto* format : videos)
        if (key_equal(geometry_key(*format), geometry)) best_formats.push_back(format);
    std::vector<const nlohmann::json*> separate_video;
    for (const auto* format : best_formats) {
        const auto& codec = python_json_dict_get(*format, "acodec");
        if (codec.is_null() || codec == "none") separate_video.push_back(format);
    }
    if (!separate_video.empty()) best_formats = std::move(separate_video);
    if (best_formats.empty()) throw std::invalid_argument("max() arg is an empty sequence");
    const auto best_video = std::max_element(best_formats.begin(), best_formats.end(),
        [](const auto* a, const auto* b) {
            return json_python_less(truth_field(*a, "tbr"), truth_field(*b, "tbr"));
        });
    std::vector<const nlohmann::json*> audios;
    visit_formats(info, [&](const auto& format) {
        if (audio_format(format)) audios.push_back(&format);
    });
    std::vector<const nlohmann::json*> separate_audio;
    for (const auto* format : audios) {
        const auto& codec = python_json_dict_get(*format, "vcodec");
        if (codec.is_null() || codec == "none") separate_audio.push_back(format);
    }
    if (!separate_audio.empty()) audios = std::move(separate_audio);
    std::vector<const nlohmann::json*> originals;
    static const nlohmann::json empty_note("");
    const nlohmann::json original_threshold(10);
    for (const auto& format : audios) {
        const auto& value = python_json_dict_get(*format, "format_note", empty_note);
        auto note = value.is_string() ? value.get<std::string>() : python_json_repr(value);
        // Only ASCII letters can lower to the literal "original" here; keep
        // other UTF-8 bytes and avoid the process locale's tolower policy.
        for (auto& letter : note) if (letter >= 'A' && letter <= 'Z') letter += 'a' - 'A';
        if (note.find("original") != std::string::npos) originals.push_back(format);
        else {
            const auto& preference = truth_field(*format, "language_preference");
            if (json_python_equal(preference, original_threshold) ||
                json_python_less(original_threshold, preference)) originals.push_back(format);
        }
    }
    if (!originals.empty()) audios = std::move(originals);
    else {
        std::vector<const nlohmann::json*> languages;
        for (const auto& format : audios) {
            const auto& language = python_json_dict_get(*format, "language");
            if (!json_truthy(language)) continue;
            if (language.is_array() || language.is_object())
                throw PythonJsonTypeError("unhashable type: '" +
                    std::string(json_compat_detail::python_type_name(language)) + "'");
            if (std::none_of(languages.begin(), languages.end(),
                [&](const auto* saved) { return saved == &language || json_python_equal(*saved, language); }))
                languages.push_back(&language);
        }
        if (languages.size() > 1)
            throw std::invalid_argument("Multiple audio languages without an identified original track; refusing an automatic dub");
    }
    if (audios.empty()) return {**best_video, std::nullopt};
    const auto audio_key = [](const auto& format) {
        const auto& abr = python_json_dict_get(format, "abr");
        const auto& rate = json_truthy(abr) ? abr : truth_field(format, "tbr");
        return std::array<const nlohmann::json*, 2>{{&truth_field(format, "language_preference"), &rate}};
    };
    const auto best_audio = std::max_element(audios.begin(), audios.end(),
        [&](const auto* a, const auto* b) { return key_less(audio_key(*a), audio_key(*b)); });
    return {**best_video, **best_audio};
}

std::string upload_timestamp(const nlohmann::json& info) {
    for (const char* key : {"timestamp", "release_timestamp"}) {
        const auto& value = python_json_dict_get(info, key);
        if (!value.is_number() && !python_json_extended_integer(value)) continue;
        // math.isfinite(int) converts to binary64 before testing the sign;
        // an oversized negative integer must therefore raise as well.
        const auto seconds = python_json_as_double(value);
        if (std::isfinite(seconds) && seconds >= 0) return python_json_int(value).decimal();
    }
    const auto& value = python_json_dict_get(info, "upload_date");
    if (!value.is_string())
        throw std::invalid_argument("Remote video has no publication timestamp or upload date");
    const auto& date = value.get_ref<const std::string&>();
    if (!python_decimal_digits(date))
        throw std::invalid_argument("Remote video has no publication timestamp or upload date");
    const auto digits = python_numeric_ascii(date);
    if (digits.size() != 8)
        throw std::invalid_argument("Remote video has no publication timestamp or upload date");
    // re.fullmatch(\d{8}) uses Unicode decimal digits. CPython _strptime's
    // %Y permits four Unicode digits, %m requires ASCII, and two-digit %d
    // permits a Unicode final digit only after ASCII 1 or 2.
    const auto suffix = utf8_tail(date, 4);
    const auto day = utf8_tail(date, 2);
    const bool ascii_month = suffix.size() >= 2 && suffix[0] >= '0' && suffix[0] <= '1' &&
        suffix[1] >= '0' && suffix[1] <= '9';
    const auto month = std::stoi(digits.substr(4, 2));
    const bool matched_day = day.front() == '1' || day.front() == '2' || day == "30" || day == "31" ||
        (day.size() == 2 && day.front() == '0' && day[1] >= '1' && day[1] <= '9');
    if (!ascii_month || month < 1 || month > 12 || !matched_day)
        throw std::invalid_argument("time data " + python_json_repr(value) +
                                    " does not match format '%Y%m%d'");
    std::tm parts{};
    const auto year = std::stoi(digits.substr(0, 4));
    if (year == 0) throw std::invalid_argument("year 0 is out of range");
    parts.tm_year = year - 1900;
    parts.tm_mon = month - 1;
    parts.tm_mday = std::stoi(digits.substr(6, 2));
    const auto requested = parts;
    errno = 0;
    const auto stamp = timegm(&parts);
    if (errno == EOVERFLOW) throw std::overflow_error("Remote upload date exceeds native time range");
    if (parts.tm_year != requested.tm_year || parts.tm_mon != requested.tm_mon ||
        parts.tm_mday != requested.tm_mday)
        throw std::invalid_argument("day is out of range for month");
    return std::to_string(stamp);
}

std::filesystem::path pathlib_path(const std::string& path) {
    // CPython pathlib POSIX spelling; keep '..' for filesystem traversal.
    // Retained license: licenses/CPython-3.10.20.txt.
    std::size_t cursor = 0;
    while (cursor < path.size() && path[cursor] == '/') ++cursor;
    std::string result = cursor == 0 ? "" : (cursor == 2 ? "//" : "/");
    while (cursor < path.size()) {
        auto end = path.find('/', cursor);
        if (end == std::string::npos) end = path.size();
        if (end != cursor && !(end == cursor + 1 && path[cursor] == '.')) {
            if (!result.empty() && result.back() != '/') result.push_back('/');
            result.append(path, cursor, end - cursor);
        }
        cursor = end;
        while (cursor < path.size() && path[cursor] == '/') ++cursor;
    }
    return std::filesystem::path(result.empty() ? "." : result);
}

bool pathlib_is_file(const std::filesystem::path& candidate) {
    const auto original = candidate.string();
    if (original.find('\0') != std::string::npos) return false;
    const auto path = pathlib_path(original);
    struct stat status;
    if (stat(path.c_str(), &status) == 0) return S_ISREG(status.st_mode);
    const int error = errno;
    if (error == ENOENT || error == ENOTDIR || error == EBADF || error == ELOOP) return false;
    throw std::system_error(error, std::generic_category(), "Cannot inspect Deno runtime: " + original);
}

std::filesystem::path home_directory() {
    // Path.home()/POSIX expanduser('~'): HOME wins even when it is empty.
    std::string home;
    if (const char* environment = std::getenv("HOME")) {
        home = environment;
    } else {
        const long suggested = sysconf(_SC_GETPW_R_SIZE_MAX);
        std::vector<char> buffer(suggested > 0 ? static_cast<std::size_t>(suggested) : 1024);
        struct passwd entry;
        struct passwd* found = nullptr;
        for (;;) {
            const int error = getpwuid_r(getuid(), &entry, buffer.data(), buffer.size(), &found);
            if (error == ERANGE) {
                if (buffer.size() > buffer.max_size() / 2) {
                    throw std::length_error("Home-directory lookup buffer is too large");
                }
                buffer.resize(buffer.size() * 2);
                continue;
            }
            if (error) throw std::system_error(error, std::generic_category(), "Home-directory lookup");
            if (!found) throw std::runtime_error("Could not determine home directory.");
            home = entry.pw_dir;
            break;
        }
    }
    const auto last = home.find_last_not_of('/');
    home = last == std::string::npos ? "/" : home.substr(0, last + 1);
    if (home.front() == '~') throw std::runtime_error("Could not determine home directory.");
    return std::filesystem::path(home);
}

nlohmann::json metadata(const std::string& url, const RemoteOptions& options, bool flat,
                        const std::function<bool()>& cancelled) {
    ProcessInterrupt::check();
    if (cancelled && cancelled()) throw std::runtime_error("Streaming metadata cancelled");
    RemoteCookies cookies(options.cookies);
    return with_remote_cookies(cookies, [&]() -> nlohmann::json {
        auto command = ytdlp_command(options.ytdlp_binary, cookies.path());
        command.insert(command.end(), {"--dump-single-json", "--skip-download"});
        if (flat) command.insert(command.end(), {"--flat-playlist", "--ignore-errors"});
        else command.push_back("--no-playlist");
        command.insert(command.end(), {"--", python_filesystem_encode(url)});
        ProcessInterrupt::check();
        if (cancelled && cancelled()) throw std::runtime_error("Streaming metadata cancelled");
        const bool cancellable = static_cast<bool>(cancelled);
        const auto encoding = python_process_text_encoding();
        ChildPipe process(command, std::nullopt, cancellable, std::nullopt, false, cancellable, true);
        auto result = process.communicate(cancelled);
        auto output = decode_process_text(std::move(result.output), encoding);
        auto errors = decode_process_text(std::move(result.errors), encoding);
        if (result.returncode) throw std::runtime_error(utf8_tail(errors, 2500));
        return parse_python_json(output);
    });
}

struct ArchiveSnapshot {
    std::vector<std::pair<nlohmann::json, std::filesystem::path>> manifests;
    std::vector<nlohmann::json> completed;

    bool is_completed(const std::string& url) const {
        const nlohmann::json key(url);
        return std::any_of(completed.begin(), completed.end(), [&](const auto& saved) {
            return json_python_equal(saved, key);
        });
    }

    std::optional<std::filesystem::path> manifest_for(const std::string& url) const {
        const nlohmann::json key(url);
        for (const auto& saved : manifests)
            if (json_python_equal(saved.first, key)) return saved.second;
        return std::nullopt;
    }
};

std::vector<std::filesystem::path> snapshot_manifest_candidates(
    const std::filesystem::path& channel_root) {
    const auto root = python_path_normalize(channel_root);
    // pathlib's root is_dir check is outside its PermissionError handler.
    if (!python_path_is_directory(root)) return {};
    std::vector<std::filesystem::path> entries;
    try {
        ProcessInterrupt::check();
        for (const auto& entry : std::filesystem::directory_iterator(root)) {
            ProcessInterrupt::check();
            entries.push_back(python_path_normalize(entry.path()));
        }
    } catch (const std::system_error& error) {
        if (error.code() == std::errc::permission_denied ||
            error.code() == std::errc::operation_not_permitted) return {};
        throw;
    }
    // The wildcard selector closes its directory listing before yielding any
    // candidates. It includes directories/symlinks and does not stat matches.
    entries.erase(std::remove_if(entries.begin(), entries.end(), [](const auto& path) {
        const auto name = path.filename().string();
        return name.size() < 8 || name.compare(name.size() - 8, 8, ".uf.json") != 0;
    }), entries.end());
    return entries;
}

ArchiveSnapshot archive_snapshot(const std::filesystem::path& channel_root,
                                 const std::optional<nlohmann::json>& pipeline) {
    ArchiveSnapshot result;
    static const nlohmann::json empty_object = nlohmann::json::object();
    for (const auto& manifest : snapshot_manifest_candidates(channel_root)) {
        ProcessInterrupt::check();
        const auto document = read_python_json_file(
            manifest, python_process_text_encoding(), true);
        const auto& url = python_json_dict_get(python_json_dict_get(document, "source", empty_object), "url");
        if (python_json_dict_get(document, "format") != "dcvc-uf-archive" || !json_truthy(url)) continue;
        if (url.is_array() || url.is_object())
            throw std::invalid_argument(url.is_array() ? "unhashable type: 'list'" : "unhashable type: 'dict'");
        const auto duplicate = std::find_if(result.manifests.begin(), result.manifests.end(),
            [&](const auto& saved) { return json_python_equal(saved.first, url); });
        if (duplicate != result.manifests.end()) {
            const auto spelling = url.is_string() ? url.get<std::string>() : python_json_repr(url);
            throw std::invalid_argument("Multiple archives for " + spelling + ": " +
                duplicate->second.string() + " and " + manifest.string());
        }
        result.manifests.emplace_back(url, manifest);
        if (!pipeline || pipeline->is_null() || python_path_is_symlink(manifest)) continue;
        const auto& artifacts = python_json_dict_get(document, "artifacts", empty_object);
        if (!json_python_equal(python_json_dict_get(document, "pipeline"), *pipeline) ||
            !archive_completion_record(document) || !python_json_string_in("video.bin", artifacts)) continue;
        if (python_json_dict_get(*pipeline, "audio") == "opus" &&
            json_truthy(python_json_dict_get(python_json_dict_get(
                python_json_dict_get(document, "source", empty_object), "media", empty_object), "audio")) &&
            !python_json_string_in("audio.opus", artifacts)) continue;
        try {
            // This is Python's metadata snapshot, not load_bundle validation.
            // Preserve prior names even when a manifest cannot be resumed.
            const Archive archive{manifest, manifest.parent_path(), document};
            if (!artifacts.is_object()) json_compat_detail::missing_attribute(artifacts, "items");
            bool complete = true;
            for (auto record = artifacts.begin(); record != artifacts.end(); ++record) {
                ProcessInterrupt::check();
                const auto artifact = artifact_path(archive, record.key());
                if (python_path_is_symlink(artifact) || !python_path_is_file(artifact) ||
                    !json_python_equal(nlohmann::json(python_path_size(artifact)), record.value().at("size"))) {
                    complete = false;
                    break;
                }
            }
            if (complete) result.completed.push_back(url);
        } catch (const std::exception& error) {
            ProcessInterrupt::check();
            const bool recoverable = dynamic_cast<const std::system_error*>(&error) ||
                dynamic_cast<const std::invalid_argument*>(&error) ||
                dynamic_cast<const std::out_of_range*>(&error) ||
                dynamic_cast<const nlohmann::json::parse_error*>(&error) ||
                dynamic_cast<const nlohmann::json::type_error*>(&error) ||
                dynamic_cast<const nlohmann::json::out_of_range*>(&error);
            if (!recoverable) throw;
            continue;
        }
    }
    return result;
}

std::string thumbnail_extension(const std::string& url) {
    const auto scheme = url.find("://");
    const auto path_start = url.find('/', scheme + 3);
    const auto query = url.find_first_of("?#", path_start);
    const auto path = path_start == std::string::npos ? std::string{} :
        url.substr(path_start, query == std::string::npos ? std::string::npos : query - path_start);
    auto extension = std::filesystem::path(path).extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    if (extension != ".jpg" && extension != ".jpeg" &&
        extension != ".png" && extension != ".webp") return ".jpg";
    return extension;
}

// Request/HTTPConnection header policy follows installed Python 3.10 urllib
// and http.client; licenses/CPython-3.10.20.txt. libcurl is the transport.
std::vector<std::string> thumbnail_request_headers(const nlohmann::json& metadata) {
    const auto& supplied = python_json_dict_get(metadata, "http_headers");
    std::map<std::string, nlohmann::json> headers;
    if (json_truthy(supplied)) {
        if (!supplied.is_object())
            throw std::invalid_argument("Thumbnail HTTP headers must be a mapping");
        for (auto item = supplied.begin(); item != supplied.end(); ++item) {
            auto name = item.key();
            // The normal ASCII header-name path of Request.add_header uses
            // capitalize(), with case-insensitive duplicate replacement.
            for (auto& character : name) {
                const auto value = static_cast<unsigned char>(character);
                if (value >= 128)
                    throw std::invalid_argument("Cannot encode HTTP header name as ASCII");
                if (character >= 'A' && character <= 'Z') character += 'a' - 'A';
            }
            if (!name.empty() && name.front() >= 'a' && name.front() <= 'z')
                name.front() -= 'a' - 'A';
            headers[name] = item.value();
        }
    }
    // urllib's default opener adds its user agent; HTTPConnection supplies
    // identity encoding unless the request has its own Accept-Encoding.
    if (!headers.count("User-agent")) headers["User-agent"] = "Python-urllib/3.10";
    if (!headers.count("Accept-encoding")) headers["Accept-encoding"] = "identity";
    headers["Connection"] = "close";
    std::vector<std::string> lines;
    // urllib does not add libcurl's default Accept: */* header.
    if (!headers.count("Accept")) lines.push_back("Accept:");
    const auto whitespace = [](char value) {
        return value == ' ' || value == '\t' || value == '\r' || value == '\n' ||
               value == '\v' || value == '\f';
    };
    for (const auto& entry : headers) {
        auto name = entry.first;
        // do_open title-cases header names before HTTPConnection.putheader.
        bool previous_letter = false;
        for (auto& character : name) {
            const bool letter = (character >= 'a' && character <= 'z') ||
                                (character >= 'A' && character <= 'Z');
            if (letter) {
                if (!previous_letter && character >= 'a' && character <= 'z') character -= 'a' - 'A';
                if (previous_letter && character >= 'A' && character <= 'Z') character += 'a' - 'A';
            }
            previous_letter = letter;
        }
        if (name.empty() || whitespace(name.front()) || name.find_first_of(":\r\n") != std::string::npos)
            throw std::invalid_argument("Invalid HTTP header name");
        std::string value;
        if (entry.second.is_string()) value = python_latin1_encode(entry.second.get_ref<const std::string&>());
        else if (entry.second.is_boolean() || entry.second.is_number_integer() ||
                 entry.second.is_number_unsigned() || python_json_extended_integer(entry.second))
            value = python_json_repr(entry.second);
        else throw std::invalid_argument("Invalid HTTP header value type");
        for (std::size_t position = 0; position < value.size(); ++position) {
            const auto character = value[position];
            const auto next = position + 1 < value.size() ? value[position + 1] : '\0';
            if ((character == '\n' && next != ' ' && next != '\t') ||
                (character == '\r' && next != ' ' && next != '\t' && next != '\n'))
                throw std::invalid_argument("Invalid HTTP header value");
        }
        // libcurl's semicolon spelling sends an empty value rather than removing
        // a built-in header. Latin-1 wire bytes bypass filesystem re-encoding.
        lines.push_back(value.empty() ? name + ";" : name + ": " + value);
    }
    return lines;
}

struct ThumbnailCurlGlobal {
    ThumbnailCurlGlobal() {
        if (curl_global_init(CURL_GLOBAL_DEFAULT) != CURLE_OK)
            throw std::runtime_error("Cannot initialize thumbnail HTTP transport");
    }
    ~ThumbnailCurlGlobal() { curl_global_cleanup(); }
};

struct ThumbnailCurl {
    CURL* handle{curl_easy_init()};
    curl_slist* headers{};
    ThumbnailCurl() {
        if (!handle) throw std::bad_alloc();
    }
    ~ThumbnailCurl() { close(); }
    void close() noexcept {
        if (handle) curl_easy_cleanup(std::exchange(handle, nullptr));
        curl_slist_free_all(std::exchange(headers, nullptr));
    }
    ThumbnailCurl(const ThumbnailCurl&) = delete;
    ThumbnailCurl& operator=(const ThumbnailCurl&) = delete;
    template<class Value>
    void option(CURLoption key, Value value) {
        const auto result = curl_easy_setopt(handle, key, value);
        if (result != CURLE_OK)
            throw std::runtime_error(std::string("Cannot set thumbnail HTTP option: ") + curl_easy_strerror(result));
    }
    void set_headers(const std::vector<std::string>& lines) {
        curl_slist* replacement = nullptr;
        try {
            for (const auto& line : lines) {
                ProcessInterrupt::check();
                if (line.find('\0') != std::string::npos)
                    throw std::invalid_argument("Embedded null byte in thumbnail HTTP header");
                auto* next = curl_slist_append(replacement, line.c_str());
                if (!next) throw std::bad_alloc();
                replacement = next;
            }
            option(CURLOPT_HTTPHEADER, replacement);
        } catch (...) {
            curl_slist_free_all(replacement);
            throw;
        }
        curl_slist_free_all(headers);
        headers = replacement;
    }
};

struct ThumbnailResponse {
    std::filesystem::path destination;
    FileDescriptor output;
    std::exception_ptr error;
    long status{};
    bool ftp{}, header_complete{}, chunked{}, finished{}, header_parse_stopped{};
    unsigned header_lines{};
    std::string current, reason, pending_header_name, pending_header_value;
    std::map<std::string, unsigned> visited;
    std::optional<std::string> location, uri, redirect, transfer_encoding;
    explicit ThumbnailResponse(std::filesystem::path path) : destination(std::move(path)) {}
    void finish_header() {
        if (pending_header_name.empty()) return;
        // Compat32 strips only CR/LF at the end of the combined source value;
        // trailing spaces and embedded continuation newlines remain significant.
        while (!pending_header_value.empty() &&
               (pending_header_value.back() == '\r' || pending_header_value.back() == '\n'))
            pending_header_value.pop_back();
        if (pending_header_name == "location" && !location) location = pending_header_value;
        else if (pending_header_name == "uri" && !uri) uri = pending_header_value;
        else if (pending_header_name == "transfer-encoding" && !transfer_encoding) {
            transfer_encoding = pending_header_value;
            auto encoding = pending_header_value;
            for (auto& character : encoding)
                if (character >= 'A' && character <= 'Z') character += 'a' - 'A';
            // libcurl may decode a compound transfer encoding too. Do not
            // accept a framing failure as bounded non-chunked early EOF.
            chunked = encoding.find("chunked") != std::string::npos;
        }
        pending_header_name.clear();
        pending_header_value.clear();
    }
    void open_output() {
        if (output.get() >= 0) return;
        for (;;) {
            ProcessInterrupt::check();
            const int descriptor = ::open(destination.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0666);
            if (descriptor >= 0) { output.reset(descriptor); return; }
            const int failure = errno;
            if (failure != EINTR)
                throw std::system_error(failure, std::generic_category(), "Cannot create remote thumbnail stage");
        }
    }
};

std::string thumbnail_redirect_url(const std::string& current, std::string_view location);

bool thumbnail_header_space(unsigned char value) {
    return value == ' ' || (value >= '\t' && value <= '\r') ||
           (value >= 0x1c && value <= 0x1f) || value == 0x85 || value == 0xa0;
}

std::size_t thumbnail_callback_size(std::size_t size, std::size_t count) {
    if (size && count > std::numeric_limits<std::size_t>::max() / size)
        throw std::overflow_error("Excessive thumbnail HTTP callback size");
    return size * count;
}

std::size_t thumbnail_response_header(char* data, std::size_t size, std::size_t count, void* user) noexcept {
    auto& response = *static_cast<ThumbnailResponse*>(user);
    try {
        ProcessInterrupt::check();
        const auto length = thumbnail_callback_size(size, count);
        auto line = std::string_view(data, length);
        if (response.ftp || response.finished) return length;
        if (length > 65536) throw std::runtime_error("Excessive thumbnail HTTP header line");
        if (line.rfind("HTTP/", 0) == 0) {
            auto position = static_cast<std::size_t>(std::find_if(line.begin(), line.end(), thumbnail_header_space) - line.begin());
            while (position < line.size() && thumbnail_header_space(line[position])) ++position;
            if (position > line.size() || line.size() - position < 3 ||
                line[position] < '0' || line[position] > '9' ||
                line[position + 1] < '0' || line[position + 1] > '9' ||
                line[position + 2] < '0' || line[position + 2] > '9' ||
                (line.size() - position > 3 && !thumbnail_header_space(line[position + 3])))
                throw std::runtime_error("Invalid thumbnail HTTP status line");
            response.status = (line[position] - '0') * 100 +
                (line[position + 1] - '0') * 10 + line[position + 2] - '0';
            if (response.status < 100) throw std::runtime_error("Invalid thumbnail HTTP status code");
            response.reason = python_strip(python_latin1_decode(line.substr(position + 3)));
            response.location.reset();
            response.uri.reset();
            response.redirect.reset();
            response.header_complete = false;
            response.chunked = false;
            response.header_lines = 0;
            response.header_parse_stopped = false;
            response.pending_header_name.clear();
            response.pending_header_value.clear();
            response.transfer_encoding.reset();
        } else if (response.header_complete) {
            // libcurl also delivers trailers through this callback. Python's
            // chunk reader discards them without updating response headers.
            return length;
        } else if (++response.header_lines > 100) {
            throw std::runtime_error("Excessive thumbnail HTTP headers");
        } else if (line == "\r\n" || line == "\n") {
            response.finish_header();
            response.header_complete = true;
            if (response.status >= 200 && response.status < 300) response.open_output();
            else if (response.status == 301 || response.status == 302 || response.status == 303 || response.status == 307) {
                const auto& target = response.location ? response.location : response.uri;
                if (!target) throw std::runtime_error("HTTP Error " + std::to_string(response.status) + ": " + response.reason);
                response.redirect = thumbnail_redirect_url(response.current, *target);
                const auto found = response.visited.find(*response.redirect);
                if ((found != response.visited.end() && found->second >= 4) || response.visited.size() >= 10)
                    throw std::runtime_error("Thumbnail HTTP redirect loop");
                ++response.visited[*response.redirect];
            } else if (response.status != 100) {
                throw std::runtime_error("HTTP Error " + std::to_string(response.status) + ": " + response.reason);
            }
        } else if (!response.header_parse_stopped) {
            if (!line.empty() && (line.front() == ' ' || line.front() == '\t')) {
                if (!response.pending_header_name.empty()) response.pending_header_value.append(line);
                return length;
            }
            response.finish_header();
            if (line.rfind("From ", 0) == 0) return length;
            const auto colon = line.find(':');
            const auto name = line.substr(0, colon);
            if (colon == std::string_view::npos ||
                std::any_of(name.begin(), name.end(), [](unsigned char value) {
                    return value < 0x21 || value > 0x7e;
                })) {
                // FeedParser treats the first non-header line as body; the
                // remaining raw header lines no longer supply field values.
                response.header_parse_stopped = true;
            } else if (colon != 0) {
                response.pending_header_name.assign(name);
                for (auto& character : response.pending_header_name)
                    if (character >= 'A' && character <= 'Z') character += 'a' - 'A';
                auto value = line.substr(colon + 1);
                while (!value.empty() && (value.front() == ' ' || value.front() == '\t')) value.remove_prefix(1);
                response.pending_header_value.assign(value);
            }
        }
        return length;
    } catch (...) {
        if (!response.error) response.error = std::current_exception();
        return 0;
    }
}

std::size_t thumbnail_response_body(char* data, std::size_t size, std::size_t count, void* user) noexcept {
    auto& response = *static_cast<ThumbnailResponse*>(user);
    try {
        ProcessInterrupt::check();
        const auto length = thumbnail_callback_size(size, count);
        if (response.finished) return length;
        if (response.ftp) response.open_output();
        if (response.output.get() < 0) return length;  // Drain redirects/error bodies.
        std::size_t offset = 0;
        while (offset < length) {
            ProcessInterrupt::check();
            const auto request = std::min(length - offset,
                static_cast<std::size_t>(std::numeric_limits<ssize_t>::max()));
            const auto written = ::write(response.output.get(), data + offset, request);
            if (written < 0 && errno == EINTR) continue;
            if (written <= 0) throw std::system_error(written < 0 ? errno : EIO,
                std::generic_category(), "Cannot write remote thumbnail");
            offset += static_cast<std::size_t>(written);
        }
        return length;
    } catch (...) {
        if (!response.error) response.error = std::current_exception();
        return 0;
    }
}

int thumbnail_response_progress(void* user, curl_off_t, curl_off_t, curl_off_t, curl_off_t) noexcept {
    auto& response = *static_cast<ThumbnailResponse*>(user);
    try { ProcessInterrupt::check(); return 0; }
    catch (...) {
        if (!response.error) response.error = std::current_exception();
        return 1;
    }
}

// URL parsing/joining adapts installed urllib.parse for the three schemes
// allowed by HTTPRedirectHandler; licenses/CPython-3.10.20.txt. Percent escapes
// stay opaque and scheme names are normalized without changing authority case.
struct ThumbnailUrl {
    std::string scheme, authority, path, parameters, query, fragment;
};

bool thumbnail_url_scheme(std::string_view scheme) {
    return scheme.empty() || scheme == "http" || scheme == "https" || scheme == "ftp";
}

void thumbnail_url_check_authority(std::string_view authority) {
    const auto opening = authority.find('['), closing = authority.find(']');
    if ((opening == std::string_view::npos) != (closing == std::string_view::npos))
        throw std::invalid_argument("Invalid IPv6 URL");
    if (opening == std::string_view::npos) return;
    // urllib checks the host after the last user-info separator, including
    // rejecting brackets that occur only in user information.
    const auto at = authority.rfind('@');
    const auto host_port = authority.substr(at == std::string_view::npos ? 0 : at + 1);
    const auto bracket = host_port.find('[');
    std::string_view host;
    if (bracket != std::string_view::npos) {
        if (bracket != 0) throw std::invalid_argument("Invalid IPv6 URL");
        const auto end = host_port.find(']', 1);
        if (end == std::string_view::npos ||
            (end + 1 < host_port.size() && host_port[end + 1] != ':'))
            throw std::invalid_argument("Invalid IPv6 URL");
        host = host_port.substr(1, end - 1);
    } else {
        host = host_port.substr(0, host_port.find(':'));
    }
    if (!host.empty() && host.front() == 'v') {
        const auto dot = host.find('.');
        if (dot == std::string_view::npos || dot <= 1 || dot + 1 == host.size() ||
            std::any_of(host.begin() + 1, host.begin() + dot, [](unsigned char value) {
                return !((value >= '0' && value <= '9') ||
                         (value >= 'a' && value <= 'f') || (value >= 'A' && value <= 'F'));
            })) throw std::invalid_argument("IPvFuture address is invalid");
        return;
    }
    const auto scope = host.find('%');
    if (scope != std::string_view::npos) {
        if (scope + 1 == host.size() || host.find('%', scope + 1) != std::string_view::npos)
            throw std::invalid_argument("Invalid IPv6 address scope");
        host = host.substr(0, scope);
    }
    const std::string address(host);
    in6_addr packed{};
    if (address.find('\0') != std::string::npos ||
        ::inet_pton(AF_INET6, address.c_str(), &packed) != 1)
        throw std::invalid_argument("Invalid IPv6 URL");
}

ThumbnailUrl thumbnail_url_parse(std::string_view source, std::string scheme = {}) {
    while (!source.empty() && static_cast<unsigned char>(source.front()) <= 0x20) source.remove_prefix(1);
    std::string storage;
    storage.reserve(source.size());
    for (const char byte : source)
        if (byte != '\t' && byte != '\r' && byte != '\n') storage.push_back(byte);
    std::string_view rest(storage);
    const auto colon = rest.find(':');
    if (colon != std::string_view::npos && colon > 0 &&
        std::all_of(rest.begin(), rest.begin() + colon, [](unsigned char value) {
            return (value >= 'a' && value <= 'z') || (value >= 'A' && value <= 'Z') ||
                   (value >= '0' && value <= '9') || value == '+' || value == '-' || value == '.';
        })) {
        scheme.assign(rest.substr(0, colon));
        for (auto& byte : scheme) if (byte >= 'A' && byte <= 'Z') byte += 'a' - 'A';
        rest.remove_prefix(colon + 1);
    }
    ThumbnailUrl result;
    result.scheme = std::move(scheme);
    if (rest.rfind("//", 0) == 0) {
        rest.remove_prefix(2);
        const auto end = rest.find_first_of("/?#");
        result.authority.assign(rest.substr(0, end));
        thumbnail_url_check_authority(result.authority);
        rest = end == std::string_view::npos ? std::string_view{} : rest.substr(end);
    }
    const auto fragment = rest.find('#');
    if (fragment != std::string_view::npos) {
        result.fragment.assign(rest.substr(fragment + 1));
        rest = rest.substr(0, fragment);
    }
    const auto query = rest.find('?');
    if (query != std::string_view::npos) {
        result.query.assign(rest.substr(query + 1));
        rest = rest.substr(0, query);
    }
    if (thumbnail_url_scheme(result.scheme)) {
        const auto slash = rest.rfind('/');
        const auto parameters = rest.find(';', slash == std::string_view::npos ? 0 : slash);
        if (parameters != std::string_view::npos) {
            result.parameters.assign(rest.substr(parameters + 1));
            rest = rest.substr(0, parameters);
        }
    }
    result.path.assign(rest);
    return result;
}

std::string thumbnail_url_unparse(const ThumbnailUrl& url) {
    auto result = url.path;
    if (!url.parameters.empty()) result += ';' + url.parameters;
    if (!url.authority.empty() || (!url.scheme.empty() && thumbnail_url_scheme(url.scheme)) ||
        result.rfind("//", 0) == 0) {
        if (!result.empty() && result.front() != '/') result.insert(result.begin(), '/');
        result = "//" + url.authority + result;
    }
    if (!url.scheme.empty()) result = url.scheme + ':' + result;
    if (!url.query.empty()) result += '?' + url.query;
    if (!url.fragment.empty()) result += '#' + url.fragment;
    return result;
}

std::vector<std::string> thumbnail_url_path_parts(std::string_view path) {
    std::vector<std::string> parts;
    for (;;) {
        const auto slash = path.find('/');
        parts.emplace_back(path.substr(0, slash));
        if (slash == std::string_view::npos) return parts;
        path.remove_prefix(slash + 1);
    }
}

std::string thumbnail_url_join(const std::string& current, const std::string& reference) {
    if (current.empty()) return reference;
    if (reference.empty()) return current;
    const auto base = thumbnail_url_parse(current);
    auto target = thumbnail_url_parse(reference, base.scheme);
    if (target.scheme != base.scheme || !thumbnail_url_scheme(target.scheme)) return reference;
    if (!target.authority.empty()) return thumbnail_url_unparse(target);
    target.authority = base.authority;
    if (target.path.empty() && target.parameters.empty()) {
        target.path = base.path;
        target.parameters = base.parameters;
        if (target.query.empty()) target.query = base.query;
        return thumbnail_url_unparse(target);
    }
    std::vector<std::string> segments;
    if (!target.path.empty() && target.path.front() == '/') {
        segments = thumbnail_url_path_parts(target.path);
    } else {
        segments = thumbnail_url_path_parts(base.path);
        if (!segments.back().empty()) segments.pop_back();
        const auto suffix = thumbnail_url_path_parts(target.path);
        segments.insert(segments.end(), suffix.begin(), suffix.end());
        if (segments.size() > 2) {
            const auto end = std::remove(segments.begin() + 1, segments.end() - 1, std::string{});
            segments.erase(end, segments.end() - 1);
        }
    }
    std::vector<std::string> resolved;
    for (const auto& segment : segments) {
        ProcessInterrupt::check();
        if (segment == "..") { if (!resolved.empty()) resolved.pop_back(); }
        else if (segment != ".") resolved.push_back(segment);
    }
    if (segments.back() == "." || segments.back() == "..") resolved.emplace_back();
    target.path.clear();
    for (std::size_t position = 0; position < resolved.size(); ++position) {
        if (position) target.path.push_back('/');
        target.path += resolved[position];
    }
    if (target.path.empty()) target.path = "/";
    return thumbnail_url_unparse(target);
}

std::string thumbnail_redirect_url(const std::string& current, std::string_view location) {
    auto target = thumbnail_url_parse(location);
    if (!thumbnail_url_scheme(target.scheme))
        throw std::runtime_error("Unsupported thumbnail redirect scheme");
    if (target.path.empty() && !target.authority.empty()) target.path = "/";
    const auto normalized = thumbnail_url_unparse(target);
    std::string quoted;
    static constexpr std::string_view punctuation = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
    static constexpr char hex[] = "0123456789ABCDEF";
    for (const unsigned char byte : normalized) {
        if ((byte >= 'a' && byte <= 'z') || (byte >= 'A' && byte <= 'Z') ||
            (byte >= '0' && byte <= '9') || punctuation.find(static_cast<char>(byte)) != std::string_view::npos)
            quoted.push_back(static_cast<char>(byte));
        else { quoted.push_back('%'); quoted.push_back(hex[byte >> 4]); quoted.push_back(hex[byte & 15]); }
    }
    return thumbnail_url_join(current, quoted);
}

void download_thumbnail(const std::string& url, std::vector<std::string> headers,
                        const std::filesystem::path& destination) {
    static const ThumbnailCurlGlobal global;
    (void)global;
    ThumbnailResponse response(destination);  // Alive until curl cleanup completes.
    std::array<char, CURL_ERROR_SIZE> error{};
    ThumbnailCurl transfer;
    try {
        transfer.option(CURLOPT_ERRORBUFFER, error.data());
        transfer.option(CURLOPT_FOLLOWLOCATION, 0L);
        transfer.option(CURLOPT_PROTOCOLS_STR, "http,https,ftp");
        transfer.option(CURLOPT_HTTP_VERSION, static_cast<long>(CURL_HTTP_VERSION_1_1));
        transfer.option(CURLOPT_HTTP_CONTENT_DECODING, 0L);
        transfer.option(CURLOPT_SUPPRESS_CONNECT_HEADERS, 1L);
        transfer.option(CURLOPT_NOSIGNAL, 1L);
        transfer.option(CURLOPT_CONNECTTIMEOUT, 30L);
        transfer.option(CURLOPT_LOW_SPEED_LIMIT, 1L);
        transfer.option(CURLOPT_LOW_SPEED_TIME, 30L);
        transfer.option(CURLOPT_PATH_AS_IS, 1L);
        transfer.option(CURLOPT_NOPROGRESS, 0L);
        transfer.option(CURLOPT_HEADERFUNCTION, static_cast<curl_write_callback>(&thumbnail_response_header));
        transfer.option(CURLOPT_HEADERDATA, static_cast<void*>(&response));
        transfer.option(CURLOPT_WRITEFUNCTION, static_cast<curl_write_callback>(&thumbnail_response_body));
        transfer.option(CURLOPT_WRITEDATA, static_cast<void*>(&response));
        transfer.option(CURLOPT_XFERINFOFUNCTION, static_cast<curl_xferinfo_callback>(&thumbnail_response_progress));
        transfer.option(CURLOPT_XFERINFODATA, static_cast<void*>(&response));
        response.current = python_filesystem_encode(url);
        if (response.current.find('\0') != std::string::npos) throw std::invalid_argument("embedded null byte");
        for (;;) {
            ProcessInterrupt::check();
            response.status = 0;
            response.ftp = response.current.size() >= 4 &&
                (response.current[0] == 'f' || response.current[0] == 'F') &&
                (response.current[1] == 't' || response.current[1] == 'T') &&
                (response.current[2] == 'p' || response.current[2] == 'P') && response.current[3] == ':';
            response.header_complete = false;
            response.chunked = false;
            response.header_lines = 0;
            response.header_parse_stopped = false;
            response.pending_header_name.clear();
            response.pending_header_value.clear();
            response.transfer_encoding.reset();
            response.reason.clear();
            response.redirect.reset();
            response.location.reset();
            response.uri.reset();
            error.fill(0);
            transfer.set_headers(headers);
            transfer.option(CURLOPT_URL, response.current.c_str());
            const auto result = curl_easy_perform(transfer.handle);
            if (response.error) std::rethrow_exception(response.error);
            // copyfileobj uses bounded reads: HTTPResponse.read(amt) accepts
            // early EOF in non-chunked bodies rather than raising IncompleteRead.
            const bool bounded_eof = result == CURLE_PARTIAL_FILE && !response.ftp && !response.chunked &&
                response.header_complete && response.status >= 200 && response.status < 300;
            if (result != CURLE_OK && !bounded_eof) throw std::runtime_error(std::string("Thumbnail download failed: ") +
                (error[0] ? error.data() : curl_easy_strerror(result)));
            if (response.ftp || (response.status >= 200 && response.status < 300)) {
                response.open_output();
                response.finished = true;
                for (;;) {
                    ProcessInterrupt::check();
                    if (::fsync(response.output.get()) == 0) break;
                    const int failure = errno;
                    if (failure != EINTR)
                        throw std::system_error(failure, std::generic_category(), "Cannot sync remote thumbnail stage");
                }
                response.output.close_checked("Cannot close remote thumbnail stage");
                transfer.close();
                if (response.error) std::rethrow_exception(response.error);
                return;
            }
            if (!response.redirect) throw std::runtime_error("Thumbnail HTTP response has no final target");
            headers.erase(std::remove_if(headers.begin(), headers.end(), [](const std::string& line) {
                return line.rfind("Content-Length:", 0) == 0 || line == "Content-Length;" ||
                       line.rfind("Content-Type:", 0) == 0 || line == "Content-Type;";
            }), headers.end());
            response.current = std::move(*response.redirect);
        }
    } catch (...) {
        const auto original = std::current_exception();
        response.finished = true;
        response.output.close_checked("Cannot close remote thumbnail stage");
        transfer.close();
        if (response.error) std::rethrow_exception(response.error);
        std::rethrow_exception(original);
    }
}

class ThumbnailStage {
public:
    explicit ThumbnailStage(const std::filesystem::path& parent) {
        path_ = python_temporary_mkdtemp(".uf-thumbnail-", {}, parent);
    }
    ~ThumbnailStage() {
        if (!cleanup_attempted_ && !path_.empty()) {
            try { python_temporary_rmtree(path_, false, false); } catch (...) {}
        }
    }
    const std::filesystem::path& path() const { return path_; }
    void close() {
        if (cleanup_attempted_) return;
        cleanup_attempted_ = true;
        python_temporary_rmtree(path_, false, false);
        path_.clear();
    }
private:
    std::filesystem::path path_;
    bool cleanup_attempted_{};
};

}  // namespace

RemoteCookies::RemoteCookies(const std::optional<std::filesystem::path>& source) {
    if (!source || source->empty()) return;
    auto temporary = python_temporary_file("uf-ytdlp-cookies-");
    auto destination = std::move(temporary.descriptor);
    path_ = std::move(temporary.path);
    try {
        try {
            if (fchmod(destination.get(), 0660) != 0) {
                const int error = errno;
                throw std::system_error(error, std::generic_category(), "Cannot set yt-dlp cookie permissions");
            }
            FileDescriptor input;
            try {
                if (source->native().find('\0') != std::string::npos)
                    throw std::invalid_argument("embedded null byte");
                for (;;) {
                    ProcessInterrupt::check();
                    const int opened = open(source->c_str(), O_RDONLY | O_CLOEXEC);
                    if (opened >= 0) { input.reset(opened); break; }
                    const int error = errno;
                    if (error == EINTR) continue;
                    throw std::system_error(error, std::generic_category(), "Cannot open yt-dlp cookies");
                }
                std::array<char, 65536> buffer{};
                for (;;) {
                    ProcessInterrupt::check();
                    const auto count = read(input.get(), buffer.data(), buffer.size());
                    if (count < 0) {
                        const int error = errno;
                        if (error == EINTR) continue;
                        throw std::system_error(error, std::generic_category(), "Cannot read yt-dlp cookies");
                    }
                    if (count == 0) break;
                    std::size_t offset = 0;
                    while (offset < static_cast<std::size_t>(count)) {
                        ProcessInterrupt::check();
                        const auto written = write(destination.get(), buffer.data() + offset,
                                                   static_cast<std::size_t>(count) - offset);
                        if (written < 0) {
                            const int error = errno;
                            if (error == EINTR) continue;
                            throw std::system_error(error, std::generic_category(), "Cannot copy yt-dlp cookies");
                        }
                        if (written == 0)
                            throw std::system_error(EIO, std::generic_category(), "Cannot copy yt-dlp cookies");
                        offset += static_cast<std::size_t>(written);
                    }
                }
                input.close_checked("Cannot close yt-dlp source cookies");
            } catch (...) {
                input.close_checked("Cannot close yt-dlp source cookies");
                throw;
            }
            destination.close_checked("Cannot close private yt-dlp cookies");
        } catch (...) {
            destination.close_checked("Cannot close private yt-dlp cookies");
            throw;
        }
    } catch (...) {
        close();
        throw;
    }
}

RemoteCookies::~RemoteCookies() {
    if (!cleanup_attempted_) try { close(); } catch (...) {}
}

void RemoteCookies::close() {
    if (cleanup_attempted_) return;
    cleanup_attempted_ = true;
    if (!path_) return;
    if (unlink(path_->c_str()) != 0) {
        const int error = errno;
        if (error != ENOENT)
            throw std::system_error(error, std::generic_category(), "Cannot remove private yt-dlp cookies");
    }
}

std::vector<std::string> ytdlp_command(
    const std::string& binary, const std::optional<std::filesystem::path>& cookies) {
    std::vector<std::string> command{binary};
    auto deno = find_executable("deno");
    if (!deno) deno = pathlib_path((home_directory() / ".deno/bin/deno").string());
    if (pathlib_is_file(*deno)) {
        command.insert(command.end(), {"--js-runtimes", "deno:" + deno->string()});
    }
    command.insert(command.end(), {"--ignore-config", "--remote-components", "ejs:github",
        "--no-progress", "--retries", "5", "--fragment-retries", "5"});
    if (cookies) command.insert(command.end(), {"--cookies", cookies->string()});
    return command;
}

std::optional<std::filesystem::path> store_remote_thumbnail(const RemoteJob& job) {
    if (!job.write_thumbnail) return std::nullopt;
    const auto url = string(job.public_metadata, "thumbnail");
    if (url.rfind("https://", 0) != 0 && url.rfind("http://", 0) != 0) {
        return std::nullopt;
    }
    const auto name = job.manifest.filename().string();
    if (name.size() < 8 || name.substr(name.size() - 8) != ".uf.json") {
        throw std::runtime_error("Invalid remote manifest for thumbnail");
    }
    const auto target = job.manifest.parent_path() /
        (name.substr(0, name.size() - 8) + thumbnail_extension(url));
    if (python_path_exists(target)) {
        if (python_path_is_symlink(target)) {
            throw std::runtime_error("Unsafe existing remote thumbnail");
        }
        return target;
    }
    ThumbnailStage temporary(target.parent_path());
    try {
        const auto staged = temporary.path() / target.filename();
        const auto headers = thumbnail_request_headers(job.public_metadata);
        download_thumbnail(url, headers, staged);
        verify_image(staged);
        try {
            std::filesystem::create_hard_link(staged, target);
        } catch (const std::filesystem::filesystem_error& error) {
            if (error.code() != std::errc::file_exists) throw;
        }
        temporary.close();
        return target;
    } catch (...) {
        temporary.close();
        throw;
    }
}

std::vector<std::string> remote_source_urls(const RemoteOptions& options) {
    std::vector<std::string> urls;
    urls.reserve(options.source_urls.size());
    for (const auto& url : options.source_urls) urls.push_back(python_filesystem_decode(url));
    for (const auto& channel : options.youtube_channels) {
        auto extension = channel.size() < 4 ? std::string{} : channel.substr(channel.size() - 4);
        std::transform(extension.begin(), extension.end(), extension.begin(),
                       [](unsigned char c) { return std::tolower(c); });
        if (extension == ".txt") {
            const auto contents = read_python_text_file(channel, python_process_text_encoding());
            for (const auto& line : python_splitlines(contents)) {
                const auto value = python_strip(line);
                if (value.empty() || value.front() == '#') continue;
                urls.push_back("https://www.youtube.com/watch?v=" +
                    safe_id(value, false));
            }
        } else {
            urls.push_back("https://www.youtube.com/channel/" +
                safe_id(python_filesystem_decode(channel), true) + "/videos");
        }
    }
    for (const auto& channel : options.twitch_channels) {
        urls.push_back("https://www.twitch.tv/" + safe_component(python_filesystem_decode(channel)) +
            "/videos?filter=archives&sort=time");
    }
    return urls;
}

std::string resolve_ytdlp_binary(const std::string& binary) {
    const auto resolved = find_executable(binary);
    if (!resolved) {
        throw std::invalid_argument("yt-dlp not found; pass --yt-dlp /path/to/yt-dlp");
    }
    return resolved->string();
}

RemoteDiscovery discover_remote_jobs(
    const RemoteOptions& settings,
    const std::function<void(RemoteJob&&)>& consume,
    const std::function<bool()>& cancelled) {
    auto options = settings;
    options.ytdlp_binary = resolve_ytdlp_binary(options.ytdlp_binary);
    if (options.max_height < 0 || (options.max_videos && *options.max_videos < 1)) {
        throw std::invalid_argument("Invalid remote source height or max videos");
    }
    const auto urls = remote_source_urls(options);
    if (urls.empty()) throw std::invalid_argument("No remote source URLs or channels");
    std::set<std::string> seen;
    const auto output_root = std::filesystem::weakly_canonical(std::filesystem::absolute(options.output_root));
    std::map<std::string, ArchiveSnapshot> snapshots;
    const auto snapshot = [&](const std::string& channel) -> const ArchiveSnapshot& {
        auto found = snapshots.find(channel);
        if (found == snapshots.end()) {
            found = snapshots.emplace(channel, archive_snapshot(
                output_root / channel, options.pipeline)).first;
        }
        return found->second;
    };
    RemoteDiscovery result;
    int scheduled = 0;
    for (const auto& url : urls) {
        if (cancelled && cancelled()) return result;
        const auto listing = metadata(url, options, true, cancelled);
        std::vector<std::pair<nlohmann::json, std::string>> pending;
        const auto type = string(listing, "_type");
        if ((type == "playlist" || type == "multi_video") &&
            listing.contains("entries") && listing.at("entries").is_array()) {
            for (const auto& entry : listing.at("entries")) {
                pending.emplace_back(entry, string(listing, "channel_id"));
            }
        } else pending.emplace_back(listing, string(listing, "channel_id"));
        for (std::size_t cursor = 0; cursor < pending.size(); ++cursor) {
            if (cancelled && cancelled()) return result;
            const auto [entry, hint] = pending[cursor];
            if (!entry.is_object()) continue;
            const auto entry_type = string(entry, "_type");
            if (entry_type == "playlist" || entry_type == "multi_video") {
                const auto sub = entry.contains("entries") && entry.at("entries").is_array()
                    ? entry : metadata(string(entry, "url"), options, true, cancelled);
                if (sub.contains("entries") && sub.at("entries").is_array()) {
                    for (const auto& item : sub.at("entries")) {
                        pending.emplace_back(item, string(sub, "channel_id").empty()
                            ? hint : string(sub, "channel_id"));
                    }
                }
                continue;
            }
            try {
            auto lower_url = url;
            std::transform(lower_url.begin(), lower_url.end(), lower_url.begin(),
                           [](unsigned char c) { return std::tolower(c); });
            const bool youtube = lower_url.find("twitch.tv") == std::string::npos;
            const auto id = youtube ? safe_id(python_json_dict_get(entry, "id"), false)
                                    : safe_component(python_json_dict_get(entry, "id"));
            if (!seen.insert(id).second) continue;
            const auto first_digit = id.find_first_not_of('v');
            const auto twitch_id = first_digit == std::string::npos
                ? std::string{} : id.substr(first_digit);
            const auto video_url = youtube ? "https://www.youtube.com/watch?v=" + id
                : (!string(entry, "webpage_url").empty() ? string(entry, "webpage_url")
                   : (!string(entry, "url").empty() ? string(entry, "url")
                      : "https://www.twitch.tv/videos/" + twitch_id));
            const auto channel_hint = string(entry, "channel_id").empty()
                ? hint : string(entry, "channel_id");
            if (!channel_hint.empty()) {
                const auto channel = youtube ? safe_id(channel_hint, true)
                                             : safe_component(channel_hint);
                if (snapshot(channel).is_completed(video_url)) {
                    ++result.resumed;
                    continue;
                }
            }
            if (options.max_videos && scheduled >= *options.max_videos) return result;
            ++scheduled;
            event("run_discovered", {{"total", static_cast<std::size_t>(scheduled) + pending.size() - cursor - 1}});
            event("stream_resolving", {{"video_id", id}});
            const auto info = metadata(video_url, options, false, cancelled);
            if (json_truthy(python_json_dict_get(info, "is_live")) ||
                python_json_dict_get(info, "live_status") == "is_upcoming") {
                ++result.skipped_live;
                event("stream_skipped_live", {{"video_id", id}});
                continue;
            }
            const auto& channel_id = python_json_dict_get(info, "channel_id");
            const auto channel = youtube ? safe_id(channel_id, true)
                : safe_component(json_truthy(channel_id)
                    ? channel_id : python_json_dict_get(info, "uploader_id"));
            const auto [video, audio] = select_formats(info, options.max_height);
            const auto width = python_media_dimension(video.at("width"));
            const auto height = python_media_dimension(video.at("height"));
            const auto& video_fps = python_json_dict_get(video, "fps");
            const auto& info_fps = python_json_dict_get(info, "fps");
            const nlohmann::json default_fps(30);
            const auto fps = python_json_as_double(json_truthy(video_fps) ? video_fps
                : (json_truthy(info_fps) ? info_fps : default_fps));
            const auto duration = python_json_as_double(truth_field(info, "duration"));
            const auto& video_format_id = video.at("format_id");
            const auto command_format = video_format_id.is_string() ? video_format_id.get<std::string>()
                : python_json_repr(video_format_id);
            std::optional<std::string> command_audio;
            if (audio) {
                const auto& audio_format_id = audio->at("format_id");
                command_audio = audio_format_id.is_string() ? audio_format_id.get<std::string>()
                    : python_json_repr(audio_format_id);
            }
            const auto& previous = snapshot(channel);
            if (previous.is_completed(video_url)) {
                ++result.resumed;
                --scheduled;
                continue;
            }
            RemoteJob job;
            job.video_id = id;
            job.channel_id = channel;
            job.url = video_url;
            job.video_format = command_format;
            job.audio_format = std::move(command_audio);
            job.media = {width, height, fps, duration, audio.has_value()};
            job.public_metadata = info;
            const auto existing = previous.manifest_for(video_url);
            job.manifest = existing ? *existing
                : output_root / channel /
                    (id + "_" + upload_timestamp(info) + ".uf.json");
            job.ytdlp_binary = options.ytdlp_binary;
            job.cookies = options.cookies;
            job.write_thumbnail = options.write_thumbnail;
            job.thumbnail_codec = options.thumbnail_codec;
            job.thumbnail_qp = options.thumbnail_qp;
            event("stream_selected", {{"video_id", id}, {"channel_id", channel},
                {"video_format", video_format_id},
                {"audio_format", audio ? audio->at("format_id") : nlohmann::json(nullptr)},
                {"audio_language", audio ? python_json_dict_get(*audio, "language") : nlohmann::json(nullptr)},
                {"audio_note", audio ? python_json_dict_get(*audio, "format_note") : nlohmann::json(nullptr)},
                {"archive", python_path_text(job.manifest)}});
            if (consume) consume(std::move(job));
            else result.jobs.push_back(std::move(job));
            } catch (const std::exception& error) {
                if (cancelled && cancelled()) return result;
                ++result.failed;
                event("stream_failed", {{"video_id", python_json_dict_get(entry, "id")}, {"error", error.what()}});
            }
        }
    }
    event("run_discovered", {{"total", scheduled}});
    return result;
}

struct RemoteAheadJobs::Impl {
    struct Item {
        std::optional<RemoteJob> job;
        bool finished{};
        std::exception_ptr error;
    };
    RemoteOptions options;
    std::atomic<bool> stop{false};
    std::mutex mutex;
    std::condition_variable space;
    std::deque<Item> queue;
    RemoteDiscovery result;
    bool consumed_end{}, joined{};
    std::thread producer;

    explicit Impl(RemoteOptions settings) : options(std::move(settings)) {
        producer = std::thread([this] {
            try {
                result = discover_remote_jobs(options,
                    [this](RemoteJob&& job) { put({std::move(job), false, {}}); },
                    [this] { return stop.load() || ProcessInterrupt::requested(); });
                put({std::nullopt, true, {}});
            } catch (...) {
                put({std::nullopt, true, std::current_exception()});
            }
        });
    }

    void put(Item item) {
        std::unique_lock<std::mutex> lock(mutex);
        space.wait(lock, [this] { return stop.load() || queue.empty(); });
        if (!stop.load()) queue.push_back(std::move(item));
    }

    ~Impl() {
        stop.store(true);
        space.notify_all();
        if (producer.joinable()) producer.join();
    }
};

RemoteAheadJobs::RemoteAheadJobs(RemoteOptions options)
    : impl_(std::make_unique<Impl>(std::move(options))) {}
RemoteAheadJobs::~RemoteAheadJobs() = default;

RemoteAheadItem RemoteAheadJobs::poll() {
    if (impl_->consumed_end) return {std::nullopt, true};
    std::unique_lock<std::mutex> lock(impl_->mutex);
    if (impl_->queue.empty()) return {};
    auto item = std::move(impl_->queue.front());
    impl_->queue.pop_front();
    impl_->consumed_end = item.finished;
    impl_->space.notify_one();
    lock.unlock();
    if (item.error) std::rethrow_exception(item.error);
    return {std::move(item.job), item.finished};
}

RemoteDiscovery RemoteAheadJobs::finish() {
    if (!impl_->consumed_end || impl_->joined) {
        throw std::logic_error("UF metadata producer must finish once after its final item");
    }
    if (impl_->producer.joinable()) impl_->producer.join();
    impl_->joined = true;
    return std::move(impl_->result);
}

}  // namespace uf
