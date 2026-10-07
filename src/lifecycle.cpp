#include "uf/lifecycle.hpp"
#include "uf/archive.hpp"
#include "uf/image_archive.hpp"
#include "uf/json_compat.hpp"
#include "uf/python_path.hpp"
#include "uf/python_json.hpp"
#include "uf/process.hpp"
#include "uf/utf8.hpp"

#include <algorithm>
#include <cctype>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <system_error>
#include <utility>
#include <vector>

namespace uf {
namespace {

bool hidden_relative(const std::filesystem::path& path) {
    for (const auto& part : path) {
        const auto name = part.string();
        if (!name.empty() && name[0] == '.') return true;
    }
    return false;
}

bool recoverable_inventory_error(const std::exception& error) {
    // Maintained inventory handles OSError, ValueError, KeyError and TypeError.
    // Resource failures and unexpected runtime errors must propagate, including
    // the typed AttributeError from Python-style dictionary/string methods.
    // Remaining subscript access still uses native parser exception classes.
    return dynamic_cast<const std::system_error*>(&error) ||
        dynamic_cast<const std::invalid_argument*>(&error) ||
        dynamic_cast<const std::out_of_range*>(&error) ||
        dynamic_cast<const nlohmann::json::parse_error*>(&error) ||
        dynamic_cast<const nlohmann::json::type_error*>(&error) ||
        dynamic_cast<const nlohmann::json::out_of_range*>(&error);
}

bool within(const std::filesystem::path& path,
            const std::filesystem::path& root) {
    return python_path_is_relative_to(path, root);
}

// CPython 3.10 pathlib selectors snapshot a directory for matching, then
// independently snapshot it again before descending. Matching has no file
// predicate; only recursion excludes symlink directories. See the retained
// CPython license and the path utilities' source notice.
enum class InventoryPattern { flat_manifest, bundle_manifest, image };

bool permission_error(const std::system_error& error) {
    return error.code() == std::errc::permission_denied ||
           error.code() == std::errc::operation_not_permitted;
}

std::optional<std::vector<std::filesystem::directory_entry>> directory_snapshot(
    const std::filesystem::path& directory) {
    try {
        ProcessInterrupt::check();
        std::vector<std::filesystem::directory_entry> entries;
        for (const auto& entry : std::filesystem::directory_iterator(directory)) {
            ProcessInterrupt::check();
            entries.push_back(entry);
        }
        return entries;
    } catch (const std::system_error& error) {
        if (permission_error(error)) return std::nullopt;
        throw;
    }
}

bool entry_is_directory(const std::filesystem::directory_entry& entry) {
    try {
        return entry.is_directory();
    } catch (const std::system_error& error) {
        if (error.code() == std::errc::no_such_file_or_directory ||
            error.code() == std::errc::not_a_directory ||
            error.code() == std::errc::bad_file_descriptor ||
            error.code() == std::errc::too_many_symbolic_link_levels) return false;
        throw;
    }
}

void visit_inventory_directory(const std::filesystem::path& directory,
    InventoryPattern pattern,
    const std::function<void(const std::filesystem::path&)>& visit) {
    ProcessInterrupt::check();
    if (pattern == InventoryPattern::bundle_manifest) {
        const auto candidate = python_path_normalize(directory / "manifest.json");
        bool exists = false;
        try {
            exists = python_path_exists(candidate);
        } catch (const std::system_error& error) {
            if (!permission_error(error)) throw;
        }
        if (exists) visit(candidate);
    } else if (const auto entries = directory_snapshot(directory)) {
        const std::string suffix = pattern == InventoryPattern::flat_manifest
            ? ".uf.json" : ".dcvci";
        for (const auto& entry : *entries) {
            ProcessInterrupt::check();
            const auto name = entry.path().filename().string();
            if (name.size() >= suffix.size() &&
                name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0)
                visit(python_path_normalize(entry.path()));
        }
    }
    const auto entries = directory_snapshot(directory);
    if (!entries) return;
    for (const auto& entry : *entries) {
        ProcessInterrupt::check();
        bool descend = false;
        try {
            descend = entry_is_directory(entry) && !entry.is_symlink();
        } catch (const std::system_error& error) {
            // PermissionError terminates this directory's recursive selector;
            // it does not erase candidates already emitted from that directory.
            if (permission_error(error)) return;
            throw;
        }
        if (descend)
            visit_inventory_directory(python_path_normalize(entry.path()), pattern, visit);
    }
}

void visit_inventory_matches(const std::filesystem::path& root,
    InventoryPattern pattern,
    const std::function<void(const std::filesystem::path&)>& visit) {
    if (!python_path_is_directory(root)) return;
    std::set<std::filesystem::path> yielded;
    visit_inventory_directory(root, pattern, [&](const auto& candidate) {
        if (yielded.insert(candidate).second) visit(candidate);
    });
}

bool in_source_scope(const std::filesystem::path& source,
    const std::optional<std::set<std::filesystem::path>>& sources) {
    if (!sources) return true;
    const auto absolute = python_path_absolute(source).native();
    // Python's filter contains absolute strings, retaining parent components
    // and anchor spelling rather than comparing canonical filesystem paths.
    return std::any_of(sources->begin(), sources->end(), [&](const auto& item) {
        return item.native() == absolute;
    });
}

std::string lower_extension(const std::filesystem::path& path) {
    auto extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return extension;
}

std::filesystem::path remap_source(const nlohmann::json& data,
                                   const std::optional<std::filesystem::path>& base_root) {
    const auto& source_data = data.at("source");
    const auto original = python_path_normalize(python_path_from_text(source_data.at("path").get_ref<const std::string&>()));
    if (!base_root || base_root->empty()) return original;
    const auto& relative_value = python_json_dict_get(source_data, "relative_path");
    if (!json_truthy(relative_value)) return original;
    const auto base = python_path_resolve(*base_root);
    const auto relative = python_path_from_text(relative_value.get_ref<const std::string&>());
    const auto source = python_path_normalize(base / relative);
    if (!within(python_path_resolve(source), base)) {
        throw std::invalid_argument("Original path escapes base_root");
    }
    return source;
}

bool has_metadata(const std::filesystem::path& source, const nlohmann::json& data) {
    auto sidecar = source;
    sidecar.replace_extension(".info.json");
    if (python_path_is_file(sidecar)) return true;
    static const nlohmann::json empty_artifacts = nlohmann::json::object();
    const auto& artifacts = python_json_dict_get(data, "artifacts", empty_artifacts);
    if (artifacts.is_object()) {
        for (auto it = artifacts.begin(); it != artifacts.end(); ++it) {
            const auto& name = it.key();
            if (name.size() >= 10 && name.compare(name.size() - 10, 10, ".info.json") == 0)
                return true;
        }
    } else if (artifacts.is_array()) {
        for (const auto& item : artifacts)
            if (python_json_endswith(item, ".info.json")) return true;
    } else if (!artifacts.is_string()) {
        throw std::invalid_argument("'" +
            std::string(json_compat_detail::python_type_name(artifacts)) + "' object is not iterable");
    }
    // Iterating a string yields single characters; none ends with .info.json.
    return false;
}

void channel_blockers(ChannelInventory& row, bool allow_missing_metadata) {
    std::set<std::filesystem::path> directories;
    std::set<std::string> covered;
    std::size_t missing_metadata = 0;
    for (const auto& item : row.archives) {
        if (item.remote || item.kind != "video") continue;
        directories.insert(python_path_normalize(item.source.parent_path()));
        covered.insert(item.source.stem().string());
        if (!allow_missing_metadata && item.exists && !item.metadata) {
            ++missing_metadata;
        }
    }
    if (missing_metadata) {
        row.blockers.push_back("Missing info.json for " + std::to_string(missing_metadata) + " original(s)");
    }
    static const std::set<std::string> videos = {
        ".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v", ".ts"};
    for (const auto& directory : directories) {
        ProcessInterrupt::check();
        if (!python_path_is_directory(directory)) continue;
        for (const auto& entry : std::filesystem::directory_iterator(directory)) {
            ProcessInterrupt::check();
            const auto name = entry.path().filename().string();
            if (name.empty() || name[0] == '.') continue;
            if (videos.count(lower_extension(entry.path())) &&
                !covered.count(entry.path().stem().string())) {
                row.blockers.push_back("Unprocessed source: " + name);
            } else if (name.size() >= 10 &&
                       name.compare(name.size() - 10, 10, ".info.json") == 0 &&
                       (name.size() < 13 ||
                        name.compare(name.size() - 13, 13, ".uf.info.json") != 0) &&
                       !covered.count(name.substr(0, name.size() - 10))) {
                row.blockers.push_back("Missing video archive for " + name);
            }
        }
    }
    if (row.blockers.size() > 10) row.blockers.resize(10);
}

}  // namespace

InventoryResult inventory_channels(const std::filesystem::path& output_root,
                                    const std::vector<std::string>& channel_ids,
                                    std::optional<std::filesystem::path> base_root,
                                    bool allow_missing_metadata,
                                    const std::optional<std::set<std::filesystem::path>>& sources) {
    const auto output = python_path_resolve(output_root);
    std::vector<std::filesystem::path> roots;
    if (channel_ids.empty()) roots.push_back(output);
    else {
        for (const auto& id : channel_ids) {
            const auto name = std::filesystem::path(id);
            const auto root = python_path_resolve(output / name);
            if (!within(root, output)) {
                throw std::invalid_argument("Channel path escapes output_root");
            }
            roots.push_back(root);
        }
    }
    InventoryResult result;
    std::map<std::string, std::size_t> channel_indices;
    const auto channel_row = [&](const std::filesystem::path& channel) -> ChannelInventory& {
        const auto key = channel.string();
        const auto found = channel_indices.find(key);
        if (found != channel_indices.end()) return result.channels[found->second];
        const auto index = result.channels.size();
        ChannelInventory row;
        row.channel = channel;
        result.channels.push_back(std::move(row));
        try {
            channel_indices.emplace(key, index);
        } catch (...) {
            result.channels.pop_back();
            throw;
        }
        return result.channels.back();
    };
    // Python's seen tuples retain scalar values/types. Numeric equality follows
    // Python (e.g. bool/int), while null and an empty digest remain distinct.
    std::map<std::string, std::vector<std::pair<nlohmann::json, nlohmann::json>>> seen;
    for (const auto& root : roots) {
        if (!python_path_exists(root)) continue;
        std::vector<std::filesystem::path> candidates;
        const auto collect = [&](const auto& path) { candidates.push_back(path); };
        visit_inventory_matches(root, InventoryPattern::flat_manifest, collect);
        visit_inventory_matches(root, InventoryPattern::bundle_manifest, collect);
        for (const auto& path : candidates) {
            if (hidden_relative(python_path_relative_to(path, root))) continue;
            try {
                const auto preview = read_python_json_file(path, python_process_text_encoding(), true);
                if (python_json_dict_get(preview, "format") != "dcvc-uf-archive") continue;
                // Python's inventory reads the manifest without loading or
                // validating its artifacts; cleanup performs full validation.
                const auto& data = preview;
                const bool flat = python_json_dict_get(data, "layout") == "flat";
                const auto channel = flat ? path.parent_path() : path.parent_path().parent_path();
                const auto& source_data = data.at("source");
                const auto& url = python_json_dict_get(source_data, "url");
                const auto source_key = json_truthy(url) ? url : source_data.at("path");
                const auto source_sha = python_json_dict_get(source_data, "sha256");
                for (const auto* value : {&source_key, &source_sha}) {
                    if (value->is_array() || value->is_object())
                        throw std::invalid_argument(value->is_array()
                            ? "unhashable type: 'list'" : "unhashable type: 'dict'");
                }
                auto& seen_channel = seen[channel.string()];
                if (std::any_of(seen_channel.begin(), seen_channel.end(), [&](const auto& key) {
                    return json_python_equal(key.first, source_key) &&
                           json_python_equal(key.second, source_sha);
                })) continue;
                seen_channel.emplace_back(source_key, source_sha);
                const auto source = remap_source(data, base_root);
                const bool remote = python_json_dict_get(source_data, "kind") == "youtube";
                bool eligible = false;
                if (in_source_scope(source, sources) && !remote) {
                    const auto& pipeline = data.at("pipeline");
                    eligible = python_json_dict_get(pipeline, "max_frames").is_null() &&
                        (!json_truthy(source_data.at("media").at("audio")) ||
                         python_json_string_in("audio.opus", data.at("artifacts"))) &&
                        archive_completion_record(data);
                }
                auto& row = channel_row(channel);
                const bool exists = !remote && python_path_exists(source);
                const bool metadata = has_metadata(source, data);
                row.archives.push_back({path, source, "video", eligible, remote, exists, metadata});
                if (remote) ++row.streamed;
                else if (!exists) ++row.absent;
                else if (eligible) {
                    ++row.originals;
                    row.bytes += static_cast<std::uint64_t>(python_path_size(source));
                } else ++row.ineligible;
            } catch (const std::exception& error) {
                ProcessInterrupt::check();
                if (!recoverable_inventory_error(error)) throw;
                result.errors.push_back({path, error.what()});
            }
        }
    }
    // Images are a separate pass over every selected root. Python deliberately
    // retains repeats from overlapping or repeated roots instead of deduping.
    for (const auto& root : roots) {
        if (!python_path_exists(root)) continue;
        visit_inventory_matches(root, InventoryPattern::image, [&](const auto& path) {
            if (hidden_relative(python_path_relative_to(path, root))) return;
            try {
                const auto image = read_intra_image(path, false);
                const auto source = remap_source(image.metadata, base_root);
                const bool eligible = in_source_scope(source, sources);
                auto& row = channel_row(path.parent_path());
                const bool exists = python_path_exists(source);
                row.archives.push_back({path, source, "image", eligible, false, exists, true});
                if (!exists) ++row.absent;
                else if (eligible) {
                    ++row.originals;
                    row.bytes += static_cast<std::uint64_t>(python_path_size(source));
                } else ++row.ineligible;
            } catch (const std::exception& error) {
                ProcessInterrupt::check();
                if (!recoverable_inventory_error(error)) throw;
                result.errors.push_back({path, error.what()});
            }
        });
    }
    for (auto& row : result.channels) channel_blockers(row, allow_missing_metadata);
    return result;
}

}  // namespace uf
