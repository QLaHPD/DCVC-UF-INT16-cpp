// Visibility/count policy adapted from the installed PyTorch CUDA helpers.
// See licenses/PyTorch.txt and PROVENANCE.md for provenance.
#include "evaluation_cuda_policy.hpp"
#include "uf/python_integer.hpp"
#include "uf/python_numeric.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <dlfcn.h>

namespace uf::evaluation {
namespace {

enum class MaskKind { ordinal, uuid, mig };
struct Visibility {
    MaskKind kind = MaskKind::ordinal;
    // Ordinals use canonical unbounded decimal strings; UUIDs preserve bytes.
    std::vector<std::string> entries;
};

Visibility visibility() {
    Visibility result;
    const auto* environment = std::getenv("CUDA_VISIBLE_DEVICES");
    if (!environment) {
        for (int index = 0; index < 64; ++index) result.entries.push_back(std::to_string(index));
        return result;
    }
    const std::string_view mask(environment);
    const std::string_view prefix = mask.substr(0, 4) == "GPU-" ? "GPU-"
        : mask.substr(0, 4) == "MIG-" ? "MIG-" : "";
    if (!prefix.empty()) result.kind = prefix == "GPU-" ? MaskKind::uuid : MaskKind::mig;
    for (std::size_t start = 0; start <= mask.size();) {
        const auto comma = mask.find(',', start);
        const auto entry = mask.substr(start, comma == std::string_view::npos ? mask.size() - start : comma - start);
        std::string value;
        if (prefix.empty()) {
            const auto digits = python_signed_digit_prefix(entry);
            value = digits.empty() ? "-1" : PythonInteger(digits).decimal();
        } else value = std::string(entry);
        // Duplicates are checked before the stop condition in Torch's parser.
        if (std::find(result.entries.begin(), result.entries.end(), value) != result.entries.end()) {
            result.entries.clear();
            return result;
        }
        if (prefix.empty() ? value.front() == '-' : entry.substr(0, prefix.size()) != prefix) break;
        result.entries.push_back(std::move(value));
        if (comma == std::string_view::npos) break;
        start = comma + 1;
    }
    return result;
}

class NvmlLibrary {
public:
    NvmlLibrary() : handle_(dlopen("libnvidia-ml.so.1", RTLD_LAZY | RTLD_LOCAL)) {}
    ~NvmlLibrary() {
        if (initialized_) shutdown_();
        if (handle_) dlclose(handle_);
    }
    NvmlLibrary(const NvmlLibrary&) = delete;
    NvmlLibrary& operator=(const NvmlLibrary&) = delete;

    std::optional<unsigned int> count() {
        if (!handle_) return std::nullopt;
        const auto init = symbol<int (*)()>("nvmlInit");
        const auto count_devices = symbol<int (*)(unsigned int*)>("nvmlDeviceGetCount_v2");
        shutdown_ = symbol<int (*)()>("nvmlShutdown");
        if (!init || !count_devices || !shutdown_) return std::nullopt;
        if (init() != 0) {
            std::cerr << "Can't initialize NVML\n";
            return std::nullopt;
        }
        initialized_ = true;
        unsigned int devices = 0;
        if (count_devices(&devices) != 0) {
            std::cerr << "Can't get nvml device count\n";
            return std::nullopt;
        }
        // The installed ctypes helper reads into a signed C int; negative
        // values request fallback. Device counts themselves are nonnegative.
        if (devices > static_cast<unsigned int>(std::numeric_limits<int>::max())) return std::nullopt;
        return devices;
    }

    std::optional<std::vector<std::string>> uuids(unsigned int count) {
        if (!count) return std::vector<std::string>{};
        const auto get_handle = symbol<int (*)(unsigned int, void**)>("nvmlDeviceGetHandleByIndex_v2");
        const auto get_uuid = symbol<int (*)(void*, char*, unsigned int)>("nvmlDeviceGetUUID");
        if (!get_handle || !get_uuid) return std::nullopt;
        std::vector<std::string> result;
        for (unsigned int index = 0; index < count; ++index) {
            void* device = nullptr;
            if (get_handle(index, &device) != 0) {
                std::cerr << "Can't get device handle\n";
                return std::nullopt;
            }
            std::array<char, 96> buffer{};
            if (get_uuid(device, buffer.data(), static_cast<unsigned int>(buffer.size())) != 0) {
                std::cerr << "Can't get device UUID\n";
                return std::nullopt;
            }
            if (std::any_of(buffer.begin(), buffer.end(), [](unsigned char byte) { return byte >= 128; })) {
                throw std::invalid_argument("NVML device UUID is not ASCII");
            }
            // Match buf.raw.decode('ascii').strip('\0'), including interior NULs.
            const std::string bytes(buffer.data(), buffer.size());
            const auto first = bytes.find_first_not_of('\0');
            result.push_back(first == std::string::npos ? std::string{} :
                bytes.substr(first, bytes.find_last_not_of('\0') - first + 1));
        }
        return result;
    }

private:
    template<class Function> Function symbol(const char* name) const {
        return reinterpret_cast<Function>(dlsym(handle_, name));
    }
    void* handle_{};
    int (*shutdown_)(){};
    bool initialized_{};
};

}  // namespace

std::optional<int> evaluation_nvml_devices() {
    const auto mask = visibility();
    if (mask.entries.empty()) return 0;
    if (mask.kind == MaskKind::mig) return std::nullopt;
    NvmlLibrary library;
    const auto raw_count = library.count();
    if (!raw_count) return std::nullopt;
    if (mask.kind == MaskKind::ordinal) {
        if (!*raw_count) return 0;
        const auto limit = std::to_string(*raw_count);
        for (std::size_t index = 0; index < mask.entries.size(); ++index) {
            const auto& value = mask.entries[index];
            if (value.size() > limit.size() || (value.size() == limit.size() && value >= limit)) {
                return static_cast<int>(index);
            }
        }
        return static_cast<int>(mask.entries.size());
    }
    const auto uuids = library.uuids(*raw_count);
    if (!uuids) return std::nullopt;
    std::vector<std::size_t> ordinals;
    for (const auto& candidate : mask.entries) {
        std::optional<std::size_t> match;
        bool ambiguous = false;
        for (std::size_t index = 0; index < uuids->size(); ++index) {
            if ((*uuids)[index].compare(0, candidate.size(), candidate) != 0) continue;
            if (match) { ambiguous = true; break; }
            match = index;
        }
        if (!match || ambiguous) break;
        if (std::find(ordinals.begin(), ordinals.end(), *match) != ordinals.end()) return 0;
        ordinals.push_back(*match);
    }
    return static_cast<int>(ordinals.size());
}

}  // namespace uf::evaluation
