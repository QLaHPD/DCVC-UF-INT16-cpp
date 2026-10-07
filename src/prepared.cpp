#include "uf/prepared.hpp"
#include "uf/int16_ops.hpp"
#include "uf/torch_pickle.hpp"
#include "uf/process.hpp"
#include "uf/python_json.hpp"

#include <nlohmann/json.hpp>
#include <openssl/evp.h>

#include <array>
#include <iomanip>
#include <memory>
#include <algorithm>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <vector>
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>

namespace uf {
namespace {

torch::IValue field(const c10::impl::GenericDict& dictionary, const std::string& name) {
    for (const auto& entry : dictionary) {
        if (entry.key().isString() && entry.key().toStringRef() == name) return entry.value();
    }
    throw std::runtime_error("Prepared UF model is missing field: " + name);
}

std::string string_field(const c10::impl::GenericDict& dictionary, const std::string& name) {
    const auto value = field(dictionary, name);
    if (!value.isString()) throw std::runtime_error("Prepared UF field is not a string: " + name);
    return value.toStringRef();
}

std::string dtype_name(const torch::Tensor& tensor) {
    switch (tensor.scalar_type()) {
    case torch::kInt16: return "torch.int16";
    case torch::kInt32: return "torch.int32";
    case torch::kUInt8: return "torch.uint8";
    case torch::kInt8: return "torch.int8";
    case torch::kInt64: return "torch.int64";
    case torch::kBool: return "torch.bool";
    case at::ScalarType::UInt16: return "torch.uint16";
    case at::ScalarType::UInt32: return "torch.uint32";
    case at::ScalarType::UInt64: return "torch.uint64";
    default: throw std::runtime_error("Prepared UF model contains a non-integer tensor");
    }
}

void update(EVP_MD_CTX* context, const void* data, std::size_t size) {
    if (EVP_DigestUpdate(context, data, size) != 1) {
        throw std::runtime_error("OpenSSL SHA-256 update failed");
    }
}

void update(EVP_MD_CTX* context, const std::string& value) {
    update(context, value.data(), value.size());
}

std::string content_hash(const PreparedModel& model) {
    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(EVP_MD_CTX_new(), EVP_MD_CTX_free);
    if (!context || EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1) {
        throw std::runtime_error("OpenSSL SHA-256 initialization failed");
    }
    // std::map object keys give Python's sorted compact header. Its json.dumps
    // also uses ensure_ascii=True, including for arbitrary tensor names.
    const nlohmann::json header = {{"format", model.format},
                                   {"arithmetic", model.arithmetic},
                                   {"variant", model.variant},
                                   {"source_sha256", model.source_sha256}};
    update(context.get(), python_ascii_canonical_json(header));
    for (const auto& [name, original] : model.tensors) {
        ProcessInterrupt::check();
        const auto dtype = dtype_name(original);
        const auto tensor = original.to(torch::kCPU).contiguous();
        std::vector<std::int64_t> shape(tensor.sizes().begin(), tensor.sizes().end());
        const nlohmann::json descriptor = nlohmann::json::array({name, dtype, shape});
        update(context.get(), python_ascii_canonical_json(descriptor));
        const auto* bytes = reinterpret_cast<const std::uint8_t*>(tensor.data_ptr());
        const auto item_size = static_cast<std::size_t>(tensor.element_size());
        const auto item_count = static_cast<std::size_t>(tensor.numel());
        if (item_count > std::numeric_limits<std::size_t>::max() / item_size)
            throw std::overflow_error("Prepared UF tensor byte count exceeds size_t");
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
        for (std::size_t i = 0; i < item_count; ++i) {
            for (std::size_t j = item_size; j-- > 0;) update(context.get(), bytes + i * item_size + j, 1);
        }
#else
        update(context.get(), bytes, item_count * item_size);
#endif
    }
    std::array<unsigned char, EVP_MAX_MD_SIZE> hash{};
    unsigned int size = 0;
    if (EVP_DigestFinal_ex(context.get(), hash.data(), &size) != 1 || size != 32) {
        throw std::runtime_error("OpenSSL SHA-256 finalization failed");
    }
    std::ostringstream text;
    text << std::hex << std::setfill('0');
    for (unsigned int i = 0; i < size; ++i) text << std::setw(2) << static_cast<unsigned>(hash[i]);
    return text.str();
}

}  // namespace

const torch::Tensor& PreparedModel::tensor(const std::string& name) const {
    const auto it = tensors.find(name);
    if (it == tensors.end()) throw std::runtime_error("Missing prepared UF tensor: " + name);
    return it->second;
}

std::string prepared_identity(const PreparedModel& model) {
    return content_hash(model);
}

void save_prepared(const PreparedModel& model, const std::filesystem::path& path) {
    if (model.format != "dcvc-uf-prepared-v1" ||
        model.arithmetic != int16::arithmetic_id ||
        model.identity != content_hash(model)) {
        throw std::invalid_argument("Prepared UF model identity is invalid");
    }
    std::filesystem::create_directories(path.parent_path().empty()
        ? std::filesystem::path(".") : path.parent_path());
    c10::impl::GenericDict tensors(c10::StringType::get(), c10::AnyType::get());
    for (const auto& [name, value] : model.tensors) tensors.insert(name, value);
    c10::impl::GenericDict data(c10::StringType::get(), c10::AnyType::get());
    data.insert("format", model.format);
    data.insert("arithmetic", model.arithmetic);
    data.insert("variant", model.variant);
    data.insert("source_sha256", model.source_sha256);
    data.insert("identity", model.identity);
    data.insert("tensors", tensors);
    const auto parent = path.parent_path().empty() ? std::filesystem::path(".") : path.parent_path();
    auto temporary = python_temporary_file("." + path.filename().native(), {}, parent);
    auto output = std::move(temporary.descriptor);
    const auto unlink_temporary = [&] {
        if (unlink(temporary.path.c_str()) != 0) {
            const int error = errno;
            if (error != ENOENT)
                throw std::system_error(error, std::generic_category(), "Cannot remove prepared UF temporary file");
        }
    };
    try {
        try {
            write_torch_pickle_cpu_fd(output.get(), torch::IValue(data));
            for (;;) {
                ProcessInterrupt::check();
                if (fsync(output.get()) == 0) break;
                const int error = errno;
                if (error != EINTR)
                    throw std::system_error(error, std::generic_category(), "Cannot sync prepared UF model");
            }
            output.close_checked("Cannot close prepared UF model");
        } catch (...) {
            output.close_checked("Cannot close prepared UF model");
            throw;
        }
        ProcessInterrupt::check();
        if (link(temporary.path.c_str(), path.c_str()) != 0) {
            if (errno != EEXIST) throw std::runtime_error("Cannot publish prepared UF model");
            load_prepared(path, model.source_sha256, model.variant, model.identity);
        }
    } catch (...) {
        unlink_temporary();
        throw;
    }
    unlink_temporary();
}

PreparedModel load_prepared(const std::filesystem::path& path,
                            const std::optional<std::string>& expected_source_sha256,
                            const std::optional<std::string>& expected_variant,
                            const std::optional<std::string>& expected_identity) {
    const auto value = load_torch_pickle_cpu(path);
    if (!value.isGenericDict()) throw std::runtime_error("Prepared UF model is not a dictionary");
    const auto dictionary = value.toGenericDict();
    PreparedModel result;
    result.format = string_field(dictionary, "format");
    result.arithmetic = string_field(dictionary, "arithmetic");
    result.variant = string_field(dictionary, "variant");
    result.source_sha256 = string_field(dictionary, "source_sha256");
    result.identity = string_field(dictionary, "identity");
    if (result.format != "dcvc-uf-prepared-v1" || result.arithmetic != int16::arithmetic_id ||
        (expected_source_sha256 && result.source_sha256 != *expected_source_sha256) ||
        (expected_variant && result.variant != *expected_variant)) {
        throw std::runtime_error("Unsupported or mismatched prepared UF model");
    }
    const auto tensor_value = field(dictionary, "tensors");
    if (!tensor_value.isGenericDict()) throw std::runtime_error("Prepared UF tensors are not a dictionary");
    for (const auto& entry : tensor_value.toGenericDict()) {
        if (!entry.key().isString() || !entry.value().isTensor()) {
            throw std::runtime_error("Prepared UF tensor entry has the wrong type");
        }
        result.tensors.emplace(entry.key().toStringRef(), entry.value().toTensor());
    }
    const auto actual_identity = content_hash(result);
    if (result.identity != actual_identity ||
        (expected_identity && actual_identity != *expected_identity)) {
        throw std::runtime_error("Prepared UF model/table identity mismatch");
    }
    return result;
}

}  // namespace uf
