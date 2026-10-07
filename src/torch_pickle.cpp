#include "uf/torch_pickle.hpp"
#include "uf/process.hpp"
#include "uf/python_path.hpp"
#include "torch_pickle_structure.hpp"
#include "checkpoint_storage_copy.hpp"

// Storage dtype checks follow PyTorch serialization.py; licenses/PyTorch.txt.

#include <caffe2/serialize/inline_container.h>
#include <c10/core/QScheme.h>
#include <torch/csrc/jit/serialization/export.h>
#include <torch/csrc/jit/serialization/pickler.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fcntl.h>
#include <functional>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unordered_map>
#include <utility>
#include <vector>
#include <unistd.h>

namespace uf {
namespace {

struct ContainerCopies {
    torch::IValue::HashIdentityIValueMap memo;
    torch::IValue::HashIdentityIValues active;
    checkpoint_detail::StorageCopies storages{torch::Device(torch::kCPU)};
    // Python serialization checks allocated source addresses before moving
    // storage to CPU. Separate StorageImpls can expose the same data address.
    std::unordered_map<const void*, c10::ScalarType> storage_dtypes;

    void observe_storage(const torch::Tensor& tensor) {
        ProcessInterrupt::check();
        if (tensor.layout() == torch::kStrided) {
            const auto storage = tensor.storage();
            if (const auto address = storage.data()) {
                const auto inserted = storage_dtypes.emplace(address, tensor.scalar_type());
                if (!inserted.second && inserted.first->second != tensor.scalar_type()) {
                    throw std::runtime_error("Cannot save multiple tensors or storages that view the same data as different types");
                }
            }
            if (tensor.is_quantized() && (tensor.qscheme() == c10::kPerChannelAffine ||
                                         tensor.qscheme() == c10::kPerChannelAffineFloatQParams)) {
                observe_storage(tensor.q_per_channel_scales());
                observe_storage(tensor.q_per_channel_zero_points());
            }
        } else if (tensor.is_sparse()) {
            observe_storage(tensor._indices());
            observe_storage(tensor._values());
        } else if (tensor.is_sparse_csr()) {
            observe_storage(tensor.crow_indices());
            observe_storage(tensor.col_indices());
            observe_storage(tensor.values());
        }
    }
};

torch::IValue python_containers(const torch::IValue& value, ContainerCopies& copies) {
    ProcessInterrupt::check();
    if (copies.active.count(value)) {
        // The underlying JIT Pickler memoizes aggregates after their children.
        // It cannot emit recursive aggregates; avoid unbounded native recursion.
        throw std::invalid_argument("Cyclic containers cannot be written by the UF checkpoint pickler");
    }
    if (const auto found = copies.memo.find(value); found != copies.memo.end()) return found->second;
    if (value.isNone() || value.isInt() || value.isDouble() || value.isBool() || value.isString()) return value;
    copies.active.insert(value);
    torch::IValue output;
    if (value.isGenericDict()) {
        c10::impl::GenericDict dictionary(c10::AnyType::get(), c10::AnyType::get());
        for (const auto& entry : value.toGenericDict()) {
            auto key = python_containers(entry.key(), copies);
            dictionary.insert(std::move(key), python_containers(entry.value(), copies));
        }
        output = dictionary;
    } else if (value.isList()) {
        // An Any list avoids Pickler's build_intlist/build_tensorlist helpers
        // as well as restore_type_tag, all absent from weights_only's allowlist.
        c10::impl::GenericList list(c10::AnyType::get());
        for (const auto& item : value.toListRef()) list.push_back(python_containers(item, copies));
        output = list;
    } else if (value.isTuple()) {
        std::vector<torch::IValue> items;
        for (const auto& item : value.toTupleRef().elements()) items.push_back(python_containers(item, copies));
        output = c10::ivalue::Tuple::create(std::move(items));
    } else if (value.isTensor()) {
        const auto tensor = value.toTensor();
        copies.observe_storage(tensor);
        // Keep shared CPU storage directly. Move each non-CPU strided storage
        // once, retaining view geometry and conjugate/negative metadata.
        auto cpu = !tensor.is_cpu() && tensor.layout() == torch::kStrided && !tensor.device().is_meta()
            ? copies.storages.copy(tensor, true) : tensor.detach().to(torch::kCPU);
        cpu.set_requires_grad(tensor.requires_grad());
        output = cpu;
    } else {
        throw std::invalid_argument("Unsupported value in UF Python checkpoint dictionary");
    }
    copies.active.erase(value);
    copies.memo.emplace(value, output);
    return output;
}

void write_pickle_archive(const torch::IValue& value,
                         caffe2::serialize::PyTorchStreamWriter& writer) {
    try {
        ProcessInterrupt::check();
        torch::NoGradGuard no_grad;
        ContainerCopies copies;
        const auto normalized = python_containers(value, copies);
        std::vector<char> pickle;
        torch::jit::Pickler pickler(
            [&](const char* data, std::size_t size) { pickle.insert(pickle.end(), data, data + size); },
            nullptr, nullptr, nullptr, nullptr, false);
        pickler.protocol();
        pickler.pushIValue(normalized);
        pickler.stop();
        torch::jit::writeArchiveAndTensors("data", pickle.data(), pickle.size(), pickler.tensorData(), writer);
        writer.writeEndOfFile();
    } catch (...) {
        const auto error = std::current_exception();
        // Python's ZIP context finalizes even after serialization failure. An
        // EOF error can replace that failure; StreamWriter marks itself
        // finalized on error so its destructor does not attempt it again.
        if (!writer.finalized()) writer.writeEndOfFile();
        std::rethrow_exception(error);
    }
}

void write_pickle_archive(const torch::IValue& value,
                         const std::function<std::size_t(const void*, std::size_t)>& output) {
    ProcessInterrupt::check();
    caffe2::serialize::PyTorchStreamWriter writer(output);
    write_pickle_archive(value, writer);
}

std::vector<char> pickle_bytes(const torch::IValue& value) {
    std::vector<char> archive;
    write_pickle_archive(value, [&](const void* data, std::size_t size) {
        if (size) {
            const auto* bytes = static_cast<const char*>(data);
            archive.insert(archive.end(), bytes, bytes + size);
        }
        return size;
    });
    return archive;
}

std::size_t write_fd(int fd, const void* data, std::size_t size) {
    const auto* bytes = static_cast<const char*>(data);
    std::size_t offset = 0;
    while (offset < size) {
        ProcessInterrupt::check();
        const auto request = std::min(size - offset, static_cast<std::size_t>(std::numeric_limits<ssize_t>::max()));
        const auto count = ::write(fd, bytes + offset, request);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) throw std::runtime_error("Cannot write UF checkpoint");
        offset += static_cast<std::size_t>(count);
    }
    return size;
}

}  // namespace

torch::IValue load_torch_pickle_cpu(const std::filesystem::path& path) {
    return pickle_detail::load_checkpoint(path);
}

std::vector<char> torch_pickle_bytes_cpu(const torch::IValue& value) { return pickle_bytes(value); }

void write_torch_pickle_cpu_fd(int fd, const torch::IValue& value) {
    write_pickle_archive(value, [&](const void* data, std::size_t size) { return write_fd(fd, data, size); });
}

void save_torch_pickle_cpu(const std::filesystem::path& path, const torch::IValue& value) {
    ProcessInterrupt::check();
    if (path.native().find('\0') != std::string::npos)
        throw std::invalid_argument("embedded null byte");
    const auto text = python_path_text(path);
    if (std::all_of(text.begin(), text.end(), [](unsigned char byte) { return byte < 128; })) {
        // Match torch.save's ASCII filename overload, including ZIP root naming
        // and the library's direct truncate/open/finalize behavior.
        caffe2::serialize::PyTorchStreamWriter writer(path.native());
        write_pickle_archive(value, writer);
        return;
    }
    // Non-ASCII names use Python's FileIO("w") path: direct owned descriptor,
    // callback ZIP output, then checked close after successful finalization.
    FileDescriptor output;
    for (;;) {
        ProcessInterrupt::check();
        const int descriptor = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0666);
        if (descriptor >= 0) { output.reset(descriptor); break; }
        const int error = errno;
        if (error != EINTR)
            throw std::system_error(error, std::generic_category(), "Cannot open UF checkpoint output");
    }
    write_torch_pickle_cpu_fd(output.get(), value);
    output.close_checked("Cannot close UF checkpoint output");
}

}  // namespace uf
