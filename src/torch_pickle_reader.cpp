// Native torch.save readers, based on the retained PyTorch serialization
// source. PyTorch BSD notice: licenses/PyTorch.txt.
#include "torch_pickle_structure.hpp"
#include "uf/process.hpp"
#include "uf/python_integer.hpp"
#include <ATen/ops/_empty_affine_quantized.h>
#include <ATen/ops/_empty_per_channel_affine_quantized.h>
#include <ATen/ops/_validate_sparse_coo_tensor_args.h>
#include <ATen/ops/_validate_sparse_compressed_tensor_args.h>
#include <ATen/ops/sparse_coo_tensor.h>
#include <ATen/ops/sparse_compressed_tensor.h>
#include <caffe2/serialize/inline_container.h>
#include <caffe2/serialize/read_adapter_interface.h>
#include <c10/core/CPUAllocator.h>
#include <torch/csrc/jit/serialization/unpickler.h>
#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace uf::pickle_detail {
namespace {
[[noreturn]] void invalid(const char* message) { throw std::runtime_error(std::string("Invalid PyTorch checkpoint: ") + message); }
std::uint64_t multiply(std::uint64_t left, std::uint64_t right) {
    if (left && right > std::numeric_limits<std::uint64_t>::max() / left) invalid("size overflow"); return left * right;
}
std::uint64_t add(std::uint64_t left, std::uint64_t right) {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) invalid("offset overflow"); return left + right;
}
std::uint64_t nonnegative(const Node* value) { const auto number = integer_value(value); if (number < 0) invalid("negative size"); return static_cast<std::uint64_t>(number); }
std::vector<std::int64_t> dimensions(const Node* value) {
    std::vector<std::int64_t> result; for (const auto* item : sequence(value)) result.push_back(integer_value(item)); return result;
}
bool little_host() { const std::uint16_t value = 1; return *reinterpret_cast<const std::uint8_t*>(&value) == 1; }

class File {
public:
    explicit File(const std::filesystem::path& path) : descriptor_(::open(path.c_str(), O_RDONLY | O_CLOEXEC)) {
        if (descriptor_.get() < 0) throw std::runtime_error("Cannot open checkpoint: " + path.string());
        struct stat status{};
        if (fstat(descriptor_.get(), &status) != 0 || !S_ISREG(status.st_mode) || status.st_size < 0 ||
            static_cast<std::uint64_t>(status.st_size) > std::numeric_limits<std::size_t>::max()) invalid("checkpoint is not a bounded regular file");
        size_ = static_cast<std::uint64_t>(status.st_size);
    }
    std::uint64_t size() const { return size_; }
    void read(std::uint64_t position, void* target, std::size_t count) const {
        if (position > size_ || count > size_ - position) invalid("truncated file");
        auto* output = static_cast<char*>(target);
        while (count) {
            ProcessInterrupt::check();
            if (position >= cache_begin_ && position - cache_begin_ < cache_size_) {
                const auto offset = static_cast<std::size_t>(position - cache_begin_);
                const auto step = std::min(count, cache_size_ - offset);
                std::memcpy(output, cache_.data() + offset, step); position += step; output += step; count -= step;
            } else if (count < cache_.size()) {
                cache_begin_ = position; cache_size_ = static_cast<std::size_t>(std::min<std::uint64_t>(cache_.size(), size_ - position));
                direct(position, cache_.data(), cache_size_);
            } else {
                const auto step = std::min(count, cache_.size()); direct(position, output, step);
                position += step; output += step; count -= step;
            }
        }
    }
private:
    void direct(std::uint64_t position, void* output, std::size_t count) const {
        for (std::size_t done = 0; done < count;) {
            ProcessInterrupt::check();
            const auto received = pread(descriptor_.get(), static_cast<char*>(output) + done, count - done,
                static_cast<off_t>(position + done));
            if (received < 0 && errno == EINTR) continue;
            if (received <= 0) invalid("cannot read checkpoint bytes");
            done += static_cast<std::size_t>(received);
        }
    }
    FileDescriptor descriptor_;
    std::uint64_t size_{};
    mutable std::array<char, 64 * 1024> cache_{};
    mutable std::uint64_t cache_begin_{};
    mutable std::size_t cache_size_{};
};
class FileAdapter : public caffe2::serialize::ReadAdapterInterface {
public:
    explicit FileAdapter(std::shared_ptr<File> file) : file_(std::move(file)) {}
    std::size_t size() const override { return static_cast<std::size_t>(file_->size()); }
    std::size_t read(std::uint64_t position, void* output, std::size_t count, const char* = "") const override {
        file_->read(position, output, count); return count;
    }
private:
    std::shared_ptr<File> file_;
};
ReadAt file_reader(const std::shared_ptr<File>& file) { return [file](std::uint64_t at, void* target, std::size_t size) { file->read(at, target, size); }; }

struct Scalar {
    c10::ScalarType type;
    std::string storage_name;
    std::size_t bytes;
    std::size_t swap_bytes;
    unsigned packing;
};
Scalar scalar(c10::ScalarType type, std::string name) {
    const auto bytes = c10::elementSize(type);
    return {type, std::move(name), bytes, c10::isComplexType(type) ? bytes / 2 : bytes,
        type == c10::kQUInt4x2 ? 2u : type == c10::kQUInt2x4 ? 4u : 1u};
}
Scalar storage_scalar(const Node* type) {
    if (type->kind != Kind::global) invalid("storage type is not a global");
    if (type->name == "UntypedStorage") return scalar(c10::kByte, "ByteStorage");
#define UF_STORAGE_SCALAR(_, scalar_name) if (type->name == #scalar_name "Storage") return scalar(c10::k##scalar_name, #scalar_name "Storage");
    AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(UF_STORAGE_SCALAR)
#undef UF_STORAGE_SCALAR
    invalid("unknown storage type");
}
std::string lower(std::string value) { for (auto& byte : value) if (byte >= 'A' && byte <= 'Z') byte += 'a' - 'A'; return value; }
Scalar tensor_scalar(const Node* type) {
    if (type->kind != Kind::global || type->text != "torch") invalid("tensor dtype is not a torch global");
    static const std::map<std::string, c10::ScalarType> aliases{
        {"float16", c10::kHalf}, {"half", c10::kHalf}, {"float32", c10::kFloat}, {"float", c10::kFloat},
        {"float64", c10::kDouble}, {"double", c10::kDouble}, {"int8", c10::kChar}, {"uint8", c10::kByte},
        {"int16", c10::kShort}, {"short", c10::kShort}, {"int32", c10::kInt}, {"int", c10::kInt},
        {"int64", c10::kLong}, {"long", c10::kLong}, {"complex32", c10::kComplexHalf},
        {"complex64", c10::kComplexFloat}, {"cfloat", c10::kComplexFloat}, {"complex128", c10::kComplexDouble},
        {"cdouble", c10::kComplexDouble}};
    if (const auto found = aliases.find(type->name); found != aliases.end()) return scalar(found->second, "");
#define UF_DTYPE_SCALAR(_, scalar_name) if (lower(type->name) == lower(#scalar_name)) return scalar(c10::k##scalar_name, #scalar_name "Storage");
    AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(UF_DTYPE_SCALAR)
#undef UF_DTYPE_SCALAR
    invalid("unknown tensor dtype");
}
struct Span { std::uint64_t begin{}, bytes{}; };
using Record = std::function<std::tuple<at::DataPtr, std::size_t>()>;
struct Storage {
    Scalar dtype;
    std::uint64_t elements{}, offset{};
    std::string root;
    Record record;
    std::optional<c10::Storage> loaded;
    std::uint64_t bytes() const { return multiply(elements, dtype.bytes); }
};
class Bank {
public:
    std::map<std::string, Storage> entries;
    bool little_endian = true;
    void root(const std::string& key, const Scalar& dtype, std::uint64_t elements) {
        if (multiply(elements, dtype.bytes) > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) invalid("storage exceeds the native byte range");
        const auto found = entries.find(key);
        if (found != entries.end()) {
            if (found->second.root != key || found->second.dtype.type != dtype.type || found->second.elements != elements) invalid("conflicting storage descriptors");
            return;
        }
        entries.emplace(key, Storage{dtype, elements, 0, key, {}, std::nullopt});
    }
    void view(const std::string& key, const std::string& parent, std::uint64_t offset, std::uint64_t elements) {
        if (entries.count(key)) return;  // Python's first registered view wins.
        const auto& source = entries.at(parent);
        if (offset > source.elements || elements > source.elements - offset) invalid("storage view exceeds its parent");
        entries.emplace(key, Storage{source.dtype, elements, add(source.offset, multiply(offset, source.dtype.bytes)),
            source.root, {}, std::nullopt});
    }
    c10::Storage get(const std::string& key) {
        auto& info = entries.at(key);
        if (info.loaded) return *info.loaded;
        const auto bytes = static_cast<std::size_t>(info.bytes());
        if (info.root != key) {
            auto root_storage = get(info.root);
            auto* base = static_cast<char*>(root_storage.mutable_data());
            auto owner = std::make_unique<c10::Storage>(root_storage);
            at::DataPtr pointer(base ? base + info.offset : nullptr, owner.get(),
                [](void* value) { delete static_cast<c10::Storage*>(value); }, c10::Device(c10::kCPU));
            owner.release();
            info.loaded.emplace(c10::Storage::use_byte_size_t(), bytes, std::move(pointer), nullptr, false);
        } else if (!bytes) {
            info.loaded.emplace(c10::Storage::use_byte_size_t(), c10::SymInt(0), c10::GetCPUAllocator(), true);
        } else {
            if (!info.record) invalid("missing storage data");
            auto record = info.record();
            if (std::get<1>(record) < bytes || !std::get<0>(record).get()) invalid("storage data is shorter than its descriptor");
            auto pointer = std::move(std::get<0>(record));
            if (little_endian != little_host() && info.dtype.swap_bytes > 1) {
                auto* data = static_cast<char*>(pointer.get());
                for (std::size_t offset = 0; offset < bytes; offset += info.dtype.swap_bytes) {
                    if (offset % (64 * 1024) == 0) ProcessInterrupt::check();
                    std::reverse(data + offset, data + offset + info.dtype.swap_bytes);
                }
            }
            info.loaded.emplace(c10::Storage::use_byte_size_t(), bytes, std::move(pointer), c10::GetCPUAllocator(), true);
        }
        return *info.loaded;
    }
    torch::Tensor flat(const std::string& key) {
        const auto& info = entries.at(key);
        auto options = torch::TensorOptions().device(torch::kCPU).dtype(info.dtype.type);
        if (c10::isQIntType(info.dtype.type)) {
            return at::_empty_affine_quantized({0}, options, 1.0, 0).set_(get(key), 0, {0}, {1});
        }
        return torch::empty({0}, options).set_(get(key), 0, {static_cast<std::int64_t>(info.elements)}, {1});
    }
};
Record raw_record(const ReadAt& read, Span span) {
    if (span.bytes > std::numeric_limits<std::size_t>::max()) invalid("storage exceeds size_t");
    return [read, span] {
        auto data = c10::GetCPUAllocator()->allocate(static_cast<std::size_t>(span.bytes));
        auto* output = static_cast<char*>(data.get());
        for (std::uint64_t offset = 0; offset < span.bytes;) {
            ProcessInterrupt::check();
            const auto step = static_cast<std::size_t>(std::min<std::uint64_t>(64 * 1024, span.bytes - offset));
            read(span.begin + offset, output + offset, step); offset += step;
        }
        return std::make_tuple(std::move(data), static_cast<std::size_t>(span.bytes));
    };
}

using StorageNodes = std::unordered_map<const Node*, std::string>;
StorageNodes describe(Graph& graph, Bank& bank, bool legacy) {
    StorageNodes references;
    const auto count = graph.nodes.size();
    for (std::size_t i = 0; i < count; ++i) {
        auto* node = graph.nodes[i].get();
        if (node->kind != Kind::persistent) continue;
        const auto& arguments = sequence(node->children.at(0));
        if (arguments.size() != (legacy ? 6u : 5u) || string_value(arguments.at(0)) != "storage") invalid("unsupported persistent identifier");
        const auto type = storage_scalar(arguments[1]);
        const auto root = key_value(arguments[2]);
        bank.root(root, type, nonnegative(arguments[4]));
        auto key = root;
        if (legacy && arguments[5]->kind != Kind::none) {
            const auto& view = sequence(arguments[5]); if (view.size() != 3) invalid("storage view metadata");
            key = key_value(view[0]); bank.view(key, root, nonnegative(view[1]), nonnegative(view[2]));
        }
        references.emplace(node, std::move(key));
    }
    return references;
}
void tensor_node(Node* node, std::vector<torch::Tensor>& tensors, torch::Tensor tensor) {
    if (tensors.size() > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) invalid("tensor table exceeds int64");
    node->kind = Kind::tensor; node->tensor_index = static_cast<std::int64_t>(tensors.size());
    node->children.clear(); node->state.clear(); tensors.push_back(std::move(tensor));
}
bool flag(const Node* node) {
    if (node->kind == Kind::boolean) return node->boolean;
    if (node->kind == Kind::integer) return integer_value(node) != 0;
    invalid("expected a tensor boolean");
}
void math_bits(torch::Tensor& tensor, const Node* value) {
    if (value->kind == Kind::none) return;
    if (value->kind != Kind::dictionary) invalid("tensor metadata is not a dictionary");
    for (std::size_t i = 0; i < value->children.size(); i += 2) {
        const auto& key = string_value(value->children[i]);
        if (key == "conj") tensor._set_conj(flag(value->children[i + 1]));
        else if (key == "neg") tensor._set_neg(flag(value->children[i + 1]));
    }
}
torch::Tensor dense_tensor(const Node* args, const StorageNodes& references, Bank& bank, bool version3) {
    const auto& values = sequence(args);
    if (values.size() < 4) invalid("short tensor rebuild arguments");
    const auto found = references.find(values[0]); if (found == references.end()) invalid("tensor storage is not a persistent identifier");
    const auto& info = bank.entries.at(found->second);
    const auto dtype = version3 ? tensor_scalar(values.at(6)) : info.dtype;
    const auto size = dimensions(values[2]), stride = dimensions(values[3]);
    const auto offset = integer_value(values[1]);
    const auto capacity = multiply(info.bytes() / dtype.bytes, dtype.packing);
    check_tensor_geometry(size, stride, offset, capacity);
    auto tensor = torch::empty({0}, torch::TensorOptions().device(torch::kCPU).dtype(dtype.type))
        .set_(bank.get(found->second), offset, size, stride);
    if (values.size() > 4) tensor.set_requires_grad(flag(values[4]));
    const auto metadata = version3 ? 7u : 6u;
    if (values.size() > metadata) math_bits(tensor, values[metadata]);
    return tensor;
}
const torch::Tensor& table_tensor(const Node* node, const std::vector<torch::Tensor>& tensors) {
    if (node->kind != Kind::tensor || node->tensor_index < 0) invalid("rebuild argument is not a tensor");
    return tensors.at(static_cast<std::size_t>(node->tensor_index));
}
double real(const Node* node) {
    return node->kind == Kind::floating ? node->floating : static_cast<double>(integer_value(node));
}
torch::Tensor numeric_tensor(const Node* node, const std::vector<torch::Tensor>& tensors, c10::ScalarType dtype) {
    if (node->kind == Kind::tensor) return table_tensor(node, tensors);
    if (node->kind != Kind::list) invalid("quantization parameters are not tensors or lists");
    const auto options = torch::TensorOptions().device(torch::kCPU).dtype(dtype);
    if (dtype == c10::kLong) {
        std::vector<std::int64_t> values;
        for (const auto* value : node->children) values.push_back(integer_value(value));
        return torch::tensor(values, options);
    }
    std::vector<double> values;
    for (const auto* value : node->children) values.push_back(real(value));
    return torch::tensor(values, options);
}
torch::Tensor quantized_tensor(const Node* args, const StorageNodes& references, Bank& bank,
                              const std::vector<torch::Tensor>& tensors) {
    const auto& values = sequence(args);
    if (values.size() != 7) invalid("quantized tensor arguments");
    const auto found = references.find(values[0]); if (found == references.end()) invalid("quantized tensor storage");
    const auto& info = bank.entries.at(found->second);
    if (!c10::isQIntType(info.dtype.type)) invalid("quantized tensor storage dtype");
    const auto size = dimensions(values[2]), stride = dimensions(values[3]);
    const auto offset = integer_value(values[1]);
    check_tensor_geometry(size, stride, offset, multiply(info.elements, info.dtype.packing));
    const auto& parameters = sequence(values[4]);
    if (parameters.empty()) invalid("empty tensor quantizer");
    const auto options = torch::TensorOptions().device(torch::kCPU).dtype(info.dtype.type);
    torch::Tensor tensor;
    if (is_global(parameters[0], "torch", "per_tensor_affine")) {
        if (parameters.size() != 3) invalid("affine tensor quantizer arguments");
        tensor = at::_empty_affine_quantized(size, options, real(parameters[1]), integer_value(parameters[2]));
    } else if (is_global(parameters[0], "torch", "per_channel_affine") ||
               is_global(parameters[0], "torch", "per_channel_affine_float_qparams")) {
        if (parameters.size() != 4) invalid("channel tensor quantizer arguments");
        const auto floating = parameters[0]->name == "per_channel_affine_float_qparams";
        const auto scales = numeric_tensor(parameters[1], tensors, floating ? c10::kFloat : c10::kDouble);
        const auto zero_points = numeric_tensor(parameters[2], tensors, floating ? c10::kFloat : c10::kLong);
        tensor = at::_empty_per_channel_affine_quantized(size, scales, zero_points, integer_value(parameters[3]), options);
    } else invalid("unsupported tensor quantization scheme");
    tensor.set_(bank.get(found->second), offset, size, stride);
    tensor.set_requires_grad(flag(values[5]));
    return tensor;
}
c10::Layout sparse_layout(const std::string& name) {
    if (name == "torch.sparse_coo") return c10::Layout::Sparse;
    if (name == "torch.sparse_csr") return c10::Layout::SparseCsr;
    if (name == "torch.sparse_csc") return c10::Layout::SparseCsc;
    if (name == "torch.sparse_bsr") return c10::Layout::SparseBsr;
    if (name == "torch.sparse_bsc") return c10::Layout::SparseBsc;
    invalid("unsupported sparse tensor layout");
}
torch::Tensor sparse_tensor(const Node* args, const std::vector<torch::Tensor>& tensors) {
    const auto& arguments = sequence(args);
    if (arguments.size() != 2) invalid("sparse tensor arguments");
    const auto layout = static_cast<c10::Layout>(integer_value(arguments[0]));
    const auto& data = sequence(arguments[1]);
    if (layout == c10::Layout::Sparse) {
        if (data.size() != 3 && data.size() != 4) invalid("COO tensor arguments");
        const auto& indices = table_tensor(data[0], tensors);
        const auto& values = table_tensor(data[1], tensors);
        const auto size = dimensions(data[2]);
        std::optional<bool> coalesced;
        if (data.size() == 4 && data[3]->kind != Kind::none) coalesced = flag(data[3]);
        at::_validate_sparse_coo_tensor_args(indices, values, size, coalesced);
        return at::sparse_coo_tensor(indices, values, size, values.options().layout(layout), coalesced);
    }
    if (layout != c10::Layout::SparseCsr && layout != c10::Layout::SparseCsc &&
        layout != c10::Layout::SparseBsr && layout != c10::Layout::SparseBsc) invalid("sparse compressed tensor layout");
    if (data.size() != 4) invalid("sparse compressed tensor arguments");
    const auto& compressed = table_tensor(data[0], tensors);
    const auto& plain = table_tensor(data[1], tensors);
    const auto& values = table_tensor(data[2], tensors);
    const auto size = dimensions(data[3]);
    at::_validate_sparse_compressed_tensor_args(compressed, plain, values, size, layout);
    return at::sparse_compressed_tensor(compressed, plain, values, size, values.options().layout(layout));
}
void materialize(Graph& graph, StorageNodes& references, Bank& bank, std::vector<torch::Tensor>& tensors) {
    const auto count = graph.nodes.size();
    // Storage placeholders keep a single tensor-table representation for
    // persistent references while reducers consume the shared backing bank.
    for (std::size_t i = 0; i < count; ++i) {
        auto* node = graph.nodes[i].get();
        if (const auto found = references.find(node); found != references.end()) tensor_node(node, tensors, bank.flat(found->second));
    }
    for (std::size_t i = 0; i < count; ++i) {
        auto* node = graph.nodes[i].get();
        if (node->kind != Kind::reduce || node->children.size() != 2) continue;
        const auto* function = node->children[0]; const auto* args = node->children[1];
        const auto& values = sequence(args);
        if (is_global(function, "torch._utils", "_rebuild_tensor") ||
            is_global(function, "torch._utils", "_rebuild_tensor_v2") ||
            is_global(function, "torch._utils", "_rebuild_tensor_v3")) {
            tensor_node(node, tensors, dense_tensor(args, references, bank, function->name == "_rebuild_tensor_v3"));
        } else if (is_global(function, "torch._utils", "_rebuild_parameter") ||
                   is_global(function, "torch._utils", "_rebuild_parameter_with_state")) {
            if (values.size() < 2 || values[0]->kind != Kind::tensor) invalid("parameter data is not a tensor");
            auto tensor = tensors.at(static_cast<std::size_t>(values[0]->tensor_index)).detach();
            tensor.set_requires_grad(flag(values[1])); tensor_node(node, tensors, std::move(tensor));
        } else if (is_global(function, "torch._tensor", "_rebuild_from_type_v2")) {
            if (values.size() != 4 || values[0]->kind != Kind::global) invalid("tensor subclass rebuild arguments");
            const auto* base = values[0];
            if (base->text != "torch._utils" || (base->name != "_rebuild_tensor" && base->name != "_rebuild_tensor_v2" && base->name != "_rebuild_tensor_v3")) invalid("unsupported tensor subclass constructor");
            tensor_node(node, tensors, dense_tensor(values[2], references, bank, base->name == "_rebuild_tensor_v3"));
        } else if (is_global(function, "torch", "Size")) {
            if (values.size() != 1) invalid("torch.Size arguments");
            node->kind = Kind::tuple; node->children = sequence(values[0]); node->state.clear();
        } else if (is_global(function, "torch.serialization", "_get_layout")) {
            if (values.size() != 1) invalid("tensor layout arguments");
            const auto& layout = string_value(values[0]);
            auto* value = graph.integer(static_cast<std::int64_t>(sparse_layout(layout)));
            node->kind = Kind::integer; node->text = value->text; node->children.clear(); node->state.clear();
        } else if (is_global(function, "torch._utils", "_rebuild_qtensor")) {
            tensor_node(node, tensors, quantized_tensor(args, references, bank, tensors));
        } else if (is_global(function, "torch._utils", "_rebuild_sparse_tensor")) {
            tensor_node(node, tensors, sparse_tensor(args, tensors));
        }
    }
}
torch::IValue decode(Graph& graph, Node* root, const std::vector<torch::Tensor>& tensors) {
    const auto pickle = graph.emit(root);
    std::size_t position = 0;
    torch::jit::Unpickler unpickler([&](char* output, std::size_t requested) {
        ProcessInterrupt::check();
        const auto count = std::min(requested, pickle.size() - position);
        if (count) std::memcpy(output, pickle.data() + position, count); position += count; return count;
    }, nullptr, c10::ArrayRef<at::Tensor>(tensors));
    return unpickler.parse_ivalue();
}

torch::IValue zip_checkpoint(const std::shared_ptr<File>& file) {
    caffe2::serialize::PyTorchStreamReader reader(std::make_shared<FileAdapter>(file));
    auto record = reader.getRecord("data.pkl");
    const auto* bytes = static_cast<const char*>(std::get<0>(record).get());
    const auto size = std::get<1>(record);
    Cursor input([&](std::uint64_t at, void* output, std::size_t count) {
        if (at > size || count > size - at) invalid("pickle archive range"); if (count) std::memcpy(output, bytes + at, count);
    }, 0, size);
    Graph graph; auto* root = graph.parse(input);
    Bank bank;
    if (reader.hasRecord("byteorder")) {
        auto order = reader.getRecord("byteorder");
        const std::string value(static_cast<const char*>(std::get<0>(order).get()), std::get<1>(order));
        if (value != "little" && value != "big") invalid("unknown archive byte order"); bank.little_endian = value == "little";
    }
    auto references = describe(graph, bank, false);
    for (auto& item : bank.entries) {
        const auto name = "data/" + item.first;
        item.second.record = [&reader, name] { ProcessInterrupt::check(); return reader.getRecord(name); };
    }
    std::vector<torch::Tensor> tensors;
    materialize(graph, references, bank, tensors);
    return decode(graph, root, tensors);
}

torch::IValue stream_checkpoint(const std::shared_ptr<File>& file) {
    const auto read = file_reader(file);
    Cursor input(read, 0, file->size());
    Graph graph;
    const auto* magic = graph.parse(input);
    const std::string expected("\x6c\xfc\x9c\x46\xf9\x20\x6a\xa8\x50\x19", 10);
    if (magic->kind != Kind::integer || magic->text != expected) invalid("legacy magic number");
    if (integer_value(graph.parse(input)) != 1001) invalid("legacy protocol version");
    const auto* system = graph.parse(input);
    if (system->kind != Kind::dictionary) invalid("legacy system metadata");
    auto* root = graph.parse(input);
    Bank bank; auto references = describe(graph, bank, true);
    const auto* keys = graph.parse(input);
    for (const auto* key_node : sequence(keys)) {
        const auto key = key_value(key_node);
        auto found = bank.entries.find(key); if (found == bank.entries.end() || found->second.root != key) invalid("unknown serialized storage");
        const auto count = input.little(8);
        if (count != found->second.elements) invalid("storage element count differs");
        const auto bytes = found->second.bytes();
        const auto begin = input.position(); input.skip(bytes);
        found->second.record = raw_record(read, {begin, bytes});
    }
    // Legacy storage records are always little-endian, independently of the
    // producer's sys_info flag (see THPStorage_writeFileRaw).
    std::vector<torch::Tensor> tensors;
    materialize(graph, references, bank, tensors);
    return decode(graph, root, tensors);
}

// The older tar format stores independent storages, tensor geometry, and the
// object pickle. Read member ranges directly; no archive extraction occurs.
struct TarMember { Span span; char type{}; };
std::string tar_text(const std::array<char, 512>& header, std::size_t start, std::size_t count) {
    auto end = start; while (end < start + count && header[end]) ++end; return {header.data() + start, end - start};
}
std::uint64_t tar_number(const char* field, std::size_t count) {
    std::uint64_t value = 0;
    if (static_cast<unsigned char>(*field) & 128) {
        if (static_cast<unsigned char>(*field) & 64) invalid("negative tar number");
        for (std::size_t i = 0; i < count; ++i) value = add(multiply(value, 256), static_cast<unsigned char>(field[i]) & (i ? 255u : 127u));
    } else {
        std::size_t i = 0; while (i < count && (field[i] == ' ' || field[i] == 0)) ++i;
        for (; i < count && field[i] != ' ' && field[i] != 0; ++i) {
            if (field[i] < '0' || field[i] > '7') invalid("bad tar number"); value = add(multiply(value, 8), field[i] - '0');
        }
    }
    return value;
}
bool tar_checksum(const std::array<char, 512>& header) {
    std::int64_t unsigned_sum = 0, signed_sum = 0;
    for (std::size_t i = 0; i < header.size(); ++i) {
        const auto byte = i >= 148 && i < 156 ? 32 : static_cast<unsigned char>(header[i]);
        unsigned_sum += byte; signed_sum += i >= 148 && i < 156 ? 32 : static_cast<signed char>(header[i]);
    }
    try { const auto expected = tar_number(header.data() + 148, 8); return expected == static_cast<std::uint64_t>(unsigned_sum) || (signed_sum >= 0 && expected == static_cast<std::uint64_t>(signed_sum)); }
    catch (const std::runtime_error&) { return false; }
}
std::map<std::string, std::string> pax_fields(const std::string& value) {
    std::map<std::string, std::string> fields;
    for (std::size_t offset = 0; offset < value.size();) {
        const auto space = value.find(' ', offset); if (space == std::string::npos) invalid("PAX record length");
        const auto digits = value.substr(offset, space - offset);
        if (digits.empty() || digits.find_first_not_of("0123456789") != std::string::npos) invalid("PAX length digits");
        const auto count = std::stoull(digits); if (!count || count > value.size() - offset || count <= space - offset + 1 || value[offset + count - 1] != '\n') invalid("PAX record range");
        const auto text = value.substr(space + 1, offset + static_cast<std::size_t>(count) - space - 2);
        const auto equals = text.find('='); if (equals == std::string::npos) invalid("PAX record field");
        fields[text.substr(0, equals)] = text.substr(equals + 1); offset += static_cast<std::size_t>(count);
    }
    return fields;
}
std::optional<std::map<std::string, TarMember>> tar_members(const std::shared_ptr<File>& file) {
    if (file->size() < 512) return std::nullopt;
    std::array<char, 512> header{}; file->read(0, header.data(), header.size());
    if (!tar_checksum(header) && std::any_of(header.begin(), header.end(), [](char byte) { return byte != 0; })) return std::nullopt;
    std::map<std::string, TarMember> result;
    std::map<std::string, std::string> global_fields, next_fields;
    std::string long_name;
    const auto read = file_reader(file);
    for (std::uint64_t position = 0; position < file->size();) {
        if (file->size() - position < 512) invalid("truncated tar header");
        file->read(position, header.data(), header.size()); position += 512;
        if (std::all_of(header.begin(), header.end(), [](char byte) { return byte == 0; })) break;
        if (!tar_checksum(header)) invalid("tar checksum");
        auto name = tar_text(header, 0, 100);
        const auto prefix = tar_text(header, 345, 155);
        if (!prefix.empty() && tar_text(header, 257, 6) == "ustar") name = prefix + "/" + name;
        auto count = tar_number(header.data() + 124, 12);
        const auto type = header[156];
        if (type == 'x' || type == 'g' || type == 'L' || type == 'K') {
            Cursor payload(read, position, file->size()); const auto value = payload.bytes(count);
            if (type == 'L') { long_name = value.substr(0, value.find('\0')); if (!long_name.empty() && long_name.back() == '\n') long_name.pop_back(); }
            else if (type != 'K') {
                auto fields = pax_fields(value); auto& target = type == 'g' ? global_fields : next_fields;
                for (auto& item : fields) target[item.first] = std::move(item.second);
            }
        } else {
            auto fields = global_fields; for (const auto& item : next_fields) fields[item.first] = item.second;
            if (!long_name.empty()) name = long_name;
            if (fields.count("path")) name = fields.at("path");
            if (fields.count("size")) {
                const auto& text = fields.at("size"); if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos) invalid("PAX size");
                count = std::stoull(text);
            }
            if (count > file->size() - position) invalid("truncated tar member");
            result[name] = {{position, count}, type};
            next_fields.clear(); long_name.clear();
        }
        const auto padded = multiply(add(count, 511) / 512, 512);
        if (padded > file->size() - position) invalid("truncated tar padding"); position += padded;
    }
    return result;
}
Cursor tar_cursor(const std::shared_ptr<File>& file, const std::map<std::string, TarMember>& members, const char* name) {
    const auto found = members.find(name); if (found == members.end()) invalid("missing legacy tar member");
    const auto& member = found->second;
    if (member.type != 0 && member.type != '0' && member.type != '7') invalid("legacy checkpoint member is not a regular file");
    return Cursor(file_reader(file), member.span.begin, member.span.begin + member.span.bytes);
}
torch::IValue tar_checkpoint(const std::shared_ptr<File>& file, const std::map<std::string, TarMember>& members) {
    const auto read = file_reader(file); Graph graph; Bank bank;
    auto storage_input = tar_cursor(file, members, "storages");
    const auto count = nonnegative(graph.parse(storage_input));
    for (std::uint64_t index = 0; index < count; ++index) {
        const auto& values = sequence(graph.parse(storage_input)); if (values.size() != 3) invalid("tar storage arguments");
        const auto key = key_value(values[0]); const auto dtype = storage_scalar(values[2]);
        const auto elements = storage_input.little(8);
        if (elements > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) invalid("tar storage element count");
        bank.root(key, dtype, elements);
        const auto bytes = bank.entries.at(key).bytes(), begin = storage_input.position(); storage_input.skip(bytes);
        bank.entries.at(key).record = raw_record(read, {begin, bytes});
    }
    const auto& views = sequence(graph.parse(storage_input));
    for (const auto* view : views) {
        const auto& values = sequence(view); if (values.size() != 4) invalid("tar storage view metadata");
        bank.view(key_value(values[0]), key_value(values[1]), nonnegative(values[2]), nonnegative(values[3]));
    }
    std::vector<torch::Tensor> tensors;
    std::map<std::string, std::size_t> objects;
    for (const auto& entry : bank.entries) { objects[entry.first] = tensors.size(); tensors.push_back(bank.flat(entry.first)); }
    auto tensor_input = tar_cursor(file, members, "tensors");
    const auto tensor_count = nonnegative(graph.parse(tensor_input));
    for (std::uint64_t index = 0; index < tensor_count; ++index) {
        const auto& values = sequence(graph.parse(tensor_input)); if (values.size() != 3) invalid("tar tensor arguments");
        const auto key = key_value(values[0]), storage = key_value(values[1]);
        const auto dimensions_count = tensor_input.little(4); tensor_input.skip(4);
        if (dimensions_count > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()) ||
            add(multiply(dimensions_count, 16), 8) > tensor_input.remaining()) invalid("tar tensor dimensions");
        std::vector<std::int64_t> size, stride;
        for (std::uint64_t axis = 0; axis < dimensions_count; ++axis) { const auto bits = tensor_input.little(8); std::int64_t value; std::memcpy(&value, &bits, 8); size.push_back(value); }
        for (std::uint64_t axis = 0; axis < dimensions_count; ++axis) { const auto bits = tensor_input.little(8); std::int64_t value; std::memcpy(&value, &bits, 8); stride.push_back(value); }
        const auto bits = tensor_input.little(8); std::int64_t offset; std::memcpy(&offset, &bits, 8);
        const auto& info = bank.entries.at(storage);
        check_tensor_geometry(size, stride, offset, multiply(info.elements, info.dtype.packing));
        auto tensor = torch::empty({0}, torch::TensorOptions().device(torch::kCPU).dtype(info.dtype.type)).set_(bank.get(storage), offset, size, stride);
        objects[key] = tensors.size(); tensors.push_back(std::move(tensor));
    }
    const auto object_start = graph.nodes.size();
    auto object_input = tar_cursor(file, members, "pickle"); auto* root = graph.parse(object_input);
    for (std::size_t index = object_start; index < graph.nodes.size(); ++index) {
        auto* node = graph.nodes[index].get();
        if (node->kind != Kind::persistent) continue;
        const auto* saved = node->children.at(0);
        if (saved->kind == Kind::tuple) invalid("serialized Python module classes are not checkpoint state dictionaries");
        const auto key = saved->kind == Kind::string || saved->kind == Kind::bytes ? PythonInteger(string_value(saved)).decimal() : key_value(saved);
        const auto found = objects.find(key); if (found == objects.end()) invalid("unknown tar persistent object");
        node->kind = Kind::tensor; node->tensor_index = static_cast<std::int64_t>(found->second); node->children.clear(); node->state.clear();
    }
    StorageNodes references;
    materialize(graph, references, bank, tensors);
    return decode(graph, root, tensors);
}
}  // namespace

torch::IValue load_checkpoint(const std::filesystem::path& path) {
    ProcessInterrupt::check(); torch::NoGradGuard no_grad;
    const auto file = std::make_shared<File>(path);
    std::array<unsigned char, 4> magic{};
    if (file->size() >= magic.size()) file->read(0, magic.data(), magic.size());
    torch::IValue result;
    if (magic == std::array<unsigned char, 4>{'P', 'K', 3, 4}) result = zip_checkpoint(file);
    else if (const auto members = tar_members(file)) result = tar_checkpoint(file, *members);
    else result = stream_checkpoint(file);
    ProcessInterrupt::check(); return result;
}
}  // namespace uf::pickle_detail
