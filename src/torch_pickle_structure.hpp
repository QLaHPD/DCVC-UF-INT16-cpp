// Native Python pickle structure used by the checkpoint readers.
// Format behavior follows CPython and PyTorch references retained under TOOLS.
#pragma once
#include <torch/torch.h>
#include <cstdint>
#include <functional>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace uf::pickle_detail {
using ReadAt = std::function<void(std::uint64_t, void*, std::size_t)>;

class Cursor {
public:
    Cursor(ReadAt reader, std::uint64_t begin, std::uint64_t end);
    std::uint64_t position() const { return position_; }
    std::uint64_t remaining() const { return end_ - position_; }
    std::uint8_t byte();
    std::uint64_t little(std::size_t count);
    std::string bytes(std::uint64_t count);
    std::string line();
    void skip(std::uint64_t count);
    void start_frame(std::uint64_t count);
    void next_opcode();
    void finish_pickle();
private:
    void require(std::uint64_t count) const;
    ReadAt reader_;
    std::uint64_t position_, end_, frame_end_{};
    bool framed_{};
};

enum class Kind { none, boolean, integer, floating, string, bytes, list, tuple,
                  dictionary, global, reduce, new_object, persistent, tensor };
struct Node {
    Kind kind{Kind::none};
    bool boolean{};
    double floating{};
    // Integer: normalized little-endian two's complement. String/bytes:
    // payload. Global: module (text) and name (name).
    std::string text, name;
    std::vector<Node*> children, state;
    std::int64_t tensor_index{-1};
};

class Graph {
public:
    Node* make(Kind kind);
    Node* integer(std::int64_t value);
    Node* string(std::string value);
    Node* global(std::string module, std::string name);
    Node* tuple(std::vector<Node*> values);
    Node* parse(Cursor& input);
    std::vector<char> emit(Node* root) const;
    // Owning arena avoids reference-count cycles in the temporary graph.
    std::vector<std::unique_ptr<Node>> nodes;
};

std::int64_t integer_value(const Node* node);
std::string integer_decimal(const Node* node);
std::string key_value(const Node* node);
const std::string& string_value(const Node* node);
const std::vector<Node*>& sequence(const Node* node);
bool is_global(const Node* node, const char* module, const char* name);
void check_tensor_geometry(const std::vector<std::int64_t>& size,
    const std::vector<std::int64_t>& stride, std::int64_t offset, std::uint64_t capacity);
torch::IValue load_checkpoint(const std::filesystem::path& path);
}  // namespace uf::pickle_detail
