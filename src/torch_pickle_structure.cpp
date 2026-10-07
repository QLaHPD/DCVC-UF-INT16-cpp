// Native checkpoint pickle structure. PyTorch/CPython format references are
// retained separately; this code does not import Python or execute globals.
#include "torch_pickle_structure.hpp"
#include "uf/process.hpp"
#include "uf/python_integer.hpp"
#include "uf/python_numeric.hpp"
#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace uf::pickle_detail {
namespace {
[[noreturn]] void invalid(const char* message) { throw std::runtime_error(std::string("Invalid checkpoint pickle: ") + message); }
std::uint64_t memo_index(const std::string& text) {
    if (text.empty()) invalid("empty memo index");
    std::uint64_t result = 0;
    for (const auto byte : text) {
        if (byte < '0' || byte > '9') invalid("invalid memo index");
        const auto digit = static_cast<unsigned>(byte - '0');
        if (result > (std::numeric_limits<std::uint64_t>::max() - digit) / 10) invalid("memo index overflow");
        result = result * 10 + digit;
    }
    return result;
}
void normalize_integer(std::string& value) {
    if (value.empty()) value.push_back(0);
    while (value.size() > 1) {
        const auto last = static_cast<unsigned char>(value.back());
        const auto previous = static_cast<unsigned char>(value[value.size() - 2]);
        if ((last == 0 && !(previous & 128)) || (last == 255 && (previous & 128))) value.pop_back();
        else break;
    }
}
std::string decimal_integer(std::string_view text) {
    const PythonInteger integer(text);
    const auto words = integer.seed_words();
    std::string result;
    for (const auto word : words) for (unsigned byte = 0; byte < 4; ++byte) result.push_back(static_cast<char>(word >> (8 * byte)));
    while (result.size() > 1 && result.back() == 0) result.pop_back();
    if (static_cast<unsigned char>(result.back()) & 128) result.push_back(0);
    if (integer.decimal().front() == '-') {
        unsigned carry = 1;
        for (auto& byte : result) {
            const auto next = (static_cast<unsigned char>(byte) ^ 255u) + carry;
            byte = static_cast<char>(next); carry = next >> 8;
        }
    }
    normalize_integer(result);
    return result;
}
void utf8(std::string& output, std::uint32_t point) {
    if (point > 0x10ffff) invalid("Unicode code point is out of range");
    if (point < 128) output.push_back(static_cast<char>(point));
    else if (point < 2048) {
        output.push_back(static_cast<char>(0xc0 | (point >> 6))); output.push_back(static_cast<char>(0x80 | (point & 63)));
    } else if (point < 65536) {
        output.push_back(static_cast<char>(0xe0 | (point >> 12))); output.push_back(static_cast<char>(0x80 | ((point >> 6) & 63)));
        output.push_back(static_cast<char>(0x80 | (point & 63)));
    } else {
        output.push_back(static_cast<char>(0xf0 | (point >> 18))); output.push_back(static_cast<char>(0x80 | ((point >> 12) & 63)));
        output.push_back(static_cast<char>(0x80 | ((point >> 6) & 63))); output.push_back(static_cast<char>(0x80 | (point & 63)));
    }
}
unsigned hex(char value) {
    if (value >= '0' && value <= '9') return value - '0';
    if (value >= 'a' && value <= 'f') return value - 'a' + 10;
    if (value >= 'A' && value <= 'F') return value - 'A' + 10;
    invalid("bad hexadecimal escape");
}
std::string unicode_literal(const std::string& raw) {
    std::string result;
    for (std::size_t i = 0; i < raw.size();) {
        if (raw[i] != '\\') { utf8(result, static_cast<unsigned char>(raw[i++])); continue; }
        const auto begin = i;
        while (i < raw.size() && raw[i] == '\\') ++i;
        const auto count = i - begin;
        if (count % 2 && i < raw.size() && (raw[i] == 'u' || raw[i] == 'U')) {
            result.append(count - 1, '\\');
            const auto digits = raw[i++] == 'u' ? 4u : 8u;
            if (digits > raw.size() - i) invalid("truncated Unicode escape");
            std::uint32_t point = 0;
            for (unsigned digit = 0; digit < digits; ++digit) point = point * 16 + hex(raw[i++]);
            utf8(result, point);
        } else result.append(count, '\\');
    }
    return result;
}
std::string string_literal(const std::string& raw) {
    if (raw.size() < 2 || (raw.front() != '\'' && raw.front() != '"') || raw.back() != raw.front()) invalid("bad quoted string");
    std::string result;
    for (std::size_t i = 1; i + 1 < raw.size(); ++i) {
        if (raw[i] != '\\') { result.push_back(raw[i]); continue; }
        if (++i + 1 >= raw.size()) invalid("truncated string escape");
        const auto value = raw[i];
        if (value == 'x') {
            if (raw.size() - i < 4) invalid("truncated hexadecimal escape");
            const auto upper = hex(raw[++i]), lower = hex(raw[++i]);
            result.push_back(static_cast<char>(upper * 16 + lower));
        } else if (value >= '0' && value <= '7') {
            unsigned number = value - '0';
            for (int count = 1; count < 3 && i + 2 < raw.size() && raw[i + 1] >= '0' && raw[i + 1] <= '7'; ++count) {
                number = number * 8 + raw[++i] - '0';
            }
            result.push_back(static_cast<char>(number));
        } else {
            switch (value) {
                case 'n': result.push_back('\n'); break; case 'r': result.push_back('\r'); break;
                case 't': result.push_back('\t'); break; case 'b': result.push_back('\b'); break;
                case 'f': result.push_back('\f'); break; case 'a': result.push_back('\a'); break;
                case 'v': result.push_back('\v'); break;
                case '\\': case '\'': case '"': result.push_back(value); break;
                default: result.push_back('\\'); result.push_back(value); break;
            }
        }
    }
    return result;
}
std::string latin1_bytes(const std::string& value, bool ascii) {
    std::string result;
    for (std::size_t i = 0; i < value.size();) {
        const auto first = static_cast<unsigned char>(value[i++]);
        unsigned point = first;
        if (first >= 128) {
            if (first < 0xc2 || first > 0xc3 || i == value.size()) invalid("byte codec input exceeds Latin-1");
            const auto second = static_cast<unsigned char>(value[i++]);
            if ((second & 0xc0) != 0x80) invalid("bad byte codec UTF-8");
            point = (first & 31) * 64 + (second & 63);
        }
        if (ascii && point > 127) invalid("byte codec input exceeds ASCII");
        result.push_back(static_cast<char>(point));
    }
    return result;
}
}  // namespace

Cursor::Cursor(ReadAt reader, std::uint64_t begin, std::uint64_t end)
    : reader_(std::move(reader)), position_(begin), end_(end) {
    if (begin > end) invalid("bad byte range");
}
void Cursor::require(std::uint64_t count) const {
    if (count > end_ - position_ || (framed_ && count > frame_end_ - position_)) invalid("truncated byte range");
}
std::uint8_t Cursor::byte() {
    require(1); std::uint8_t value; reader_(position_, &value, 1); ++position_; return value;
}
std::uint64_t Cursor::little(std::size_t count) {
    if (count > 8) invalid("integer byte width");
    std::uint64_t value = 0;
    for (std::size_t i = 0; i < count; ++i) value |= static_cast<std::uint64_t>(byte()) << (8 * i);
    return value;
}
std::string Cursor::bytes(std::uint64_t count) {
    require(count);
    if (count > std::string{}.max_size()) invalid("payload exceeds the native string range");
    std::string value(static_cast<std::size_t>(count), 0);
    for (std::size_t offset = 0; offset < value.size();) {
        ProcessInterrupt::check();
        const auto step = std::min<std::size_t>(64 * 1024, value.size() - offset);
        reader_(position_ + offset, value.data() + offset, step); offset += step;
    }
    position_ += count; return value;
}
std::string Cursor::line() {
    std::string value;
    for (;;) { const auto next = byte(); if (next == '\n') return value; value.push_back(static_cast<char>(next)); }
}
void Cursor::skip(std::uint64_t count) { require(count); position_ += count; }
void Cursor::start_frame(std::uint64_t count) {
    if (framed_) invalid("nested pickle frame");
    require(count); frame_end_ = position_ + count; framed_ = true;
}
void Cursor::next_opcode() { if (framed_ && position_ == frame_end_) framed_ = false; }
void Cursor::finish_pickle() {
    // STOP returns even if a frame has unread bytes; the next independent
    // pickle begins at the actual consumed position.
    framed_ = false;
}

Node* Graph::make(Kind kind) {
    auto node = std::make_unique<Node>(); node->kind = kind; auto* pointer = node.get();
    nodes.push_back(std::move(node)); return pointer;
}
Node* Graph::integer(std::int64_t value) { auto* node = make(Kind::integer); node->text = decimal_integer(std::to_string(value)); return node; }
Node* Graph::string(std::string value) { auto* node = make(Kind::string); node->text = std::move(value); return node; }
Node* Graph::global(std::string module, std::string name) { auto* node = make(Kind::global); node->text = std::move(module); node->name = std::move(name); return node; }
Node* Graph::tuple(std::vector<Node*> values) { auto* node = make(Kind::tuple); node->children = std::move(values); return node; }

Node* Graph::parse(Cursor& input) {
    std::vector<Node*> stack;
    std::vector<std::size_t> marks;
    std::unordered_map<std::uint64_t, Node*> memo;
    auto pop = [&]() { if (stack.empty() || !stack.back()) invalid("empty value stack"); auto* node = stack.back(); stack.pop_back(); return node; };
    auto top = [&]() { if (stack.empty() || !stack.back()) invalid("empty value stack"); return stack.back(); };
    auto marked = [&]() {
        if (marks.empty()) invalid("missing MARK");
        const auto offset = marks.back(); marks.pop_back();
        if (offset >= stack.size() || stack[offset]) invalid("bad MARK stack");
        std::vector<Node*> values(stack.begin() + static_cast<std::ptrdiff_t>(offset + 1), stack.end());
        if (std::find(values.begin(), values.end(), nullptr) != values.end()) invalid("unclosed MARK");
        stack.resize(offset); return values;
    };
    auto number = [&](std::string bytes) { auto* node = make(Kind::integer); normalize_integer(bytes); node->text = std::move(bytes); stack.push_back(node); };
    auto reduce = [&](Kind kind, Node* function, Node* arguments) {
        if (arguments->kind != Kind::tuple) invalid("constructor arguments are not a tuple");
        if (is_global(function, "collections", "OrderedDict") || is_global(function, "builtins", "dict") ||
            is_global(function, "__builtin__", "dict")) {
            auto* node = make(Kind::dictionary);
            if (arguments->children.size() > 1) invalid("dictionary constructor arguments");
            if (!arguments->children.empty()) {
                const auto* values = arguments->children.front();
                if (values->kind == Kind::dictionary) node->children = values->children;
                else for (const auto* pair : sequence(values)) {
                    const auto& entries = sequence(pair); if (entries.size() != 2) invalid("dictionary pair length");
                    node->children.insert(node->children.end(), entries.begin(), entries.end());
                }
            }
            stack.push_back(node);
        } else if (is_global(function, "_codecs", "encode") && arguments->children.size() >= 2) {
            const auto& codec = string_value(arguments->children[1]);
            if (codec == "latin1" || codec == "latin-1" || codec == "ascii" || codec == "utf-8" || codec == "utf8") {
                auto* node = make(Kind::bytes);
                const auto& value = string_value(arguments->children[0]);
                node->text = codec == "utf-8" || codec == "utf8" ? value : latin1_bytes(value, codec == "ascii");
                stack.push_back(node);
            } else { auto* node = make(kind); node->children = {function, arguments}; stack.push_back(node); }
        } else { auto* node = make(kind); node->children = {function, arguments}; stack.push_back(node); }
    };
    for (;;) {
        ProcessInterrupt::check(); input.next_opcode();
        const auto opcode = input.byte();
        switch (opcode) {
            case 0x80: if (input.byte() > 5) invalid("unsupported pickle protocol"); break;
            case 0x95: { const auto count = input.little(8); input.start_frame(count); break; }
            case '.': {
                if (stack.size() != 1 || !stack.back() || !marks.empty()) invalid("STOP stack");
                input.finish_pickle(); return stack.back();
            }
            case '(': marks.push_back(stack.size()); stack.push_back(nullptr); break;
            case '0': if (stack.empty()) invalid("POP stack"); if (!stack.back()) { if (marks.empty()) invalid("POP MARK"); marks.pop_back(); } stack.pop_back(); break;
            case '1': static_cast<void>(marked()); break;
            case '2': stack.push_back(top()); break;
            case 'N': stack.push_back(make(Kind::none)); break;
            case 0x88: case 0x89: { auto* node = make(Kind::boolean); node->boolean = opcode == 0x88; stack.push_back(node); break; }
            case 'I': case 'L': {
                auto text = input.line();
                if (opcode == 'I' && (text == "00" || text == "01")) {
                    auto* node = make(Kind::boolean); node->boolean = text == "01"; stack.push_back(node);
                } else { if (opcode == 'L' && !text.empty() && text.back() == 'L') text.pop_back(); number(decimal_integer(text)); }
                break;
            }
            case 'J': { const auto bits = static_cast<std::uint32_t>(input.little(4)); std::int32_t signed_value; std::memcpy(&signed_value, &bits, 4); stack.push_back(integer(signed_value)); break; }
            case 'K': stack.push_back(integer(input.byte())); break;
            case 'M': stack.push_back(integer(static_cast<std::int64_t>(input.little(2)))); break;
            case 0x8a: { const auto count = input.byte(); number(input.bytes(count)); break; }
            case 0x8b: { const auto count = input.little(4); number(input.bytes(count)); break; }
            case 'F': { auto* node = make(Kind::floating); node->floating = python_float(input.line()); stack.push_back(node); break; }
            case 'G': {
                std::uint64_t bits = 0; for (int i = 0; i < 8; ++i) bits = (bits << 8) | input.byte();
                auto* node = make(Kind::floating); std::memcpy(&node->floating, &bits, 8); stack.push_back(node); break;
            }
            case 'S': stack.push_back(string(string_literal(input.line()))); break;
            case 'V': stack.push_back(string(unicode_literal(input.line()))); break;
            case 'U': case 'T': case 'X': case 'B': case 'C': case 0x8c: case 0x8d: case 0x8e: case 0x96: {
                const auto width = opcode == 'U' || opcode == 'C' || opcode == 0x8c ? 1 : opcode == 0x8d || opcode == 0x8e || opcode == 0x96 ? 8 : 4;
                const auto count = input.little(width);
                auto* node = make(opcode == 'B' || opcode == 'C' || opcode == 0x8e || opcode == 0x96 ? Kind::bytes : Kind::string);
                node->text = input.bytes(count); stack.push_back(node); break;
            }
            case ']': stack.push_back(make(Kind::list)); break;
            case ')': stack.push_back(make(Kind::tuple)); break;
            case '}': stack.push_back(make(Kind::dictionary)); break;
            case 'l': case 't': case 'd': {
                auto values = marked(); if (opcode == 'd' && values.size() % 2) invalid("dictionary items");
                auto* node = make(opcode == 'l' ? Kind::list : opcode == 't' ? Kind::tuple : Kind::dictionary);
                node->children = std::move(values); stack.push_back(node); break;
            }
            case 0x85: case 0x86: case 0x87: {
                const auto count = static_cast<std::size_t>(opcode - 0x84);
                if (stack.size() < count) invalid("short tuple stack");
                std::vector<Node*> values(stack.end() - static_cast<std::ptrdiff_t>(count), stack.end());
                if (std::find(values.begin(), values.end(), nullptr) != values.end()) invalid("tuple MARK");
                stack.resize(stack.size() - count); stack.push_back(tuple(std::move(values))); break;
            }
            case 'a': { auto* value = pop(); auto* node = top(); if (node->kind != Kind::list) invalid("APPEND target"); node->children.push_back(value); break; }
            case 'e': {
                auto values = marked(); auto* node = top(); if (node->kind != Kind::list) invalid("APPENDS target");
                node->children.insert(node->children.end(), values.begin(), values.end()); break;
            }
            case 's': {
                auto* value = pop(); auto* key = pop(); auto* node = top(); if (node->kind != Kind::dictionary) invalid("SETITEM target");
                node->children.push_back(key); node->children.push_back(value); break;
            }
            case 'u': {
                auto values = marked(); auto* node = top(); if (node->kind != Kind::dictionary || values.size() % 2) invalid("SETITEMS target");
                node->children.insert(node->children.end(), values.begin(), values.end()); break;
            }
            case 'q': case 'r': case 'p': {
                const auto index = opcode == 'q' ? input.little(1) : opcode == 'r' ? input.little(4) : memo_index(input.line());
                memo[index] = top(); break;
            }
            case 0x94: { auto* node = top(); memo[static_cast<std::uint64_t>(memo.size())] = node; break; }
            case 'h': case 'j': case 'g': {
                const auto index = opcode == 'h' ? input.little(1) : opcode == 'j' ? input.little(4) : memo_index(input.line());
                const auto found = memo.find(index); if (found == memo.end()) invalid("missing memo"); stack.push_back(found->second); break;
            }
            case 'c': { auto module = input.line(); auto name = input.line(); stack.push_back(global(std::move(module), std::move(name))); break; }
            case 0x93: { auto* name = pop(); auto* module = pop(); stack.push_back(global(string_value(module), string_value(name))); break; }
            case 'R': case 0x81: { auto* arguments = pop(); auto* function = pop(); reduce(opcode == 'R' ? Kind::reduce : Kind::new_object, function, arguments); break; }
            case 0x92: {
                auto* keywords = pop(); auto* arguments = pop(); auto* function = pop();
                if (keywords->kind != Kind::dictionary || !keywords->children.empty()) invalid("NEWOBJ_EX keyword arguments are not supported");
                reduce(Kind::new_object, function, arguments); break;
            }
            case 'b': { auto* state = pop(); top()->state.push_back(state); break; }
            case 'Q': { auto* saved = pop(); auto* node = make(Kind::persistent); node->children = {saved}; stack.push_back(node); break; }
            case 'P': { auto* saved = string(input.line()); auto* node = make(Kind::persistent); node->children = {saved}; stack.push_back(node); break; }
            default: invalid("unsupported opcode or out-of-band buffer");
        }
    }
}

std::int64_t integer_value(const Node* node) {
    if (node->kind == Kind::boolean) return node->boolean ? 1 : 0;
    if (node->kind != Kind::integer || node->text.size() > 8) invalid("integer exceeds int64");
    std::uint64_t bits = static_cast<unsigned char>(node->text.back()) & 128 ? std::numeric_limits<std::uint64_t>::max() : 0;
    for (std::size_t i = 0; i < node->text.size(); ++i) {
        bits &= ~(std::uint64_t(255) << (8 * i)); bits |= std::uint64_t(static_cast<unsigned char>(node->text[i])) << (8 * i);
    }
    std::int64_t value; std::memcpy(&value, &bits, 8); return value;
}
std::string integer_decimal(const Node* node) {
    if (node->kind == Kind::boolean) return node->boolean ? "1" : "0";
    if (node->kind != Kind::integer) invalid("expected integer");
    auto magnitude = node->text; const auto negative = (static_cast<unsigned char>(magnitude.back()) & 128) != 0;
    if (negative) {
        unsigned carry = 1;
        for (auto& byte : magnitude) { const auto next = (static_cast<unsigned char>(byte) ^ 255u) + carry; byte = static_cast<char>(next); carry = next >> 8; }
    }
    std::string decimal = "0";
    for (auto iterator = magnitude.rbegin(); iterator != magnitude.rend(); ++iterator) {
        unsigned carry = static_cast<unsigned char>(*iterator);
        for (auto digit = decimal.rbegin(); digit != decimal.rend(); ++digit) {
            const auto next = static_cast<unsigned>(*digit - '0') * 256 + carry; *digit = static_cast<char>('0' + next % 10); carry = next / 10;
        }
        while (carry) { decimal.insert(decimal.begin(), static_cast<char>('0' + carry % 10)); carry /= 10; }
    }
    if (negative && decimal != "0") decimal.insert(decimal.begin(), '-'); return decimal;
}
std::string key_value(const Node* node) {
    if (node->kind == Kind::string || node->kind == Kind::bytes) return node->text;
    return integer_decimal(node);
}
const std::string& string_value(const Node* node) {
    if (node->kind != Kind::string && node->kind != Kind::bytes) invalid("expected string");
    return node->text;
}
const std::vector<Node*>& sequence(const Node* node) {
    if (node->kind != Kind::tuple && node->kind != Kind::list) invalid("expected sequence"); return node->children;
}
bool is_global(const Node* node, const char* module, const char* name) { return node->kind == Kind::global && node->text == module && node->name == name; }
void check_tensor_geometry(const std::vector<std::int64_t>& size, const std::vector<std::int64_t>& stride,
    std::int64_t offset, std::uint64_t capacity) {
    if (size.size() != stride.size() || offset < 0) invalid("tensor dimensions or storage offset");
    bool empty = false;
    for (std::size_t i = 0; i < size.size(); ++i) { if (size[i] < 0 || stride[i] < 0) invalid("negative tensor geometry"); empty |= size[i] == 0; }
    if (empty) return;
    auto last = static_cast<std::uint64_t>(offset);
    for (std::size_t i = 0; i < size.size(); ++i) {
        const auto count = static_cast<std::uint64_t>(size[i] - 1), step = static_cast<std::uint64_t>(stride[i]);
        if (count && step > (std::numeric_limits<std::uint64_t>::max() - last) / count) invalid("tensor geometry overflow");
        last += count * step;
    }
    if (last >= capacity) invalid("tensor view exceeds its storage");
}

std::vector<char> Graph::emit(Node* root) const {
    std::vector<char> output{static_cast<char>(0x80), 2};
    std::unordered_map<const Node*, std::uint32_t> memo;
    const auto little = [&](std::uint64_t value, unsigned count) { for (unsigned i = 0; i < count; ++i) output.push_back(static_cast<char>(value >> (8 * i))); };
    const auto put = [&](const Node* node) {
        if (memo.size() >= std::numeric_limits<std::uint32_t>::max()) invalid("memo exceeds uint32");
        const auto index = static_cast<std::uint32_t>(memo.size()); memo.emplace(node, index);
        output.push_back('r'); little(index, 4);
    };
    const auto text = [&](const std::string& value) {
        if (value.size() > std::numeric_limits<std::uint32_t>::max()) invalid("string exceeds protocol 2");
        output.push_back('X'); little(value.size(), 4); output.insert(output.end(), value.begin(), value.end());
    };
    struct Work { enum class Action { node, memo, opcode, finish_tuple } action; const Node* value{}; char opcode{}; };
    std::vector<Work> work{{Work::Action::node, root}};
    while (!work.empty()) {
        ProcessInterrupt::check();
        const auto current = work.back(); work.pop_back();
        if (current.action == Work::Action::opcode) { output.push_back(current.opcode); continue; }
        if (current.action == Work::Action::memo) { put(current.value); continue; }
        if (current.action == Work::Action::finish_tuple) {
            // A tuple can be encountered again through a list/dict cycle.
            // LibTorch cannot consume the POP_MARK needed to reuse it.
            if (memo.count(current.value)) invalid("cyclic tuples are not supported by the IValue reader");
            output.push_back('t'); put(current.value);
            continue;
        }
        const auto* node = current.value;
        if (const auto found = memo.find(node); found != memo.end()) { output.push_back('j'); little(found->second, 4); continue; }
        const auto defer_children = [&](const std::vector<Node*>& children) {
            for (auto child = children.rbegin(); child != children.rend(); ++child) work.push_back({Work::Action::node, *child});
        };
        if (node->kind == Kind::list || node->kind == Kind::dictionary) {
            output.push_back(node->kind == Kind::list ? ']' : '}'); put(node);
            if (!node->children.empty()) { output.push_back('('); work.push_back({Work::Action::opcode, nullptr, node->kind == Kind::list ? 'e' : 'u'}); defer_children(node->children); }
            continue;  // OrderedDict Python attributes are not data entries.
        }
        for (auto state = node->state.rbegin(); state != node->state.rend(); ++state) {
            work.push_back({Work::Action::opcode, nullptr, 'b'}); work.push_back({Work::Action::node, *state});
        }
        if (node->kind != Kind::tuple || node->children.empty()) work.push_back({Work::Action::memo, node});
        switch (node->kind) {
            case Kind::none: output.push_back('N'); break;
            case Kind::boolean: output.push_back(static_cast<char>(node->boolean ? 0x88 : 0x89)); break;
            case Kind::integer:
                // LibTorch's protocol-2 reader accepts LONG1 only at width 8.
                output.push_back(static_cast<char>(0x8a)); output.push_back(8);
                little(static_cast<std::uint64_t>(integer_value(node)), 8); break;
            case Kind::floating: { std::uint64_t bits; std::memcpy(&bits, &node->floating, 8); output.push_back('G'); for (int i = 7; i >= 0; --i) output.push_back(static_cast<char>(bits >> (8 * i))); break; }
            case Kind::string: case Kind::bytes: text(node->text); break;
            case Kind::global:
                output.push_back('c'); output.insert(output.end(), node->text.begin(), node->text.end()); output.push_back('\n');
                output.insert(output.end(), node->name.begin(), node->name.end()); output.push_back('\n'); break;
            case Kind::tuple:
                if (node->children.empty()) output.push_back(')');
                else { output.push_back('('); work.push_back({Work::Action::finish_tuple, node}); defer_children(node->children); }
                break;
            case Kind::reduce: case Kind::new_object:
                work.push_back({Work::Action::opcode, nullptr, node->kind == Kind::reduce ? 'R' : static_cast<char>(0x81)}); defer_children(node->children); break;
            case Kind::persistent:
                work.push_back({Work::Action::opcode, nullptr, 'Q'}); defer_children(node->children); break;
            case Kind::tensor:
                output.push_back('c');
                { const std::string name = "torch.jit._pickle\nbuild_tensor_from_id\n"; output.insert(output.end(), name.begin(), name.end()); }
                output.push_back('('); output.push_back(static_cast<char>(0x8a)); output.push_back(8); little(static_cast<std::uint64_t>(node->tensor_index), 8);
                output.push_back('t'); output.push_back('R'); break;
            default: invalid("unhandled emission kind");
        }
    }
    output.push_back('.'); return output;
}
}  // namespace uf::pickle_detail
