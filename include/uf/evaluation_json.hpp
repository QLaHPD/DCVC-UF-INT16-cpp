// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#pragma once

#include <nlohmann/json_fwd.hpp>
#include <filesystem>
#include <iosfwd>

namespace uf::evaluation {

using Json = nlohmann::ordered_json;

// Shared Python numeric parsing preserves exact integers, NaN/Infinity,
// binary64 overflow/underflow and object insertion order. Lone surrogate
// codepoints use the shared WTF-8 representation. Nonnegative float_digits
// follows the maintained custom .Nf formatter, including lowercase nonfinite tokens.
Json read_json(std::istream& input);
// Text file counterpart: locale decoding, universal newlines, numeric parsing
// and checked close follow the shared json.load-style owned file reader.
Json read_json_file(const std::filesystem::path& path);
void write_json(const Json& value, std::ostream& output, int float_digits = -1, int indent = 2);
// Owned report output, with checked close on both success and failure.
// Parent directory creation remains the caller's responsibility.
void write_json_file(const Json& value, const std::filesystem::path& path,
                     int float_digits = -1, int indent = 2);

}  // namespace uf::evaluation
