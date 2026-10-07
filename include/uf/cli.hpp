#pragma once

#include <cstdint>
#include <exception>
#include <iosfwd>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace uf::cli {

using Options = std::unordered_map<std::string, std::string>;

class ParseError : public std::invalid_argument {
public:
    using std::invalid_argument::invalid_argument;
};

class HelpRequested : public std::exception {
public:
    const char* what() const noexcept override { return "Command help requested"; }
};

// Archive, training and evaluation commands use Python-compatible aliases, repetitions, attached
// values, nargs and booleans. List values preserve each argv item, including
// empty strings and embedded newlines. DCVC/TCM switches also retain optional
// values (a bare switch means true); TCM and HEM reject unknown boolean words.
// Private training commands declare rank
// options separately, keeping them out of the public schemas.
// Maintained root commands apply semantic checks after parsing accepts all
// tokens; encode/stream defer theirs to the explicit post-preset boundary.
Options parse(int argc, char** argv);
// Root encode/stream semantic checks run after target-PSNR selection and
// channel expansion. Parsing retains conversion, arity, groups and required
// arguments; this boundary reports subsequent failures as ParseError.
void validate_encode_options(const Options& values);
// Help includes maintained option descriptions and required-argument labels.
// A failed output stream raises; no additional flush is performed.
void help(std::ostream& output, const std::string& command);
std::vector<std::string> list(const Options& values, const std::string& option);
void set_list(Options& values, const std::string& option, const std::vector<std::string>& items);

// UTF-8 Python decimal syntax, with Unicode decimal digits/spaces and digit
// separators. Floating conversion is independent of the process locale and
// allows IEEE overflow/underflow, as Python float() does. Range checks belong
// to each option. Integer helpers enforce the native destination's width.
int integer(const std::string& value, const std::string& option);
std::int64_t signed_integer(const std::string& value, const std::string& option);
std::uint64_t unsigned_integer(const std::string& value, const std::string& option);
double number(const std::string& value, const std::string& option);

}  // namespace uf::cli
