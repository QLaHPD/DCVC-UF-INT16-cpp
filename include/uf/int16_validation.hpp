#pragma once
#include <filesystem>
#include <iosfwd>

namespace uf {
// Native counterpart of scripts/check_uf_int16.py. Calling this command
// deliberately loads models and performs inference; porting it does not run it.
int run_int16_validation_command(int argc, char** argv,
                                 const std::filesystem::path& checkpoint_directory,
                                 std::ostream& output);
}  // namespace uf
