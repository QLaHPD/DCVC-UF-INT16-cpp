#pragma once

#include <torch/torch.h>

#include <filesystem>
#include <vector>

namespace uf {

// torch.save ZIP archives, legacy streams and historical uncompressed tar
// checkpoints containing tensors and ordinary Python containers. OrderedDict
// entries become insertion-ordered IValue dictionaries; Python attributes and
// custom classes are outside this state-dictionary interface. All storage is
// mapped to CPU before materializing tensors. See CHECKPOINTS.md for scope.
torch::IValue load_torch_pickle_cpu(const std::filesystem::path& path);
std::vector<char> torch_pickle_bytes_cpu(const torch::IValue& value);
// Stream a CPU ZIP archive to a borrowed descriptor for a new empty file.
// The caller owns syncing, closing and publishing that file. This avoids
// retaining an additional full archive byte vector during file saves.
void write_torch_pickle_cpu_fd(int fd, const torch::IValue& value);
// Uses protocol 2 without JIT container tags, for Python weights_only=True
// readers. Tensor data is saved on CPU; filename output directly truncates,
// like maintained torch.save. ASCII names use StreamWriter's filename API;
// other names use an owned POSIX descriptor. Parent creation belongs to caller.
// Retains repeated objects and strided storage views; cyclic writer containers
// are rejected because the underlying JIT Pickler cannot encode them.
// Allocated source storage shared across different dtypes is rejected.
void save_torch_pickle_cpu(const std::filesystem::path& path, const torch::IValue& value);

}  // namespace uf
