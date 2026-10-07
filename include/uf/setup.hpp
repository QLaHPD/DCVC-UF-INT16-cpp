#pragma once

#include <filesystem>
#include <iosfwd>

namespace uf {

// Inspect linked backends and list checkpoints without opening a model,
// allocating tensors, or running an encode/decode operation.
// Report writes check stream state before later inspection phases and return;
// output errors propagate without adding a final flush.
int report_setup(std::ostream& output, const std::filesystem::path& checkpoints,
                 bool require_native = false, bool require_int16_native = false);

}  // namespace uf
