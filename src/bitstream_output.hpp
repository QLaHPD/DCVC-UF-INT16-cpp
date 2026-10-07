#pragma once

#include <filesystem>
#include <iosfwd>
#include <memory>

namespace uf {

// Python Codec.encode opens with 'xb'. Keep the exclusive descriptor through
// buffered writes, flush, fsync and close; an existing file is never truncated.
class BitstreamOutput {
public:
    explicit BitstreamOutput(const std::filesystem::path& path);
    ~BitstreamOutput();
    BitstreamOutput(const BitstreamOutput&) = delete;
    BitstreamOutput& operator=(const BitstreamOutput&) = delete;
    std::ostream& stream();
    void finish();
    // Flush and close on exceptional exit without fsync. A close error may
    // replace a flush/body error; repeated calls after closure are harmless.
    void close_checked();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace uf
