#include "bitstream_output.hpp"

#include "uf/process.hpp"

#include <cerrno>
#include <cstring>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <streambuf>
#include <string>
#include <system_error>
#include <vector>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace uf {
namespace {

int open_output(const std::filesystem::path& path) {
    if (path.native().find('\0') != std::string::npos)
        throw std::invalid_argument("embedded null byte");
    int descriptor;
    do {
        descriptor = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0666);
        if (descriptor < 0 && errno == EINTR) ProcessInterrupt::check();
    } while (descriptor < 0 && errno == EINTR);
    if (descriptor < 0) throw std::system_error(errno, std::generic_category(),
        "Cannot exclusively create UF bitstream: " + path.string());
    return descriptor;
}

std::size_t output_buffer_size(int descriptor) {
    struct stat status{};
    if (::fstat(descriptor, &status) < 0) {
        if (errno == EBADF)
            throw std::system_error(errno, std::generic_category(), "Cannot stat UF bitstream");
        return 8192;
    }
    if (S_ISDIR(status.st_mode))
        throw std::system_error(EISDIR, std::generic_category(), "UF bitstream is a directory");
    if (status.st_blksize <= 1) return 8192;
    // CPython 3.10 passes its selected buffering size through a C int.
    if (status.st_blksize > std::numeric_limits<int>::max())
        throw std::overflow_error("UF bitstream buffer exceeds the native size interface");
    return static_cast<std::size_t>(status.st_blksize);
}

class FileBuffer : public std::streambuf {
public:
    explicit FileBuffer(FileDescriptor& descriptor) : descriptor_(descriptor) {}
    void initialize() {
        bytes_.resize(output_buffer_size(descriptor_.get()));
        setp(bytes_.data(), bytes_.data() + bytes_.size());
    }
    void flush_for_close() { flush_buffer(false); }
protected:
    int sync() override {
        flush_buffer(true);
        return 0;
    }
    int_type overflow(int_type value) override {
        sync();
        if (!traits_type::eq_int_type(value, traits_type::eof())) {
            *pptr() = traits_type::to_char_type(value);
            pbump(1);
        }
        return traits_type::not_eof(value);
    }
    std::streamsize xsputn(const char* source, std::streamsize size) override {
        if (size <= 0) return 0;
        const auto available = static_cast<std::streamsize>(epptr() - pptr());
        if (size <= available) {
            std::memcpy(pptr(), source, static_cast<std::size_t>(size));
            pbump(static_cast<int>(size));
            return size;
        }
        // BufferedWriter flushes old bytes before consuming an input that
        // does not fit. A full buffer remains pending until the next write.
        sync();
        std::streamsize written = 0;
        while (size - written > static_cast<std::streamsize>(bytes_.size())) {
            const auto count = write_once(source + written,
                static_cast<std::size_t>(size - written), true);
            written += static_cast<std::streamsize>(count);
            ProcessInterrupt::check();
        }
        const auto remaining = static_cast<std::size_t>(size - written);
        if (remaining) std::memcpy(pptr(), source + written, remaining);
        pbump(static_cast<int>(remaining));
        return size;
    }
private:
    std::size_t write_once(const char* source, std::size_t size, bool interruptible) {
        ssize_t count;
        do {
            if (interruptible) ProcessInterrupt::check();
            count = ::write(descriptor_.get(), source, size);
        } while (count < 0 && errno == EINTR);
        if (count < 0) throw std::system_error(errno, std::generic_category(), "Cannot write UF bitstream");
        if (count == 0) throw std::runtime_error("Cannot write UF bitstream: zero-byte write");
        return static_cast<std::size_t>(count);
    }
    void flush_buffer(bool interruptible) {
        const auto size = static_cast<std::size_t>(pptr() - pbase());
        std::size_t written = 0;
        try {
            while (written < size) {
                written += write_once(bytes_.data() + written, size - written, interruptible);
                if (interruptible) ProcessInterrupt::check();
            }
        } catch (...) {
            // Keep only the pending suffix after partial success. A close
            // flush must not replay bytes already accepted by the file.
            const auto remaining = size - written;
            if (written && remaining) std::memmove(bytes_.data(), bytes_.data() + written, remaining);
            setp(bytes_.data(), bytes_.data() + bytes_.size());
            pbump(static_cast<int>(remaining));
            throw;
        }
        setp(bytes_.data(), bytes_.data() + bytes_.size());
    }
    FileDescriptor& descriptor_;
    std::vector<char> bytes_;
};

}  // namespace

struct BitstreamOutput::Impl {
    FileDescriptor descriptor;
    FileBuffer buffer;
    std::ostream output;
    explicit Impl(const std::filesystem::path& path)
        : descriptor(open_output(path)),
          buffer(descriptor), output(&buffer) {
        try {
            buffer.initialize();
            output.exceptions(std::ios::badbit | std::ios::failbit);
        } catch (...) {
            descriptor.close_checked("Cannot close UF bitstream");
            throw;
        }
    }
    ~Impl() {
        // Fallback only; codec callers explicitly close so errors propagate.
        try { close_checked(); } catch (...) {}
    }
    void close_checked() {
        if (descriptor.get() < 0) return;
        try {
            // Buffered close retries pending bytes even after a write error.
            // Cleanup does not reject an already latched foreground interrupt.
            buffer.flush_for_close();
        } catch (...) {
            descriptor.close_checked("Cannot close UF bitstream");
            throw;
        }
        descriptor.close_checked("Cannot close UF bitstream");
    }
};

BitstreamOutput::BitstreamOutput(const std::filesystem::path& path)
    : impl_(std::make_unique<Impl>(path)) {}

BitstreamOutput::~BitstreamOutput() = default;

std::ostream& BitstreamOutput::stream() {
    if (impl_->descriptor.get() < 0) throw std::logic_error("UF bitstream output is closed");
    return impl_->output;
}

void BitstreamOutput::finish() {
    if (impl_->descriptor.get() < 0) return;
    ProcessInterrupt::check();
    impl_->output.flush();
    for (;;) {
        if (::fsync(impl_->descriptor.get()) == 0) break;
        if (errno != EINTR) throw std::runtime_error("Cannot sync UF bitstream");
        ProcessInterrupt::check();
    }
    impl_->descriptor.close_checked("Cannot close UF bitstream");
}

void BitstreamOutput::close_checked() { impl_->close_checked(); }

}  // namespace uf
