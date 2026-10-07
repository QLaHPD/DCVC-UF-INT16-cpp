// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "training_loader_process.hpp"
#include "uf/training_loader.hpp"
#include "uf/process.hpp"
#include "uf/python_json.hpp"
#include "pareto_random.hpp"

#include <ATen/CPUGeneratorImpl.h>
#include <c10/util/thread_name.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cerrno>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <limits>
#include <mutex>
#include <optional>
#include <poll.h>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>

namespace uf::training {
namespace {

using Json = nlohmann::json;
using Check = std::function<void()>;
constexpr int worker_socket = 3;
constexpr std::uint32_t wire_magic = 0x55464c44;
constexpr std::uint32_t wire_version = 1;
constexpr std::size_t tensor_alignment = 64;

struct WireHeader {
    std::uint32_t magic = wire_magic, version = wire_version;
    std::uint64_t metadata_bytes = 0, tensor_bytes = 0;
};
static_assert(sizeof(WireHeader) == 24, "Training loader wire header layout");

std::runtime_error system_error(const char* operation) {
    return std::runtime_error(std::string(operation) + ": " + std::strerror(errno));
}

std::size_t checked_size(std::uint64_t size) {
    if (!size || size > std::numeric_limits<std::size_t>::max() ||
        size > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max())) {
        throw std::overflow_error("Training worker shared storage size is invalid");
    }
    return static_cast<std::size_t>(size);
}

std::size_t align_tensor(std::size_t offset) {
    if (offset > std::numeric_limits<std::size_t>::max() - (tensor_alignment - 1)) {
        throw std::overflow_error("Training shared tensor alignment overflow");
    }
    return (offset + tensor_alignment - 1) & ~(tensor_alignment - 1);
}

std::size_t tensor_bytes(const std::vector<std::int64_t>& sizes) {
    bool empty = false;
    for (const auto size : sizes) {
        if (size < 0) throw std::invalid_argument("Negative shared tensor dimension");
        empty = empty || size == 0;
    }
    if (empty) return 0;
    std::uint64_t count = 1;
    for (const auto size : sizes) {
        if (static_cast<std::uint64_t>(size) > static_cast<std::uint64_t>(
                std::numeric_limits<std::int64_t>::max()) / count) {
            throw std::overflow_error("Training shared tensor element count overflow");
        }
        count *= static_cast<std::uint64_t>(size);
    }
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(std::uint32_t)) {
        throw std::overflow_error("Training shared tensor byte count overflow");
    }
    return static_cast<std::size_t>(count) * sizeof(std::uint32_t);
}

FileDescriptor memory_file(const char* name, std::size_t bytes) {
    checked_size(bytes);
    FileDescriptor file(memfd_create(name, MFD_CLOEXEC | MFD_ALLOW_SEALING));
    if (file.get() < 0) throw system_error("Cannot create training shared storage");
    if (ftruncate(file.get(), static_cast<off_t>(bytes)) != 0) {
        throw system_error("Cannot size training shared storage");
    }
    return file;
}

void verify_file(int descriptor, std::size_t bytes) {
    struct stat status {};
    if (fstat(descriptor, &status) != 0) throw system_error("Cannot inspect training shared storage");
    if (!S_ISREG(status.st_mode) || status.st_size < 0 ||
        static_cast<std::uint64_t>(status.st_size) != bytes) {
        throw std::runtime_error("Training shared storage size does not match its message");
    }
    const int seals = fcntl(descriptor, F_GET_SEALS);
    if (seals < 0 || (seals & (F_SEAL_GROW | F_SEAL_SHRINK)) != (F_SEAL_GROW | F_SEAL_SHRINK)) {
        throw std::runtime_error("Training shared storage is not protected against resizing");
    }
}

struct Mapping {
    FileDescriptor file;
    const std::size_t size;
    void* data = MAP_FAILED;
    Mapping(FileDescriptor descriptor, std::size_t bytes, bool writable)
        : file(std::move(descriptor)), size(bytes) {
        verify_file(file.get(), size);
        if (!writable) {
            // Jetson rejects MAP_SHARED on a write-sealed memfd opened
            // O_RDWR even when the requested protection is only PROT_READ.
            // Reopen our owned descriptor read-only without relaxing seals
            // or copying a potentially large metadata document.
            const auto path = "/proc/self/fd/" + std::to_string(file.get());
            FileDescriptor readonly(open(path.c_str(), O_RDONLY | O_CLOEXEC));
            if (readonly.get() < 0) throw system_error("Cannot open read-only training metadata");
            verify_file(readonly.get(), size);
            file = std::move(readonly);
        }
        data = mmap(nullptr, size, writable ? PROT_READ | PROT_WRITE : PROT_READ,
                    MAP_SHARED, file.get(), 0);
        if (data == MAP_FAILED) throw system_error("Cannot map training shared storage");
    }
    ~Mapping() { if (data != MAP_FAILED) munmap(data, size); }
    Mapping(const Mapping&) = delete;
    Mapping& operator=(const Mapping&) = delete;
};

void wait_socket(int socket, short events, const Check& check) {
    for (;;) {
        check();
        struct pollfd descriptor {socket, events, 0};
        const int result = poll(&descriptor, 1, 100);
        if (result < 0 && errno == EINTR) continue;
        if (result < 0) throw system_error("Cannot poll training worker socket");
        if (!result) continue;
        if (descriptor.revents & POLLNVAL) throw std::runtime_error("Training worker socket is invalid");
        if (descriptor.revents & (events | POLLERR | POLLHUP)) return;
    }
}

// Every control document lives in a sealed memfd. The socket packet is small
// even for a dataset with millions of paths or a large list of sample indices.
// SCM_RIGHTS transfers ownership independently of the sender's mapping.
void send_message(int socket, const Json& value, const Check& check,
                  int tensor_file = -1, std::size_t tensor_size = 0) {
    check();
    const auto serialized = python_compact_json(value);
    auto metadata = memory_file("uf_training_metadata", serialized.size());
    write_descriptor(metadata.get(), serialized.data(), serialized.size());
    if (fcntl(metadata.get(), F_ADD_SEALS, F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_WRITE | F_SEAL_SEAL) < 0) {
        throw system_error("Cannot seal training metadata");
    }
    if ((tensor_file >= 0) != (tensor_size != 0)) throw std::invalid_argument("Invalid training tensor descriptor");
    WireHeader header;
    header.metadata_bytes = serialized.size();
    header.tensor_bytes = tensor_size;
    std::array<int, 2> files{metadata.get(), tensor_file};
    const auto count = tensor_file < 0 ? 1u : 2u;
    struct iovec bytes {&header, sizeof(header)};
    alignas(struct cmsghdr) std::array<unsigned char, CMSG_SPACE(2 * sizeof(int))> control{};
    struct msghdr message {};
    message.msg_iov = &bytes;
    message.msg_iovlen = 1;
    message.msg_control = control.data();
    message.msg_controllen = CMSG_SPACE(count * sizeof(int));
    auto* rights = CMSG_FIRSTHDR(&message);
    rights->cmsg_level = SOL_SOCKET;
    rights->cmsg_type = SCM_RIGHTS;
    rights->cmsg_len = CMSG_LEN(count * sizeof(int));
    std::memcpy(CMSG_DATA(rights), files.data(), count * sizeof(int));
    for (;;) {
        wait_socket(socket, POLLOUT, check);
        const auto written = sendmsg(socket, &message, MSG_NOSIGNAL | MSG_DONTWAIT);
        if (written < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) continue;
        if (written < 0) throw system_error("Cannot send training worker message");
        if (written != sizeof(header)) throw std::runtime_error("Incomplete training worker socket packet");
        return;
    }
}

struct Message {
    Json value;
    FileDescriptor tensor_file;
    std::size_t tensor_size = 0;
};

std::optional<Message> receive_message(int socket, const Check& check) {
    for (;;) {
        wait_socket(socket, POLLIN, check);
        WireHeader header;
        struct iovec bytes {&header, sizeof(header)};
        alignas(struct cmsghdr) std::array<unsigned char, CMSG_SPACE(2 * sizeof(int))> control{};
        struct msghdr message {};
        message.msg_iov = &bytes;
        message.msg_iovlen = 1;
        message.msg_control = control.data();
        message.msg_controllen = control.size();
        std::vector<FileDescriptor> files;
        // Reserve before recvmsg: an allocation failure must not leak newly
        // installed SCM_RIGHTS descriptors.
        files.reserve(2);
        const auto received = recvmsg(socket, &message, MSG_DONTWAIT | MSG_CMSG_CLOEXEC);
        if (received < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) continue;
        if (received < 0) throw system_error("Cannot receive training worker message");
        // Own every received descriptor before reporting malformed metadata.
        for (auto* rights = CMSG_FIRSTHDR(&message); rights; rights = CMSG_NXTHDR(&message, rights)) {
            if (rights->cmsg_level == SOL_SOCKET && rights->cmsg_type == SCM_RIGHTS &&
                rights->cmsg_len >= CMSG_LEN(0)) {
                const auto count = (rights->cmsg_len - CMSG_LEN(0)) / sizeof(int);
                for (std::size_t index = 0; index < count; ++index) {
                    int file;
                    std::memcpy(&file, CMSG_DATA(rights) + index * sizeof(int), sizeof(file));
                    files.emplace_back(file);
                }
            }
        }
        if (!received) return std::nullopt;
        if (received != sizeof(header) || (message.msg_flags & (MSG_TRUNC | MSG_CTRUNC)) ||
            header.magic != wire_magic || header.version != wire_version ||
            files.size() != (header.tensor_bytes ? 2u : 1u)) {
            throw std::runtime_error("Invalid training worker socket packet");
        }
        const auto metadata_size = checked_size(header.metadata_bytes);
        Mapping metadata(std::move(files.front()), metadata_size, false);
        const auto* begin = static_cast<const char*>(metadata.data);
        Message result;
        result.value = parse_python_json(std::string_view(begin, metadata_size));
        if (header.tensor_bytes) {
            result.tensor_size = checked_size(header.tensor_bytes);
            result.tensor_file = std::move(files.at(1));
        }
        return result;
    }
}

struct Field {
    std::vector<std::int64_t> sizes;
    torch::ScalarType dtype;
    std::size_t offset, bytes;
};

const char* dtype_name(torch::ScalarType dtype) {
    if (dtype == torch::kFloat32) return "float32";
    if (dtype == torch::kInt32) return "int32";
    throw std::invalid_argument("Training shared tensors require float32 or int32");
}

torch::ScalarType parse_dtype(const Json& value) {
    if (value == "float32") return torch::kFloat32;
    if (value == "int32") return torch::kInt32;
    throw std::invalid_argument("Invalid training shared tensor dtype");
}

torch::Tensor mapped_tensor(const std::shared_ptr<Mapping>& mapping, const Field& field) {
    return torch::from_blob(static_cast<unsigned char*>(mapping->data) + field.offset, field.sizes,
        [owner = mapping](void*) {}, torch::TensorOptions().dtype(field.dtype).device(torch::kCPU));
}

struct SharedBatch {
    Json fields;
    std::shared_ptr<Mapping> storage;
    DatasetSample tensors;
};

SharedBatch collate_shared(const std::vector<DatasetSample>& samples) {
    if (samples.empty() || samples.front().empty() ||
        samples.size() > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
        throw std::invalid_argument("Invalid training batch for shared collation");
    }
    const auto count = samples.front().size();
    for (const auto& sample : samples) {
        if (sample.size() != count) throw std::invalid_argument("Training samples have different list lengths");
    }
    std::vector<Field> fields;
    fields.reserve(count);
    std::size_t size = 0;
    SharedBatch batch{Json::array(), {}, {}};
    for (std::size_t index = 0; index < count; ++index) {
        ProcessInterrupt::check();
        const auto& first = samples.front().at(index);
        if (!first.defined() || first.device() != torch::Device(torch::kCPU) ||
            first.layout() != torch::kStrided || first.requires_grad()) {
            throw std::invalid_argument("Training workers collate ordinary CPU tensors without gradients");
        }
        const auto dtype = dtype_name(first.scalar_type());
        for (const auto& sample : samples) {
            const auto& tensor = sample.at(index);
            if (!tensor.defined() || tensor.sizes() != first.sizes() ||
                tensor.scalar_type() != first.scalar_type() || tensor.device() != first.device() ||
                tensor.layout() != first.layout() || tensor.requires_grad()) {
                throw std::invalid_argument("Training batch tensors have incompatible shapes or types");
            }
        }
        auto sizes = first.sizes().vec();
        sizes.insert(sizes.begin(), static_cast<std::int64_t>(samples.size()));
        const auto bytes = tensor_bytes(sizes);
        size = align_tensor(size);
        if (bytes > std::numeric_limits<std::size_t>::max() - size) {
            throw std::overflow_error("Training shared batch size overflow");
        }
        fields.push_back({std::move(sizes), first.scalar_type(), size, bytes});
        batch.fields.push_back({{"sizes", fields.back().sizes}, {"dtype", dtype}, {"offset", size}});
        size += bytes;
    }
    size = checked_size(std::max<std::size_t>(size, 1));
    auto file = memory_file("uf_training_batch", size);
    if (fcntl(file.get(), F_ADD_SEALS, F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL) < 0) {
        throw system_error("Cannot protect training shared batch");
    }
    batch.storage = std::make_shared<Mapping>(std::move(file), size, true);
    batch.tensors.reserve(count);
    std::vector<torch::Tensor> values;
    values.reserve(samples.size());
    for (std::size_t index = 0; index < count; ++index) {
        ProcessInterrupt::check();
        auto output = mapped_tensor(batch.storage, fields[index]);
        values.clear();
        for (const auto& sample : samples) values.push_back(sample[index]);
        // As in Python's worker default_collate, stack into shared storage
        // directly; no second full-batch allocation or socket copy is needed.
        torch::stack_out(output, values, 0);
        batch.tensors.push_back(std::move(output));
    }
    return batch;
}

DatasetSample receive_tensors(Message& message) {
    if (message.tensor_file.get() < 0 || !message.tensor_size) {
        throw std::runtime_error("Training batch has no shared tensor storage");
    }
    const auto& values = message.value.at("fields");
    if (!values.is_array() || values.empty()) throw std::runtime_error("Training batch has no tensor fields");
    std::vector<Field> fields;
    fields.reserve(values.size());
    std::size_t size = 0;
    for (const auto& value : values) {
        auto sizes = value.at("sizes").get<std::vector<std::int64_t>>();
        const auto bytes = tensor_bytes(sizes);
        size = align_tensor(size);
        if (value.at("offset").get<std::uint64_t>() != size || size > message.tensor_size ||
            bytes > message.tensor_size - size) {
            throw std::runtime_error("Training shared tensor exceeds its storage");
        }
        fields.push_back({std::move(sizes), parse_dtype(value.at("dtype")), size, bytes});
        size += bytes;
    }
    if (std::max<std::size_t>(size, 1) != message.tensor_size) {
        throw std::runtime_error("Training shared batch storage has trailing bytes");
    }
    auto mapping = std::make_shared<Mapping>(std::move(message.tensor_file), message.tensor_size, true);
    DatasetSample tensors;
    tensors.reserve(fields.size());
    for (const auto& field : fields) tensors.push_back(mapped_tensor(mapping, field));
    return tensors;
}

}  // namespace

namespace loader_detail {

struct ProcessWorker::Impl {
    const std::size_t worker;
    const std::int64_t batch_size;
    FileDescriptor socket;
    std::unique_ptr<ChildProcess> process;

    Impl(const Json& dataset, std::size_t id, std::int64_t seed, std::int64_t size)
        : worker(id), batch_size(size) {
        int pair[2];
        if (socketpair(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC | SOCK_NONBLOCK, 0, pair) != 0) {
            throw system_error("Cannot create training worker socket");
        }
        socket.reset(pair[0]);
        FileDescriptor child_socket(pair[1]);
        const auto executable = std::filesystem::read_symlink("/proc/self/exe").string();
        process = std::make_unique<ChildProcess>(
            std::vector<std::string>{executable, "_train_loader_worker"},
            std::vector<std::pair<int, int>>{{child_socket.get(), worker_socket}}, true);
        child_socket.reset();
        const Check check = [this] { ProcessInterrupt::check(); check_alive(); };
        send_message(socket.get(), {{"kind", "init"}, {"dataset", dataset}, {"worker", worker},
            {"base_seed", seed}, {"batch_size", batch_size}, {"parent", getpid()}}, check);
        auto reply = receive_message(socket.get(), check);
        if (!reply || reply->tensor_size || reply->value.at("kind") != "ready" ||
            reply->value.at("worker").get<std::size_t>() != worker) {
            throw std::runtime_error("Training worker did not acknowledge dataset initialization");
        }
    }

    void check_alive() {
        if (const auto status = process->poll()) {
            throw std::runtime_error("Training loader worker " + std::to_string(worker) +
                " (pid " + std::to_string(process->pid()) + ") exited with status " + std::to_string(*status));
        }
    }
};

ProcessWorker::ProcessWorker(const Json& dataset, std::size_t worker,
                             std::int64_t base_seed, std::int64_t batch_size)
    : impl_(std::make_unique<Impl>(dataset, worker, base_seed, batch_size)) {}
ProcessWorker::~ProcessWorker() = default;

ProcessBatch ProcessWorker::fetch(std::size_t id, const std::vector<std::int64_t>& indices, const Check& check) {
    if (indices.size() != static_cast<std::size_t>(impl_->batch_size)) {
        throw std::invalid_argument("Training process worker needs one complete batch");
    }
    const Check alive = [&] { check(); impl_->check_alive(); };
    send_message(impl_->socket.get(), {{"kind", "batch"}, {"id", id}, {"indices", indices}}, alive);
    auto reply = receive_message(impl_->socket.get(), alive);
    if (!reply) throw std::runtime_error("Training loader worker closed its result socket");
    if (reply->value.at("id").get<std::size_t>() != id) {
        throw std::runtime_error("Training loader worker returned an out-of-order batch");
    }
    ProcessBatch batch{id, {}, {}};
    if (reply->value.at("kind") == "error" && !reply->tensor_size) {
        batch.error = std::make_exception_ptr(std::runtime_error(
            "In training loader worker process " + std::to_string(impl_->worker) + ": " +
            reply->value.at("message").get<std::string>()));
    } else if (reply->value.at("kind") == "batch") {
        batch.tensors = receive_tensors(*reply);
    } else throw std::runtime_error("Unexpected training worker response");
    return batch;
}

void ProcessWorker::check_alive() { impl_->check_alive(); }
void ProcessWorker::stop(bool interrupt) noexcept {
    // Keep the FD allocated until the transport thread joins. Closing it here
    // could reuse its number while that thread is polling/receiving.
    shutdown(impl_->socket.get(), SHUT_RDWR);
    if (interrupt) impl_->process->signal(SIGTERM);
}
bool ProcessWorker::reap() { return impl_->process->poll_for_cleanup().has_value(); }
void ProcessWorker::kill() noexcept { impl_->process->signal(SIGKILL); }

}  // namespace loader_detail

namespace {

int training_loader_worker_main() {
    const auto parent = getppid();
    if (prctl(PR_SET_PDEATHSIG, SIGTERM) != 0) throw system_error("Cannot watch training worker parent");
    if (getppid() != parent) return 0;
    c10::setThreadName("pt_data_worker");
    const Check check = [parent] {
        ProcessInterrupt::check();
        if (getppid() != parent) throw std::runtime_error("Training loader parent exited");
    };
    auto init = receive_message(worker_socket, check);
    if (!init) return 0;
    if (init->tensor_size || init->value.at("kind") != "init" ||
        init->value.at("parent").get<std::int64_t>() != parent) {
        throw std::runtime_error("Invalid training worker initialization");
    }
    const auto worker = init->value.at("worker").get<std::uint64_t>();
    const auto base_seed = init->value.at("base_seed").get<std::int64_t>();
    const auto batch_size = init->value.at("batch_size").get<std::int64_t>();
    if (base_seed < 0 || batch_size <= 0 ||
        static_cast<std::uint64_t>(batch_size) > std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument("Invalid training worker seed or batch size");
    }
    pareto_detail::PythonRandom random(base_seed, worker);
    torch::set_num_threads(1);
    {
        // Only seed the CPU generator. Dataset workers must not initialize a
        // CUDA context; Python's manual_seed also defers CUDA initialization.
        auto generator = at::detail::getDefaultCPUGenerator();
        const std::lock_guard<std::mutex> lock(generator.mutex());
        generator.set_current_seed(static_cast<std::uint64_t>(base_seed) + worker);
    }
    torch::NoGradGuard no_grad;
    const auto& state = init->value.at("dataset");
    std::unique_ptr<ImageFolder> images;
    std::unique_ptr<VideoFolder> video;
    if (state.at("kind") == "image") images = std::make_unique<ImageFolder>(state);
    else if (state.at("kind") == "video") video = std::make_unique<VideoFolder>(state);
    else throw std::invalid_argument("Unknown training worker dataset kind");
    init.reset();  // Release the potentially large initialization document.
    const RandomBelow draw = [&](std::uint64_t upper) { return random.below(upper); };
    send_message(worker_socket, {{"kind", "ready"}, {"worker", worker}}, check);
    for (;;) {
        auto task = receive_message(worker_socket, check);
        if (!task) return 0;
        if (task->tensor_size || task->value.at("kind") != "batch") {
            throw std::runtime_error("Invalid training worker task");
        }
        const auto id = task->value.at("id").get<std::uint64_t>();
        const auto indices = task->value.at("indices").get<std::vector<std::int64_t>>();
        if (indices.size() != static_cast<std::size_t>(batch_size)) {
            throw std::runtime_error("Training worker task is not a complete batch");
        }
        task.reset();
        std::vector<DatasetSample> samples;
        samples.reserve(indices.size());
        std::optional<SharedBatch> batch;
        std::string error;
        try {
            for (const auto index : indices) {
                check();
                samples.push_back(images ? images->get(index, draw) : video->get(index, draw));
            }
            batch = collate_shared(samples);
        } catch (const std::exception& failure) { error = failure.what(); }
        catch (...) { error = "Unknown dataset or collation failure"; }
        samples.clear();
        check();  // Cancellation takes priority over publishing a sample error.
        if (batch) {
            send_message(worker_socket, {{"kind", "batch"}, {"id", id}, {"fields", batch->fields}}, check,
                         batch->storage->file.get(), batch->storage->size);
        } else {
            send_message(worker_socket, {{"kind", "error"}, {"id", id}, {"message", error}}, check);
        }
    }
}

}  // namespace

int run_training_loader_worker() {
    ProcessInterrupt interruption;
    try { return training_loader_worker_main(); }
    catch (...) {
        // Controller cleanup already reports the foreground failure. Keep
        // expected TERM/INT teardown from printing one error per worker.
        if (ProcessInterrupt::requested()) return 128 + ProcessInterrupt::requested();
        throw;
    }
}

}  // namespace uf::training
