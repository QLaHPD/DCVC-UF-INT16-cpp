// Standalone UF integer determinism report, adapted from scripts/check_uf_int16.py.
#include "uf/int16_validation.hpp"
#include "uf/cli.hpp"
#include "uf/int16_ops.hpp"
#include "uf/integer_model.hpp"
#include "uf/prepare.hpp"
#include "uf/process.hpp"
#include "uf/json_compat.hpp"
#include "uf/python_json.hpp"
#include "uf/python_path.hpp"
#include "uf/utf8.hpp"
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/ops/randint.h>
#include <ATen/record_function.h>
#include <torch/cuda.h>
#ifdef UF_HAS_INT16_VALIDATION_CUDA
#include <cuda_runtime_api.h>
#include <c10/cuda/CUDACachingAllocator.h>
#endif
#include <openssl/evp.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <ostream>
#include <sstream>
#include <stdexcept>

namespace uf {
namespace {
using Json = nlohmann::ordered_json;

class IntegerOnly {
public:
    IntegerOnly() {
        if (active_) throw std::logic_error("Nested UF integer-only observation");
        active_ = this;
        try {
            handle_ = at::addThreadLocalCallback(at::RecordFunctionCallback(&start, &end)
                .needsInputs(true).needsOutputs(true).scopes({at::RecordScope::FUNCTION}));
        } catch (...) { active_ = nullptr; throw; }
    }
    ~IntegerOnly() { at::removeCallback(handle_); active_ = nullptr; }
    IntegerOnly(const IntegerOnly&) = delete;
    IntegerOnly& operator=(const IntegerOnly&) = delete;
    void check() const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!failure_.empty()) throw std::runtime_error(failure_);
    }
private:
    struct Observation : at::ObserverContext {
        explicit Observation(IntegerOnly* value) : owner(value) {}
        IntegerOnly* owner;
    };
    void observe(const c10::IValue& value, const char* operation) {
        if (value.isTensor()) {
            const auto& tensor = value.toTensor();
            if (tensor.defined() && (tensor.is_floating_point() || tensor.is_complex())) {
                std::lock_guard<std::mutex> lock(mutex_);
                if (failure_.empty()) failure_ = std::string("Floating-point tensor in integer inference: ") + operation;
            }
        } else if (value.isTensorList()) {
            for (const auto& child : value.toTensorVector()) observe(c10::IValue(child), operation);
        } else if (value.isOptionalTensorList()) {
            for (const auto& child : value.toOptionalTensorVector()) if (child) observe(c10::IValue(*child), operation);
        } else if (value.isTuple()) {
            for (const auto& child : value.toTupleRef().elements()) observe(child, operation);
        } else if (value.isList()) {
            for (const auto& child : value.toListRef()) observe(child, operation);
        } else if (value.isGenericDict()) {
            for (const auto& child : value.toGenericDict()) { observe(child.key(), operation); observe(child.value(), operation); }
        }
    }
    static std::unique_ptr<at::ObserverContext> start(const at::RecordFunction& function) {
        if (!active_) return {};
        for (const auto& input : function.inputs()) active_->observe(input, function.name());
        for (const auto& input : function.kwinputs()) active_->observe(input.second, function.name());
        return std::make_unique<Observation>(active_);
    }
    static void end(const at::RecordFunction& function, at::ObserverContext* context) {
        if (const auto observation = static_cast<Observation*>(context))
            for (const auto& output : function.outputs()) observation->owner->observe(output, function.name());
    }
    // LibTorch catches observer exceptions. Record the violation in the callback
    // and reject the packet at its call boundary, before accepting any report.
    at::RecordFunctionGuard enabled_{true};
    at::CallbackHandle handle_{};
    mutable std::mutex mutex_;
    std::string failure_;
    static thread_local IntegerOnly* active_;
};
thread_local IntegerOnly* IntegerOnly::active_ = nullptr;

std::string digest(const void* bytes, std::size_t count) {
    std::array<unsigned char, EVP_MAX_MD_SIZE> output{};
    unsigned int length = 0;
    const auto pointer = count ? bytes : static_cast<const void*>("");
    if (EVP_Digest(pointer, count, output.data(), &length, EVP_sha256(), nullptr) != 1)
        throw std::runtime_error("Cannot hash UF integer report data");
    std::ostringstream result;
    result << std::hex << std::setfill('0');
    for (unsigned int i = 0; i < length; ++i) result << std::setw(2) << static_cast<unsigned int>(output[i]);
    return result.str();
}
Json tensor_hash(const torch::Tensor& value) {
    if (!value.defined()) return nullptr;
    if (value.is_floating_point() || value.is_complex())
        throw std::runtime_error("Floating-point tensor in integer state/reconstruction");
    const auto cpu = value.detach().to(torch::kCPU).contiguous();
    const auto width = static_cast<std::size_t>(cpu.element_size());
    if (static_cast<std::uint64_t>(cpu.numel()) > std::numeric_limits<std::size_t>::max() / width)
        throw std::overflow_error("UF report tensor byte count exceeds size_t");
    const auto count = static_cast<std::size_t>(cpu.numel()) * width;
    const std::uint16_t word = 1;
    if (*reinterpret_cast<const unsigned char*>(&word) == 1 || width == 1)
        return digest(cpu.data_ptr(), count);
    // NumPy astype(dtype.newbyteorder('<')) retains dtype width and changes bytes.
    const auto first = static_cast<const unsigned char*>(cpu.data_ptr());
    std::vector<unsigned char> little(count);
    for (std::size_t offset = 0; offset < count; offset += width)
        std::reverse_copy(first + offset, first + offset + width, little.begin() + offset);
    return digest(little.data(), little.size());
}
Json state_record(const IntegerModel& model, const std::vector<torch::Tensor>& frames,
                  const std::vector<std::uint8_t>& bits, bool image) {
    Json reconstruction = nullptr;
    if (image) {
        if (frames.size() != 1) throw std::runtime_error("UF image reconstruction must contain one tensor");
        reconstruction = tensor_hash(frames.front());
    } else if (!frames.empty()) {
        reconstruction = Json::array();
        for (const auto& frame : frames) reconstruction.push_back(tensor_hash(frame));
    }
    return Json{{"bitstream", digest(bits.data(), bits.size())}, {"bytes", bits.size()},
                {"reconstruction", reconstruction},
                {"state", Json{{"reference", tensor_hash(model.reference())},
                               {"memory", tensor_hash(model.memory())},
                               {"context", tensor_hash(model.context())}}}};
}
bool equal_json(const Json& first, const Json& second) {
    return json_python_equal(nlohmann::json(first), nlohmann::json(second));
}
void empty_cuda_cache() {
#ifdef UF_HAS_INT16_VALIDATION_CUDA
    c10::cuda::CUDACachingAllocator::emptyCache();
#endif
}
std::string device_name(int index) {
#ifdef UF_HAS_INT16_VALIDATION_CUDA
    cudaDeviceProp properties{};
    const auto status = cudaGetDeviceProperties(&properties, index);
    if (status != cudaSuccess) throw std::runtime_error(std::string("Cannot inspect CUDA device: ") + cudaGetErrorString(status));
    return properties.name;
#else
    throw std::runtime_error("check-int16 device reporting requires CUDA runtime development files at build time");
#endif
}
struct PacketRecord { bool image, reset; std::vector<std::uint8_t> bits; };
}

int run_int16_validation_command(int argc, char** argv,
                                 const std::filesystem::path& checkpoints,
                                 std::ostream& output) {
    ProcessInterrupt interruption;
    const auto options = cli::parse(argc, argv);
    const auto variant = options.count("__variant") ? options.at("__variant") : "hts";
    const auto cuda_index = options.count("__cuda_idx") ? cli::integer(options.at("__cuda_idx"), "--cuda_idx") : 0;
    const auto size = options.count("__size") ? cli::integer(options.at("__size"), "--size") : 32;
    const bool cpu_reference = options.count("__cpu_reference") != 0;
    const auto explicit_path = [&](const char* key) -> std::optional<std::filesystem::path> {
        if (options.count(key)) return std::filesystem::path(options.at(key));
        return std::nullopt;
    };
    std::array<ResolvedPreparedModel, 2> prepared{
        resolve_prepared(checkpoints / "cvpr2026_image.pth.tar", "image", explicit_path("__prepared_i")),
        resolve_prepared(checkpoints / ("cvpr2026_video_" + variant + ".pth.tar"), variant, explicit_path("__prepared_p"))};
    if (!torch::cuda::is_available()) throw std::runtime_error("UF integer validation requires CUDA");
    if (cuda_index < 0 || cuda_index > std::numeric_limits<c10::DeviceIndex>::max())
        throw std::out_of_range("CUDA device index is outside LibTorch's native range");
    const torch::Device device(torch::kCUDA, static_cast<c10::DeviceIndex>(cuda_index));
    auto generator = at::detail::createCPUGenerator(437);
    const auto source = [&](std::int64_t channels) {
        return at::randint(-256, 257, {1, channels, size, size}, generator,
                          torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt16)).to(device);
    };
    Json records = Json::array();
    std::vector<PacketRecord> packets;
    const auto started = std::chrono::steady_clock::now();
    torch::NoGradGuard no_grad;
    {
        IntegerModel image(prepared[0].data, device), video(prepared[1].data, device);
        IntegerOnly integer_only;
        auto result = image.compress(source(3), 36);
        integer_only.check();
        records.push_back(state_record(image, result.frames, result.payload, true));
        packets.push_back({true, false, result.payload});
        video.add_ref_feature_from_frame(result.frames.at(0));
        for (const bool reset : {false, true, false}) {
            result = video.compress(source(variant == "ld" ? 3 : 24), 30, reset);
            integer_only.check();
            records.push_back(state_record(video, result.frames, result.payload, false));
            packets.push_back({false, reset, result.payload});
        }
        integer_only.check();
    }
    empty_cuda_cache();
    generator.set_current_seed(437);
    {
        IntegerModel image(prepared[0].data, device), video(prepared[1].data, device, false);
        IntegerOnly integer_only;
        auto result = image.compress(source(3), 36);
        integer_only.check();
        if (!equal_json(state_record(image, result.frames, result.payload, true), records.at(0)))
            throw std::runtime_error("Feature-only encoder diverged at I-frame");
        video.add_ref_feature_from_frame(result.frames.at(0));
        std::size_t index = 1;
        for (const bool reset : {false, true, false}) {
            result = video.compress(source(variant == "ld" ? 3 : 24), 30, reset);
            integer_only.check();
            const auto actual = state_record(video, result.frames, result.payload, false);
            if (!result.frames.empty()) throw std::runtime_error("Feature-only encoder returned P-frame pixels");
            for (const auto* key : {"bitstream", "bytes", "state"})
                if (!equal_json(actual.at(key), records.at(index).at(key)))
                    throw std::runtime_error("Feature-only encoder diverged at packet " + std::to_string(index));
            output << "feature-only encode: packet " << index << ", reset=" << (reset ? "True" : "False")
                   << ", bytes/state match" << std::endl;
            if (!output) throw std::runtime_error("Cannot write UF integer validation progress");
            ++index;
        }
        integer_only.check();
    }
    empty_cuda_cache();
    std::vector<torch::Device> backends{device};
    if (cpu_reference) backends.emplace_back(torch::kCPU);
    for (const auto& backend : backends) {
        {
            IntegerModel image(prepared[0].data, backend), video(prepared[1].data, backend);
            IntegerOnly integer_only;
            for (std::size_t index = 0; index < packets.size(); ++index) {
                const auto& packet = packets[index];
                auto& model = packet.image ? image : video;
                const auto frames = model.decompress(packet.bits, size, size, packet.image ? 36 : 30, 1, packet.reset);
                integer_only.check();
                if (!equal_json(state_record(model, frames, packet.bits, packet.image), records.at(index)))
                    throw std::runtime_error(backend.str() + " diverged at packet " + std::to_string(index));
                if (packet.image) video.add_ref_feature_from_frame(frames.at(0));
                output << backend.str() << ": packet " << index << ", reset=" << (packet.reset ? "True" : "False")
                       << ", integer state matches" << std::endl;
                if (!output) throw std::runtime_error("Cannot write UF integer validation progress");
            }
            integer_only.check();
        }
        empty_cuda_cache();
    }
    Json report{{"arithmetic", int16::arithmetic_id}, {"variant", variant}, {"size", size}, {"seed", 437},
                {"prepared", Json::array({prepared[0].data.identity, prepared[1].data.identity})}, {"records", records},
                {"cpu_reference_checked", cpu_reference}, {"device", device_name(cuda_index)},
                {"seconds", std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count()}};
    if (options.count("__compare") && !options.at("__compare").empty()) {
        const auto previous = read_python_ordered_json_file(
            options.at("__compare"), python_process_text_encoding(), true);
        for (const auto* key : {"arithmetic", "variant", "size", "seed", "prepared", "records"})
            if (!equal_json(previous.at(key), report.at(key)))
                throw std::runtime_error(std::string("Result differs from comparison report: ") + key);
    }
    const auto path = python_path_normalize(std::filesystem::path(options.at("__output")));
    std::filesystem::create_directories(python_path_normalize(path.parent_path()));
    // Path.write_text receives an already serialized string. Preserve an
    // existing report if serialization fails before the output is opened.
    const auto content = python_pretty_json(report) + '\n';
    if (content.size() > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max()))
        throw std::overflow_error("UF integer report exceeds streamsize");
    std::ofstream file(path, std::ios::trunc);
    if (!file) throw std::runtime_error("Cannot open UF integer report output");
    bool close_attempted = false;
    const auto close = [&]() {
        close_attempted = true;
        file.clear();
        file.close();
        if (!file) throw std::runtime_error("Cannot close UF integer report output");
    };
    try {
        file.write(content.data(), static_cast<std::streamsize>(content.size()));
        if (!file) throw std::runtime_error("Cannot write UF integer report output");
        close();
    } catch (...) {
        if (!close_attempted) close();
        throw;
    }
    output << "Passed. Report: " << path.string() << std::endl;
    if (!output) throw std::runtime_error("Cannot write UF integer validation completion");
    return 0;
}
}  // namespace uf
