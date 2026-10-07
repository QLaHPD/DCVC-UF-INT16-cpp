// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
// Optimizer restoration follows PyTorch; BSD notice: licenses/PyTorch.txt.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "uf/training_checkpoint.hpp"
#include "uf/torch_pickle.hpp"
#include "uf/process.hpp"
#include "checkpoint_storage_copy.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <initializer_list>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <sys/stat.h>
#include <unistd.h>

namespace uf::training {
namespace {

using Dictionary = c10::impl::GenericDict;

std::vector<std::filesystem::path> status_paths(const std::filesystem::path& save_dir) {
    std::vector<std::filesystem::path> paths;
    for (const auto& entry : std::filesystem::directory_iterator(save_dir)) {
        ProcessInterrupt::check();
        if (entry.path().filename().string().find("status_epo") != std::string::npos) {
            paths.push_back(entry.path());
        }
    }
    return paths;
}

double status_modified(const std::filesystem::path& path) {
    struct stat info{};
    if (::stat(path.c_str(), &info) != 0) {
        throw std::system_error(errno, std::generic_category(), "Cannot inspect training status: " + path.string());
    }
    // os.path.getmtime sorts CPython's binary64 st_mtime, not the exact
    // nanosecond timestamp. Equal converted values keep listdir order.
    const double fraction = 1e-9 * static_cast<double>(info.st_mtim.tv_nsec);
    return static_cast<double>(info.st_mtim.tv_sec) + fraction;
}

Dictionary dictionary(const torch::IValue& value) {
    if (!value.isGenericDict()) throw std::invalid_argument("Training checkpoint value must be a dictionary");
    return value.toGenericDict();
}

Dictionary string_dictionary() { return Dictionary{c10::StringType::get(), c10::AnyType::get()}; }

// Plain UF checkpoint deepcopy, with separate identity and untyped-storage
// memos. Python copies groups but casts parameter state from the original.
// Container rules follow copy.py; notice: licenses/CPython-3.10.20.txt.
class GroupCopies {
public:
    torch::IValue copy(const torch::IValue& value) {
        ProcessInterrupt::check();
        if (const auto found = memo_.find(value); found != memo_.end()) return found->second;
        if (value.isTensor()) {
            const auto source = value.toTensor();
            if (!source.is_leaf()) throw std::runtime_error("Only leaf tensors support optimizer group deepcopy");
            torch::NoGradGuard no_grad;
            auto output = source.is_sparse() || source.device().is_meta()
                ? source.clone() : storages_.copy(source);
            if (!source.is_sparse() && !source.device().is_meta()) {
                if (source.is_conj()) output = output.conj_physical();
                if (source.is_neg()) output = output.neg();
            }
            output.set_requires_grad(source.requires_grad());
            if (source.grad().defined()) output.mutable_grad() = copy(source.grad()).toTensor();
            memo_.emplace(value, output);
            return output;
        }
        if (value.isGenericDict()) {
            Dictionary output(c10::AnyType::get(), c10::AnyType::get());
            memo_.emplace(value, output);
            for (const auto& entry : value.toGenericDict()) {
                auto item = copy(entry.value());
                auto key = copy(entry.key());
                output.insert(std::move(key), std::move(item));
            }
            return output;
        }
        if (value.isList()) {
            c10::impl::GenericList output(c10::AnyType::get());
            memo_.emplace(value, output);
            for (const auto& item : value.toListRef()) output.push_back(copy(item));
            return output;
        }
        if (value.isTuple()) {
            std::vector<torch::IValue> items;
            bool changed = false;
            for (const auto& item : value.toTupleRef().elements()) {
                items.push_back(copy(item));
                changed = changed || !item.is(items.back());
            }
            // A recursive mutable child may already have finished this tuple.
            if (const auto found = memo_.find(value); found != memo_.end()) return found->second;
            if (!changed) return value;
            torch::IValue output = c10::ivalue::Tuple::create(std::move(items));
            memo_.emplace(value, output);
            return output;
        }
        if (value.isNone() || value.isInt() || value.isDouble() || value.isBool() || value.isString()) return value;
        throw std::invalid_argument("Unsupported value in optimizer group deepcopy");
    }

private:
    torch::IValue::HashIdentityIValueMap memo_;
    checkpoint_detail::StorageCopies storages_;
};

std::optional<torch::IValue> optional_field(const Dictionary& value, const std::string& key) {
    const auto found = value.find(torch::IValue(key));
    if (found == value.end()) return std::nullopt;
    return found->value();
}

torch::IValue field(const Dictionary& value, const std::string& key) {
    const auto found = optional_field(value, key);
    if (!found) throw std::invalid_argument("Missing training checkpoint field: " + key);
    return *found;
}

CheckpointState tensor_state(torch::IValue value, bool unwrap) {
    auto entries = dictionary(value);
    if (unwrap) {
        for (const auto& key : {"state_dict", "net"}) {
            if (const auto nested = optional_field(entries, key)) entries = dictionary(*nested);
        }
    }
    CheckpointState state;
    for (const auto& entry : entries) {
        if (!entry.key().isString() || !entry.value().isTensor()) {
            throw std::invalid_argument("Training network state must map names to tensors");
        }
        const auto name = entry.key().toStringRef();
        if (name.rfind("module.", 0) != 0) state[name] = entry.value().toTensor();
    }
    // consume_prefix_in_state_dict_if_present overwrites a plain name if a
    // prefixed name also exists.
    for (const auto& entry : entries) {
        const auto name = entry.key().toStringRef();
        if (name.rfind("module.", 0) == 0) state[name.substr(7)] = entry.value().toTensor();
    }
    return state;
}

Dictionary tensor_dictionary(const torch::nn::Module& model) {
    auto output = string_dictionary();
    for (const auto& entry : export_training_state(model)) output.insert(entry.first, entry.second);
    return output;
}

double number(const torch::IValue& value) {
    if (value.isBool()) return value.toBool() ? 1.0 : 0.0;
    if (value.isDouble()) return value.toDouble();
    if (value.isInt()) return static_cast<double>(value.toInt());
    if (value.isTensor() && value.toTensor().numel() == 1) return value.toTensor().item<double>();
    throw std::invalid_argument("Training optimizer value must be a scalar number");
}

bool boolean(const Dictionary& options, const std::string& key, bool fallback = false) {
    const auto value = optional_field(options, key);
    if (!value) return fallback;
    if (!value->isBool()) throw std::invalid_argument("Training optimizer option must be boolean: " + key);
    return value->toBool();
}

std::optional<bool> implementation_option(const Dictionary& options, const std::string& key) {
    const auto value = optional_field(options, key);
    if (!value || value->isNone()) return std::nullopt;
    if (!value->isBool()) throw std::invalid_argument("Invalid AdamW implementation option: " + key);
    return value->toBool();
}

Dictionary extra_fields(const Dictionary& values, std::initializer_list<const char*> known) {
    Dictionary output(c10::AnyType::get(), c10::AnyType::get());
    for (const auto& entry : values) {
        bool standard = false;
        if (entry.key().isString()) {
            for (const auto* name : known) if (entry.key().toStringRef() == name) standard = true;
        }
        if (!standard) output.insert(entry.key(), entry.value());
    }
    return output;
}

Dictionary extra_dictionary(const torch::IValue& values) {
    Dictionary output(c10::AnyType::get(), c10::AnyType::get());
    if (values.isNone()) return output;
    for (const auto& entry : dictionary(values)) output.insert(entry.key(), entry.value());
    return output;
}

AdamWOptions adamw_options(const Dictionary& packed) {
    const auto betas_value = field(packed, "betas");
    std::vector<torch::IValue> betas;
    if (betas_value.isTuple()) {
        for (const auto& beta : betas_value.toTupleRef().elements()) betas.push_back(beta);
    } else if (betas_value.isList()) {
        for (const auto& beta : betas_value.toListRef()) betas.push_back(beta);
    } else throw std::invalid_argument("AdamW betas must be a pair");
    if (betas.size() != 2) throw std::invalid_argument("AdamW betas must be a pair");
    const auto saved_lr = field(packed, "lr");
    // Python does not extract a saved tensor LR's value during restoration.
    AdamWOptions options(saved_lr.isTensor() ? 1e-3 : number(saved_lr));
    if (saved_lr.isTensor()) options.tensor_lr(saved_lr.toTensor());
    options.betas(std::make_tuple(number(betas[0]), number(betas[1])))
        .eps(number(field(packed, "eps"))).weight_decay(number(field(packed, "weight_decay")))
        .amsgrad(boolean(packed, "amsgrad")).maximize(boolean(packed, "maximize"))
        .capturable(boolean(packed, "capturable")).differentiable(boolean(packed, "differentiable"))
        .foreach(implementation_option(packed, "foreach")).fused(implementation_option(packed, "fused"));
    options.extra_options(extra_fields(packed, {"lr", "betas", "eps", "weight_decay", "amsgrad",
        "maximize", "capturable", "differentiable", "foreach", "fused", "params"}));
    return options;
}

// Optimizer._cast descends through dictionaries/lists/tuples. Dictionary keys
// determine the step policy; entering a list/tuple clears that key. A no-op
// Tensor.to keeps its alias, as in Python, and shape checks belong to step().
torch::IValue cast_optimizer_value(const torch::IValue& value, const torch::Tensor& parameter,
                                  const AdamWOptions& policy, const torch::IValue& key = {}) {
    ProcessInterrupt::check();
    if (value.isTensor()) {
        const auto tensor = value.toTensor();
        if (key.isString() && key.toStringRef() == "step") {
            if (policy.capturable() || policy.fused().value_or(false)) {
                return tensor.to(parameter.device(), torch::kFloat32);
            }
            return tensor;
        }
        const auto dtype = parameter.is_floating_point() ? parameter.scalar_type() : tensor.scalar_type();
        return tensor.to(parameter.device(), dtype);
    }
    if (value.isGenericDict()) {
        Dictionary output(c10::AnyType::get(), c10::AnyType::get());
        for (const auto& entry : value.toGenericDict()) {
            output.insert(entry.key(), cast_optimizer_value(entry.value(), parameter, policy, entry.key()));
        }
        return output;
    }
    if (value.isList()) {
        c10::impl::GenericList output(c10::AnyType::get());
        for (const auto& item : value.toListRef()) output.push_back(cast_optimizer_value(item, parameter, policy));
        return output;
    }
    if (value.isTuple()) {
        std::vector<torch::IValue> output;
        for (const auto& item : value.toTupleRef().elements()) output.push_back(cast_optimizer_value(item, parameter, policy));
        return c10::ivalue::Tuple::create(std::move(output));
    }
    return value;
}

torch::Tensor scalar_step_tensor(const torch::IValue& value, const torch::Tensor& parameter,
                                 const AdamWOptions& options) {
    const auto device = options.capturable() || options.fused().value_or(false)
        ? parameter.device() : torch::Device(torch::kCPU);
    return torch::tensor(number(value), torch::TensorOptions()
        .dtype(adamw_step_dtype(options.fused().value_or(false))).device(device));
}

}  // namespace

CheckpointState read_training_state(const std::filesystem::path& checkpoint) {
    return tensor_state(load_torch_pickle_cpu(checkpoint), true);
}

std::vector<std::filesystem::path> latest_status_paths(const std::filesystem::path& save_dir) {
    struct Candidate { std::filesystem::path path; double modified; };
    std::vector<Candidate> entries;
    // Snapshot directory names before querying their sort keys, as listdir does.
    for (const auto& path : status_paths(save_dir)) {
        ProcessInterrupt::check();
        entries.push_back({path, status_modified(path)});
    }
    std::stable_sort(entries.begin(), entries.end(),
        [](const Candidate& left, const Candidate& right) { return left.modified < right.modified; });
    std::vector<std::filesystem::path> output;
    const auto start = entries.size() > 2 ? entries.size() - 2 : 0;
    for (auto index = start; index < entries.size(); ++index) output.push_back(entries[index].path);
    return output;
}

ExistingTraining load_existing_weights(const std::filesystem::path& save_dir, torch::nn::Module& model,
    int rank, const std::optional<std::filesystem::path>& pretrain_path) {
    ExistingTraining output;
    bool status_metadata_loaded = false;
    const auto statuses = latest_status_paths(save_dir);
    for (auto status = statuses.rbegin(); status != statuses.rend(); ++status) {
        ProcessInterrupt::check();
        try {
            const auto data = dictionary(load_torch_pickle_cpu(*status));
            const auto optimizer = field(data, "opt");
            // The reference assigns opt before reading epoch, and retains
            // those assignments if network restoration fails afterward.
            if (optimizer.isNone()) output.optimizer.reset();
            else output.optimizer = optimizer;
            const auto epoch = field(data, "epoch");
            if (!epoch.isInt() && !epoch.isBool()) {
                throw std::invalid_argument("Invalid saved training epoch");
            }
            const auto saved_epoch = epoch.isBool() ? static_cast<std::int64_t>(epoch.toBool()) : epoch.toInt();
            if (saved_epoch == std::numeric_limits<std::int64_t>::max()) {
                throw std::invalid_argument("Invalid saved training epoch");
            }
            output.begin_epoch = saved_epoch + 1;
            status_metadata_loaded = true;
            output.status_path = *status;
            if (rank <= 0) load_training_state(model, tensor_state(field(data, "net"), false));
            if (rank <= 0) {
                std::cout << "load status from " << status->string() << '\n';
                if (!std::cout) throw std::runtime_error("Cannot write training status path");
                std::cout << "begin epoch " << output.begin_epoch << '\n';
                if (!std::cout) throw std::runtime_error("Cannot write resumed training epoch");
            }
            return output;
        } catch (...) {
            ProcessInterrupt::check();
            // The reference tries the older of the two newest status files.
        }
    }
    if (!status_metadata_loaded && pretrain_path && rank <= 0) {
        load_training_state(model, tensor_state(load_torch_pickle_cpu(*pretrain_path), true));
        output.pretrained = true;
        std::cout << "load pretrained weights from " << pretrain_path->string() << '\n';
        if (!std::cout) throw std::runtime_error("Cannot write pretrained training path");
    }
    return output;
}

torch::IValue adamw_state_dict(const AdamW& optimizer) {
    torch::NoGradGuard no_grad;
    auto output = string_dictionary();
    auto states = extra_dictionary(optimizer.unmapped_state());
    c10::impl::GenericList groups(c10::AnyType::get());
    std::unordered_map<void*, std::int64_t> identifiers;
    std::int64_t next = 0;
    for (const auto& group : optimizer.param_groups()) {
        const auto& options = dynamic_cast<const AdamWOptions&>(group.options());
        auto packed = extra_dictionary(options.extra_options());
        packed.insert("lr", options.tensor_lr().defined()
            ? torch::IValue(options.tensor_lr()) : torch::IValue(options.lr()));
        packed.insert("betas", c10::ivalue::Tuple::create({std::get<0>(options.betas()), std::get<1>(options.betas())}));
        packed.insert("eps", options.eps());
        packed.insert("weight_decay", options.weight_decay());
        packed.insert("amsgrad", options.amsgrad());
        packed.insert("maximize", options.maximize());
        packed.insert("foreach", options.foreach() ? torch::IValue(*options.foreach()) : torch::IValue{});
        packed.insert("capturable", options.capturable());
        packed.insert("differentiable", options.differentiable());
        packed.insert("fused", options.fused() ? torch::IValue(*options.fused()) : torch::IValue{});
        c10::impl::GenericList parameters(c10::AnyType::get());
        if (group.params().size() > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max() - next)) {
            throw std::overflow_error("Too many AdamW parameters");
        }
        // Python packs indices by position, including duplicate positions.
        // Its group comprehension keeps the last occurrence of a new tensor
        // while tensors already mapped by earlier groups keep their old ID.
        std::unordered_map<void*, std::int64_t> additions;
        for (std::size_t index = 0; index < group.params().size(); ++index) {
            auto* identity = group.params()[index].unsafeGetTensorImpl();
            if (!identifiers.count(identity)) additions[identity] = next + static_cast<std::int64_t>(index);
        }
        identifiers.insert(additions.begin(), additions.end());
        for (const auto& parameter : group.params()) parameters.push_back(identifiers.at(parameter.unsafeGetTensorImpl()));
        next += static_cast<std::int64_t>(group.params().size());
        packed.insert("params", parameters);
        groups.push_back(packed);
    }
    for (const auto& entry : optimizer.state()) {
        ProcessInterrupt::check();
        const auto identity = identifiers.find(entry.first);
        if (identity == identifiers.end()) throw std::invalid_argument("AdamW state has an unbound parameter");
        const auto& state = dynamic_cast<const AdamWParamState&>(*entry.second);
        auto values = extra_dictionary(state.extra_state());
        if (state.step().defined()) values.insert("step", state.step());
        if (state.exp_avg().defined()) values.insert("exp_avg", state.exp_avg());
        if (state.exp_avg_sq().defined()) values.insert("exp_avg_sq", state.exp_avg_sq());
        if (state.max_exp_avg_sq().defined()) values.insert("max_exp_avg_sq", state.max_exp_avg_sq());
        states.insert_or_assign(identity->second, values);
    }
    output.insert("state", states);
    output.insert("param_groups", groups);
    return output;
}

void load_adamw_state(AdamW& optimizer, const torch::IValue& value) {
    const auto data = dictionary(value);
    const auto saved_state = dictionary(field(data, "state"));
    const auto saved_groups = field(data, "param_groups");
    GroupCopies group_copies;
    const auto copied_groups = group_copies.copy(saved_groups);
    if (!copied_groups.isList() || copied_groups.toListRef().size() != optimizer.param_groups().size()) {
        throw std::invalid_argument("AdamW status has a different number of parameter groups");
    }
    std::vector<std::unique_ptr<AdamWOptions>> options;
    std::map<void*, std::unique_ptr<AdamWParamState>> states;
    std::map<std::int64_t, torch::Tensor> parameters_by_id;
    std::map<std::int64_t, std::size_t> policy_by_id;
    Dictionary unmapped(c10::AnyType::get(), c10::AnyType::get());
    const auto groups = copied_groups.toListRef();
    for (std::size_t index = 0; index < groups.size(); ++index) {
        ProcessInterrupt::check();
        const auto packed = dictionary(groups[index]);
        const auto saved_parameters = field(packed, "params");
        const auto& parameters = optimizer.param_groups()[index].params();
        if (!saved_parameters.isList() || saved_parameters.toListRef().size() != parameters.size()) {
            throw std::invalid_argument("AdamW status parameter count differs from the model");
        }
        const auto identifiers = saved_parameters.toListRef();
        for (std::size_t parameter_index = 0; parameter_index < parameters.size(); ++parameter_index) {
            const auto& id = identifiers[parameter_index];
            if (!id.isInt()) throw std::invalid_argument("AdamW parameter identifier must be an integer");
            // dict(zip(...)) uses the last parameter for a repeated saved ID;
            // the casting policy searches the saved groups from the front.
            parameters_by_id.insert_or_assign(id.toInt(), parameters[parameter_index]);
            policy_by_id.emplace(id.toInt(), index);
        }
        options.push_back(std::make_unique<AdamWOptions>(adamw_options(packed)));
    }
    for (const auto& entry : saved_state) {
        ProcessInterrupt::check();
        const auto parameter = entry.key().isInt() ? parameters_by_id.find(entry.key().toInt()) : parameters_by_id.end();
        if (parameter == parameters_by_id.end()) {
            // Python retains state not assigned to model parameters as is.
            unmapped.insert(entry.key(), entry.value());
            continue;
        }
        const auto values = dictionary(cast_optimizer_value(entry.value(), parameter->second,
            *options[policy_by_id.at(entry.key().toInt())]));
        auto state = std::make_unique<AdamWParamState>();
        Dictionary extras(c10::AnyType::get(), c10::AnyType::get());
        for (const auto& item : values) {
            bool assigned = false;
            if (item.key().isString() && item.value().isTensor()) {
                const auto name = item.key().toStringRef();
                const auto tensor = item.value().toTensor();
                if (name == "step") { state->step(tensor); assigned = true; }
                else if (name == "exp_avg") { state->exp_avg(tensor); assigned = true; }
                else if (name == "exp_avg_sq") { state->exp_avg_sq(tensor); assigned = true; }
                else if (name == "max_exp_avg_sq") { state->max_exp_avg_sq(tensor); assigned = true; }
            }
            if (!assigned) extras.insert(item.key(), item.value());
        }
        state->extra_state(extras);
        states[parameter->second.unsafeGetTensorImpl()] = std::move(state);
    }
    // Python commits state/groups before AdamW.__setstate__ normalizes old
    // scalar counters. A normalization error retains that committed state.
    optimizer.state().clear();
    for (auto& entry : states) optimizer.state()[entry.first] = std::move(entry.second);
    optimizer.unmapped_state(unmapped);
    for (std::size_t index = 0; index < options.size(); ++index) {
        optimizer.param_groups()[index].set_options(std::move(options[index]));
    }
    for (const auto& group : optimizer.param_groups()) {
        const auto& group_options = dynamic_cast<const AdamWOptions&>(group.options());
        for (const auto& parameter : group.params()) {
            ProcessInterrupt::check();
            const auto found = optimizer.state().find(parameter.unsafeGetTensorImpl());
            if (found == optimizer.state().end()) continue;
            auto& state = dynamic_cast<AdamWParamState&>(*found->second);
            if (state.step().defined()) continue;
            auto extras = dictionary(state.extra_state());
            if (!state.exp_avg().defined() && !state.exp_avg_sq().defined() &&
                !state.max_exp_avg_sq().defined() && extras.empty()) continue;
            state.step(scalar_step_tensor(field(extras, "step"), parameter, group_options));
            (void)extras.erase(torch::IValue("step"));
        }
    }
}

void save_training_checkpoint(const std::filesystem::path& save_dir, const torch::nn::Module& model) {
    auto data = string_dictionary();
    data.insert("state_dict", tensor_dictionary(model));
    const auto path = save_dir / "ckpt.pth.tar";
    save_torch_pickle_cpu(path, data);
    std::cout << "save final checkpoint to " << path.string() << '\n';
    if (!std::cout) throw std::runtime_error("Cannot write final training checkpoint message");
}

void save_training_status(const std::filesystem::path& save_dir, const torch::nn::Module& model,
                          const AdamW& optimizer, std::int64_t epoch) {
    auto data = string_dictionary();
    data.insert("epoch", epoch);
    data.insert("net", tensor_dictionary(model));
    data.insert("opt", adamw_state_dict(optimizer));
    const auto path = save_dir / ("status_epo" + std::to_string(epoch) + ".pth.tar");
    save_torch_pickle_cpu(path, data);
    std::cout << "save model epoch " << epoch << '\n';
    if (!std::cout) throw std::runtime_error("Cannot write saved training epoch");
    // Python snapshots listdir before deleting entries. Its exists check
    // suppresses stat errors, and os.remove unlinks files/symlinks only.
    for (const auto& previous : status_paths(save_dir)) {
        ProcessInterrupt::check();
        struct stat info{};
        if (previous != path && ::stat(previous.c_str(), &info) == 0) {
            if (::unlink(previous.c_str()) != 0) {
                throw std::system_error(errno, std::generic_category(), "Cannot remove training status: " + previous.string());
            }
        }
    }
}

}  // namespace uf::training
