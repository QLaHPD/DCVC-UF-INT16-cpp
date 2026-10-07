// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#include "uf/training_video_model.hpp"
#include "uf/training_ht_model.hpp"
#include "uf/training_ld_model.hpp"

#include <stdexcept>
#include <utility>

namespace uf::training {

VideoTrainingModel::VideoTrainingModel(std::string structure) : structure_(std::move(structure)) {
    if (structure_ == "ld") graph_ = std::make_shared<DmcLdTrainingImpl>();
    else if (structure_ == "hts" || structure_ == "htl") graph_ = std::make_shared<DmcHtTrainingImpl>(structure_);
    else throw std::invalid_argument("Unknown video training model structure: " + structure_);
}

torch::nn::Module& VideoTrainingModel::module() { return *graph_; }
const torch::nn::Module& VideoTrainingModel::module() const { return *graph_; }
std::shared_ptr<torch::nn::Module> VideoTrainingModel::module_ptr() const { return graph_; }
const std::string& VideoTrainingModel::model_structure() const noexcept { return structure_; }
std::int64_t VideoTrainingModel::frame_delay() const noexcept {
    return structure_ == "ld" ? DmcLdTrainingImpl::frame_delay : DmcHtTrainingImpl::frame_delay;
}

void VideoTrainingModel::clear_dpb() {
    if (structure_ == "ld") static_cast<DmcLdTrainingImpl&>(*graph_).clear_dpb();
    else static_cast<DmcHtTrainingImpl&>(*graph_).clear_dpb();
}

void VideoTrainingModel::set_use_checkpointing(bool enabled) {
    if (structure_ == "ld") static_cast<DmcLdTrainingImpl&>(*graph_).set_use_checkpointing(enabled);
    else static_cast<DmcHtTrainingImpl&>(*graph_).set_use_checkpointing(enabled);
}

void VideoTrainingModel::add_reference(const torch::Tensor& frame) {
    if (structure_ == "ld") static_cast<DmcLdTrainingImpl&>(*graph_).add_ref_feature_from_frame(frame);
    else static_cast<DmcHtTrainingImpl&>(*graph_).add_ref_feature_from_frame(frame);
}

TrainingStep VideoTrainingModel::forward(const torch::Tensor& input, const torch::Tensor& qp,
    const torch::Tensor& lambdas, bool get_loss_info, std::int64_t current_poc) {
    if (structure_ == "ld") {
        auto output = static_cast<DmcLdTrainingImpl&>(*graph_).forward(input, qp, lambdas, get_loss_info, current_poc);
        return {std::move(output.loss.mean), std::move(output.info)};
    }
    auto output = static_cast<DmcHtTrainingImpl&>(*graph_).forward(input, qp, lambdas, get_loss_info, current_poc);
    return {std::move(output.loss.mean), std::move(output.info)};
}

TrainingStep VideoTrainingModel::forward(const std::vector<torch::Tensor>& groups, const torch::Tensor& qp,
    const torch::Tensor& lambdas, bool get_loss_info, std::int64_t current_poc) {
    if (structure_ == "ld") {
        auto output = static_cast<DmcLdTrainingImpl&>(*graph_).forward(groups, qp, lambdas, get_loss_info, current_poc);
        return {std::move(output.loss.mean), std::move(output.info)};
    }
    auto output = static_cast<DmcHtTrainingImpl&>(*graph_).forward(groups, qp, lambdas, get_loss_info, current_poc);
    return {std::move(output.loss.mean), std::move(output.info)};
}

}  // namespace uf::training
