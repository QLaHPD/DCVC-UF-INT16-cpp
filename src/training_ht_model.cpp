// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#include "uf/training_ht_model.hpp"
#include "uf/process.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <utility>

namespace uf::training {
namespace {
constexpr std::int64_t intra_source_channels = 3 * 8 * 8;
constexpr std::int64_t group_frames = 8;
constexpr std::int64_t source_channels = intra_source_channels * group_frames;
constexpr std::int64_t latent_channels = 256;
constexpr std::int64_t hyper_channels = 128;
constexpr std::int64_t decoder_channels = 512;
constexpr std::int64_t memory_channels = 512;
constexpr std::int64_t reconstruction_channels = 256;

bool hts_structure(const std::string& structure) {
    if (structure != "hts" && structure != "htl") {
        throw std::invalid_argument("HT training requires hts or htl model structure");
    }
    return structure == "hts";
}

torch::nn::Sequential reconstruction_branch(std::int64_t depth_blocks) {
    torch::nn::Sequential branch;
    branch->push_back(DepthConvBlock(decoder_channels, reconstruction_channels));
    for (std::int64_t index = 1; index < depth_blocks; ++index) {
        branch->push_back(DepthConvBlock(reconstruction_channels, reconstruction_channels));
    }
    branch->push_back(torch::nn::Conv2d(
        torch::nn::Conv2dOptions(reconstruction_channels, intra_source_channels, 1)));
    return branch;
}
}  // namespace

HtDepthStackImpl::HtDepthStackImpl(std::int64_t in, std::int64_t out,
                                 std::int64_t count, bool dcb2, std::int64_t final_channels) {
    if (count <= 0) throw std::invalid_argument("HT convolution stack must contain a depth block");
    conv = register_module("conv", torch::nn::Sequential());
    conv->push_back(DepthConvBlock(in, out, dcb2));
    for (std::int64_t index = 1; index < count; ++index) {
        conv->push_back(DepthConvBlock(out, out, dcb2));
    }
    if (final_channels > 0) {
        conv->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(out, final_channels, 1)));
    }
}

torch::Tensor HtDepthStackImpl::forward(const torch::Tensor& input) {
    return run_tensor([layers = conv](const CheckpointInputs& args) mutable {
        return layers->forward(args.at(0));
    }, {input});
}

torch::Tensor HtDepthStackImpl::forward(const torch::Tensor& input, const torch::Tensor& context) {
    return run_tensor([layers = conv](const CheckpointInputs& args) mutable {
        return layers->forward(torch::cat({args.at(0), args.at(1)}, 1));
    }, {input, context});
}

HtEncoderImpl::HtEncoderImpl(bool is_hts) {
    conv1 = register_module("conv1", torch::nn::Sequential());
    conv1->push_back(DepthConvBlock(source_channels + decoder_channels, decoder_channels, is_hts));
    for (int index = 1; index < (is_hts ? 6 : 7); ++index) {
        conv1->push_back(DepthConvBlock(decoder_channels, decoder_channels, is_hts));
    }
    down = register_module("down",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(decoder_channels, latent_channels, 3)
                         .stride(2).padding(1)));
}

torch::Tensor HtEncoderImpl::forward(const torch::Tensor& input, const torch::Tensor& context,
                                   const torch::Tensor& step) {
    return run_tensor([layers = conv1, reduction = down](const CheckpointInputs& args) mutable {
        const auto unpacked = torch::pixel_unshuffle(args.at(0), 8);
        const auto feature = layers->forward(torch::cat({unpacked, args.at(1)}, 1));
        return reduction->forward(feature * args.at(2));
    }, {input, context, step});
}

HtDecoderImpl::HtDecoderImpl(bool is_hts) {
    up = register_module("up",
        SubpelConv2x(latent_channels, decoder_channels, is_hts ? 1 : 3, is_hts ? 0 : 1, !is_hts));
    conv1 = register_module("conv1", torch::nn::Sequential());
    conv1->push_back(DepthConvBlock(decoder_channels * 2, decoder_channels, is_hts));
    for (int index = 1; index < (is_hts ? 7 : 11); ++index) {
        conv1->push_back(DepthConvBlock(decoder_channels, decoder_channels, is_hts));
    }
}

torch::Tensor HtDecoderImpl::forward(const torch::Tensor& input, const torch::Tensor& context,
                                   const torch::Tensor& step) {
    return run_tensor([upsample = up, layers = conv1](const CheckpointInputs& args) mutable {
        const auto feature = upsample->forward(args.at(0));
        return layers->forward(torch::cat({feature, args.at(1)}, 1)) * args.at(2);
    }, {input, context, step});
}

HtHyperEncoderImpl::HtHyperEncoderImpl(bool is_hts) {
    conv = register_module("conv", torch::nn::Sequential());
    conv->push_back(DepthConvBlock(latent_channels, latent_channels));
    conv->push_back(ResidualBlockWithStride2(latent_channels, latent_channels, false, !is_hts));
    conv->push_back(ResidualBlockWithStride2(latent_channels, hyper_channels, false, !is_hts));
}

torch::Tensor HtHyperEncoderImpl::forward(const torch::Tensor& input) {
    return run_tensor([layers = conv](const CheckpointInputs& args) mutable {
        return layers->forward(args.at(0));
    }, {input});
}

HtHyperDecoderImpl::HtHyperDecoderImpl(bool is_hts) {
    conv = register_module("conv", torch::nn::Sequential());
    conv->push_back(ResidualBlockUpsample(hyper_channels, latent_channels, false, !is_hts, !is_hts));
    conv->push_back(ResidualBlockUpsample(latent_channels, latent_channels, false, !is_hts, !is_hts));
    conv->push_back(DepthConvBlock(latent_channels, latent_channels));
}

torch::Tensor HtHyperDecoderImpl::forward(const torch::Tensor& input) {
    return run_tensor([layers = conv](const CheckpointInputs& args) mutable {
        return layers->forward(args.at(0));
    }, {input});
}

HtTemporalPriorImpl::HtTemporalPriorImpl(bool is_hts) {
    conv = register_module("conv",
        ResidualBlockWithStride2(decoder_channels, latent_channels * 2, false, !is_hts));
}

torch::Tensor HtTemporalPriorImpl::forward(const torch::Tensor& input, const torch::Tensor& step) {
    return run_tensor([layer = conv](const CheckpointInputs& args) mutable {
        return layer->forward(args.at(0) * args.at(1));
    }, {input, step});
}

HtReconHeadImpl::HtReconHeadImpl(bool small) : is_hts(small) {
    if (is_hts) {
        conv1 = register_module("conv1", torch::nn::ModuleList());
        for (int index = 0; index < group_frames / 2; ++index) {
            torch::nn::Sequential common;
            common->push_back(DepthConvBlock(decoder_channels, decoder_channels));
            conv1->push_back(common);
        }
        conv2 = register_module("conv2", torch::nn::ModuleList());
        for (int index = 0; index < group_frames; ++index) {
            conv2->push_back(reconstruction_branch(3));
        }
    } else {
        conv = register_module("conv", torch::nn::ModuleList());
        for (int index = 0; index < group_frames; ++index) {
            conv->push_back(reconstruction_branch(5));
        }
    }
}

std::vector<torch::Tensor> HtReconHeadImpl::forward(const torch::Tensor& input) {
    return run_tensors([small = is_hts, first = conv1, second = conv2, independent = conv]
                       (const CheckpointInputs& args) mutable {
        CheckpointInputs output;
        output.reserve(group_frames);
        torch::Tensor common;
        for (std::size_t index = 0; index < group_frames; ++index) {
            ProcessInterrupt::check();
            torch::Tensor image;
            if (small) {
                if (index % 2 == 0) common = first->at<torch::nn::SequentialImpl>(index / 2).forward(args.at(0));
                image = second->at<torch::nn::SequentialImpl>(index).forward(common);
            } else image = independent->at<torch::nn::SequentialImpl>(index).forward(args.at(0));
            output.push_back(torch::pixel_shuffle(image, 8));
        }
        return output;
    }, {input});
}

torch::Tensor HtReconHeadImpl::for_reset(const torch::Tensor& input) {
    ProcessInterrupt::check();
    return run_tensor([small = is_hts, first = conv1, second = conv2, independent = conv]
                      (const CheckpointInputs& args) mutable {
        if (small) {
            const auto common = first->at<torch::nn::SequentialImpl>(group_frames / 2 - 1).forward(args.at(0));
            return second->at<torch::nn::SequentialImpl>(group_frames - 1).forward(common);
        }
        return independent->at<torch::nn::SequentialImpl>(group_frames - 1).forward(args.at(0));
    }, {input});
}

DmcHtTrainingImpl::DmcHtTrainingImpl(const std::string& structure)
    : CompressionTrainingImpl(hyper_channels), is_hts(hts_structure(structure)) {
    feature_adaptor_i = register_module("feature_adaptor_i",
        HtDepthStack(intra_source_channels, memory_channels, is_hts ? 4 : 3, is_hts));
    feature_adaptor_m = register_module("feature_adaptor_m",
        HtDepthStack(memory_channels + decoder_channels, memory_channels, is_hts ? 6 : 10, is_hts));
    feature_extractor = register_module("feature_extractor",
        HtDepthStack(memory_channels, decoder_channels, is_hts ? 5 : 2, is_hts));
    encoder = register_module("encoder", HtEncoder(is_hts));
    hyper_encoder = register_module("hyper_encoder", HtHyperEncoder(is_hts));
    hyper_decoder = register_module("hyper_decoder", HtHyperDecoder(is_hts));
    temporal_prior_encoder = register_module("temporal_prior_encoder", HtTemporalPrior(is_hts));
    y_prior_fusion = register_module("y_prior_fusion",
        HtDepthStack(latent_channels * 3, latent_channels * 3, 3, false, latent_channels * 3));
    y_spatial_prior_reduction = register_module("y_spatial_prior_reduction",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(latent_channels * 3, latent_channels, 1)));
    y_spatial_prior_adaptor_1 = register_module("y_spatial_prior_adaptor_1",
        DepthConvBlock(latent_channels * 2, latent_channels * 2, false, false, true));
    y_spatial_prior_adaptor_2 = register_module("y_spatial_prior_adaptor_2",
        DepthConvBlock(latent_channels * 2, latent_channels * 2, false, false, true));
    y_spatial_prior_adaptor_3 = register_module("y_spatial_prior_adaptor_3",
        DepthConvBlock(latent_channels * 2, latent_channels * 2, false, false, true));
    y_spatial_prior = register_module("y_spatial_prior",
        HtDepthStack(latent_channels * 2, latent_channels * 2, 3, false, latent_channels * (is_hts ? 1 : 2)));
    decoder = register_module("decoder", HtDecoder(is_hts));
    recon_head = register_module("recon_head", HtReconHead(is_hts));
    // Python's q_* torch.ones factories follow the default floating dtype.
    const auto options = torch::TensorOptions();
    q_encoder = register_parameter("q_encoder", torch::ones({qp_count, decoder_channels}, options));
    q_decoder = register_parameter("q_decoder", torch::ones({qp_count, decoder_channels}, options));
    q_feature = register_parameter("q_feature", torch::ones({qp_count, decoder_channels}, options));
    initialize_weights();
}

void DmcHtTrainingImpl::set_use_checkpointing(bool enabled) {
    feature_adaptor_i->set_use_checkpointing(enabled);
    feature_adaptor_m->set_use_checkpointing(enabled);
    feature_extractor->set_use_checkpointing(enabled);
    temporal_prior_encoder->set_use_checkpointing(enabled);
    encoder->set_use_checkpointing(enabled);
    decoder->set_use_checkpointing(enabled);
    recon_head->set_use_checkpointing(enabled);
    hyper_encoder->set_use_checkpointing(enabled);
    hyper_decoder->set_use_checkpointing(enabled);
    y_prior_fusion->set_use_checkpointing(enabled);
    y_spatial_prior->set_use_checkpointing(enabled);
}

void DmcHtTrainingImpl::clear_dpb() {
    ref_feature = torch::Tensor{};
    memory = torch::Tensor{};
    ctx = torch::Tensor{};
}

void DmcHtTrainingImpl::add_ref_feature_from_frame(const torch::Tensor& frame) {
    ProcessInterrupt::check();
    ref_feature = torch::pixel_unshuffle(frame, 8);
}

void DmcHtTrainingImpl::apply_feature_adaptor() {
    ProcessInterrupt::check();
    if (!ref_feature.defined()) throw std::runtime_error("HT training requires an initial reference frame");
    if (!memory.defined()) memory = feature_adaptor_i->forward(ref_feature);
    else memory = feature_adaptor_m->forward(memory, ref_feature);
    ctx = feature_extractor->forward(memory);
}

void DmcHtTrainingImpl::set_ref_feature(const torch::Tensor& feature, bool reset_feature_memory) {
    ref_feature = feature;
    if (reset_feature_memory) {
        const auto reconstructed = recon_head->for_reset(feature);
        clear_dpb();
        ref_feature = reconstructed;
    }
}

HtRateDistortion DmcHtTrainingImpl::forward_one_frame(const torch::Tensor& input,
    const torch::Tensor& qp, bool reset_feature_memory) {
    ProcessInterrupt::check();
    // Match Python's QP selection followed by tensor operations. A shared QP
    // may broadcast across the batch; do not impose one index per input item.
    const auto q_enc = select_q(q_encoder, qp);
    const auto q_dec = select_q(q_decoder, qp);
    const auto q_ctx = select_q(q_feature, qp);
    apply_feature_adaptor();
    const auto y = encoder->forward(input, ctx, q_enc);
    ProcessInterrupt::check();
    const auto z = hyper_encoder->forward(y);
    const auto z_hat = quantize(z);
    const auto temporal_parameters = temporal_prior_encoder->forward(memory, q_ctx);
    const auto hyper_parameters = hyper_decoder->forward(z_hat);
    const auto parameters = y_prior_fusion->forward(hyper_parameters, temporal_parameters);
    const std::array<Combine, 3> adaptors{
        [this](const torch::Tensor& reconstructed, const torch::Tensor& common) {
            return y_spatial_prior_adaptor_1->forward(torch::cat({reconstructed, common}, 1));
        },
        [this](const torch::Tensor& reconstructed, const torch::Tensor& common) {
            return y_spatial_prior_adaptor_2->forward(torch::cat({reconstructed, common}, 1));
        },
        [this](const torch::Tensor& reconstructed, const torch::Tensor& common) {
            return y_spatial_prior_adaptor_3->forward(torch::cat({reconstructed, common}, 1));
        }
    };
    const auto prior = prior_four(y, torch::Tensor{}, torch::Tensor{}, parameters,
        [this](const torch::Tensor& common) { return y_spatial_prior_reduction->forward(common); },
        adaptors, [this](const torch::Tensor& prediction) { return y_spatial_prior->forward(prediction); },
        !is_hts);
    const auto feature = decoder->forward(prior.reconstruction, ctx, q_dec);
    HtRateDistortion result;
    result.reconstructions = recon_head->forward(feature);
    set_ref_feature(feature, reset_feature_memory);
    const auto y_for_bits = add_noise(prior.residual);
    const auto z_for_bits = add_noise(z);
    const auto bit_y = y_bits(y_for_bits, prior.scales);
    const auto bit_z = z_bits(z_for_bits, qp);
    const auto originals = input.chunk(frame_delay, 1);
    const auto paired_frames = std::min(originals.size(), result.reconstructions.size());
    result.mse.reserve(paired_frames);
    for (std::size_t index = 0; index < paired_frames; ++index) {
        ProcessInterrupt::check();
        result.mse.push_back(mse(originals[index], result.reconstructions[index]));
    }
    result.bits_y = bit_y.sum({1, 2, 3});
    result.bits_z = bit_z.sum({1, 2, 3});
    // bpp here is per group divided by one frame's pixels; the reference does
    // not divide by frame_delay in its training rate/distortion objective.
    result.bpp = (result.bits_y + result.bits_z) / (input.size(2) * input.size(3));
    return result;
}

RateDistortion DmcHtTrainingImpl::rd_info(const HtRateDistortion& result) {
    RateDistortion output;
    output.bits_y = result.bits_y;
    output.bits_z = result.bits_z;
    output.bpp = result.bpp;
    output.mse = mse_eight_frame_sum(result.mse, {1.5, 0.16, 0.4});
    return output;
}

HtTrainingForward DmcHtTrainingImpl::forward(const torch::Tensor& input, const torch::Tensor& qp,
    const torch::Tensor& lambdas, bool get_loss_info, std::int64_t) {
    HtTrainingForward output;
    output.result = forward_one_frame(input, qp);
    output.weighted = rd_info(output.result);
    output.loss = rate_distortion_loss(output.weighted, lambdas);
    ref_feature = ref_feature.detach();
    memory = memory.detach();
    if (get_loss_info) output.info = loss_info(output.weighted, output.loss, input.size(2) * input.size(3));
    return output;
}

HtTrainingForward DmcHtTrainingImpl::forward(const std::vector<torch::Tensor>& groups,
    const torch::Tensor& qp, const torch::Tensor& lambdas, bool get_loss_info, std::int64_t) {
    HtTrainingForward output;
    std::vector<torch::Tensor> losses;
    losses.reserve(groups.size());
    for (const auto& group : groups) {
        ProcessInterrupt::check();
        output.result = forward_one_frame(group, qp);
        output.weighted = rd_info(output.result);
        output.loss = rate_distortion_loss(output.weighted, lambdas);
        losses.push_back(output.loss.mean);
    }
    if (get_loss_info) output.info = loss_info(output.weighted, output.loss, groups.at(0).size(2) * groups.at(0).size(3));
    output.loss.mean = torch::stack(losses).mean();
    return output;
}

}  // namespace uf::training
