// Model topology follows Microsoft's DCVC-UF modules (MIT license).
#include "uf/model_schema.hpp"

#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace uf {
namespace {

using Shapes = std::map<std::string, std::vector<std::int64_t>>;

struct Schema {
    Shapes shapes;

    void parameter(const std::string& name, std::vector<std::int64_t> dimensions) {
        if (!shapes.emplace(name, std::move(dimensions)).second) {
            throw std::logic_error("Duplicate UF model schema parameter: " + name);
        }
    }

    void conv(const std::string& name, int input, int output, int kernel = 1,
              bool bias = true, int groups = 1) {
        parameter(name + ".weight", {output, input / groups, kernel, kernel});
        if (bias) parameter(name + ".bias", {output});
    }

    void depth(const std::string& name, int input, int output,
               bool dcb2 = false, bool force_adaptor = false) {
        const int hidden = output / (dcb2 ? 2 : 1);
        if (input != output || force_adaptor) conv(name + ".adaptor", input, output);
        conv(name + ".dc.0", output, hidden);
        conv(name + ".dc.2", hidden, hidden, 3, true, hidden);
        conv(name + ".dc.3", hidden, output);
        conv(name + ".ffn.0", output, hidden * 4);
        conv(name + ".ffn.2", hidden, output);
    }

    void depths(const std::string& name, int count, int input, int output,
                bool dcb2 = false) {
        for (int i = 0; i < count; ++i) {
            depth(name + "." + std::to_string(i), i ? output : input, output, dcb2);
        }
    }

    void subpel(const std::string& name, int input, int output,
                int kernel = 1, bool force_bias = false) {
        conv(name + ".conv.0", input, output * 4, kernel, kernel > 1 || force_bias);
    }

    void up(const std::string& name, int input, int output,
            bool dcb2 = false, bool force_bias = false) {
        subpel(name + ".up", input, output, 1, force_bias);
        depth(name + ".conv", output, output, dcb2);
    }

    void down(const std::string& name, int input, int output, bool dcb2 = false) {
        conv(name + ".down", input * 4, output);
        depth(name + ".conv", output, output, dcb2);
    }

    void prior(const std::string& name, int input, int hidden, int output,
                int blocks, bool dcb2 = false) {
        depths(name, blocks, input, hidden, dcb2);
        conv(name + "." + std::to_string(blocks), hidden, output);
    }

    void entropy() {
        parameter("bit_estimator_z.h", {64, 128, 4});
        parameter("bit_estimator_z.b", {64, 128, 4});
        parameter("bit_estimator_z.a", {64, 128, 3});
    }
};

Schema image_schema() {
    Schema s;
    s.entropy();
    s.depth("enc.enc_1", 192, 384);
    s.depths("enc.enc_2", 6, 384, 384);
    s.conv("enc.enc_2.6", 384, 256, 3);
    s.depth("hyper_enc.conv.0", 256, 128);
    s.down("hyper_enc.conv.1", 128, 128);
    s.down("hyper_enc.conv.2", 128, 128);
    s.up("hyper_dec.conv.0", 128, 128);
    s.up("hyper_dec.conv.1", 128, 128);
    s.depth("hyper_dec.conv.2", 128, 256);
    s.prior("y_prior_fusion.conv", 256, 512, 512, 3);
    s.conv("y_spatial_prior_reduction", 512, 256);
    for (int i = 1; i <= 3; ++i) {
        s.depth("y_spatial_prior_adaptor_" + std::to_string(i), 512, 512, false, true);
    }
    s.prior("y_spatial_prior.conv", 512, 512, 512, 3);
    s.up("dec.dec_1.0", 256, 384);
    for (int i = 1; i <= 12; ++i) s.depth("dec.dec_1." + std::to_string(i), 384, 384);
    s.depth("dec.dec_2", 384, 192);
    s.parameter("q_scale_enc", {64, 384});
    s.parameter("q_scale_dec", {64, 384});
    s.parameter("q_scale_y_enc", {64, 256});
    s.parameter("q_scale_y_dec", {64, 256});
    return s;
}

Schema ht_schema(bool hts) {
    Schema s;
    s.entropy();
    s.depths("feature_adaptor_i.conv", hts ? 4 : 3, 192, 512, hts);
    s.depths("feature_adaptor_m.conv", hts ? 6 : 10, 1024, 512, hts);
    s.depths("feature_extractor.conv", hts ? 5 : 2, 512, 512, hts);
    s.depths("encoder.conv1", hts ? 6 : 7, 2048, 512, hts);
    s.conv("encoder.down", 512, 256, 3);
    s.depth("hyper_encoder.conv.0", 256, 256);
    s.down("hyper_encoder.conv.1", 256, 256);
    s.down("hyper_encoder.conv.2", 256, 128);
    s.up("hyper_decoder.conv.0", 128, 256, false, !hts);
    s.up("hyper_decoder.conv.1", 256, 256, false, !hts);
    s.depth("hyper_decoder.conv.2", 256, 256);
    s.down("temporal_prior_encoder.conv", 512, 512);
    s.prior("y_prior_fusion.conv", 768, 768, 768, 3);
    s.conv("y_spatial_prior_reduction", 768, 256);
    for (int i = 1; i <= 3; ++i) {
        s.depth("y_spatial_prior_adaptor_" + std::to_string(i), 512, 512, false, true);
    }
    s.prior("y_spatial_prior.conv", 512, 512, hts ? 256 : 512, 3);
    s.subpel("decoder.up", 256, 512, hts ? 1 : 3, !hts);
    s.depths("decoder.conv1", hts ? 7 : 11, 1024, 512, hts);
    if (hts) {
        for (int i = 0; i < 4; ++i) {
            s.depth("recon_head.conv1." + std::to_string(i) + ".0", 512, 512);
        }
        for (int i = 0; i < 8; ++i) {
            s.prior("recon_head.conv2." + std::to_string(i), 512, 256, 192, 3);
        }
    } else {
        for (int i = 0; i < 8; ++i) {
            s.prior("recon_head.conv." + std::to_string(i), 512, 256, 192, 5);
        }
    }
    s.parameter("q_encoder", {64, 512});
    s.parameter("q_decoder", {64, 512});
    s.parameter("q_feature", {64, 512});
    return s;
}

Schema ld_schema() {
    Schema s;
    s.entropy();
    s.depths("feature_adaptor_i.conv", 4, 192, 256, true);
    s.depths("feature_adaptor_m.conv", 4, 512, 256, true);
    s.depths("feature_extractor.conv", 5, 256, 256, true);
    s.depths("encoder.conv1", 2, 448, 256, true);
    s.depth("encoder.conv2", 256, 256, true);
    s.conv("encoder.down", 256, 128, 3);
    s.depth("hyper_encoder.conv.0", 128, 128, true);
    s.down("hyper_encoder.conv.1", 128, 128, true);
    s.down("hyper_encoder.conv.2", 128, 128, true);
    s.up("hyper_decoder.conv.0", 128, 128, true);
    s.up("hyper_decoder.conv.1", 128, 128, true);
    s.depth("hyper_decoder.conv.2", 128, 128, true);
    s.down("temporal_prior_encoder.conv", 256, 256, true);
    s.prior("y_prior_fusion.conv", 384, 384, 384, 3, true);
    s.prior("y_spatial_prior.conv", 512, 256, 128, 2, true);
    s.subpel("decoder.up", 128, 256);
    s.depths("decoder.conv1", 3, 512, 256, true);
    s.conv("decoder.conv2", 256, 256);
    s.depths("recon_head.conv", 3, 256, 256, true);
    s.conv("recon_head.head", 256, 192);
    s.parameter("q_encoder", {64, 256});
    s.parameter("q_decoder", {64, 256});
    s.parameter("q_feature", {64, 256});
    return s;
}

}  // namespace

void validate_checkpoint_state(const CheckpointState& state, const std::string& variant) {
    Schema schema;
    if (variant == "image") schema = image_schema();
    else if (variant == "hts" || variant == "htl") schema = ht_schema(variant == "hts");
    else if (variant == "ld") schema = ld_schema();
    else throw std::invalid_argument("Unknown UF checkpoint variant");
    for (const auto& [name, shape] : schema.shapes) {
        const auto entry = state.find(name);
        if (entry == state.end()) throw std::runtime_error("UF checkpoint is missing parameter: " + name);
        const auto& tensor = entry->second;
        if (!tensor.defined() || tensor.layout() != torch::kStrided ||
            tensor.sizes().vec() != shape) {
            throw std::runtime_error("UF checkpoint parameter has an invalid shape or layout: " + name);
        }
    }
    for (const auto& [name, tensor] : state) {
        if (!schema.shapes.count(name)) throw std::runtime_error("UF checkpoint has unexpected parameter: " + name);
    }
}

}  // namespace uf
