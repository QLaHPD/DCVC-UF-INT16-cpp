#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "uf/cli.hpp"

#include "uf/python_integer.hpp"
#include "uf/python_numeric.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <initializer_list>
#include <optional>
#include <ostream>
#include <set>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace uf::cli {
namespace {

bool digit(char value) { return value >= '0' && value <= '9'; }

std::string lower(std::string value) {
    for (auto& letter : value) {
        if (letter >= 'A' && letter <= 'Z') letter += 'a' - 'A';
    }
    return value;
}

std::string integer_text(const std::string& value, const std::string& option) {
    try {
        return PythonInteger(value).decimal();
    } catch (const std::invalid_argument& error) {
        throw std::invalid_argument(option + " must be an integer: " + error.what());
    }
}

std::string number_text(std::string_view value, const std::string& option) {
    try { return python_float_text(value); }
    catch (const std::invalid_argument&) {
        throw std::invalid_argument(option + " must be a decimal number");
    }
}

std::string key(const std::string& option) {
    auto result = option;
    std::replace(result.begin(), result.end(), '-', '_');
    return result;
}

enum class Kind { text, integer, cuda_index, qp, complexity, number, positive_number, threshold,
                  boolean, legacy_boolean, python_boolean, device };
enum class Shape { one, many, flag, optional_value };

struct Spec {
    std::string spelling, destination;
    Kind kind{Kind::text};
    Shape shape{Shape::one};
    bool flag_value{true};
    std::vector<std::string> choices;
};

class UnknownOption : public std::invalid_argument {
public:
    using std::invalid_argument::invalid_argument;
};

class Schema {
public:
    std::vector<Spec> specs;

    void add(std::string spelling, Kind kind = Kind::text, Shape shape = Shape::one,
             std::string destination = {}, std::vector<std::string> choices = {},
             bool flag_value = true) {
        if (destination.empty()) destination = key(spelling);
        specs.push_back({std::move(spelling), std::move(destination), kind, shape,
                         flag_value, std::move(choices)});
    }

    const Spec* exact(std::string_view spelling) const {
        for (const auto& spec : specs) if (std::string_view(spec.spelling) == spelling) return &spec;
        return nullptr;
    }

    const Spec& resolve(const std::string& spelling) const {
        if (const auto* spec = exact(spelling)) return *spec;
        // Retain the native CLI's full hyphen/underscore aliases. Abbreviations
        // use the declared Python spellings, including ambiguity between aliases.
        if (spelling.rfind("--", 0) == 0) {
            for (const auto& spec : specs) {
                if (key(spec.spelling) == key(spelling)) return spec;
            }
            const Spec* found = nullptr;
            for (const auto& spec : specs) {
                if (spec.spelling.rfind(spelling, 0) != 0) continue;
                if (found) throw std::invalid_argument("Ambiguous option: " + spelling);
                found = &spec;
            }
            if (found) return *found;
        }
        throw UnknownOption("Unknown option: " + spelling);
    }
};

void model_options(Schema& schema, bool automatic_runtime, Shape cuda_indices = Shape::one) {
    for (const auto* name : {"--model_path_i", "--model_path_p", "--prepared_i", "--prepared_p"}) {
        schema.add(name);
    }
    schema.add("--device", Kind::device);
    schema.add("--cuda", Kind::boolean);
    schema.add("--cuda_idx", Kind::cuda_index, cuda_indices);
    schema.add("--runtime", Kind::text, Shape::one, {}, automatic_runtime
        ? std::vector<std::string>{"auto", "fp16", "int16"}
        : std::vector<std::string>{"fp16", "int16"});
}

void structure(Schema& schema) {
    schema.add("--model_structure", Kind::text, Shape::one, {}, {"hts", "htl", "ld"});
}

void policy(Schema& schema) {
    for (const auto* name : {"--auto-delete", "--keep-originals", "--cleanup-dry-run",
                              "--allow-missing-metadata"}) {
        schema.add(name, Kind::text, Shape::flag);
    }
    schema.add("--ui", Kind::text, Shape::one, {}, {"auto", "plain", "tui"});
}

Schema schema_for(const std::string& command) {
    Schema schema;
    schema.add("--help", Kind::text, Shape::flag);
    schema.add("-h", Kind::text, Shape::flag, "__help");
    if (command == "check-setup") {
        schema.add("--require-native", Kind::text, Shape::flag);
        schema.add("--require-int16-native", Kind::text, Shape::flag);
    } else if (command == "check-int16") {
        schema.add("--variant", Kind::text, Shape::one, {}, {"hts", "htl", "ld"});
        schema.add("--cuda_idx", Kind::integer);
        schema.add("--size", Kind::integer);
        schema.add("--cpu-reference", Kind::text, Shape::flag);
        for (const auto* name : {"--prepared_i", "--prepared_p", "--compare", "--output"}) schema.add(name);
    } else if (command == "encode" || command == "stream") {
        model_options(schema, false, Shape::many);
        structure(schema);
        schema.add("--input_file");
        schema.add("--input-file", Kind::text, Shape::one, "__input_file");
        schema.add("--base_root");
        schema.add("--input_folder", Kind::text, Shape::one, "__base_root");
        schema.add("--output_root");
        for (const auto* name : {"--source_urls", "--youtube_channels", "--twitch_channels", "--channel_ids"}) {
            schema.add(name, Kind::text, Shape::many);
        }
        schema.add("--cookies");
        schema.add("--ytdlp_bin");
        schema.add("--yt-dlp", Kind::text, Shape::one, "__ytdlp_bin");
        schema.add("--source-max-height", Kind::integer);
        schema.add("--max-videos", Kind::integer);
        schema.add("--no-thumbnail", Kind::text, Shape::flag);
        schema.add("--recursive", Kind::text, Shape::flag);
        schema.add("--no-recursive", Kind::text, Shape::flag, "__recursive", {}, false);
        schema.add("--qi", Kind::qp, Shape::one, "__qp_i");
        schema.add("--qp_i", Kind::qp);
        schema.add("--qp", Kind::qp, Shape::one, "__qp_p");
        schema.add("--qp_p", Kind::qp);
        schema.add("--target-psnr", Kind::positive_number);
        schema.add("--fps", Kind::positive_number);
        for (const auto* name : {"--resolution", "--procs", "--input_threads", "--prefetch_frames",
                                  "--reset_interval", "--intra_period", "--max_frames"}) {
            schema.add(name, Kind::integer);
        }
        schema.add("--ffmpeg_prefetch", Kind::integer, Shape::one, "__prefetch_frames");
        schema.add("--force_intra_period", Kind::integer, Shape::one, "__intra_period");
        schema.add("--skip_thres", Kind::threshold);
        schema.add("--ff_hwaccel", Kind::text, Shape::one, {}, {"none", "auto", "jetson"});
        schema.add("--audio", Kind::text, Shape::one, {}, {"opus", "none"});
        schema.add("--opus_channels", Kind::text, Shape::one, {}, {"mono", "stereo"});
        schema.add("--opus_bitrate");
        schema.add("--opus_frame_ms", Kind::number, Shape::one, {}, {"2.5", "5", "10", "20", "40", "60"});
        schema.add("--opus_complexity", Kind::complexity);
        schema.add("--opus_vbr", Kind::text, Shape::one, {}, {"on", "off", "constrained"});
        schema.add("--thumbnail_codec", Kind::text, Shape::one, {}, {"keep", "dcvc-intra"});
        schema.add("--thumbnail_qp", Kind::qp);
        schema.add("--shared-work", Kind::text, Shape::flag);
        schema.add("--shared-instance");
        schema.add("--shared-lease-seconds", Kind::integer);
        policy(schema);
    } else if (command == "decode" || command == "verify" || command == "view" || command == "cleanup") {
        model_options(schema, true, command == "decode" ? Shape::many : Shape::one);
        schema.add("--input");
        schema.add("--input_file", Kind::text, Shape::one, "__input");
        if (command == "decode") {
            schema.add("--worker", Kind::integer);
            schema.add("-w", Kind::integer, Shape::one, "__worker");
            schema.add("--input_folder");
            schema.add("--output_folder");
            schema.add("--output_file");
            schema.add("--input_kind", Kind::text, Shape::one, {}, {"all", "videos", "images"});
            schema.add("--recursive", Kind::text, Shape::flag);
        } else if (command == "view") {
            schema.add("--recursive", Kind::text, Shape::flag);
            schema.add("--fps", Kind::positive_number);
            for (const auto* name : {"--start_frame", "--max_width", "--max_height"}) {
                schema.add(name, Kind::integer);
            }
        } else if (command == "cleanup") {
            schema.add("--base_root");
            schema.add("--yes", Kind::text, Shape::flag);
        }
    } else if (command == "manage") {
        for (const auto* name : {"--output_root", "--base_root", "--model_path_i", "--model_path_p",
                                  "--prepared_i", "--prepared_p"}) schema.add(name);
        schema.add("--channel_ids", Kind::text, Shape::many);
        schema.add("--device", Kind::device);
        schema.add("--cuda", Kind::boolean);  // Existing native extension.
        schema.add("--cuda_idx", Kind::cuda_index);
        policy(schema);
    } else if (command == "prepare-int16" || command == "pareto-search") {
        structure(schema);
        for (const auto* name : {"--model_path_i", "--model_path_p", "--prepared_i", "--prepared_p"}) {
            schema.add(name);
        }
        if (command == "pareto-search") {
            schema.add("--input_file");
            schema.add("--input-file", Kind::text, Shape::one, "__input_file");
            schema.add("--output_json");
            schema.add("--device", Kind::device);  // Existing native extension.
            for (const auto* name : {"--resolution", "--cuda_idx", "--trials", "--initial_samples",
                                      "--seed", "--input_threads"}) {
                schema.add(name, std::string_view(name) == "--cuda_idx" ? Kind::cuda_index : Kind::integer);
            }
            schema.add("--skip_thresholds", Kind::threshold, Shape::many);
            schema.add("--resume", Kind::text, Shape::flag);
            schema.add("--no-resume", Kind::text, Shape::flag, "__resume", {}, false);
        }
    } else if (command == "encode-raw") {
        model_options(schema, false);
        schema.add("--input_file");
        schema.add("--output_file");
        schema.add("--variant", Kind::text, Shape::one, {}, {"hts", "htl", "ld"});
        for (const auto* name : {"--width", "--height", "--reset_interval", "--intra_period", "--max_frames"}) {
            schema.add(name, Kind::integer);
        }
        schema.add("--qp_i", Kind::qp);
        schema.add("--qp_p", Kind::qp);
        schema.add("--skip_thres", Kind::threshold);
    } else if (command == "encode-image") {
        for (const auto* name : {"--input_file", "--output_file", "--output_root", "--model_path_i",
                                  "--prepared_i", "--base_root", "--shared-instance"}) schema.add(name);
        schema.add("--device", Kind::device);
        schema.add("--cuda", Kind::boolean);
        schema.add("--cuda_idx", Kind::cuda_index, Shape::many);
        schema.add("--runtime", Kind::text, Shape::one, {}, {"fp16", "int16"});
        schema.add("--thumbnail_codec", Kind::text, Shape::one, {}, {"keep", "dcvc-intra"});
        schema.add("--thumbnail_qp", Kind::qp);
        schema.add("--skip_thres", Kind::threshold);
        schema.add("--shared-work", Kind::text, Shape::flag);
        schema.add("--shared-lease-seconds", Kind::integer);
    } else if (command == "train-image" || command == "train-video" ||
               command == "_train_image_worker" || command == "_train_video_worker") {
        const bool video = command == "train-video" || command == "_train_video_worker";
        schema.add("--batch_size", Kind::integer);
        schema.add("-e", Kind::integer, Shape::one, "__epochs");
        schema.add("--epochs", Kind::integer);
        schema.add("--lambdas", Kind::number, Shape::many);
        schema.add("-n", Kind::integer, Shape::one, "__num_workers");
        schema.add("--num_workers", Kind::integer);
        schema.add("--save_dir");
        schema.add("--train_dataset");
        if (video) {
            schema.add("--model_path_i");
            structure(schema);
            schema.add("--pretrain_path");
            schema.add("--training_scheduling", Kind::text, Shape::one, {},
                       {"stage0", "stage1", "stage2", "stage3"});
        }
        if (command == "_train_image_worker" || command == "_train_video_worker") {
            schema.add("--_rank", Kind::integer, Shape::one, "__rank");
            schema.add("--_world_size", Kind::integer, Shape::one, "__world_size");
        }
    } else if (command == "dcvc2021-test-video") {
        for (const auto* name : {"--i_frame_model_name", "--test_config", "--cuda_device",
                                  "--recon_bin_path", "--output_json_result_path", "--model_type"}) schema.add(name);
        for (const auto* name : {"--i_frame_model_path", "--model_path"}) schema.add(name, Kind::text, Shape::many);
        schema.add("--worker", Kind::integer);
        schema.add("-w", Kind::integer, Shape::one, "__worker");
        for (const auto* name : {"--cuda", "--write_stream", "--write_recon_frame"}) {
            schema.add(name, Kind::legacy_boolean, Shape::optional_value);
        }
    } else if (command == "dcvc-tcm-test-video") {
        for (const auto* name : {"--i_frame_model_name", "--test_config", "--cuda_device", "--stream_path",
                                  "--decoded_frame_path", "--output_path"}) schema.add(name);
        for (const auto* name : {"--i_frame_model_path", "--model_path"}) schema.add(name, Kind::text, Shape::many);
        for (const auto* name : {"--force_frame_num", "--force_intra_period", "--worker", "--verbose"})
            schema.add(name, Kind::integer);
        schema.add("-w", Kind::integer, Shape::one, "__worker");
        for (const auto* name : {"--force_intra", "--cuda", "--write_stream"})
            schema.add(name, Kind::python_boolean, Shape::optional_value);
        schema.add("--save_decoded_frame", Kind::python_boolean);
    } else if (command == "dcvc-hem-test-video") {
        for (const auto* name : {"--i_frame_model_path", "--model_path", "--test_config", "--force_root_path",
                                  "--cuda_device", "--stream_path", "--decoded_frame_path", "--output_path"})
            schema.add(name);
        for (const auto* name : {"--i_frame_q_scales", "--p_frame_y_q_scales", "--p_frame_mv_y_q_scales"})
            schema.add(name, Kind::number, Shape::many);
        for (const auto* name : {"--rate_num", "--force_frame_num", "--force_intra_period", "--worker", "--verbose"})
            schema.add(name, Kind::integer);
        schema.add("-w", Kind::integer, Shape::one, "__worker");
        for (const auto* name : {"--force_intra", "--cuda", "--write_stream"})
            schema.add(name, Kind::python_boolean, Shape::optional_value);
        schema.add("--save_decoded_frame", Kind::python_boolean);
    } else if (command == "dcvc-dc-test-video") {
        for (const auto* name : {"--i_frame_model_path", "--p_frame_model_path", "--test_config", "--cuda_device",
                                  "--stream_path", "--decoded_frame_path", "--output_path"}) schema.add(name);
        for (const auto* name : {"--stream_part_i", "--stream_part_p", "--rate_num", "--force_frame_num",
                                  "--force_intra_period", "--worker", "--verbose"}) schema.add(name, Kind::integer);
        schema.add("-w", Kind::integer, Shape::one, "__worker");
        for (const auto* name : {"--i_frame_q_indexes", "--p_frame_q_indexes"}) schema.add(name, Kind::integer, Shape::many);
        for (const auto* name : {"--ec_thread", "--force_intra", "--cuda", "--write_stream"})
            schema.add(name, Kind::legacy_boolean, Shape::optional_value);
        for (const auto* name : {"--yuv420", "--calc_ssim", "--save_decoded_frame"}) schema.add(name, Kind::legacy_boolean);
    } else if (command == "dcvc-dc-data-to-png") {
        schema.add("--input"); schema.add("--output");
        schema.add("--width", Kind::integer); schema.add("--height", Kind::integer);
    } else if (command == "dcvc2021-download-checkpoints" || command == "dcvc-hem-download-checkpoints" ||
               command == "dcvc-dc-download-checkpoints") {
        schema.add("--output_directory");
    } else if (command == "test-video") {
        schema.add("--skip_thres", Kind::number);
        for (const auto* name : {"--model_path_i", "--model_path_p", "--test_config",
                                  "--force_root_path", "--stream_path", "--output_path"}) schema.add(name);
        for (const auto* name : {"--rate_num", "--force_frame_num", "--force_intra_period",
                                  "--reset_interval", "--worker", "--verbose"}) schema.add(name, Kind::integer);
        schema.add("-w", Kind::integer, Shape::one, "__worker");
        for (const auto* name : {"--qp_i", "--qp_p", "--cuda_idx"}) schema.add(name, Kind::integer, Shape::many);
        for (const auto* name : {"--force_intra", "--calc_ssim", "--check_existing",
                                  "--save_decoded_frame", "--verbose_json"}) schema.add(name, Kind::legacy_boolean);
        structure(schema);
    } else if (command == "test-compress-time") {
        schema.add("--img", Kind::text, Shape::flag);
        structure(schema);
        schema.add("--output_path");
    } else if (command == "compare-bd-rate") {
        schema.add("--compare_between", Kind::text, Shape::one, {}, {"class", "sequence"});
        schema.add("--compare_frame_type", Kind::text, Shape::one, {}, {"default", "all"});
        for (const auto* name : {"--base_method", "--output_path", "--plot_path"}) schema.add(name);
        schema.add("--log_paths", Kind::text, Shape::many);
        schema.add("--plot_scheme", Kind::text, Shape::one, {}, {"combined", "separate"});
        schema.add("--distortion_metrics", Kind::text, Shape::many, {},
                   {"psnr", "msssim", "psnr_y", "psnr_u", "psnr_v", "msssim_y", "msssim_u", "msssim_v"});
        schema.add("--plot_rd_curve", Kind::integer, Shape::one, {}, {"0", "1"});
        schema.add("--rate_exclude", Kind::text, Shape::many);
    } else {
        throw std::invalid_argument("No option schema for " + command);
    }
    return schema;
}

std::string converted(const Spec& spec, const std::string& value,
                      bool check_cuda_index = true) {
    std::string result = value;
    std::optional<double> numeric;
    if (spec.kind == Kind::integer || spec.kind == Kind::cuda_index ||
        spec.kind == Kind::qp || spec.kind == Kind::complexity) {
        result = integer_text(value, spec.spelling);
        if (check_cuda_index && spec.kind == Kind::cuda_index && result.front() == '-') {
            throw std::invalid_argument("--cuda_idx must be nonnegative");
        }
        if (spec.kind == Kind::qp || spec.kind == Kind::complexity) {
            const auto amount = integer(result, spec.spelling);
            const auto maximum = spec.kind == Kind::qp ? 63 : 10;
            if (amount < 0 || amount > maximum) {
                throw std::invalid_argument(spec.spelling + " must be between 0 and " + std::to_string(maximum));
            }
        }
    } else if (spec.kind == Kind::number || spec.kind == Kind::positive_number || spec.kind == Kind::threshold) {
        result = number_text(value, spec.spelling);
        numeric = number(result, spec.spelling);
        if (spec.kind == Kind::positive_number && (!std::isfinite(*numeric) || *numeric <= 0)) {
            throw std::invalid_argument(spec.spelling + " must be positive and finite");
        }
        if (spec.kind == Kind::threshold && (!std::isfinite(*numeric) || *numeric < 0 || *numeric > 16)) {
            throw std::invalid_argument(spec.spelling + " must be between 0 and 16");
        }
    } else if (spec.kind == Kind::legacy_boolean || spec.kind == Kind::python_boolean) {
        const auto normalized = lower(value);
        const auto positive = normalized == "true" || normalized == "t" || normalized == "yes" ||
                              normalized == "y" || normalized == "1";
        const auto negative = normalized == "false" || normalized == "f" || normalized == "no" ||
                              normalized == "n" || normalized == "0";
        if (spec.kind == Kind::python_boolean && !positive && !negative)
            throw std::invalid_argument(spec.spelling + ": Boolean value expected.");
        result = positive ? "true" : "false";
    } else if (spec.kind == Kind::boolean) {
        const auto normalized = lower(value);
        if (normalized == "true" || normalized == "1" || normalized == "yes") result = "true";
        else if (normalized == "false" || normalized == "0" || normalized == "no") result = "false";
        else throw std::invalid_argument(spec.spelling + " expects true or false");
    } else if (spec.kind == Kind::device && value != "cpu" && value != "cuda") {
        // cuda:N is a documented native extension to Python's cpu/cuda choices.
        if (value.rfind("cuda:", 0) != 0 || value.size() == 5) {
            throw std::invalid_argument("--device must be cpu, cuda or cuda:N");
        }
        for (std::size_t index = 5; index < value.size(); ++index) {
            if (!digit(value[index])) throw std::invalid_argument("--device CUDA index must be nonnegative");
        }
    }
    if (!spec.choices.empty()) {
        bool matches = false;
        for (const auto& choice : spec.choices) {
            if (numeric ? *numeric == number(choice, spec.spelling) : result == choice) matches = true;
        }
        if (!matches) throw std::invalid_argument("Invalid value for " + spec.spelling + ": " + value);
    }
    return result;
}

enum class TokenKind { value, optional, delimiter };

struct Token {
    TokenKind kind{TokenKind::value};
    std::string argument, spelling;
    const Spec* spec{};
    std::optional<std::string> attached;
};

std::vector<Token> classify(int argc, char** argv, const Schema& schema) {
    std::vector<Token> result;
    result.reserve(static_cast<std::size_t>(argc - 2));
    bool positional_only = false;
    for (int index = 2; index < argc; ++index) {
        Token token;
        token.argument = argv[index];
        const auto& argument = token.argument;
        if (!positional_only && argument == "--") {
            token.kind = TokenKind::delimiter;
            positional_only = true;
        } else if (!positional_only && argument.size() > 1 && argument.front() == '-') {
            token.spec = schema.exact(argument);
            token.spelling = argument;
            const auto equals = argument.find('=');
            if (!token.spec && equals != std::string::npos) {
                const auto spelling = argument.substr(0, equals);
                if (const auto* spec = schema.exact(spelling)) {
                    token.spec = spec;
                    token.spelling = spelling;
                    token.attached = argument.substr(equals + 1);
                }
            }
            if (!token.spec && argument.rfind("--", 0) == 0) {
                token.spelling = argument.substr(0, equals);
                try { token.spec = &schema.resolve(token.spelling); }
                catch (const UnknownOption&) {}
                if (token.spec && equals != std::string::npos) token.attached = argument.substr(equals + 1);
            } else if (!token.spec && argument[1] != '-') {
                token.spelling = argument.substr(0, 2);
                token.spec = schema.exact(token.spelling);
                if (token.spec) token.attached = argument.substr(2);
            }
            // None of the schemas declares a negative-number-shaped option.
            // Known options/short prefixes take priority over literal spaces.
            if (token.spec || (!python_negative_number(argument) && argument.find(' ') == std::string::npos)) {
                token.kind = TokenKind::optional;
            }
        }
        result.push_back(std::move(token));
    }
    return result;
}

void group(std::set<std::string>& seen, const std::string& destination,
           std::initializer_list<const char*> members) {
    bool belongs = false;
    for (const auto* member : members) if (destination == member) belongs = true;
    if (!belongs) return;
    for (const auto* member : members) {
        if (destination != member && seen.count(member)) {
            throw std::invalid_argument("Conflicting options: " + destination + " and " + member);
        }
    }
    seen.insert(destination);
}

bool present(const Options& values, const char* name) {
    const auto found = values.find(name);
    return found != values.end() && !found->second.empty();
}

bool root_command(const std::string& command) {
    return command == "encode" || command == "stream" || command == "manage" ||
        command == "pareto-search" || command == "prepare-int16" ||
        command == "decode" || command == "verify" || command == "view" || command == "cleanup";
}

// Help metadata only: parsing and its validation phases remain above/below.
bool required_help_option(const std::string& command, const std::string& name) {
    if (command == "encode" || command == "stream" || command == "manage")
        return name == "__output_root";
    if (command == "pareto-search") return name == "__input_file" || name == "__output_json";
    if (command == "check-int16") return name == "__output";
    if (command == "train-image" || command == "train-video")
        return name == "__lambdas" || name == "__save_dir" || name == "__train_dataset" ||
               (command == "train-video" && name == "__model_path_i");
    if (command == "test-video") return name == "__test_config" || name == "__output_path";
    if (command == "compare-bd-rate") return name == "__base_method" || name == "__log_paths";
    if (command == "encode-raw") return name == "__input_file" || name == "__output_file" ||
                                      name == "__width" || name == "__height";
    return false;
}

const char* option_description(const std::string& command, const std::string& name) {
    if (name == "__help") return "Show this help message and exit.";
    if (command == "encode" || command == "stream") {
        if (name == "__source_urls") return "YouTube video or channel URLs; stream without keeping source videos.";
        if (name == "__youtube_channels") return "YouTube channel IDs or channel_ID.txt video lists.";
        if (name == "__twitch_channels") return "Twitch channel names.";
        if (name == "__cookies") return "Cookie file; each yt-dlp call uses a private temporary copy.";
        if (name == "__target_psnr") return "Select the fastest measured configuration meeting benchmark PSNR; smaller bytes break ties.";
        if (name == "__resolution") return "Short edge; preserve aspect ratio. Default: original size.";
        if (name == "__device") return "CPU is the INT16 reference backend.";
        if (name == "__prepared_i") return "Prepared integer image model/table file.";
        if (name == "__prepared_p") return "Prepared integer video model/table file.";
        if (name == "__input_threads") return "FFmpeg decoder threads per worker (default: INT16 CPU budget, up to 4; FP16 1).";
        if (name == "__prefetch_frames") return "Read ahead frames (default: INT16 8, FP16 0); queue capped at 8 MiB or one frame.";
        if (name == "__skip_thres") return "Do not code latent residuals whose predicted scale is at or below this value.";
        if (name == "__shared_work") return "UF always uses cooperative claims.";
        if (name == "__ui") return "Timestamped events, also persisted beside archives.";
    }
    if (command == "encode" || command == "stream" || command == "manage") {
        if (name == "__allow_missing_metadata") return "Permit cleanup without matching info.json files.";
        if (name == "__auto_delete") return "Verify archives and delete only their local originals.";
        if (name == "__keep_originals") return "Retain originals and exit without waiting for TUI cleanup.";
        if (name == "__cleanup_dry_run") return "Preview exact original paths without deleting.";
    }
    if (command == "manage" && name == "__base_root")
        return "Local mount path of originals when reviewing another machine's archives.";
    if (command == "prepare-int16") {
        if (name == "__prepared_i") return "Prepared image output path.";
        if (name == "__prepared_p") return "Prepared video output path.";
    }
    if (command == "decode" || command == "verify" || command == "view" || command == "cleanup") {
        if (name == "__input") return "Flat .bin/.uf.json, .dcvci image or legacy UF archive.";
        if (command == "decode" && name == "__output_file")
            return "Lossless reconstructed Matroska (.mkv) or intra image (.png).";
        if (command == "cleanup" && name == "__base_root")
            return "Remap original relative path on another machine.";
        if (command == "cleanup" && name == "__yes")
            return "Verify archive and delete only its original video.";
    }
    if (command == "train-image" || command == "train-video") {
        if (name == "__num_workers") return "Dataloader workers per trainer.";
        if (name == "__save_dir") return "Path to save models.";
        if (name == "__training_scheduling") return "How to schedule the training strategy.";
    }
    if (command == "test-video" && name == "__worker") return "Worker number.";
    if (command == "check-setup") {
        if (name == "__require_native") return "Fail unless the UF native runtime is ready too.";
        if (name == "__require_int16_native") return "Also require the integer CUDA runtime and entropy decoder API.";
    }
    if (command == "check-int16" && name == "__compare")
        return "Previously saved report from another run/device.";
    if (command == "compare-bd-rate") {
        if (name == "__compare_between") return "Compare performance between different classes or sequences.";
        if (name == "__compare_frame_type") return "Frame type to compare BD-Rate; default is I, P, and all.";
        if (name == "__base_method") return "Name of the anchor model.";
        if (name == "__log_paths") return "Model test-result paths: model name followed by file path.";
        if (name == "__output_path") return "Print to console or save to a TXT or CSV file.";
        if (name == "__plot_path") return "Path to save the plots.";
        if (name == "__plot_scheme") return "RD curve plot scheme.";
        if (name == "__distortion_metrics") return "Distortion metrics used to calculate BD-Rate and plot.";
        if (name == "__plot_rd_curve") return "If 1, plot RD curves.";
    }
    return nullptr;
}

void validate(const Options& values, const std::string& command, const Schema& schema,
              bool semantic_checks) {
    const auto option_name = [&](const char* destination) {
        for (const auto& spec : schema.specs) {
            if (spec.destination == destination) return spec.spelling;
        }
        return std::string(destination);
    };
    const auto require = [&](const char* name) {
        if (!values.count(name)) throw std::invalid_argument("Missing required option: " + option_name(name));
    };
    const auto positive = [&](const char* name) {
        const auto found = values.find(name);
        if (found != values.end() && (found->second == "0" || found->second.front() == '-')) {
            throw std::invalid_argument(option_name(name) + " must be positive");
        }
    };
    if (command == "encode" || command == "stream") {
        if (!values.count("__output_root")) throw std::invalid_argument("Missing --output_root");
        // argparse's required arguments belong to parsing. UF main applies
        // the target preset and expands channels before semantic validation.
        if (!semantic_checks) return;
        if (!present(values, "__input_file") && !present(values, "__base_root") &&
            !present(values, "__source_urls") && !present(values, "__youtube_channels") &&
            !present(values, "__twitch_channels")) {
            throw std::invalid_argument("Provide a local input or remote URLs/channels");
        }
        const bool remote_channels = present(values, "__youtube_channels") ||
            (present(values, "__twitch_channels") && !present(values, "__base_root"));
        if (remote_channels && (present(values, "__input_file") || present(values, "__base_root"))) {
            throw std::invalid_argument("Remote channel inputs cannot be combined with local input");
        }
        for (const auto* name : {"__resolution", "__max_frames", "__procs", "__input_threads", "__max_videos"}) positive(name);
        for (const auto* name : {"__prefetch_frames", "__source_max_height"}) {
            if (values.count(name) && values.at(name).front() == '-') {
                throw std::invalid_argument(std::string(name) + " must be nonnegative");
            }
        }
        if (values.count("__shared_lease_seconds") &&
            signed_integer(values.at("__shared_lease_seconds"), "--shared-lease-seconds") < 30) {
            throw std::invalid_argument("--shared-lease-seconds must be at least 30");
        }
        if (values.count("__reset_interval") && integer(values.at("__reset_interval"), "--reset_interval") < -1) {
            throw std::invalid_argument("--reset_interval must be -1, 0 or positive");
        }
        if (values.count("__intra_period")) {
            const int period = integer(values.at("__intra_period"), "--intra_period");
            const auto variant = values.count("__model_structure") ? values.at("__model_structure") : "hts";
            if (period == 0 || period < -1 || (variant != "ld" && period > 1 && period % 8)) {
                throw std::invalid_argument("--intra_period must be -1, 1, or a valid period for the selected structure");
            }
        }
        const auto runtime = values.count("__runtime") ? values.at("__runtime") : "fp16";
        if ((values.count("__cuda") && values.at("__cuda") == "false") ||
            (values.count("__device") && values.at("__device") == "cpu")) {
            if (runtime != "int16") throw std::invalid_argument("--device cpu requires --runtime int16");
        }
    } else if (command == "check-int16") {
        require("__output");
        if (!semantic_checks) return;
        const auto size = values.count("__size") ? integer(values.at("__size"), "--size") : 32;
        if (size < 16 || size % 16) throw std::invalid_argument("--size must be a positive multiple of 16");
    } else if (command == "pareto-search") {
        if (!values.count("__input_file") || !values.count("__output_json")) {
            throw std::invalid_argument("Pareto search requires --input_file and --output_json");
        }
        if (!semantic_checks) return;
        for (const auto* name : {"__resolution", "__trials", "__initial_samples", "__input_threads"}) positive(name);
        const auto trials = values.count("__trials") ? integer(values.at("__trials"), "--trials") : 2000;
        const auto initial = values.count("__initial_samples") ? integer(values.at("__initial_samples"), "--initial_samples") : 800;
        if (initial > trials) throw std::invalid_argument("--initial_samples cannot exceed --trials");
    } else if (command == "train-image" || command == "train-video" ||
               command == "_train_image_worker" || command == "_train_video_worker") {
        for (const auto* name : {"__save_dir", "__train_dataset", "__lambdas"}) require(name);
        if (command == "train-video" || command == "_train_video_worker") require("__model_path_i");
        if (command == "_train_image_worker" || command == "_train_video_worker") {
            require("__rank"); require("__world_size");
        }
    } else if (command == "dcvc2021-test-video") {
        require("__test_config"); require("__output_json_result_path");
    } else if (command == "test-video" || command == "dcvc-tcm-test-video" || command == "dcvc-hem-test-video" ||
               command == "dcvc-dc-test-video") {
        require("__test_config"); require("__output_path");
    } else if (command == "compare-bd-rate") {
        require("__base_method"); require("__log_paths");
    } else if (command == "manage") {
        if (!values.count("__output_root")) throw std::invalid_argument("Missing --output_root");
    } else if (command == "decode" || command == "verify" || command == "view" || command == "cleanup") {
        if (!semantic_checks) return;
        if (!present(values, "__input") && !(command == "decode" && present(values, "__input_folder"))) {
            throw std::invalid_argument("Provide --input, --input_file, --input_folder or a viewer path");
        }
        if (command == "decode") {
            if (!present(values, "__output_file") && !present(values, "__output_folder")) {
                throw std::invalid_argument("Decode requires --output_file or --output_folder");
            }
            if (present(values, "__input") && present(values, "__input_folder")) {
                throw std::invalid_argument("Select one of --input and --input_folder");
            }
        }
    }
    if (semantic_checks && root_command(command)) {
        for (const auto& index : list(values, "__cuda_idx")) {
            if (!index.empty() && index.front() == '-') {
                throw std::invalid_argument("--cuda_idx must be nonnegative");
            }
        }
        // UF main checks device indexes before the decode worker count.
        if (command == "decode") positive("__worker");
    }
}

}  // namespace

int integer(const std::string& value, const std::string& option) {
    try { return std::stoi(integer_text(value, option)); }
    catch (const std::out_of_range&) { throw std::out_of_range(option + " exceeds the native integer range"); }
}

std::int64_t signed_integer(const std::string& value, const std::string& option) {
    try { return std::stoll(integer_text(value, option)); }
    catch (const std::out_of_range&) { throw std::out_of_range(option + " exceeds the native int64 range"); }
}

std::uint64_t unsigned_integer(const std::string& value, const std::string& option) {
    const auto decimal = integer_text(value, option);
    if (decimal.front() == '-') throw std::invalid_argument(option + " must be nonnegative");
    try { return std::stoull(decimal); }
    catch (const std::out_of_range&) { throw std::out_of_range(option + " exceeds the native uint64 range"); }
}

double number(const std::string& value, const std::string& option) {
    try { return python_float(value); }
    catch (const std::invalid_argument&) { throw std::invalid_argument(option + " must be a number"); }
}

std::vector<std::string> list(const Options& values, const std::string& option) {
    const auto found = values.find(option);
    if (found == values.end()) return {};
    const auto& joined = found->second;
    // execve argv items cannot contain NUL. A terminal NUL identifies the
    // lossless native representation; read old programmatic newline lists too.
    const char separator = joined.find('\0') != std::string::npos ? '\0' : '\n';
    std::vector<std::string> result;
    std::size_t start = 0;
    while (start < joined.size()) {
        const auto end = joined.find(separator, start);
        result.push_back(joined.substr(start, end - start));
        if (end == std::string::npos) break;
        start = end + 1;
    }
    return result;
}

void set_list(Options& values, const std::string& option, const std::vector<std::string>& items) {
    if (items.empty()) { values.erase(option); return; }
    std::string joined;
    for (const auto& item : items) {
        if (item.find('\0') != std::string::npos) throw std::invalid_argument("CLI list item contains NUL");
        joined += item;
        joined.push_back('\0');
    }
    values.insert_or_assign(option, std::move(joined));
}

static Options parse_options(int argc, char** argv) {
    if (argc < 2) throw std::invalid_argument("Missing command");
    const std::string command(argv[1]);
    const auto schema = schema_for(command);
    // argparse classifies every token before taking actions. An ambiguous
    // option after --help is therefore an error; unknown extras, in contrast,
    // remain deferred until after actions/required arguments (and help exits).
    const auto tokens = classify(argc, argv, schema);
    Options result;
    std::set<std::string> sources, cleanup;
    std::optional<std::string> viewer_path;
    std::vector<std::string> extras;
    struct Action {
        const Spec* spec;
        std::string spelling;
        std::vector<std::string> values;
    };
    for (std::size_t index = 0; index < tokens.size();) {
        const auto& token = tokens[index++];
        if (token.kind == TokenKind::delimiter) {
            // A positional action consumes its separator. Commands without a
            // positional action retain it in argparse's unrecognized arguments.
            // A viewer path consumed before an intervening option also leaves
            // a later separator unrecognized.
            if (command != "view" || viewer_path) extras.push_back(token.argument);
            continue;
        }
        if (token.kind == TokenKind::value) {
            if (command == "view" && !viewer_path) {
                viewer_path = token.argument;
                // nargs='?' consumes a directly following positional separator
                // as part of the same action's match.
                if (index < tokens.size() && tokens[index].kind == TokenKind::delimiter) ++index;
            }
            else extras.push_back(token.argument);
            continue;
        }
        if (!token.spec) {
            extras.push_back(token.argument);
            continue;
        }
        std::vector<Action> actions;
        const Spec* selected = token.spec;
        auto spelling = token.spelling;
        auto attached = token.attached;
        for (;;) {
            if (selected->shape == Shape::flag) {
                if (attached) {
                    if (spelling.size() != 2 || spelling[1] == '-' || attached->empty()) {
                        throw std::invalid_argument(spelling + " does not take a value");
                    }
                    // Complete a short-option bundle before taking its actions.
                    // Unknown bundle tails or a missing final argument must fail
                    // before a bundled help action can exit, as in argparse.
                    actions.push_back({selected, spelling, {}});
                    const auto next = std::string("-") + attached->front();
                    selected = schema.exact(next);
                    if (!selected) throw std::invalid_argument(spelling + " has an unknown short-option tail");
                    spelling = next;
                    auto tail = attached->substr(1);
                    if (tail.empty()) attached.reset(); else attached = std::move(tail);
                    continue;
                }
                actions.push_back({selected, spelling, {}});
            } else {
                std::vector<std::string> values;
                if (attached) {
                    // An explicit argument supplies exactly one nargs item.
                    values.push_back(*attached);
                } else {
                    if (index == tokens.size() || tokens[index].kind != TokenKind::value) {
                        if (selected->shape == Shape::optional_value) values.push_back("true");
                        else throw std::invalid_argument(spelling + " needs a value");
                    } else values.push_back(tokens[index++].argument);
                    if (selected->shape == Shape::many) {
                        while (index < tokens.size() && tokens[index].kind == TokenKind::value) {
                            values.push_back(tokens[index++].argument);
                        }
                    }
                }
                actions.push_back({selected, spelling, std::move(values)});
            }
            break;
        }
        for (const auto& action : actions) {
            const auto& spec = *action.spec;
            std::string value;
            std::vector<std::string> items;
            if (spec.shape == Shape::flag) {
                if (spec.destination == "__help") throw HelpRequested();
                value = spec.flag_value ? "true" : "false";
            } else {
                const bool check_cuda_index = !root_command(command);
                if (spec.shape == Shape::one || spec.shape == Shape::optional_value) {
                    value = converted(spec, action.values.front(), check_cuda_index);
                } else {
                    for (const auto& item : action.values) {
                        items.push_back(converted(spec, item, check_cuda_index));
                    }
                }
            }
            if (command == "encode" || command == "stream") {
                group(sources, spec.destination, {"__input_file", "__base_root", "__source_urls"});
            }
            group(cleanup, spec.destination, {"__auto_delete", "__keep_originals", "__cleanup_dry_run"});
            if (spec.shape == Shape::many) {
                set_list(result, spec.destination, items);
            } else {
                result.insert_or_assign(spec.destination, std::move(value));
            }
        }
    }
    if (viewer_path && !present(result, "__input")) result["__input"] = std::move(*viewer_path);
    // Python's path fallback and command dispatch use string truthiness.
    const bool separate_workflow = command == "test-video" || command == "test-compress-time" ||
        command == "compare-bd-rate" || command == "train-image" || command == "train-video" ||
        command == "_train_image_worker" || command == "_train_video_worker";
    if (!separate_workflow) {
        for (const auto* name : {"__model_path_i", "__model_path_p"}) {
            if (result.count(name) && result.at(name).empty()) result.erase(name);
        }
    }
    if (command == "encode" || command == "stream") {
        for (const auto* name : {"__input_file", "__base_root"}) {
            if (result.count(name) && result.at(name).empty()) result.erase(name);
        }
    } else if (command == "decode") {
        for (const auto* name : {"__input", "__input_folder", "__output_folder"}) {
            if (result.count(name) && result.at(name).empty()) result.erase(name);
        }
    }
    validate(result, command, schema, false);
    if (!extras.empty()) {
        std::string error = "Unrecognized arguments:";
        for (const auto& extra : extras) error += " " + extra;
        throw std::invalid_argument(error);
    }
    // These UF commands have no preset/channel transformation. Apply semantic
    // checks after parse_args has accepted every token, including the integer
    // validation script's size check.
    if ((root_command(command) && command != "encode" && command != "stream") ||
        command == "check-int16") {
        validate(result, command, schema, true);
    }
    return result;
}

Options parse(int argc, char** argv) {
    try { return parse_options(argc, argv); }
    catch (const std::invalid_argument& error) { throw ParseError(error.what()); }
    catch (const std::out_of_range& error) { throw ParseError(error.what()); }
}

void validate_encode_options(const Options& values) {
    try { validate(values, "encode", schema_for("encode"), true); }
    catch (const std::invalid_argument& error) { throw ParseError(error.what()); }
    catch (const std::out_of_range& error) { throw ParseError(error.what()); }
}

void help(std::ostream& output, const std::string& command) {
    const auto schema = schema_for(command);
    output << "Usage: dcvc-uf " << command << " [options]" << (command == "view" ? " [PATH]" : "") << "\n\nOptions:\n";
    std::set<std::string> emitted;
    for (const auto& spec : schema.specs) {
        if (!emitted.insert(spec.destination).second) continue;
        output << "  ";
        bool first = true;
        for (const auto& alias : schema.specs) {
            if (alias.destination != spec.destination) continue;
            if (!first) output << ", ";
            output << alias.spelling;
            first = false;
        }
        if (spec.shape == Shape::optional_value) output << " [VALUE]";
        else if (spec.shape != Shape::flag) output << " VALUE" << (spec.shape == Shape::many ? "..." : "");
        if (!spec.choices.empty()) {
            output << " (";
            for (std::size_t index = 0; index < spec.choices.size(); ++index) {
                if (index) output << ", ";
                output << spec.choices[index];
            }
            output << ')';
        }
        if (required_help_option(command, spec.destination)) output << " (required)";
        output << '\n';
        if (const auto* description = option_description(command, spec.destination))
            output << "      " << description << '\n';
    }
    if (command == "decode")
        output << "\nProvide one input: --input or --input_folder. Provide --output_file or --output_folder.\n";
    else if (command == "view")
        output << "\nProvide --input or a positional PATH.\n";
    else if (command == "verify" || command == "cleanup")
        output << "\nProvide --input (or --input_file).\n";
    if (!output) throw std::runtime_error("Cannot write UF command help");
}

}  // namespace uf::cli
