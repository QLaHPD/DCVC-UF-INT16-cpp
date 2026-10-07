// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#include "uf/training_workflow.hpp"
#include "uf/process.hpp"
#include "uf/cli.hpp"

#include <torch/cuda.h>

#include <chrono>
#include <csignal>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace uf::training {
namespace {

struct Arguments {
    ImageTrainingOptions image;
    VideoTrainingOptions video;
    TrainingRank rank;
};

std::int64_t integer(const std::string& value, const std::string& name) {
    return cli::signed_integer(value, name);
}

double number(const std::string& value, const std::string& name) {
    return cli::number(value, name);
}

int rank_number(const std::string& value, const std::string& name) {
    const auto result = integer(value, name);
    if (result < 0 || result > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(name + " is outside the rank range");
    }
    return static_cast<int>(result);
}

Arguments training_arguments(const cli::Options& values, bool video, bool private_worker) {
    Arguments args;
    auto& common = video ? static_cast<CommonTrainingOptions&>(args.video)
                         : static_cast<CommonTrainingOptions&>(args.image);
    auto& epochs = video ? args.video.epochs : args.image.epochs;
    if (values.count("__batch_size")) common.batch_size = integer(values.at("__batch_size"), "--batch_size");
    if (values.count("__epochs")) epochs = integer(values.at("__epochs"), "--epochs");
    if (values.count("__num_workers")) {
        common.num_workers = integer(values.at("__num_workers"), "--num_workers");
    }
    common.save_dir = values.at("__save_dir");
    common.train_dataset = values.at("__train_dataset");
    common.lambdas.clear();
    for (const auto& value : cli::list(values, "__lambdas")) common.lambdas.push_back(number(value, "--lambdas"));
    if (video) {
        args.video.model_path_i = values.at("__model_path_i");
        if (values.count("__model_structure")) args.video.model_structure = values.at("__model_structure");
        if (values.count("__pretrain_path")) args.video.pretrain_path = values.at("__pretrain_path");
        if (values.count("__training_scheduling")) args.video.training_scheduling = values.at("__training_scheduling");
    }
    if (private_worker) {
        args.rank.rank = rank_number(values.at("__rank"), "--_rank");
        args.rank.world_size = rank_number(values.at("__world_size"), "--_world_size");
    }
    if (private_worker && (args.rank.world_size <= 0 || args.rank.rank >= args.rank.world_size)) {
        throw std::invalid_argument("Invalid private training worker rank/world size");
    }
    return args;
}

void help(bool video, std::ostream& output = std::cout) {
    cli::help(output, video ? "train-video" : "train-image");
}

void stop_workers(std::vector<std::unique_ptr<ChildProcess>>& workers) noexcept {
    for (const auto& worker : workers) worker->signal(SIGTERM);
    // Match the reference spawn failure cleanup: allow TERM for 30 seconds,
    // then let ChildProcess destruction kill and reap any surviving groups.
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (std::chrono::steady_clock::now() < deadline) {
        bool finished = true;
        for (const auto& worker : workers) {
            try { if (!worker->poll_for_cleanup()) finished = false; }
            catch (...) { finished = false; }
        }
        if (finished) return;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    for (const auto& worker : workers) worker->signal(SIGKILL);
}

int launch_workers(int argc, char** argv, bool video, int world_size) {
    const auto executable = std::filesystem::read_symlink("/proc/self/exe").string();
    std::vector<std::unique_ptr<ChildProcess>> workers;
    workers.reserve(static_cast<std::size_t>(world_size));
    try {
        for (int rank = 0; rank < world_size; ++rank) {
            ProcessInterrupt::check();
            std::vector<std::string> command{executable, video ? "_train_video_worker" : "_train_image_worker"};
            for (int index = 2; index < argc; ++index) command.emplace_back(argv[index]);
            command.insert(command.end(), {"--_rank", std::to_string(rank), "--_world_size", std::to_string(world_size)});
            workers.push_back(std::make_unique<ChildProcess>(command, std::vector<std::pair<int, int>>{}, true));
        }
        for (;;) {
            ProcessInterrupt::check();
            bool finished = true;
            for (std::size_t rank = 0; rank < workers.size(); ++rank) {
                const auto status = workers[rank]->poll();
                if (!status) finished = false;
                else if (*status != 0) {
                    throw std::runtime_error("Training rank " + std::to_string(rank) +
                                             " exited with status " + std::to_string(*status));
                }
            }
            if (finished) return 0;
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    } catch (...) {
        stop_workers(workers);
        throw;
    }
}

}  // namespace

int run_training_command(int argc, char** argv, bool video, bool private_worker) {
    cli::Options values;
    try { values = cli::parse(argc, argv); }
    catch (const cli::HelpRequested&) { help(video); return 0; }
    catch (const cli::ParseError& error) {
        help(video, std::cerr);
        std::cerr << "dcvc-uf " << (video ? "train-video" : "train-image")
                  << ": error: " << error.what() << '\n';
        return 2;
    }
    // The schema checks syntax, choices and required option presence. Supplied
    // empty strings and numeric domain errors belong to the training workflow,
    // rather than argparse's exit-2 path. Native integer-width failures also
    // occur here, after the Python-compatible token conversion.
    const auto args = training_arguments(values, video, private_worker);
    ProcessInterrupt interruption;
    const auto count = torch::cuda::device_count();
    if (!private_worker && count > 1) {
        if (count > static_cast<std::size_t>(std::numeric_limits<c10::DeviceIndex>::max()) + 1) {
            throw std::overflow_error("Visible CUDA device count exceeds the LibTorch device index range");
        }
        const auto batch_size = video ? args.video.batch_size : args.image.batch_size;
        if (batch_size % static_cast<std::int64_t>(count) != 0) {
            throw std::invalid_argument("Training batch size must be divisible by visible CUDA device count");
        }
        if (!training_nccl_available()) {
            throw std::runtime_error("Multiple CUDA devices are visible, but this build needs UF_BUILD_TRAINING_NCCL with NCCL-enabled LibTorch");
        }
        return launch_workers(argc, argv, video, static_cast<int>(count));
    }
    if (video) train_video(args.video, args.rank);
    else train_image(args.image, args.rank);
    return 0;
}

}  // namespace uf::training
