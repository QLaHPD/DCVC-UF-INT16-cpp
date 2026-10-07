// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#include "uf/training_schedules.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace uf::training {
namespace {

std::size_t strategy_index(std::int64_t epoch, std::size_t row_count) {
    if (!row_count) throw std::logic_error("Empty training strategy");
    if (epoch >= 0) {
        return static_cast<std::size_t>(std::min<std::uint64_t>(
            static_cast<std::uint64_t>(epoch), row_count - 1));
    }
    // Python uses rows[min(len(rows) - 1, epoch)]; a negative epoch
    // indexes from the end. Avoid negating INT64_MIN directly.
    const auto from_end = static_cast<std::uint64_t>(-(epoch + 1)) + 1;
    if (from_end > row_count) throw std::out_of_range("Training strategy index out of range");
    return row_count - static_cast<std::size_t>(from_end);
}

}  // namespace

std::vector<ImageTrainingStage> image_training_strategy() {
    std::vector<ImageTrainingStage> rows;
    const auto add = [&](std::size_t count, ImageTrainingStage row) { rows.insert(rows.end(), count, row); };
    add(45, {0, 2e-4, 256, 256});
    add(25, {49, 5e-5, 256, 256});
    add(20, {69, 1e-5, 256, 256});
    add(5, {90, 2e-4, 512, 512});
    add(4, {95, 5e-5, 512, 512});
    add(4, {99, 1e-5, 512, 512});
    add(2, {103, 1e-6, 512, 512});
    add(1, {105, 1e-6, 512, 512});
    return rows;
}

std::vector<VideoTrainingStage> video_training_strategy(
    const std::string& scheduling, std::int64_t frame_delay, const std::string& model_structure) {
    if (model_structure != "ld" && model_structure != "hts" && model_structure != "htl") {
        throw std::invalid_argument("Unknown training model structure: " + model_structure);
    }
    std::vector<VideoTrainingStage> rows;
    const auto add = [&](std::size_t count, VideoTrainingStage row) { rows.insert(rows.end(), count, row); };
    if (model_structure == "ld") {
        if (scheduling == "stage0") {
            add(5, {0, 1e-4, 2, false, 256, 256});
            add(5, {5, 1e-4, 3, false, 256, 256});
            add(45, {10, 1e-4, 6, false, 256, 256});
            add(1, {55, 1e-4, 6, false, 256, 256});
        } else if (scheduling == "stage1") {
            add(5, {0, 5e-5, 8, true, 256, 256});
            add(5, {5, 5e-5, 16, true, 256, 256});
            add(5, {10, 5e-5, 24, true, 256, 256});
            add(15, {15, 5e-5, 32, true, 256, 256});
            add(7, {30, 5e-6, 32, true, 256, 256});
            add(1, {37, 5e-6, 32, true, 256, 256});
        } else if (scheduling == "stage2") {
            add(14, {0, 5e-5, 33, true, 512, 512});
            add(4, {14, 5e-6, 33, true, 512, 512});
            add(7, {18, 2e-5, 49, true, 512, 512});
            add(2, {25, 2e-6, 49, true, 512, 512});
            add(7, {27, 5e-6, 65, true, 512, 512});
            add(6, {34, 2e-6, 65, true, 512, 512});
            add(1, {40, 2e-6, 65, true, 512, 512});
        } else if (scheduling == "stage3") {
            add(2, {0, 2e-6, 97, true, 512, 512});
            add(2, {2, 5e-7, 129, true, 512, 512});
            add(1, {4, 5e-7, 129, true, 512, 512});
        } else throw std::invalid_argument("Unknown training scheduling: " + scheduling);
        return rows;
    }
    if (frame_delay <= 0 || frame_delay > (std::numeric_limits<std::int64_t>::max() - 1) / 4) {
        throw std::invalid_argument("Invalid training frame delay");
    }
    if (scheduling == "stage0") {
        add(5, {0, 1e-4, 1 + frame_delay, false, 256, 256});
        add(5, {5, 1e-4, 1 + 2 * frame_delay, false, 256, 256});
        add(35, {10, 1e-4, 1 + 4 * frame_delay, false, 256, 256});
        add(1, {45, 1e-4, 1 + 4 * frame_delay, false, 256, 256});
    } else if (scheduling == "stage1") {
        add(2, {0, 5e-5, 17, true, 256, 256});
        add(1, {2, 5e-5, 25, true, 256, 256});
        add(3, {3, 5e-5, 33, true, 256, 256});
        add(4, {6, 5e-6, 33, true, 256, 256});
        add(1, {10, 5e-6, 33, true, 256, 256});
    } else if (scheduling == "stage2") {
        add(10, {0, 5e-5, 33, true, 512, 512});
        add(10, {10, 5e-5, 49, true, 512, 512});
        add(12, {20, 1e-5, 65, true, 512, 512});
        add(8, {32, 2e-6, 65, true, 512, 512});
        add(1, {40, 2e-6, 65, true, 512, 512});
    } else if (scheduling == "stage3") {
        add(2, {0, 1e-5, 97, true, 512, 512});
        add(2, {2, 2e-6, 129, true, 512, 512});
        add(1, {4, 2e-6, 129, true, 512, 512});
    } else throw std::invalid_argument("Unknown training scheduling: " + scheduling);
    return rows;
}

ImageTrainingStage image_training_stage(std::int64_t epoch) {
    const auto rows = image_training_strategy();
    return rows[strategy_index(epoch, rows.size())];
}

VideoTrainingStage video_training_stage(std::int64_t epoch, const std::string& scheduling,
    std::int64_t frame_delay, const std::string& model_structure) {
    const auto rows = video_training_strategy(scheduling, frame_delay, model_structure);
    return rows[strategy_index(epoch, rows.size())];
}

std::vector<double> training_lambdas(const std::vector<double>& endpoints, std::int64_t qp_count) {
    if (endpoints.size() < 2 || qp_count < 0) {
        throw std::invalid_argument("Training lambdas require two endpoints and a nonnegative QP count");
    }
    std::vector<double> output;
    if (static_cast<std::uint64_t>(qp_count) > output.max_size()) {
        throw std::overflow_error("Training QP count exceeds addressable memory");
    }
    output.reserve(static_cast<std::size_t>(qp_count));
    if (!qp_count) return output;
    const auto start = std::log(endpoints[0]), stop = std::log(endpoints[1]);
    const auto delta = stop - start;
    const auto divisor = qp_count > 1 ? static_cast<double>(qp_count - 1) : 0.0;
    const auto step = qp_count > 1 ? delta / divisor : std::numeric_limits<double>::quiet_NaN();
    for (std::int64_t qp = 0; qp < qp_count; ++qp) {
        // np.linspace separates the multiplication and addition, forces the
        // final endpoint, and has a special path when step underflows to zero.
        volatile double offset = qp_count == 1 ? 0.0 : static_cast<double>(qp);
        if (qp_count > 1) {
            if (step == 0.0) { offset = offset / divisor; offset = offset * delta; }
            else offset = offset * step;
        } else offset = offset * delta;
        const auto value = qp_count > 1 && qp == qp_count - 1 ? stop : start + offset;
        output.push_back(std::exp(value));
    }
    return output;
}

}  // namespace uf::training
