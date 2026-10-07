#pragma once

#include "uf/cli.hpp"

#include <cstddef>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

namespace uf {

struct DecodeJob {
    std::filesystem::path input, output;
    cli::Options options;
};

struct DecodeOutcome {
    std::filesystem::path input, output;
    // Python's shared execute_jobs calls a successful decode "encoded".
    std::string status, error;
    int returncode{};
};

struct DecodeRun {
    std::vector<DecodeOutcome> outcomes;
    std::size_t decoded{}, failed{};
};

// Each archive gets a fresh executable. Device slots cycle through devices at
// initialization and return to the queue in observed completion order.
DecodeRun decode_jobs(const std::vector<DecodeJob>& jobs, int workers,
                      const std::vector<std::string>& devices);

using DecodeTask = std::function<int(const cli::Options&)>;

// Private _decode_worker entry point. Anonymous RAM files carry the job (fd 0)
// and result (fd 3); fd 8 carries the normal UF event stream to the parent.
int run_decode_worker(const DecodeTask& decode, int input_descriptor = 0,
                      int result_descriptor = 3, int event_descriptor = 8);

}  // namespace uf
