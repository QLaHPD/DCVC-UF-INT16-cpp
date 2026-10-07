// Native counterpart of the original DCVC download_compressai_models.py.
#include "uf/dcvc2021_evaluation.hpp"
#include "uf/cli.hpp"
#include "uf/process.hpp"
#include <array>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace uf::dcvc2021 {
int run_checkpoint_download_command(int argc, char** argv) {
    const auto options = cli::parse(argc, argv);
    const ProcessInterrupt interruption;
    const auto destination = options.count("__output_directory") ?
        std::filesystem::path(options.at("__output_directory")) : std::filesystem::path(".");
    if (!std::filesystem::is_directory(destination)) throw std::invalid_argument("Checkpoint output directory must exist");
    static constexpr std::array<const char*, 8> names{
        "bmshj2018-hyperprior-ms-ssim-3-92dd7878.pth.tar",
        "bmshj2018-hyperprior-ms-ssim-4-4377354e.pth.tar",
        "bmshj2018-hyperprior-ms-ssim-5-c34afc8d.pth.tar",
        "bmshj2018-hyperprior-ms-ssim-6-3a6d8229.pth.tar",
        "cheng2020-anchor-3-e49be189.pth.tar",
        "cheng2020-anchor-4-98b0b468.pth.tar",
        "cheng2020-anchor-5-23852949.pth.tar",
        "cheng2020-anchor-6-4c052b1a.pth.tar"};
    const std::string root = "https://compressai.s3.amazonaws.com/models/v1/";
    for (const auto* name : names) {
        ProcessInterrupt::check();
        std::cout << "downloading " << name << '\n' << std::flush;
        // As with urlretrieve, downloads replace an existing file. No automatic
        // retry, archive extraction or model loading is performed.
        ChildProcess transfer(std::vector<std::string>{"curl", "--fail", "--location", "--silent",
            "--show-error", "--output", (destination / name).string(), root + name}, {}, true);
        if (transfer.wait() != 0) throw std::runtime_error("Cannot download checkpoint: " + std::string(name));
    }
    return 0;
}
}  // namespace uf::dcvc2021
