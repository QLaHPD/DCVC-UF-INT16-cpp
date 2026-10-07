// Native counterpart of DCVC-family/DCVC-HEM/checkpoints/download.py.
#include "uf/dcvc_hem_evaluation.hpp"
#include "uf/cli.hpp"
#include "uf/process.hpp"
#include <array>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace uf::hem {
int run_checkpoint_download_command(int argc, char** argv) {
    const auto options = cli::parse(argc, argv);
    const ProcessInterrupt interruption;
    const auto destination = options.count("__output_directory") ?
        std::filesystem::path(options.at("__output_directory")) : std::filesystem::path(".");
    if (!std::filesystem::is_directory(destination))
        throw std::invalid_argument("Checkpoint output directory must exist");
    static constexpr std::array<std::pair<const char*, const char*>, 4> transfers{{
        {"https://onedrive.live.com/download?cid=2866592D5C55DF8C&resid=2866592D5C55DF8C%211220&authkey=AMRg1W3PVt_F3yc", "acmmm2022_image_psnr.pth.tar"},
        {"https://onedrive.live.com/download?cid=2866592D5C55DF8C&resid=2866592D5C55DF8C%211219&authkey=ACJnPOPf1ntw_w0", "acmmm2022_image_ssim.pth.tar"},
        {"https://onedrive.live.com/download?cid=2866592D5C55DF8C&resid=2866592D5C55DF8C%211217&authkey=AKpdgXQtvs-OxRs", "acmmm2022_video_psnr.pth.tar"},
        {"https://onedrive.live.com/download?cid=2866592D5C55DF8C&resid=2866592D5C55DF8C%211218&authkey=ANxapLv3PcCJ4Vw", "acmmm2022_video_ssim.pth.tar"}
    }};
    for (const auto& transfer : transfers) {
        ProcessInterrupt::check();
        std::cout << "downloading " << transfer.second << '\n' << std::flush;
        ChildProcess download(std::vector<std::string>{"curl", "--fail", "--location", "--silent",
            "--show-error", "--output", (destination / transfer.second).string(), transfer.first}, {}, true);
        if (download.wait() != 0)
            throw std::runtime_error("Cannot download checkpoint: " + std::string(transfer.second));
        std::cout << "downloaded " << transfer.second << '\n';
    }
    return 0;
}
}  // namespace uf::hem
