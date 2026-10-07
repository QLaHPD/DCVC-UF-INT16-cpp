// Native counterpart of DCVC-family/DCVC-DC/checkpoints/download.py.
#include "uf/dcvc_dc_media.hpp"
#include "uf/cli.hpp"
#include "uf/process.hpp"
#include <array>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace uf::dc {
int run_checkpoint_download_command(int argc, char** argv) {
    const auto options = cli::parse(argc, argv); const ProcessInterrupt interruption;
    const auto destination = options.count("__output_directory") ?
        std::filesystem::path(options.at("__output_directory")) : std::filesystem::path(".");
    if (!std::filesystem::is_directory(destination)) throw std::invalid_argument("Checkpoint output directory must exist");
    static constexpr std::array<std::pair<const char*, const char*>, 6> transfers{{
        {"https://onedrive.live.com/download?cid=2866592D5C55DF8C&resid=2866592D5C55DF8C%211259&authkey=AO_gFvTcYZUFd9U", "cvpr2023_image_psnr.pth.tar"},
        {"https://onedrive.live.com/download?cid=2866592D5C55DF8C&resid=2866592D5C55DF8C%211260&authkey=AFWlIyBB5PIudtw", "cvpr2023_image_ssim.pth.tar"},
        {"https://onedrive.live.com/download?cid=2866592D5C55DF8C&resid=2866592D5C55DF8C%211261&authkey=AOB9I7Jv25RbyGY", "cvpr2023_image_yuv420_psnr.pth.tar"},
        {"https://onedrive.live.com/download?cid=2866592D5C55DF8C&resid=2866592D5C55DF8C%211256&authkey=ACzRzK3XgbQxEyk", "cvpr2023_video_psnr.pth.tar"},
        {"https://onedrive.live.com/download?cid=2866592D5C55DF8C&resid=2866592D5C55DF8C%211258&authkey=AIRQMQyZqJWG15k", "cvpr2023_video_ssim.pth.tar"},
        {"https://onedrive.live.com/download?cid=2866592D5C55DF8C&resid=2866592D5C55DF8C%211257&authkey=AEQHk7O606IiqGA", "cvpr2023_video_yuv420_psnr.pth.tar"}
    }};
    for (const auto& transfer : transfers) {
        ProcessInterrupt::check(); std::cout << "downloading " << transfer.second << '\n' << std::flush;
        ChildProcess download(std::vector<std::string>{"curl", "--fail", "--location", "--silent", "--show-error",
            "--output", (destination / transfer.second).string(), transfer.first}, {}, true);
        if (download.wait() != 0) throw std::runtime_error("Cannot download checkpoint: " + std::string(transfer.second));
        std::cout << "downloaded " << transfer.second << '\n';
    }
    return 0;
}
}  // namespace uf::dc
