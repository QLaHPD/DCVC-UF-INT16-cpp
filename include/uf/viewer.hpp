#pragma once

#include "uf/frame_source.hpp"

namespace uf {

struct ViewerOptions {
    std::filesystem::path input;
    bool recursive{};
    std::optional<double> fps;
    std::int64_t start_frame{};
    int max_width{1280}, max_height{720};
};

int view_archive(const ViewerOptions& options, ViewerCodecLoader loader);

}  // namespace uf
