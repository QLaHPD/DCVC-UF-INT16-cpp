# Dependencies and reproducible builds

CMake only discovers SDKs; it does not fetch dependencies or models.

| Dependency | Requirement / role |
|---|---|
| Linux / glibc | glibc 2.34+ for closefrom process spawning |
| Compiler / CMake | C++17; GCC 11 tested; CMake 3.20+ |
| LibTorch | Compatible CPU/CUDA SDK; tested NVIDIA PyTorch 2.5.0a0+872d972e41.nv24.08 |
| OpenSSL, nlohmann/json | Development headers/libraries and CMake packages |
| libcurl | Headers/library, version 7.85+ |
| FFmpeg, ffprobe | Runtime media/audio executables |
| yt-dlp | External provider tool; tested 2026.08.19 |

Match LibTorch's C++ ABI, CUDA build, and architecture. Other SDK combinations need validation. A compatible installed PyTorch package provides headers/libraries/CMake files:

```sh
/path/to/python -c 'import torch; print(torch.utils.cmake_prefix_path)'
```

Supply the resulting directory and other SDK prefixes in `CMAKE_PREFIX_PATH`.

Typical Ubuntu 24.04 development packages:

```sh
sudo apt-get install build-essential cmake pkg-config libssl-dev \
  nlohmann-json3-dev libcurl4-openssl-dev ffmpeg \
  libjpeg-dev libpng-dev libwebp-dev libtiff-dev libavif-dev \
  libcairo2-dev tcl-dev tk-dev liblcms2-dev
```

Install yt-dlp into a separate tool environment and make its executable available on PATH (or supply `--ytdlp_bin`):

```sh
python3 -m venv /path/to/yt-dlp-env
/path/to/yt-dlp-env/bin/python -m pip install yt-dlp==2026.08.19
export PATH="/path/to/yt-dlp-env/bin:$PATH"
```

This pins the version used in the recorded extractor tests. Provider behavior can change independently; update and revalidate it when required. Obtain LibTorch from the [PyTorch SDK distribution](https://pytorch.org/) or your platform's NVIDIA package channel; use a package matching the host CUDA/ABI. The selected platform package supplies the Torch CMake prefix queried above.

This is a user setup example, not an installer executed by the project. Older distributions may have incompatible SDK versions (notably libcurl). Obtain compatible LibTorch/PyTorch and CUDA separately.

## CUDA / CUTLASS

INT16 CUDA requires CUDA-enabled LibTorch and nvcc; a CPU SDK retains the slow integer reference path.

FP16 requires NVIDIA nvcc/toolkit 11.8+, an explicit `UF_FP16_SM` of at least 80, and external CUTLASS:

```sh
git clone https://github.com/NVIDIA/cutlass.git /path/to/cutlass
git -C /path/to/cutlass checkout 4370102f9dacab813282e1d67722fceb0b90a019
```

Set `UF_CUTLASS_ROOT`. SM90 requires CUDA 12.0+; SM100 requires 12.8+. FP16 checks the GPU against its compiled capability. The verified host used CUDA 12.6 / SM87.

INT16-only build:

```sh
scripts/build_one_core.sh \
  -DCMAKE_PREFIX_PATH="/path/to/sdk;/path/to/libtorch" \
  -DCMAKE_CUDA_ARCHITECTURES=87 -DUF_BUILD_FP16=OFF \
  -DUF_BUILD_TRAINING=OFF -DUF_BUILD_EVALUATION=OFF -DUF_BUILD_VIEWER=OFF
```

Some NVIDIA PyTorch SDKs require explicit NVRTC overrides in addition to the normal configure flags:

```sh
cmake --preset uf [OTHER_OPTIONS] \
  -DCUDA_nvrtc_LIBRARY=/path/to/cuda/lib/libnvrtc.so \
  -DCUDA_NVRTC_LIB=/path/to/cuda/lib/libnvrtc.so
```

Replace bracketed placeholders with concrete flags. Use libraries from the selected toolkit. CMake isolates compiler-compatible libstdc++ inside the build directory to avoid older Conda runtime libraries taking precedence.

## Optional providers

- Native JPEG/PNG/WebP/TIFF (4.5+)/AVIF (1.0+) with development SDKs; missing formats use FFmpeg. AVIF needs a decoder backend.
- Select external AVIF via `UF_AVIF_INCLUDE_DIR` / `UF_AVIF_LIBRARY`; TIFF via `TIFF_INCLUDE_DIR` / `TIFF_LIBRARY`.
- Cairo RD plots: `UF_CAIRO_INCLUDE_DIR` / `UF_CAIRO_LIBRARY` overrides.
- Tcl/Tk viewer: installed SDK plus a graphical display.
- LittleCMS2 Lab conversion and NCCL distributed training were unavailable during validation. NCCL needs compatible LibTorch/NCCL and `UF_BUILD_TRAINING_NCCL=ON`.

Default UF switches retain training/evaluation/viewer and keep historical DCVC family targets OFF. SDK caches/libraries and generated third-party builds are external, not published.

CTest references need Python, NumPy, SciPy, Pillow, compatible PyTorch, and the pinned UF Python checkout described in [TESTING.md](TESTING.md). Codec application inference is C++/LibTorch; Python is needed for reference testing and independently by yt-dlp.

## Generated target-PSNR report

Generated Pareto reports are not shipped. Run `pareto-search` to create a HT-L INT16 144p report using your representative clip; then pass `-DUF_PARETO_BENCHMARK=/path/to/report.json` when configuring. Alternatively put it at `benchmarks/uf-int16-htl-144p-pareto.json` beside the executable or in the working directory. Without a report, ordinary explicit-QP encoding remains available and `--target-psnr` reports the missing input.
