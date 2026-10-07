# DCVC-UF INT16 C++

Native C++17/CUDA compression, archives, decoding, and streaming for DCVC-UF. This experimental source release validates **UF HT-L INT16** on Jetson Orin Nano (SM87).

UF FP16, training, evaluation, and a Tcl/Tk viewer are included; complete learned-model validation for those paths is pending. Historical DCVC-family targets are disabled by the UF preset.

## Verified results

- C++→Python and Python→C++ compression/decompression on 120 synthetic frames.
- Identical compressed video bytes and reconstructed pixels across both implementations.
- Seeded HT-L model/state parity and CPU/CUDA reference comparisons.
- A real channel update: 29 new archives, 24,637 frames fully decoded, stored hashes verified, originals retained.
- Ten bounded CTest entries covering primitives, CLI, reports, archives, lifecycle, terminals, viewer startup, plotting, and yt-dlp loopback.
- C++ processing at 626×352: 11.31 FPS encoding / 10.07 FPS decoding on synthetic video; 10.41 / 10.53 FPS on the real channel. Startup/downloads excluded. See [speed report](UF_CROSS_LANGUAGE_SPEED_REPORT.md).

## Build

See [DEPENDENCIES.md](DEPENDENCIES.md) for SDK requirements and the pinned CUTLASS revision. Models are separate; CMake downloads nothing.

```sh
git clone https://github.com/QLaHPD/DCVC-UF-INT16-cpp.git
cd DCVC-UF-INT16-cpp
# Replace SDK paths and capability with your configuration (87 for Orin).
cmake --preset uf \
  -DCMAKE_PREFIX_PATH="/path/to/sdk;/path/to/libtorch" \
  -DUF_FP16_SM=87 -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DUF_CUTLASS_ROOT=/path/to/cutlass -DCMAKE_CUDA_FLAGS=--threads=1
cmake --build --preset uf --parallel 1
```

The build preset uses one job. `scripts/build_one_core.sh` pins configure/build to one CPU core and forwards its arguments to CMake. INT16-only builds can set `UF_BUILD_FP16=OFF` and an explicit CUDA architecture; CUTLASS is then unnecessary.

## Encode, verify, decode

Obtain image/HT-L models and integer artifacts as described in [CHECKPOINTS.md](CHECKPOINTS.md).

```sh
build/uf/dcvc-uf encode --input_file clip.mp4 --output_root archives \
  --runtime int16 --model_structure htl --resolution 352 --fps 1 \
  --qp_i 0 --qp_p 20 --reset_interval 8 --intra_period -1 --skip_thres 0.2 \
  --audio opus --opus_channels mono --opus_bitrate 6k --opus_frame_ms 60 \
  --procs 1 --cuda_idx 0 --ui plain --keep-originals \
  --model_path_i /path/to/image.pth.tar --model_path_p /path/to/video_htl.pth.tar \
  --prepared_i /path/to/image.int16.pt --prepared_p /path/to/video_htl.int16.pt

# Replace MANIFEST with the emitted .uf.json file.
build/uf/dcvc-uf check-archive MANIFEST
build/uf/dcvc-uf verify --input MANIFEST --cuda_idx 0 \
  --prepared_i /path/to/image.int16.pt --prepared_p /path/to/video_htl.int16.pt
build/uf/dcvc-uf decode --input MANIFEST --output_file restored.mkv --cuda_idx 0 \
  --prepared_i /path/to/image.int16.pt --prepared_p /path/to/video_htl.int16.pt
```

`check-archive` checks stored hashes/structure; `verify` performs full neural decoding. `decode` writes a lossless container of the reconstructed video; compression remains lossy relative to the source.

## Streaming

The application reuses **yt-dlp**, FFmpeg, and ffprobe as external executables. yt-dlp may require Python; it is not rewritten in C++.

```sh
build/uf/dcvc-uf stream --source_urls "https://www.youtube.com/channel/CHANNEL_ID/videos" \
  --output_root archives --runtime int16 --model_structure htl \
  --resolution 352 --fps 1 --procs 1 --cuda_idx 0 --ui plain --keep-originals \
  --model_path_i /path/to/image.pth.tar --model_path_p /path/to/video_htl.pth.tar \
  --prepared_i /path/to/image.int16.pt --prepared_p /path/to/video_htl.int16.pt
```

Cookie files supplied with `--cookies` belong outside Git. Configuration defaults to FP16/HT-S unless overridden; the verified examples select INT16/HT-L explicitly. These examples retain originals; cleanup requires separate explicit selection and verification.

## Testing and scope

[TESTING.md](TESTING.md) contains CTest and the portable, explicit-model interoperability test. [PORT_STATUS.md](PORT_STATUS.md) lists verified and pending work. [TRAINING.md](TRAINING.md) and [EVALUATION.md](EVALUATION.md) describe optional interfaces.

Full FP16 model parity, HT-S/LD model parity, complete training, NCCL, live GUI rendering, and destructive cleanup are not claimed as verified.

## License and source

[LICENSE.txt](LICENSE.txt), [NOTICE.txt](NOTICE.txt), and [licenses/README.md](licenses/README.md) retain notices. [PROVENANCE.md](PROVENANCE.md) identifies reference revisions.

Models, binaries, generated archives, datasets, caches, cookies, local configuration, and runtime test output are excluded from Git.
