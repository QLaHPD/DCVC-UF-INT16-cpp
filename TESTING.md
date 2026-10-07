# Testing

Validation runs sequentially on CPU core 0 with one GPU worker. Models, fixtures, media, and logs are external/generated and excluded from Git.

## Python reference

```sh
git clone https://github.com/QLaHPD/DCVC-RT-INT16.git /path/to/DCVC-UF-INT16
git -C /path/to/DCVC-UF-INT16 checkout 9df8dccd6372d210922e196be6464e6cde2019ff
```

Despite its repository name, this revision contains the UF entry point, INT16 implementation, and parity scripts used by the tests. Configure that project's Python dependencies and models separately.

## Model-free CTest

Add these flags to the SDK configuration in [DEPENDENCIES.md](DEPENDENCIES.md):

```sh
cmake --preset uf [SDK_OPTIONS] \
  -DUF_BUILD_TESTS=ON -DPython3_EXECUTABLE=/path/to/reference-python \
  -DUF_TEST_REFERENCE_ROOT=/path/to/DCVC-UF-INT16 \
  -DUF_TEST_YTDLP=/path/to/yt-dlp
cmake --build --preset uf --parallel 1
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  taskset -c 0 ctest --test-dir build/uf --parallel 1 --output-on-failure
```

Replace `[SDK_OPTIONS]` with real dependency flags. Extractor tests require `UF_TEST_YTDLP`; plotting tests require Cairo. The full tested configuration has ten entries, covering 150 native primitive groups, 26 CLI, 5 report, 66 archive, 27 lifecycle, 8 terminal, 13 viewer startup, 9 plot, and 4 loopback extractor cases.

These bounded tests do not load full learned models. The complete reference-fixture suite currently requires a CUDA-capable PyTorch environment even though a CPU-only application can be built. Viewer startup does not prove live rendering; loopback uses a temporary localhost clip rather than an external provider.

## Explicit model-backed compatibility / FPS test

`tests/int16_cross_language.py` creates a 120-frame synthetic video and sine audio track. It runs three C++ encodes and one Python encode, then decodes each archive with both implementations to raw YUV. Compressed video and reconstructed pixels must match exactly. Model jobs are sequential and pinned to one core.

```sh
/path/to/reference-python tests/int16_cross_language.py \
  --binary build/uf/dcvc-uf --reference-root /path/to/DCVC-UF-INT16 \
  --reference-python /path/to/reference-python \
  --model-i /path/to/image.pth.tar --model-p /path/to/video_htl.pth.tar \
  --prepared-i /path/to/image.int16.pt --prepared-p /path/to/video_htl.int16.pt \
  --state-parity --output-dir build/cross-language-run
```

The output directory must be new. Without `--output-dir`, a unique directory is created under `build/`. `--cpu` defaults to 0. `--state-parity` additionally compares seeded native/Python HT-L state at size 32 and CPU/CUDA reference decoding.

Configuration: HT-L INT16, 626×352, source sampling 1 FPS, QP I=0/P=20, reset interval 8, intra period −1, skip threshold 0.2, Opus mono 6k / 60 ms. FPS divides actual output frames by synchronized processing seconds; command wall time includes model setup.

The test downloads no model, contacts no external video provider, and deletes no source. A fresh reference environment may compile CUDA extensions; affinity and `MAX_JOBS=1` limit that work.

## Evidence and limits

Both cross-language directions pass with identical encoded video bytes and all four reconstructed videos. Seeded CPU/CUDA state comparisons also pass.

A separate channel update published 29 archives and verified 24,637 frames. Three audio transfer failures recovered on retry. All 68,977 baseline files remained present with unchanged sizes/timestamps; baseline byte hashes were not recorded. Old RT intra-image inventory warnings did not alter those files.

See [speed report](UF_CROSS_LANGUAGE_SPEED_REPORT.md), [PORT_STATUS.md](PORT_STATUS.md), and [publication audit](UF_COMPLETION_AUDIT.md).

## Public source audit

Run `python3 scripts/audit_public_source.py` inside a Git checkout. It checks tracked text/source files, required notices, relative documentation links, model/media/build exclusions, private paths, and common credential signatures. It reports filenames without printing matched credential values. This bounded scan is not a claim that every possible secret format can be detected.
