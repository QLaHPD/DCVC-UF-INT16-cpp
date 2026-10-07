# Current status

Experimental UF C++ source release. Default presets disable DCVC2021/TCM/HEM/DC.

## Verified

- Linux AArch64 / GCC 11 / CUDA 12.6 / NVIDIA LibTorch 2.5 development build / SM87 compilation.
- UF HT-L INT16 seeded numerical/state/bitstream parity and CPU/CUDA reference decoding.
- C++→Python and Python→C++ archive compatibility on synthetic video.
- Stored hashes, packet structure, full decoding, Opus checks, resume preservation and pipeline identities.
- Real channel processing: 29 new archives, 24,637 decoded frames, originals retained.
- Ten bounded CTest entries, including optional native TIFF/AVIF and Cairo configured with external SDK inputs.

## Pending full validation

FP16 learned-model codec parity; HT-S/LD model parity; complete training/evaluation/persistence; live Tcl/Tk rendering/controls; destructive cleanup; untested worker races/provider failures; NCCL/LittleCMS2 unavailable on the validation host.

These limits define the experimental release's evidence. This document supersedes earlier source-only review statements. See [TESTING.md](TESTING.md), [PROVENANCE.md](PROVENANCE.md), and [speed report](UF_CROSS_LANGUAGE_SPEED_REPORT.md).
