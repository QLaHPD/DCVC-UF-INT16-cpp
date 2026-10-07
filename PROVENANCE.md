# Source provenance

The C++/CUDA application adapts the UF Python source and retains upstream/third-party notices.

- UF Python: [QLaHPD/DCVC-RT-INT16 revision 9df8dccd6372d210922e196be6464e6cde2019ff](https://github.com/QLaHPD/DCVC-RT-INT16/tree/9df8dccd6372d210922e196be6464e6cde2019ff), containing UF entry point, archive/INT16 code and parity test.
- Original source: [Microsoft/DCVC](https://github.com/microsoft/DCVC). Copied inference/rANS files retain Microsoft and CompressAI-related notices.
- External CUTLASS: [NVIDIA/CUTLASS revision 4370102f9dacab813282e1d67722fceb0b90a019](https://github.com/NVIDIA/cutlass/tree/4370102f9dacab813282e1d67722fceb0b90a019).
- Adapted behavior/source references include CPython 3.10.20, NumPy, SciPy, Pillow, PyTorch, PyTorch-MSSSIM, Matplotlib, MT19937, SipHash, tqdm and Ryg rANS. Component notices/upstream links are in [licenses/README.md](licenses/README.md).

Adaptations replace Python bindings/application helpers with LibTorch/native vectors, integer models, archive/process/stream/lifecycle handling, training/evaluation utilities and optional image/viewer/plot backends.

The copied inference source's development revision was 8c6346eb00abb7cd91cf0ea66a718f4212ca740c. Later workflow comparisons use the pinned revision above.

[LICENSE.txt](LICENSE.txt), [NOTICE.txt](NOTICE.txt) and texts under `licenses/` are retained. External SDKs and models are obtained separately. Private development receipts, original local paths, generated benchmarks and runtime artifacts are excluded from publication. See [TESTING.md](TESTING.md) for current evidence.
