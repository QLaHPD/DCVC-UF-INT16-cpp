# Native UF FP16 inference

Adapted from the UF Python reference's inference and py_rans sources, excluding Python bindings/binaries. Development reference: 8c6346eb00abb7cd91cf0ea66a718f4212ca740c. Microsoft notices remain.

LibTorch headers and native vectors/tuples replace extension interfaces. Checked rANS packets retain lane splitting/reversal/offset formats. GPU queries and tensor pools follow worker state.

CUTLASS is external; set `UF_CUTLASS_ROOT` and `UF_FP16_SM`. See [../../../DEPENDENCIES.md](../../../DEPENDENCIES.md).

Sources compile and selected primitive tests pass on SM87/CUDA 12.6. Full FP16 model parity is unverified; the validated release path is HT-L INT16. See [../../../TESTING.md](../../../TESTING.md) and [../../../PROVENANCE.md](../../../PROVENANCE.md).
