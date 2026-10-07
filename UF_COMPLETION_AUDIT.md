# Publication audit

This audit covers release preparation of the UF C++ source project. It supersedes the earlier no-model-loading audit; models have been authorized and loaded in the tests.

## Completed evidence

- HT-L INT16 synthetic model/state/CPU/CUDA parity and both archive language directions pass.
- 29 real channel archives / 24,637 frames fully decoded; originals retained.
- The published test harness accepts external checkpoints/reference paths and generates its own video, archive, decoded pixels, logs and timing reports.
- Public documentation now identifies tested scope, current results, SDK requirements and unverified paths.
- Licenses/notices retained; private development receipts and generated media/models/build data excluded from Git.
- CMake no longer requires an unpublished generated Pareto report. Target-PSNR selection accepts a separately supplied report.

## Publication readiness checks passed

- A fresh Git checkout configured, compiled and linked the full UF preset on core 0 with one job, using explicitly selected existing external SDKs. This verifies a fresh source checkout, not a fresh OS or every SDK combination.
- All ten CTest entries passed; native output confirmed 150/150 checks. CTest elapsed time: 163.56 seconds.
- The public portable harness passed seeded native/Python/CPU/CUDA comparisons, three C++ encodes, one Python encode, both language directions and both same-language decodes. All 120-frame raw outputs and compressed video bytes matched exactly.
- Repeated clean-build C++ measurements: 11.314 encode FPS and 10.074 decode FPS, excluding model setup.
- All 342 tracked source/text files passed the public-source audit: required notices present, local documentation links resolved, no detected private paths/common credentials, and no model/media/build artifacts tracked. The scan is bounded by its explicit patterns.
- Production and reference-test source matches the tested checkout byte for byte.

Tested source manifest SHA-256 (308 sorted build/test files; relative name + NUL + file SHA-256 bytes): `90eae923717a022d1573ca4caa60500670b32342aacf1f0c5928f975ac76f257`.

Clean-checkout executable SHA-256: `6ceeb76dc2246a603c25c812b56b9ddcf49b4e2f3cfd8679495d1f812f144f47`. The executable is not published.

Repository delivery is verified against the remote Git ref after pushing. Model and runtime test output stays local; [TESTING.md](TESTING.md) explains reproduction.

## Limits

Full FP16/HT-S/LD learned-model parity, complete training/evaluation/persistence, live GUI rendering, destructive cleanup, NCCL and LittleCMS2 remain outside the verified runtime evidence. Historical DCVC-family targets remain disabled.

See [TESTING.md](TESTING.md), [PORT_STATUS.md](PORT_STATUS.md), and [UF_VALIDATION_GATES.json](UF_VALIDATION_GATES.json).
