# UF HT-L INT16 compatibility and speed

Measured on Jetson Orin Nano, CUDA 12.6, MAXN_SUPER, CPU core 0 and one GPU worker. Clock settings were not changed. Observed CPU/GPU frequencies were 1728/1020 MHz.

## Compatibility

Both C++→Python and Python→C++ compression/decompression pass on a generated 120-frame video. All three native encodes and the Python encode produce identical video bitstreams. All four encoder/decoder combinations write byte-identical reconstructed raw YUV videos, including the partial final HT-L chunk. Stored archive hashes/structure and Opus audio checks pass. Originals remain present.

Configuration: HT-L INT16, 640×360 source resized to 626×352, source sampling 1 FPS, QP I=0/P=20, reset interval 8, intra period −1, skip threshold 0.2, Opus mono 6 kbit/s / 60 ms. Input is FFV1 testsrc2 plus 440 Hz sine audio.

## Processing speed

| Workload | C++ encode FPS | C++ decode FPS |
|---|---:|---:|
| 120 synthetic frames | 11.31 | 10.07 |
| 29 channel videos / 24,637 frames | 10.41 | 10.53 |

Synthetic encode samples: 11.313, 11.309, 11.315 FPS. Decode measured twice, once per encoder archive. Averages use total frames divided by total processing seconds.

Processing excludes model loading/startup and downloads. Encode includes source reads and bitstream writes; synthetic decode includes YUV conversion, hashing and raw output writes. Channel decoding was full verification without writing a reconstructed file.

Including setup, models, audio handling and finalization, the synthetic C++ commands averaged 7.99 encode / 8.55 decode FPS (roughly 15 seconds per encode, 14 per decode).

Python reference processing measured 13.26 encode FPS (one run), 11.63 decode FPS (two runs). C++ was slower per frame in this test; it does not establish a per-frame C++ speedup. Python encode command wall time was 22.24 seconds.

## Reproduce

See [TESTING.md](TESTING.md) and [tests/int16_cross_language.py](tests/int16_cross_language.py). The test creates reports/logs/media locally under an ignored output directory. None of those generated artifacts is published.

Encoded video SHA-256: `94cd22893956e85850c48efa04e2ef6329ad11f8f6e957346f2ad16d4ec18825`.

Reconstructed YUV SHA-256: `ae2fcf6646e18510e3948e58c8f83073b5c0b8db6101a965a6482e8970760394`.

These results cover the configured HT-L INT16 path and resolution; other variants/runtimes/resolutions were not benchmarked.

## Clean release checkout repeat

The portable published test was repeated against a freshly compiled UF checkout: all state/CPU/CUDA checks and both language directions passed. C++ processing averaged **11.314 encode FPS / 10.074 decode FPS**. Configuration and timing definition match the synthetic protocol above.
