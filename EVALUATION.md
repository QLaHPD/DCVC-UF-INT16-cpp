# Optional evaluation

Evaluation, metrics, BD-rate and optional Cairo plots are compiled by the UF preset. Bounded metric/report/plot tests pass; full learned-model evaluation across datasets/variants/runtimes remains unverified.

```sh
build/uf/dcvc-uf test-video --test_config /path/to/test-config.json \
  --output_path runs/evaluation.json --model_structure htl \
  --model_path_i /path/to/image.pth.tar --model_path_p /path/to/video_htl.pth.tar
build/uf/dcvc-uf test-compress-time --model_structure htl --output_path runs/timing.json
build/uf/dcvc-uf compare-bd-rate --base_method anchor \
  --log_paths anchor /path/to/anchor.json candidate /path/to/candidate.json \
  --compare_frame_type all --plot_rd_curve 0
```

Use `--help` for current options. Numeric comparison remains available without Cairo using `--plot_rd_curve 0`.

The release interoperability/FPS protocol is `tests/int16_cross_language.py`. Its archive timing differs from the optional inference microbenchmark. See [TESTING.md](TESTING.md) and [speed report](UF_CROSS_LANGUAGE_SPEED_REPORT.md).
