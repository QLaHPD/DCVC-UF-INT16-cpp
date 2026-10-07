# Optional training

Training source is compiled by the UF preset. Primitive optimizer/recomputation/loss/dataset/worker tests pass; complete learned-model training and persistence remain unverified.

```sh
build/uf/dcvc-uf train-image --train_dataset /path/to/images \
  --save_dir runs/image --lambdas 10 2048 -e 105 -n 4
build/uf/dcvc-uf train-video --train_dataset /path/to/video_frames \
  --save_dir runs/video --model_path_i /path/to/image.pth.tar \
  --model_structure htl --training_scheduling stage0 --lambdas 1 768 -e 45 -n 4
```

These are interface examples, not validated recipes. Use `--help`; training can require much more RAM than inference. Disable with `UF_BUILD_TRAINING=OFF`. NCCL distributed training additionally needs compatible SDKs and `UF_BUILD_TRAINING_NCCL=ON`; that provider was unavailable. Outputs/datasets are not published.

See [DEPENDENCIES.md](DEPENDENCIES.md) and [PORT_STATUS.md](PORT_STATUS.md).
