# External checkpoints and preparation

No models or automatic model downloader are included. Obtain UF image/HT-L checkpoints through the [pinned Python reference](https://github.com/QLaHPD/DCVC-RT-INT16/tree/9df8dccd6372d210922e196be6464e6cde2019ff) and its setup instructions. Use explicit paths.

```sh
build/uf/dcvc-uf prepare-int16 --model_structure htl \
  --model_path_i /path/to/image.pth.tar --model_path_p /path/to/video_htl.pth.tar \
  --prepared_i /path/to/image.int16.pt --prepared_p /path/to/video_htl.int16.pt
```

Preparation loads checkpoints and creates artifacts; it was not comprehensively parity-tested in this release. Existing compatible prepared artifacts can be supplied directly. Checkpoints are trusted inputs.

## Tested identities

Arithmetic: `uf-int16-q9-w13-acc64-rna-sat-v1`.

| Input | SHA-256 / payload identity |
|---|---|
| Image source | `b3b900de23f30e4fc437010ddffe6a5413a56d6cfd97fb6235adbcd2b6973302` |
| HT-L source | `934bde4a12fc0b6b0c679ec6ae21d41cb7722debf8b631ee0dd750d030b1a6b3` |
| Prepared image | `119cd77a1d16c241e8f5c43890bd6cef0dd8786c6a66b87301ff6d005ab7d147` |
| Prepared HT-L | `57b5223e46b683f79c1aac272725ffd19237aed79280857b57bb42c067bab852` |

Prepared identities describe payloads, not necessarily serialized file byte hashes. Archives retain source/pipeline identities; decoders reject mismatched models.

Use `inspect-prepared FILE` for metadata; [TESTING.md](TESTING.md) documents supplied-model parity tests. Models, prepared files, and generated archives remain excluded from Git.
