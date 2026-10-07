> Historical implementation notes. These optional DCVC family targets are disabled in the UF presets and are outside the verified UF HT-L INT16 release scope. Status statements below describe earlier source reviews; see [PORT_STATUS.md](PORT_STATUS.md) for current results.

# DCVC-TCM source port

The native TCM I/P-frame source is declared as `uf_dcvc_tcm`, selected by
`UF_BUILD_DCVC_TCM` (default ON). Public interfaces are in
`include/uf/dcvc_tcm.hpp` and `include/uf/dcvc_tcm_image.hpp`. The original DCVC and TCM targets share the existing
GDN/bit-parameter, CDF/rANS and image entropy/layer source through the internal
`uf_dcvc_legacy` library. The TCM motion grid and residual blocks have their
own source because their dtype, cache and activation behavior differs.

`dcvc-tcm-test-video` maps TCM's `test_video.py` through the existing process
worker scheduler when `UF_BUILD_EVALUATION` is also selected.
No configuration, compilation, project execution or tests have been performed.
The library source does not establish working checkpoint or bitstream parity.

## P-frame interfaces

| Python behavior | Native interface/source |
| --- | --- |
| Motion estimation and warping | `Spynet`, `MEBasic`, `flow_warp`, bilinear up/down; `dcvc_tcm_layers.cpp` |
| Feature extraction and three-scale context fusion | `FeatureExtractor`, `MultiScaleContextFusion` |
| Contextual analysis/synthesis and temporal prior | `ContextualEncoder`, `ContextualDecoder`, `TemporalPriorEncoder` |
| Feature/image reconstruction | `ReconGeneration`, retaining the reference's residual-first concatenation |
| Learned hyperlatent probabilities | `BitEstimator`, with existing `Bitparm` math and TCM coder/table ownership |
| Buffered joint entropy stream | `EntropyCoder`, `GaussianEncoder`; `dcvc_tcm_entropy.cpp` |
| P-frame inference and loss fields | `DMC::forward_one_frame`; `dcvc_tcm_model.cpp` |
| Latent compression and frame decompression | `DMC::compress`, `DMC::decompress` |
| Likelihood/stream roundtrip wrapper | `DMC::encode_decode` |
| State-dictionary loading | `DMC::load_dict` / `load_checkpoint`, with strict key/shape handling and module-prefix removal |
| P-frame length prefix and padding/crop | `write_p`, `read_p`, `pad`, `crop`; `dcvc_tcm_stream.cpp` |
| Differentiable RGB MS-SSIM 0.2.0 | `MSSSIM`; `dcvc_tcm_msssim.cpp` |

The model keeps the Python channel counts: motion latent 128, context 64,
image latent 96. Weighted modules are constructed in the Python source order.
The I-reference adaptor starts from a reconstructed RGB frame; later P frames
use the prior frame's reconstructed 64-channel feature map. Reference features
stay at padded resolution. Padding adds zeros on the right/bottom to a multiple
of 64; cropping keeps the top-left original rectangle.

`forward_one_frame` produces the unclamped reconstruction and feature,
warped image, latents, temporal parameters, motion, MSE/MS-SSIM values,
absolute bit estimates and component/total bpp. It always rounds the latents,
including when the module is in training mode. The Python DMC does not implement
the generic module `forward`; that native method throws and callers use the
named frame method. The likelihood branch of `encode_decode` retains the
unclamped image; the evaluation workflow must clamp the next reference frame
as its Python counterpart does. Stream decompression clamps the image itself.

## Entropy and file format

`DMC::update` attaches learned/table models to an entropy coder. The reference
evaluation calls `update(true)` before coding. The four latent groups are
submitted in this order, using one persistent rANS decoder state:

1. Motion hyperlatent `mv_z`.
2. Motion residual latent `mv_y`.
3. Image hyperlatent `z`.
4. Image residual latent `y`.

The buffered encoder preserves each submitted group's CDF snapshot, then
encodes the combined symbol operations in reverse. CDF/rANS arithmetic uses
the existing pinned 64-bit state, 32-bit little-endian words, precision 16 and
4-bit bypass implementation. The P-file container has one big-endian uint32
payload length followed by that combined stream. It carries no dimensions;
decompression receives original height/width from the caller. The reader
leaves trailing bytes unread, following the reference.

The TCM GaussianEncoder defaults to Laplace with 256 log-spaced scales from
0.01 to 64. Its indexes use clamped log/step values converted to int32,
without the original DCVC encoder's threshold-search rule. If configured as
`gaussian`, the Python implementation uses Normal only for the support-center
search and still builds a Laplace PMF; the native source retains that behavior.
Learned bit-estimator tails use the final column of the shared maximum-length
grid, including channels with shorter PMFs.

`compress` reports payload bits. `encode_decode` writes and reads the P file,
reports bits including its four-byte header, returns zero component-bit fields
in stream mode, and measures the decompression call with wall time. The
likelihood branch returns separate estimated component bits and zero decoding
time. No GPU synchronization is added to that timing boundary.

## I-frame interfaces and entropy

`uf::tcm::image::IntraNoAR` retains the residual analysis/synthesis graph,
hyperprior and channel count inferred from `g_s.0.conv1.weight`. Its weighted
modules are created in Python order: bottleneck, `g_a`, `h_a`, `h_s`, `g_s`,
Gaussian conditional. The base weight initializer runs before any convolution
exists. Image residual blocks expose the configurable LeakyReLU slope; the
standalone `ResidualBlockType2` and simplified `GDN1` also have source mappings.
Image GDN, bounds, parametrizers and stride/upsampling residual blocks reuse
the corresponding shared implementations.

| Python behavior | Native interface/source |
| --- | --- |
| Base parameters and auxiliary parameters | `CompressionModel::parameters`, `aux_parameters` |
| Likelihood reconstruction | `IntraNoAR::forward` |
| Unquantized image/hyperlatents | `get_y_z` |
| Reconstruction and tensor bit estimate | `get_recon_wo_stream` |
| Independent y/z entropy strings and z shape | `compress`, `decompress` |
| Likelihood or file roundtrip | `encode_decode` |
| CDF buffer resizing and strict state loading | `load_state_dict` |
| Channel inference and legacy table removal | `from_state_dict` |
| Raw or wrapped checkpoint loading | `from_checkpoint` |

The I-frame Gaussian uses 64 logarithmic scales from 0.11 to 256, Normal PMFs,
and `clamp((log(max(scale, 1e-5)) - log_min) / log_step + 1, 0, 63)` converted
to int32. Its scale table is an unregistered CPU attribute, absent from the
state dictionary and unchanged by module dtype/device conversion. Gaussian
`update` always rebuilds the integer CDF buffers on CPU. Learned bottleneck
offsets and lengths follow the quantiles' device. Module dtype conversion
preserves integer buffers. `d_quant` provides the source's detached rounding
residual helper; the model forward itself always uses ordinary dequantization.

The mutable `from_state_dict` overload removes
`gaussian_conditional.scale_table` from the caller's map, as Python does; its
const overload copies first. Direct loading keeps strict key checks and resizes
only empty entropy buffers. Populated bottleneck offsets cause `update(false)`
to skip coder initialization, including after a fresh checkpoint load. Actual
stream evaluation uses `update(true)` for both I/P models.

I files reuse the identical four-uint32 big-endian header: height, width,
y-string length and z-string length, then independent y/z rANS payloads.
The file roundtrip reports all file bits including the 16-byte header and
returns a clamped reconstruction. Likelihood mode returns an unclamped image
and the summed y/z likelihood bits. Primary parameter iteration excludes the
entire entropy bottleneck. `aux_loss` throws because the local Python
bottleneck has no `loss()` implementation. Batch stream limits follow the
shared bottleneck's mean/index validation.

## Evaluation workflow

The command source accepts TCM's original options and defaults. Example after
a later authorized build:

```sh
dcvc-uf dcvc-tcm-test-video --test_config test.json --output_path results.json \
  --i_frame_model_path intra.pth --model_path video.pth --worker 1
```

Bare `--force_intra`, `--cuda` and `--write_stream` mean true; each also accepts
a boolean word. `--save_decoded_frame` requires a value. TCM's accepted words
are yes/true/t/y/1 and no/false/f/n/0, with unknown values rejected. Defaults
include CPU execution, one worker, no stream or PNG output, `IntraNoAR`,
`out_bin`, `decoded_frames`, verbosity zero and frame/GOP overrides of -1.

The configuration is a dataset-name object with `test`, `base_path` and a
`sequences` object containing `gop` and `frames`. Datasets whose `test` equals
zero are skipped. Forced intra replaces P-checkpoint paths with I paths and
sets GOP to one; a positive forced period subsequently overrides that GOP,
following the source's option order. Combining forced intra with a period
that reaches a P frame fails because the reference has no P model then.

Workers reload models and seed Torch to zero for every submitted point, use
one Torch thread, disable cuDNN benchmarking and enable deterministic
algorithms regardless of stream mode. GPU assignment uses the one-based
worker index modulo the visible GPU count; unavailable CUDA falls back to
CPU. The parent sets the optional visibility string and
`CUBLAS_WORKSPACE_CONFIG=:4096:8`. NumPy's seeded RNG is unused by this path.
Results retain submission order through the shared spawn/IPC scheduler.

Frames use `im1.png` or `im00001.png` (short names take precedence), RGB
Float64-to-Float32 CHW conversion and division by 255. Each frame is padded
with zeros at right/bottom to 64; metrics and saved PNGs use the original
rectangle. I frames clear the recurrent feature; P frames retain it. Every
reference image is clamped in place before the next frame. Original pixel
counts, rather than padded counts, divide the rates. Both PSNR and pinned
0.2.0 MS-SSIM are calculated for every frame.

Optional stream output is
`STREAM_PATH/bit_stream/SEQUENCE/MODEL_INDEX/FRAME_INDEX.bin`; PNG output is
`DECODED_FRAME_PATH/decoded_video/SEQUENCE/MODEL_INDEX/FRAME_INDEX.png`.
Indexes start at zero. Directories are created only for enabled outputs.
PNG pixels use round-to-even conversion after multiplication by 255.

JSON keys follow dataset/sequence/checkpoint-basename nesting. They include
I/P counts, separate I/P bpp/PSNR/MS-SSIM averages, P component rates, all
per-frame rate/metric/type arrays and `test_time`. All-frame averages are
recomputed from the I/P frame counts, as `filter_dict` does. Original basename
prefix matching and last-result collisions are retained. Component rates are
zero for I frames and for P frames in stream mode. Nonfinite metric values
use the shared Python-compatible JSON writer. Verbose output reports frame
bits/metrics and mean P decoding milliseconds. Timing uses wall-clock
boundaries without additional CUDA synchronization. The Python tqdm progress
display is not reproduced by the shared native scheduler.

## Source-specific behavior and limits

- Repeated `DMC::update(false)` replaces the model's coder pointer while already
  initialized entropy models retain the prior coder, matching the Python code.
  `update(true)` refreshes all attachments and tables.
- Loading new weights preserves the separately cached entropy lists, as in
  Python; an explicit forced update regenerates them.
- The warp cache keys only on flow shape within the nine Python device slots.
  Its first dtype is retained. CUDA index 8 shares CPU's final slot, including
  the reference's potential device conflict; higher CUDA indexes are rejected.
- Stream hyperlatent coding requires batch one. The reference passes only
  `indexes[0]` to its unsafe extension for larger batches; the native path
  throws a bounded shape error. Frame-forward/loss math can process batches.
- The MS-SSIM window is an unregistered Float32 CPU attribute, as in the pinned
  module, and is transferred for each level. Version 0.2.0 applies valid width
  convolution before height convolution; this differs from the separately
  retained 0.2.1 evaluation implementation. Its five levels, weights, ReLU,
  odd-size pooling and minimum image size checks are retained.
- `load_checkpoint` accepts a raw tensor dictionary or its `state_dict` wrapper,
  and traverses insertion order before prefix normalization. The C++ map-based
  `load_dict` follows that map's ordering. Shared pickle scope and limits are
  in [CHECKPOINTS.md](CHECKPOINTS.md).
- Native malformed-input errors and integer limits differ from the old unsafe
  binding. Empty encoder flushes use a bounded state buffer; the source
  extension's zero-sized reverse-write buffer is not a supported memory access.

## Provenance and remaining work

The original Python tree remains read-only. Source hashes, the pinned rANS
revision and its header hash are recorded in
`/path/to/dependencies/ryg-rans-legacy/dcvc_tcm_sources.json`. No additional rANS
checkout or binding package was needed. Microsoft MIT, InterDigital Apache-2.0
and rANS notices are retained under `licenses`.

TCM pins pytorch-msssim 0.2.0. Its 5,924-byte published source archive was
downloaded into `/path/to/dependencies/pytorch-msssim-reference`, its PyPI
SHA-256 checked, and only the metric source copied out without package setup,
installation or import. `SOURCE-0.2.0.json` records URL/hash provenance;
the retained MIT notice is `licenses/PyTorch-MSSSIM.txt`. The installed PyTorch
Normal CDF source was copied locally to `pytorch-reference/dcvc-tcm`, with a
path/hash receipt and the existing PyTorch BSD notice. No model checkpoint,
dataset or media was downloaded.

The I/P models, entropy/containers, checkpoint factories and evaluation command
now have source mappings. Native API, initialization RNG, loss/gradient,
dtype/device, checkpoint, bitstream, CLI error/help and worker/output parity
remain unverified and require later authorized execution. Native numeric
widths and malformed-input exception classes differ from Python. HEM, DC,
FM, RT and EVC still require their family-specific source ports.
