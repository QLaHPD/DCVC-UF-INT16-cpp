> Historical implementation notes. These optional DCVC family targets are disabled in the UF presets and are outside the verified UF HT-L INT16 release scope. Status statements below describe earlier source reviews; see [PORT_STATUS.md](PORT_STATUS.md) for current results.

# DCVC-HEM source

The native `uf_dcvc_hem` library maps the image/video models in
`DCVC-family/DCVC-HEM`. Public model interfaces are in
`include/uf/dcvc_hem.hpp`; evaluation/image helpers are in
`include/uf/dcvc_hem_evaluation.hpp`. `UF_BUILD_DCVC_HEM` defaults to ON.
This source has not been configured, compiled, imported or executed.
Checkpoint, gradient, numerical and bitstream compatibility remain unverified.

## Source mapping

| Python reference | Native source |
| --- | --- |
| `src/layers/layers.py` | `dcvc_hem_layers.cpp`: residual down/up blocks, subpixel convolutions, squeeze/excitation and UNet |
| `src/models/video_net.py` | The same layer source: lower-bound autograd, bilinear helpers, cached border warp, ResBlock and four-level SPyNet |
| `src/models/common_model.py` | `dcvc_hem_common.cpp`: initialization, quantization/noise, likelihood bits, masked dual prior, coder setup and checkpoint loading |
| `src/entropy_models/entropy_models.py` | `dcvc_hem_entropy.cpp`: learned CDFs, Gaussian/Laplace scale tables, CDF helpers and combined buffered entropy groups |
| `src/cpp/ops/ops.cpp` and `src/cpp/rans/rans_interface.*` | Existing native 64-bit legacy rANS arithmetic plus the HEM buffered interface; no Python binding |
| `src/models/image_model.py` | `dcvc_hem_image.cpp`: IntraNoAR graph, q scales, loss/rate forward and stream coding |
| `src/models/video_model.py` | `dcvc_hem_video.cpp`: DMC 1.19 graph, multiscale contexts, four reference tensors, motion/image dual priors and loss/stream paths |
| `src/utils/stream_helper.py` | `dcvc_hem_stream.cpp`: padding/cropping, checkpoint normalization, quality rounding and I/P files |
| `src/utils/png_reader.py` and `test_video.py` | `dcvc_hem_evaluation.cpp`: RGB image helpers, PNGReader, rates, workers, GOP references, metrics, streams/decoded PNGs and result JSON |
| `checkpoints/download.py` | `dcvc_hem_download.cpp`: the four original URLs/filenames and sequential overwrite transfers through curl |

The pinned MS-SSIM 0.2.0 implementation is shared with TCM through
`uf_dcvc_legacy`. Its default scalar reduction is retained; HEM model losses
request the source's per-batch reduction.

## Models and reference state

`uf::hem::IntraNoAR` defaults to N=192 and four anchors. Its Gaussian
image prior includes a learned channel quality tensor and a dual spatial prior.
`uf::hem::DMC` defaults to four anchors, with motion channels=64,
context channels=64 and image latent channels=96. The model version is 1.19.
Both constructors retain convolution initialization order and the source's
Xavier-normal gain and bias policy.

The video `Dpb` contains `ref_frame`, `ref_feature`, `ref_y` and
`ref_mv_y`. An I-frame resets the last three tensors to undefined.
P-frames recover and retain both quality-scaled latent tensors for the next
frame's priors. Stream decoding clamps RGB, while forward/compression
reconstructions remain unclamped until the evaluation worker clamps the
reference in place.

Model forwards return the source's batch rates, summed-channel MSE and
MS-SSIM. Training uses straight-through rounding for reconstruction and
independent uniform noise for likelihood rates in Python's draw order.
Compression rounds hyperlatents directly; dual-prior quantization follows the
module training flag. Select eval mode and a no-grad context for evaluation.
HEM residual down/up blocks have two LeakyReLU slopes and no GDN.

Checkpoint loading uses the shared native CPU pickle reader. It unwraps
`state_dict`, then `net`, and removes `module.` in a second dictionary
pass. Prefixed entries overwrite an existing unprefixed key, following the
installed PyTorch helper. Strict loading checks parameter names/shapes.
Plain CDF/cache tensors are not registered buffers; loading weights or moving
a module does not rebuild them. Reader scope is in [CHECKPOINTS.md](CHECKPOINTS.md).

## Entropy and files

Call `update(true)` before stream coding. The source creates a new coder on
each update, while estimators with an existing coder may skip update(false);
the native source preserves that behavior. Gaussian image scales use 256 log
levels from 0.11 to 64. Laplace video scales use 256 levels from 0.01 to 64.
The selected distribution determines both support search and PMF values.
Scale tables and learned CDFs remain plain CPU attributes.

The dual prior codes two complementary checkerboard groups across channel
halves. Odd latent spatial dimensions retain the source's mask-shape failure.
The combined I stream group order is Z, image Y group 0, image Y group 1.
P groups are motion Z, motion Y groups 0/1, image Z, image Y groups 0/1.
All use the existing pinned 64-bit rANS state, 16-bit CDF precision and
4-bit bypass values. Hyperlatent coding is restricted to batch size one.

File headers are big endian:

| File | Header | Payload |
| --- | --- | --- |
| I | uint32 height, uint32 width, uint16 q index, uint32 length (14 bytes) | Combined I stream |
| P | uint16 motion q index, uint16 image q index, uint32 length (8 bytes) | Combined P stream |

Quality values are clipped to [0.01, 655], multiplied by 100 and rounded
to even; the recovered scale is index/100. The Tensor overload follows the
installed NumPy 1.26.1 default legacy promotion: a zero-dimensional checkpoint
scalar is converted to Float64 before clipping with Python floating bounds.
This is inferred from the installed NumPy fallback and Torch binding sources,
consistent with NumPy's documented
[scalar promotion rules](https://numpy.org/neps/nep-0050-scalar-promotion.html).
Native behavior for alternate NumPy promotion settings remains unsupported.
Rate helpers include these file headers in reported bits. P files do not
store dimensions; callers must supply the original image size and reference.
Readers leave trailing bytes unread and reject truncated or oversized fields.

`CompressedFrame.dbp` retains the Python compress result's spelling
`"dbp"`. Forward/file results use `dpb`. Stream-mode component bit fields
are absent, represented by empty optionals. The Python GaussianEncoder helper
`set_decoder_cdf` calls a missing coder method; the native helper throws a
logic error. Empty entropy flush and malformed input use bounded native
behavior in place of unsafe accesses in the original extension.

## Evaluation and checkpoint download commands

`dcvc-uf dcvc-hem-test-video` maps HEM's test script and requires
`--test_config` and `--output_path`. I/P checkpoint paths are single
values; q-scale overrides accept lists. Defaults are four rate points,
one worker and CPU execution. Optional boolean values use the same strict
Python vocabulary; `--save_decoded_frame` requires a value.

Configuration uses `root_path` and `test_classes`. Each active class
specifies `base_path`, PNG `src_type` and sequences with width, height,
frames and GOP. Empty `--force_root_path` is still an explicit override.
PNGReader prefers im1.png to im00001.png, returns CHW Float32 RGB and checks
the configured dimensions. Missing frames set EOF. Its close method resets
the index and retains EOF, as in the source. The native reader exposes RGB
output only.

The command reads checkpoint q tables even when explicit scales are given.
It selects matching checkpoint tables or interpolates logarithmically between
first and last values. Supplying image P overrides requires motion overrides;
motion overrides alone are ignored. Checkpoint scalar dtypes are carried
through IPC; explicit/interpolated scales are Float64. The native endpoint
log and exponential use host math, whose last-bit results against NumPy
remain unverified. Quality tables support Float16, Float32 and Float64.

Workers reload weights per sequence/rate, set the same seed, single-thread and
deterministic policies, and use one-based worker-index modulo GPU-count
selection. They pad right/bottom to 64, update entropy with force=true for
streams, carry all reference tensors and calculate RGB PSNR and MS-SSIM.
The source's stream/decoded paths are retained:

- Streams: STREAM_PATH/SEQUENCE/RATE_INDEX/FRAME_INDEX.bin.
- Decoded frames: DECODED_FRAME_PATH_DMC_RATE_INDEX/SEQUENCE/RATE_INDEX/FRAME_INDEX.png.

Results include all per-frame/aggregate metrics, frame pixel count, time,
dataset, sequence and rate fields. Dataset/sequence nesting uses three-digit
rate keys and six fixed fractional digits, preserving bare NaN/Infinity
through the shared native JSON writer. Final frame counts include every rate
point. Shared process transport replaces Python multiprocessing; progress-bar
presentation and exception text differ.

`dcvc-uf dcvc-hem-download-checkpoints` transfers the four original
image/video PSNR/SSIM checkpoints in source order using external curl.
It overwrites existing files, defaults to the current directory and accepts
the native `--output_directory` extension. That directory must exist.
The helper has not been invoked. No checkpoint, dataset or media was downloaded.

## Outstanding review

All source/API integration, checkpoint loading, RNG/gradient/numerical behavior,
CDFs, bytes, reconstructed references, metrics, timing and process behavior
remain unverified without compilation/execution. Native frame/dimension/count
widths and positive-rate limits use bounded errors; Python's error priority and
formatting are not fully replicated. Shared image decoder differences remain
recorded in [PORT_STATUS.md](PORT_STATUS.md). No tests were added or run.
Original/native source hashes are recorded in
`/path/to/dependencies/ryg-rans-legacy/dcvc_hem_sources.json`.
