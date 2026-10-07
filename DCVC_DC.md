> Historical implementation notes. These optional DCVC family targets are disabled in the UF presets and are outside the verified UF HT-L INT16 release scope. Status statements below describe earlier source reviews; see [PORT_STATUS.md](PORT_STATUS.md) for current results.

# DCVC-DC source

`uf_dcvc_dc` maps the image/video model, layer, entropy and stream source in
`DCVC-family/DCVC-DC`. Its public interface is
`include/uf/dcvc_dc.hpp`, in namespace `uf::dc`.
`UF_BUILD_DCVC_DC` defaults to ON. This source has not been configured,
compiled, imported or executed. DC evaluation, dataset conversion and
checkpoint download commands now have native source mappings. No model or
checkpoint was loaded into RAM during this work.

## Source mapping

| Python reference | Native source |
| --- | --- |
| `src/models/layers.py` | `dcvc_dc_layers.cpp`: residual down/up/adaptor blocks, depthwise convolutions, plain/gated ConvFFN and DepthConvBlock variants |
| `src/models/video_net.py` | The same source: Float32 warp cache, bilinear helpers, preactivation ResBlock, four-level SPyNet, UNet/UNet2 and hyperprior factories |
| `src/models/common_model.py` | `dcvc_dc_common.cpp`: plain rounding, quality selection, likelihood bits, latent replicate padding/cropping, four-part masks/prior and strict state loading |
| `src/models/entropy_models.py` | `dcvc_dc_entropy.cpp`: AEHelper, learned hyperlatent CDFs, Gaussian/Laplace log tables, signed-int16 symbol/index conversion and partitioned streams |
| `src/cpp/rans/*` and `src/cpp/py_rans/*` | Existing native pinned 64-bit rANS arithmetic plus DC partition headers, per-call splits, asynchronous flush and parallel decode |
| `src/utils/stream_helper.py` | `dcvc_dc_stream.cpp`: checkpoint wrappers/prefix pass, right/bottom padding, I/P files and flags |
| `src/models/image_model.py` | `dcvc_dc_image.cpp`: IntraEncoder/Decoder, IntraNoAR graph, learned/fine quality tables, forward and coding paths |
| `src/models/video_model.py` | `dcvc_dc_video.cpp`: grouped offset diversity, motion feature recurrence, context transforms, feature-index adaptation, DMC forward/coding and CUDA file timing |
| `src/transforms/functional.py` | `dcvc_dc_transforms.cpp`, `dcvc_dc_splines.cpp`: BT.709 array/tensor conversions, box downsampling, SciPy-coordinate chroma zoom with orders 0..5 and spline prefilter |
| `src/utils/video_reader.py`, `video_writer.py` | `dcvc_dc_media.cpp`: PNG naming/EOF, raw planar RGB at 8..16 bits, YUV420 skipping, native-endian samples and ties-to-even output |
| `src/utils/common.py`, `metrics.py` | `dcvc_dc_metrics.cpp`: DC PSNR, log aggregation, boolean/string/folder helpers and fixed JSON; shared Float64 SSIM/MS-SSIM implementation |
| `test_video.py` | `dcvc_dc_evaluation.cpp`, shared `evaluation_workers.cpp`: persistent worker models, checkpoint/fine-index rate selection, RGB/YUV evaluation, decoded output and nested JSON |
| `test_data_to_png.py` | `dcvc_dc_media.cpp`: `convert_one_seq_to_png` and `dcvc-dc-data-to-png` |
| `checkpoints/download.py` | `dcvc_dc_download.cpp`: original six filenames/URLs through sequential external curl transfers |

The library shares `uf_dcvc_legacy` for the pinned rANS and common checkpoint
infrastructure. The models retain DC's four-part prior and family stream
formats.

## Layers and prior

Residual down/up blocks use slopes 0.01/0.1; upsampling uses the source's
1x1 subpixel path. A plain residual block adapts its identity when channel
counts differ. DepthConv uses a 1x1 transform, grouped spatial convolution,
1x1 output and an identity adaptor. Its stride-2 adaptor has kernel size 2;
the source's mismatched odd-size branch shapes are retained.

ConvFFN uses the source's capped expansion formula. ConvFFN2 gates two
channel halves and always uses slope 0.1, including when another slope was
passed. The UNet variants select plain or gated blocks. Preactivation ResBlock
keeps its first activation non-inplace and registers its final activation
alias when enabled. Constructor defaults, weight shapes and module names
follow Python; DC has no HEM-style extra Xavier initialization pass.

Warp cache entries use Float32 grids on the flow's device and a nine-slot
CPU/GPU cache keyed only by flow size. The CPU/GPU-8 slot collision and
source dtype mixing remain. Native access is serialized with a mutex.

The four-part prior splits channels into quarters. Across four stages their
spatial mask assignments are:

| Stage | Channel quarters 0, 1, 2, 3 |
| --- | --- |
| 0 | 0, 1, 2, 3 |
| 1 | 3, 2, 1, 0 |
| 2 | 2, 3, 0, 1 |
| 3 | 1, 0, 3, 2 |

Each stage combines its quarters into one symbol/scale group. Three adaptor
convolutions refine later-stage parameters. Forward/compression combine
recovered values in spatial-mask order; decoding accumulates stage outputs,
retaining Python's different floating addition order.
Mask construction supports odd dimensions. Per-model mask cache keys omit
dtype/device, so reuse after moving/converting a model retains its original
cached values. Hyperanalysis replicates latent edges to a multiple of four;
hyperdecoded parameters are cropped back to the original latent shape.
Quantization always rounds, including in training mode.

## Entropy streams and containers

Entropy tables are plain CPU attributes, not registered buffers. Updating
replaces the coder pointer before checking whether cached CDFs can be reused.
Therefore `update(false)` reconnects an existing table to the new coder;
`update(true)` regenerates it. Loading weights does not invalidate those
tables. Gaussian/Laplace tables have 256 log levels ending at 64, starting at
0.11/0.01 respectively. Likelihood Gaussian scales use DC's minimum 1e-5.

Symbols are clamped to [-30000, 30000] before signed-int16 conversion.
Indexes also convert to signed int16. Encoding skips negative indexes;
the source decoder reads offsets before checking for negatives, so the native
decoder rejects those invalid accesses. Decoded values wrap to int16,
stage through the default CPU floating dtype and convert to the caller's
requested dtype/device in the estimator's original order.

Every entropy encode call divides its flattened input among stream_part
lanes, with the last lane receiving the remainder. Native buffers correctly
allocate that final remainder; the original binding's final index buffer
can overrun when the length is not divisible by stream_part.
The entropy payload begins with a flag: upper bits store lane count minus one,
and low bits select 2- or 4-byte little-endian length fields. All lanes except
the last have a stored size. Their 64-bit rANS payloads follow consecutively.
Native support bounds stream_part to 1..16, matching the header capacity.

Threaded mode or multiple parts use asynchronous per-lane flush; requesting
the stream waits for completion. Decode partitions run concurrently.
Native buffering schedules work differently from the reference's persistent
encoder queues, so runtime timing/lifetime behavior still needs comparison.
Reset waits for outstanding native work and retains previous stream bytes.
Requesting a threaded stream before flush raises a bounded error. Empty
flushes and malformed lengths use bounded behavior in place of unsafe
extension memory accesses.

Outer file headers use big-endian fields:

| File | Header |
| --- | --- |
| I | uint32 height, uint32 width, quality flag byte, uint32 payload length (13 bytes) |
| P | quality flag byte, frame-index byte, uint32 payload length (6 bytes) |

The quality flag stores checkpoint-table selection in its high bit and a
six-bit quality index shifted left by one. Its unused low bit is ignored on
read. Native writers check index 0..63 and frame index 0..255.
P files require caller-supplied dimensions. Payload lengths are bounded to
uint32; readers leave trailing bytes unread.

## Models, qualities and references

`IntraNoAR` defaults to N=256, four anchors, one entropy part and
non-inplace activations. Encoder/decoder quality tensors are applied to
intermediate features. State loading builds 64 fine scales by logarithmic
interpolation between anchor entries 0 and 3. Fine tables stay CPU Float64
attributes, following installed NumPy 1.26.1's default legacy promotion.
The retained NumPy notice is in `licenses/NumPy.txt`.
Endpoint log/exp last-bit behavior against NumPy is unverified; alternate
NumPy promotion settings and non-CPU post-load interpolation are unsupported.
Checkpoint helpers unwrap state_dict, then net, and remove module. in a
second pass so prefixed values overwrite unprefixed duplicates.

Image forward returns unclamped reconstruction and tensor bit/batch rate
fields. Compression/decompression clamp the image; file rates include the
outer header. The image entropy groups are Z followed by four image Y groups.
Decode recovers hyperlatents at ceil(H/64), ceil(W/64), then crops parameters
to ceil(H/16), ceil(W/16).

DMC uses context channels 48/64/96 and image latents 128; motion and motion
hyperlatents have 64 channels. Grouped offset diversity uses 16 groups and
two offsets, with the source's offset/mask layout and maximum residual 40.
The generic inherited forward remains unimplemented; use forward_one_frame.
P reconstruction is clamped in both forward and coding paths.

The `Dpb` contains ref_frame, ref_feature, ref_mv_feature, ref_y and
ref_mv_y. Motion features condition the next motion encoder; image/motion
latents condition their respective priors. I frames must reset all four
non-image tensors. Recurrent feature adaptation selects modules with the
frame-index pattern 0,1,0,2. Native negative indices use Python's modulo rule.

P entropy group order is motion Z, image Z, four motion Y groups, four image Y
groups. Compression flushes before reconstruction so asynchronous coding can
overlap that computation. File encode_decode retains Python's unconditional
CUDA synchronization: this P file/timing path requires a CUDA input and fails
on CPU. Likelihood forward, direct coding methods and I files do not add that
restriction. File timing includes write/read boundaries as in the source.

## Media and transforms

`include/uf/dcvc_dc_media.hpp` represents NumPy image arrays as CPU CHW
floating tensors. `MediaFrame` contains either RGB or Y/UV tensors; undefined
tensors represent EOF. PNGReader prefers im1.png when both naming conventions
exist, otherwise uses im00001.png. Its close operation resets the index and
retains EOF. PNGWriter always writes im00001.png onward, using the supplied
image shape without checking the constructor's configured dimensions.

RGBReader appends .rgb to a source without that suffix. RGBWriter instead
appends /out.rgb to a destination. Raw samples are planar, uint8 at eight bits
or native-endian uint16 at nine through sixteen bits. YUVReader appends .yuv;
YUVWriter appends /out.yuv. Raw writers require an existing parent directory.
YUV skipping retains the reference's empty-read EOF rule; nonempty partial
frames fail at conversion. An invalid odd-size chroma shape also fails.

Array color functions use the source's BT.709 coefficients and clipping
boundaries. Box reductions retain Float32 accumulation for Float32 arrays,
with the small NumPy pairwise reduction arithmetic adapted under its BSD
notice. Chroma zoom uses SciPy 1.15.3's endpoint-based coordinates, constant
boundary handling and mirrored spline footprints. Orders 2..5 prefilter all
three axes in Float64, including the unchanged channel axis. Interpolated
arrays retain their input Float32/Float64 dtype. SciPy's unsupported Float16
zoom remains unsupported. Reconstruction PNG/YUV writes round before clipping.

Torch color functions retain autograd/device and do not clip. Their 420/444
helpers expose concatenated and tuple interfaces separately. The original
nearest mode passes align_corners=False to interpolate and raises; the native
helper retains that failure. These tensor helpers differ from the array zoom.

Spline reference files and receipts are under
`/path/to/dependencies/scipy-reference/dcvc-dc`; NumPy reduction references are
under `/path/to/dependencies/numpy-reference/dcvc-dc`. They are source references,
with no Python import or separately installed numerical library.

## Commands

`dcvc-uf dcvc-dc-test-video` requires `UF_BUILD_DCVC_DC` and
`UF_BUILD_EVALUATION`. It maps the Python root_path/test_classes configuration.

| Option | Default / behavior |
| --- | --- |
| `--test_config`, `--output_path` | Required dataset and result paths |
| `--i_frame_model_path`, `--p_frame_model_path` | One checkpoint each; P is unused with force_intra |
| `--rate_num` | 4 |
| `--i_frame_q_indexes`, `--p_frame_q_indexes` | Optional integer lists; the original ignores the P list |
| `--ec_thread` | False; a bare switch means true |
| `--stream_part_i`, `--stream_part_p` | 1 each |
| `--force_intra`, `--cuda`, `--write_stream` | False; bare switches mean true |
| `--force_frame_num`, `--force_intra_period` | -1; positive values override sequence configuration |
| `--worker`, `-w` | 1; must be positive |
| `--cuda_device` | Optional CUDA_VISIBLE_DEVICES string; empty leaves it unchanged |
| `--yuv420`, `--calc_ssim`, `--save_decoded_frame` | False; require a boolean value |
| `--stream_path`, `--decoded_frame_path` | out_bin and decoded_frames |
| `--verbose` | 0; 1 prints mean P timings, 2 also prints frame statistics |

Boolean conversion accepts yes/y/true/t/1 case-insensitively as true and all
other strings as false. Four matching checkpoint rates use anchor indexes
0..3. Explicit I indexes use fine tables even when four entries were supplied.
Other counts 2..64 use rounded Float64 linspace indexes across 0..63. P frames
use the I index and the frame-index pattern modulo four, as in Python.

When executed, each worker initializes seeded, deterministic I/P models once
and reuses them across points. Visible GPU assignment uses the one-based
worker index modulo device count, with CPU fallback when there is no GPU.
Stream entropy tables update once at initialization. The shared scheduler
retains submission order and handles cancellation/parent death. Runtime worker
scheduling and model construction RNG parity remain unverified.

PNG/YUV source readers and decoded writers follow src_type independently of
the metric representation. Padding replicates the right/bottom edges to 16.
I frames reset all four non-image reference tensors. RGB PSNR retains tensor
arithmetic and can be infinite; RGB MS-SSIM uses pinned pytorch-msssim 0.2.0.
YUV PSNR is Float64 with the source's 999.9 small-MSE result and no UF 99.9 cap.
YUV MS-SSIM uses five levels from 176 pixels, four from 88, and fails below 88
for any plane. Combined YUV metrics use weights 6:1:1. Floating library
logarithms, FFT/reduction order and exact numerical compatibility remain
unverified.

Decoded folders are renamed after each point to
RATE_BPP4DEC_PSNR4DEC. The YUV file remains open across that POSIX rename.
Point logs contain I/P/all averages and optional Y/U/V averages, with no
per-frame arrays even at verbose=2. Results retain ds_name/video_path/rate_idx
inside dataset/sequence/three-digit-rate keys. DC's patched JSON float formatter
uses six fractional digits and lowercase nan/inf tokens, including its invalid
standard-JSON output for nonfinite values. Worker IPC retains those floats
through the shared nonfinite transport format. Native output follows DC's
formatter instead of the root UF JSON formatter.

`dcvc-uf dcvc-dc-data-to-png` maps the YUV420 conversion loop. Its optional
`--input`, `--output`, `--width`, `--height` expose a native override of the
Python script's hardcoded source_yuv_path/destination_png_path and 1920x1080.
It reads RGB using order-1 array zoom, writes five-digit PNG names and prints
the source path and processed frame count. The converter has not been invoked.

`dcvc-uf dcvc-dc-download-checkpoints [--output_directory DIR]` writes to an
existing directory, defaulting to the current directory as in Python. It
retains the three cvpr2023_image and three cvpr2023_video filenames, transfer
order and overwrite behavior. It uses external curl, without automatic retry.
No checkpoint URL was fetched during this work.

## Outstanding work

Review all native module/API integration, checkpoints, construction RNG,
forward/gradient values, CDFs, bytes, references, metrics and timing against
Python when compilation/execution and model loading are authorized. No tests
were added or run. Models/checkpoints have not been loaded into RAM.
Shared pickle/image limitations remain in CHECKPOINTS.md and PORT_STATUS.md.
Original/native source hashes are recorded in
`/path/to/dependencies/ryg-rans-legacy/dcvc_dc_sources.json`.
