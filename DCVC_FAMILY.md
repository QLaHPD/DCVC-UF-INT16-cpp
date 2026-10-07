> Historical implementation notes. These optional DCVC family targets are disabled in the UF presets and are outside the verified UF HT-L INT16 release scope. Status statements below describe earlier source reviews; see [PORT_STATUS.md](PORT_STATUS.md) for current results.

# Original DCVC family source

The `uf_dcvc2021` library maps the original `DCVC-family/DCVC` P-frame model
and its five I-frame architectures. Public interfaces are in
`include/uf/dcvc2021.hpp` and `include/uf/dcvc2021_image.hpp`. CMake includes
this library when `UF_BUILD_DCVC2021` is enabled (the default). It has not been
configured, compiled or executed. The native `dcvc2021-test-video` evaluation
workflow and `dcvc2021-download-checkpoints` helper also have source mappings.
Runtime, checkpoint and bitstream compatibility remain unverified.

## Source mapping

| Python reference | Native source |
| --- | --- |
| `src/models/video_net.py` | `dcvc2021_layers.cpp`: lower-bound gradient, video GDN, preactivation residuals, LeakyReLU residuals, four-level SPyNet and bilinear border warping |
| `src/layers/layers.py:MaskedConv2d` and `subpel_conv3x3` | Registered root weight/bias/mask and pixel-shuffle sequences in the layer/model sources |
| `src/entropy_models/video_entropy_models.py` | `dcvc2021_entropy.cpp`: four-layer learned CDFs, update/force caches, Laplace scale tables, channel indexes and tensor/symbol layout |
| `src/cpp/ops/ops.cpp` and `src/cpp/rans/rans_interface.cpp` | CDF quantization, 64-bit rANS, sentinel symbols, signed bypasses and incremental decoding, without Python bindings |
| `src/models/DCVC_net.py` | `dcvc2021_model.cpp`: all registered layers, likelihood forward, compress/decompress, file encode/decode and raw checkpoint state loading |
| `src/utils/stream_helper.py` | `dcvc2021_layers.cpp` padding/cropping and `dcvc2021_stream.cpp` I/P file containers |
| `src/layers/gdn.py`, image residual layers, `src/ops/parametrizers.py` and `bound_ops.py` | `dcvc2021_image_layers.cpp`: registered reparametrizers, custom lower-bound gradient, image GDN, residual down/up blocks and subpixel layers |
| `src/entropy_models/entropy_models.py` | `dcvc2021_image_entropy.cpp`: entropy bottleneck parameters/medians, Gaussian conditionals, likelihoods, scale indexes, learned CDFs and batch string interfaces |
| `src/models/priors.py`, `waseda.py`, `utils.py` and `src/zoo/image.py` | `dcvc2021_image_model.cpp`: five registered graphs, likelihood forward, coding methods, dynamic CDF buffer loading and raw checkpoint factories |
| `test_video.py` and its `pytorch_msssim` dependency | `dcvc2021_evaluation.cpp` and `dcvc2021_evaluation_metrics.cpp`: dataset/checkpoint points, GOP and rate paths, reconstruction saving, PSNR/MS-SSIM and result JSON; shared spawn/IPC scheduler in `evaluation_workers.cpp` |
| `checkpoints/download_compressai_models.py` | `dcvc2021_download.cpp`: the same eight filenames/URLs, sequential transfers and overwrite behavior through external curl |

## Model and stream interfaces

`uf::dcvc2021::VideoModel` is a LibTorch module. Its `load_checkpoint`
accepts a Python `torch.save` ZIP state dictionary through the existing native
pickle reader. `load_dict` strips one leading `module.` prefix and checks all
parameter/buffer names and shapes. The module names, sequential indexes,
convolution sizes and GDN parameters follow the Python model.
Checkpoint prefix normalization follows the original dictionary order so
duplicate normalized names retain Python's last-entry behavior.
Video GDN retains its optional beta minimum, gamma initialization and
reparametrization offset, plus the source's five-dimensional reshape path.
Warp grids are generated on CPU and retained by device/flow shape, including
their original dtype. Native concurrent access to that cache uses a mutex.

Call `update` before entropy coding. `update(false)` retains existing CDF
caches; `update(true)` rebuilds them. Loading different weights after an
update does not automatically rebuild the cached CDFs, matching Python.
Hyperlatent CDFs are built on the module's device; the 64-level Laplace table
stays on CPU. Entropy inputs require batch size one.

The original dataset workflow uses RGB frames with dimensions divisible by
64. The P-frame model consumes those frames directly. It does not implicitly
pad or crop them. The standalone `pad` and `crop` helpers retain the centered
zero-padding behavior used by the I-frame helpers.

The likelihood forward rounds latents before reconstruction and separately
estimates their rates with means and Laplace scales. Stream compression and
decompression instead reconstruct each latent pixel from its decoded neighbors,
using the hyperprior and motion/context priors. Symbols are encoded in HWC order
for the autoregressive latents and CHW order for the hyperlatents.

The Python masked convolution multiplies `weight.data` by its registered mask
when its forward method is called. Its coding helpers read the stored weight
directly. The native paths preserve that distinction and do not silently alter
a checkpoint's autoregressive weights at load time.

`compress` returns the four strings and their payload-only rates.
`encode` writes a P-frame file; `decode` reads it; `encode_decode` replaces
the reconstruction with the decoded result. Forward/compression reconstruction
remains unclamped; decompression clamps RGB to [0, 1], as in Python.
Evaluation callers must select eval mode, device and their no-grad context.

The P-frame header is six big-endian uint32 values: height, width, then the
lengths of motion Y, motion Z, image Y and image Z. Those four strings follow in
the same order. The I-frame container has four uint32 values: height, width,
image Y length and image Z length, then both strings. Readers ignore trailing
bytes as the Python helper does. These containers are distinct from UF packets.
Their entropy payload uses the original single 64-bit rANS state with
little-endian uint32 words, 16-bit CDF precision and 4-bit bypass symbols.

## I-frame interfaces and reference behavior

`uf::dcvc2021::image::Model` selects these architectures:

| Architecture name | Graph |
| --- | --- |
| `bmshj2018-factorized` | Factorized entropy bottleneck |
| `bmshj2018-hyperprior` | Scale hyperprior |
| `mbt2018-mean` | Mean/scale hyperprior |
| `mbt2018` | Joint autoregressive hyperprior |
| `cheng2020-anchor` | Cheng residual autoregressive hyperprior |

`from_checkpoint` loads a raw Python ZIP state dictionary and infers channel
counts from the same convolution keys as the original factory. I-frame keys
are used directly, without the P-frame prefix normalization. Strict loading
checks parameter/buffer names and shapes. Entropy buffers resize only when
empty, matching the embedded model loader.

`forward` returns reconstruction and Y/Z likelihood tensors. `compress` returns
one Y string group for the factorized model or two Y/Z groups for the other
models, plus the latent shape. The factorized reference has no file
encode/decode helper; the unified native file methods reject that architecture.
The hyperprior variants use the I-frame container described above and retain
the Python payload/file rate calculations. No implicit input padding is added.
Forward reconstruction is unclamped; hyperprior decompression clamps to [0, 1].

The learned entropy bottleneck quantizes around its median. Gaussian parameters
are split into scales then means, unlike the P-frame parameter order.
Nonautoregressive latents and hyperlatents use CHW symbols; joint/Cheng image
latents use raster HWC symbols. Their context coding reads the stored masked
convolution weights directly, as the Python helpers do.

Module device/dtype conversion preserves integer CDF buffers and converts
existing parameter gradients, following Python's `nn.Module.to()` policy.
CDF generation and scale-table preparation retain the source's CPU/device
placement. Decoding stages integer lists through the default floating dtype
before assignment to an integer CDF-device tensor, then dequantizes using the
means. Joint/Cheng reconstruction starts with default-dtype tensors on the
hyperlatent device. Callers must select eval mode and their no-grad context.

Several behaviors belong to this particular embedded reference:

- Joint/Cheng training forward requests `noise`, which this copy's quantizer
  does not support. The native path rejects it too.
- `CompressionModel.aux_loss()` calls a missing entropy-bottleneck `loss()`
  method; the native interface reports that unsupported path. Primary parameter
  iteration excludes the whole bottleneck, while auxiliary iteration includes it.
- Bottleneck decoder mean-shape checks reject batches larger than one.
- The joint hyperprior decoder omits the activation after its first deconvolution.
  The layer sequence and the constructors' discarded initialization draws are
  preserved rather than replaced with a newer upstream model graph.

Gaussian table quantiles use a small native adaptation of SciPy 1.15.3's Cephes
inverse-normal routine in `dcvc2021_ndtri.hpp`. This adds no SciPy runtime.

## Evaluation and checkpoint commands

`dcvc-uf dcvc2021-test-video` maps this family's `test_video.py`. It requires
both `UF_BUILD_DCVC2021` and `UF_BUILD_EVALUATION`. This is a distinct command
from the root UF `test-video`; it accepts the original flat dataset JSON whose
top-level dataset objects contain `base_path` and `sequences` with `frames`/`gop`.

| Option | Default / behavior |
| --- | --- |
| `--test_config` | Required dataset JSON |
| `--output_json_result_path` | Required result JSON; parent directory must exist |
| `--i_frame_model_name` | `cheng2020-anchor`; five architecture names above |
| `--i_frame_model_path` | List with an I checkpoint for each P checkpoint; extra I entries remain unused |
| `--model_path` | P checkpoint list, one rate point per checkpoint and sequence |
| `--worker`, `-w` | 1 worker; must be at least one |
| `--cuda` | False; a bare switch means true |
| `--cuda_device` | Optional literal `CUDA_VISIBLE_DEVICES` value; empty leaves the environment unchanged |
| `--write_stream`, `--write_recon_frame` | False; bare switches mean true |
| `--recon_bin_path` | `recon_bin_path` |
| `--model_type` | `psnr`; any other string selects MS-SSIM, as in Python |

Boolean values retain yes/y/true/t/1 as true, case-insensitively; other strings
are false. Checkpoint list order and repeated/attached option behavior use the
shared CLI parser. Options with optional boolean values retain `nargs='?'`.

The shared executable worker pool now selects a model backend. Original DCVC
workers reconstruct and load their models for every point, reseed Torch with
zero and use one CPU thread. GPU assignment uses the one-based worker index
modulo the visible GPU count. CUDA-disabled workers and CUDA requests without
an available device use CPU. The parent sets `CUBLAS_WORKSPACE_CONFIG=:4096:8`;
stream points disable cuDNN benchmarking and require deterministic algorithms.
Worker failure/interruption uses the existing process-group teardown and
parent-death handling. Completed results are retained in submission order.

Input frames use `im1.png` or `im00001.png`, with short naming taking precedence.
RGB pixels follow the source's Float64-to-Float32 CHW conversion and division
by 255, without an additional EXIF transpose. The evaluator does not pad or crop
frames. It compares pixel counts across a sequence, as the reference does.
Every GOP boundary uses the I model; later frames use the previous clamped
reconstruction as the P reference. Stream mode forces both entropy updates and
encodes/decodes each frame. Otherwise rates come from likelihoods. Stream bpp
counts entropy payloads and excludes the container headers.

The reconstruction folder is always created at
`recon_bin_path/sequence/P_checkpoint_basename[:-4]`. Streams are `index.bin`;
optional RGB reconstructions are `recon_frame_index.png`, using round-to-even
pixel conversion. Rates retain the motion-Y, motion-Z, image-Y and image-Z
components. Result JSON retains the original I/P/all-frame averages, nesting,
checkpoint-prefix matching and overwrite behavior, including bare Infinity/NaN.

PSNR retains tensor-dtype reductions and the source's square-root/log formula;
perfect images produce Infinity. MS-SSIM uses the package's default separable
valid Gaussian filters, five scale weights, odd-size pooling padding and ReLU
before scale products. This differs from the root UF metric implementation.
The package requires spatial dimensions larger than 160. The original dependency
is unpinned and is absent from the inspected environment: the native workflow
adapts the published 0.2.1 default RGB path and retains a 1.0.0 comparison source.
No package was installed or executed. Its unused general volume API is not
exposed by this RGB evaluation command.

`dcvc-uf dcvc2021-download-checkpoints` maps the eight sequential CompressAI
downloads in the Python helper. Its output directory defaults to the current
directory; the native `--output_directory` option selects another existing
directory. It prints each filename and invokes curl with an argument vector,
replacing existing files as `urlretrieve` does. It does not fetch P checkpoints
or load models. HTTP/proxy/TLS behavior belongs to curl, so it is not identical
to Python urllib's transport. This helper was written but not invoked during
the port; no model checkpoint was downloaded for this work.

Malformed configuration errors and integer overflow use native exceptions;
Python exception classes/messages and unlimited integer widths are not retained.
Progress display uses the existing native scheduler rather than tqdm.

## Reference and dependency provenance

The mathematical rANS reference was cloned separately into
`/path/to/dependencies/ryg-rans-legacy` at the revision specified in the
Python family's dependency file:
`c9d162d996fd600315af9ae8eb89d832576cb32d`.
The native coder adapts its arithmetic with bounded byte reads and owned
buffers. It requires no pybind11 or new linked library.

Original project files remain read-only. `dcvc2021_sources.json` in that
separate reference folder records the original paths and source hashes used
for this adaptation. The installed PyTorch Laplace CDF source is copied under
`/path/to/dependencies/pytorch-reference`; the retained BSD license already
covers this reference. Apache-2.0 and ryg_rans notices are under `licenses`.
The installed PyTorch module conversion source is also copied there with a
separate path/hash receipt. SciPy 1.15.3 Cephes headers and its license were
downloaded into `/path/to/dependencies/scipy-reference`; installed version and
license metadata were copied without importing SciPy. That folder's
`SOURCE.json` records URLs and hashes. SciPy/Moshier notices are retained in
`licenses/SciPy.txt`.
MS-SSIM reference sources, the published 0.2.1 source archive and URL/hash
receipts are under `/path/to/dependencies/pytorch-msssim-reference`. Its source
was read without importing or installing it. The retained MIT notice is
`licenses/PyTorch-MSSSIM.txt`.
No dependency configure, compile, installation or test was performed.

## Work still required

- Verify the shared native checkpoint reader's ZIP, legacy stream and
  historical tar formats against real family weights when execution is
  authorized. Reader source and OrderedDict normalization are connected;
  supported data and remaining limits are in [CHECKPOINTS.md](CHECKPOINTS.md).
- Confirm checkpoint compatibility, model/gradient numerics, entropy CDFs,
  reference bitstreams, decoded frames, rate calculations, malformed-input
  failure behavior and native dependency integration when execution is
  authorized. The original unsafe binding could assert or overrun on malformed
  streams; the native implementation throws bounded-read/range errors.
- TCM's I/P-frame models, entropy/stream wrappers, checkpoint factories,
  pinned loss and evaluation command now have source mappings recorded in
  [DCVC_TCM.md](DCVC_TCM.md). HEM's I/P models, dual priors, four-tensor
  references, entropy/containers and evaluation/download commands also have
  source mappings in [DCVC_HEM.md](DCVC_HEM.md).
  DC model/layer/four-part prior, partitioned entropy, quality, I/P containers,
  RGB/YUV media helpers and evaluation/conversion/download command source is
  recorded in [DCVC_DC.md](DCVC_DC.md). Port the remaining FM, RT and EVC families.

The source review does not establish runtime or bitstream equivalence.
