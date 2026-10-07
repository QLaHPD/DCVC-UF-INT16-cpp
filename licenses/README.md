# Attribution and source adaptation notes

These detailed notes describe the development history. Current build and runtime evidence is in [../TESTING.md](../TESTING.md). Local machine paths have been replaced with placeholders. See [../PROVENANCE.md](../PROVENANCE.md) for the public source references.

# Third-party source notices

## UF Pareto progress

The configured UF Pareto stderr bar in src/pareto_progress.hpp adapts installed
tqdm 4.67.3 std.py/utils.py progress formatting, EMA/miniter updates, postfix
ordering, terminal columns and final display policy. The header retains its
MPL-2.0 AND MIT notices; [the installed license](tqdm-LICENCE) was copied locally.
The package code was read without importing or executing it. Native code uses
standard C++ and POSIX APIs, with no tqdm/Python runtime dependency. No source/
dependency download or install, progress/helper/library/model/build/test
execution. Environment, monitor, encoding/display-cell and full runtime parity
remain unverified.

Pareto progress display-width intervals were exported from reference CPython
3.10.20 stdlib Unicode 13 F/W properties. The same existing
[CPython notice](CPython-3.10.20.txt) covers this source-data policy. ANSI width/
trimming and COLUMNS/LINES fallback follow the installed tqdm utilities under
their retained notice. The export imported stdlib unicodedata only; no tqdm,
project, renderer, model or test was executed and no dependency was downloaded.

## UF training epoch and output boundaries

Negative epoch schedule indexing and checked device/progress/skip/resume/save
message boundaries follow maintained train_image.py, train_video.py and
src/utils/common.py under the project's existing Microsoft MIT notice.
Installed DistributedSampler source and CPUGenerator declarations were read
under the existing [PyTorch license](PyTorch.txt); signed epoch reset and
UInt64 seed conversion reuse existing native code. Standard C++ streams and
exceptions implement output checks without introducing a flush or dependency.
No external implementation was copied or downloaded, and no trainer, library,
checkpoint, model, compiler or test was executed. Full parity is unverified.

## UF encode preflight order

Preset selection, channel expansion and subsequent encode semantic validation
follow maintained main.py/presets.py/streaming.py. Installed argparse.py was
read to trace required-argument and unknown-argument order under the existing
[CPython license](CPython-3.10.20.txt). Existing native text/filesystem/list and
remote helpers are reused; no external implementation was copied or downloaded.
No parser/helper/model/library/build/test execution; full parity is unverified.

Other root commands and check-int16 now apply their semantic checks after
argument parsing, retaining required/conversion checks inside the parser.
CUDA-index and decode worker-check order follows main.py; the validation size
boundary follows scripts/check_uf_int16.py. Installed argparse.py was read only;
existing project and CPython notices apply, with no copied implementation.
The integer report's comparison truth check, Path construction, serialization
before open and checked write/close follow the maintained validator and installed
pathlib.py. Existing [CPython license](CPython-3.10.20.txt) and project notices
apply. Native code reuses the existing path/JSON helpers and standard streams;
no external implementation was copied or downloaded and no new dependency was
added. No validator/helper/library/model/build/test execution.

## INT16 tensor buffer ownership

The maintained INT16 model/layer/arithmetic and py_rans ownership paths were
read alongside installed ATen from_blob/Tensor declarations. Native vectors
now transfer storage ownership through standard C++ shared pointers and ATen
deleters; no SDK implementation was copied. Existing project/dependency notices
apply. No new dependency, download or install; source comparison does not prove
compiler, tensor lifetime or numerical/runtime parity. No model/build/test ran.

## UF buffered output failure cleanup

Exclusive open, buffer selection, short-write tracking and flush/close ordering
were compared with primary CPython 3.10.20
[bufferedio.c](https://raw.githubusercontent.com/python/cpython/v3.10.20/Modules/_io/bufferedio.c),
[fileio.c](https://raw.githubusercontent.com/python/cpython/v3.10.20/Modules/_io/fileio.c)
and [_iomodule.c](https://raw.githubusercontent.com/python/cpython/v3.10.20/Modules/_io/_iomodule.c).
The existing [CPython license](CPython-3.10.20.txt) covers these algorithm/policy
adaptations. Browser inspection saved no dependency implementation source.
Native code uses standard C++ buffering, POSIX I/O and the existing descriptor
helper; no new library/package/install. No write/close/encoder/model/build/test
execution; interruption, native buffer limits and compiler/runtime parity remain
unverified.

## UF decoder owned input close and finite checks

The shared archive decoder and its IntegerCodec inheritance were read for
valid-frame finite checks, with-open close ordering and returned timing.
Native code uses existing std::ifstream, standard C++ callbacks and installed
LibTorch isfinite; installed GCC fstream declarations/close implementation and
ATen operator declarations were inspected only. No GCC/LibTorch implementation
was copied or downloaded and no new dependency/source/package was installed.
Existing project source notices and standard library/dependency notices apply.
No reader, close helper, library, decoder, model, compiler or test ran; full
file/error/timing/numerical and compiler/runtime parity remains unverified.

## UF shared dashboard transitions and handler events

UF imports the retained RT dashboard/progress presentation helpers through
archive/rt_managed.py; reading those helpers is within the UF application
source review and does not add an RT codec implementation to this port.
Approval selection, wall-clock elapsed display, lifecycle active counts and
percentage width follow that retained shared presentation source. Managed
backend error/completion event order follows archive/dashboard.py. Existing
project source notices remain applicable. Standard C++ chrono and existing
native event transport are reused; no new dependency/source/package/download
or install. Native exception diagnostics, process isolation, terminal behavior
and compiler/link/runtime parity remain unverified; no library/helper/terminal/
handler/event/thread/process/model/build/test execution.

## UF calibration event and progress log ordering

Event field-default precedence additionally follows maintained storage.py.
The timestamp auto-timespec rule follows installed Python 3.10 datetime.py
under the existing [CPython license](CPython-3.10.20.txt). No implementation
was downloaded or loaded; clock, locale/year-range and runtime fidelity remain
unverified.

The maintained archive/stream_pipeline.py, storage.py and workflow.py were
read for calibration critical sections and event serialization/delivery/log
ordering. Native code reuses its existing event transport and FileDescriptor
checked close helper, POSIX open/write/close and existing uf_archive/Threads
linkage. These changes adapt maintained UF application source under its
retained project notices. No external implementation, dependency, package or
source was downloaded or installed. Source inspection does not establish
complete event serialization, interruption, exception, concurrency or runtime
parity; no helper, library, event, lock, process, model, build or test ran.

Event delivery now checks std::cout state after insertion and flush before any
progress-log append; pipe EINTR retries consult the existing interrupt helper.
Installed GCC ostream insertion/flush implementations were read to trace their
badbit policy. No GCC implementation was copied or executed; these are standard
C++ stream operations with existing POSIX/native helpers and dependencies.

## UF native thumbnail URL resolution

The HTTP/HTTPS/FTP component parsing, reconstruction and relative joining policy
adapts installed Python 3.10 urllib.parse and urllib.request under the existing
[CPython license](CPython-3.10.20.txt). Bracketed authority/IPvFuture/scope checks
follow installed urllib.parse/ipaddress; IPv6 validity uses declared POSIX
inet_pton with a typed in6_addr destination. CURLU resolution is removed; libcurl
remains the transport. Installed source/headers were read without downloads or
parser/library/network/helper/model/build/test execution. Unicode/NFKC/IDNA,
libc/parser/error/resource and compiler/link/runtime fidelity remain unverified.

## UF native libcurl thumbnail transport

libcurl is reused from the installed Conda prefix; this pass downloads no curl
implementation or dependency source. Its public SDK carries the curl license
notice and [upstream license link](https://curl.se/docs/copyright.html). Native
client ownership/callback wiring is project source; header/redirect/response
policies adapt installed urllib/http.client/email/urllib.parse under the existing
[CPython license](CPython-3.10.20.txt). The Latin-1 decoder shares the existing
UTF-8 codepoint writer. Response field handling retains continuation/trailing
whitespace, first-field selection and trailer exclusion; redirect controls are
removed before byte quoting. Public API behavior was read in the official
[header-list](https://curl.se/libcurl/c/CURLOPT_HTTPHEADER.html),
[callback](https://curl.se/libcurl/c/CURLOPT_HEADERFUNCTION.html) and
[URL](https://curl.se/libcurl/c/curl_url_set.html) documentation without saving it.
SDK/package metadata, FindCURL and selected exports were inspected only; no
library/global/HTTP/callback/helper/model/configure/compiler/build/test ran.
Full URI/proxy/TLS/FTP/socket/framing/error/resource and runtime parity is unverified.

## UF JPEG verification and thumbnail HTTP headers

The header-only JPEG marker/frame/DQT and selected application-field checks
adapt installed Pillow JpegImagePlugin/ImageFile policies, covered by the
[Pillow license](Pillow.txt). They do not decode compressed scan data. Request
header normalization, byte grammar, Latin-1 and integer values and default
request fields follow installed Python 3.10 urllib/http.client, covered by the
[CPython license](CPython-3.10.20.txt). The Latin-1 helper reuses the existing
UTF-8 unit reader. Curl empty-header/config/glob/HTTP-version option behavior was
read in its [official documentation](https://curl.se/docs/manpage.html#-H) without
saving source. No implementation/dependency was downloaded or parser/library/
helper/network/image/model/build/test executed. All metadata/Unicode/special
header/transport/error/resource and compiler/runtime fidelity remains unverified.

## UF remote thumbnail verification

PNG header and CRC verification follows installed Pillow PNG/ImageFile policies,
using the existing metadata/CRC code and [Pillow license](Pillow.txt). WebP opens
its animation decoder and reads canvas metadata without requesting a frame,
consistent with installed WebPImagePlugin and the
[corresponding upstream extension](https://github.com/python-pillow/Pillow/blob/12.1.1/src/_webp.c#L362).
Browser source was read without saving it. Other formats retain FFmpeg fallback;
acceptance, warning/error/resource and compiler/runtime parity remain unverified.
No image/library/helper/model/build/test ran or dependency was downloaded.

## UF checkpoint filename output

Direct checkpoint/status filename handling and ZIP context cleanup follow
installed serialization.py and maintained common.py under the existing
[PyTorch notice](PyTorch.txt). Official constructor bindings and EOF source were
browser-read as an upstream comparison without saving implementation files.
Existing units/SDK/public linkage are reused; no downloads or writer/model/
library/build/test execution. File/byte/error/resource and runtime parity remain
unverified.

## UF PNG output

Minimal RGB PNG metadata, file opening and encoder-failure cleanup follow
installed Pillow Image.py/PngImagePlugin.py, matching retained copies under the
existing [Pillow license](Pillow.txt). Libpng declarations were read only and
existing PNG::PNG linkage is reused; no library implementation was copied or
downloaded. No encoder/helper/model/build/test ran. Compression/fallback/error/
resource/longjmp and full compiler/link/runtime parity remain unverified.

## UF temporary directory and filename policy

Named files, shared mkdtemp name policy, Linux TemporaryFile O_TMPFILE/fallback
and TemporaryDirectory permission/missing-path recovery follow installed Python
3.10 tempfile/shutil/os. Maintained UF image/PNG/thumbnail contexts and the
UF-used RT Jetson diagnostic policy were read only. Existing
native diagnostic suffix/scan code now propagates file inspection/read errors
and checks interruption/offset bounds; Jetson waits before reading diagnostics
and preserves a failed read without another diagnostic attempt. These follow
the maintained shared JetsonPipe.wait and UF streaming error-read boundaries
using standard C++/POSIX interfaces. No implementation was copied or downloaded.
Existing
[Python license](CPython-3.10.20.txt) and [MT19937 notice](MT19937.txt) cover retained
policy/algorithm adaptations. No implementation or dependency was downloaded;
no tempfile/permission/tree/helper/model/build/test ran. Buffered/error/resource/
race/cancellation, HTTP/PIL and full compiler/link/runtime parity are unverified.

## UF remote cookie and prepared-download cleanup

Cookie-copy context and prepared-download cleanup follow maintained
streaming.py/stream_pipeline.py. Installed contextlib/shutil and POSIX descriptor
declarations were read only; native ownership/finally helpers reuse existing
source units and linkage. The existing [Python license](CPython-3.10.20.txt)
covers retained Python policy adaptations. No external implementation/dependency
was downloaded and no cookie/copy/context/helper/process/model/build/test ran.
Temporary directory/name and buffered/error/resource/monitor/runtime parity
remain unverified; explicit native early descriptor ownership is documented.
Ordinary downloader output/wait/diagnostic finally boundaries now follow
streaming.download_pipe through the existing native process/descriptor APIs.
Video/audio body failures receive explicit cleanup before cookie removal;
diagnostic close failures can replace earlier failures. Project and existing
Python notices apply, without copied implementation or a new dependency.
No context/downloader/helper/library/model/build/test execution or download/install;
full close/signal/exception/compiler/runtime parity remains unverified.

## UF remote text, argv and dashboard paths

Remote channel ID file reading, subprocess argument text conversion and
dashboard path/name handling follow maintained streaming.py/dashboard.py and
installed Python pathlib policies. Existing native descriptor/text/JSON helpers
are reused; the Python policy adaptation remains covered by the retained
[Python license](CPython-3.10.20.txt). No external implementation or dependency
was copied/downloaded, and no helper/model/configure/compiler/build/test ran.
Other HTTP/display/locale/error/resource and full runtime parity are unverified.
Primary [CPython subprocess argv conversion](https://github.com/python/cpython/blob/v3.10.20/Modules/_posixsubprocess.c#L742)
was read through the browser without saving source.

## UF filesystem text and JSON paths

Remote format selection additionally follows the maintained streaming.py change
in commit 9df8dcc; its prior revision hash was checked through read-only Git.
No external source or dependency was downloaded for that adjustment.

Filesystem surrogateescape and strict UTF-8 identity policy follow retained
CPython 3.10.20 unicodeobject.c/initconfig.c, covered by the existing
[Python license](CPython-3.10.20.txt). Locale conversion uses installed libc
iconv declarations; no implementation was copied or downloaded. Native path
boundaries follow maintained storage/workflow/thumbnails/lifecycle and UF-used
shared_work.py. Only source/declaration inspection and metadata/hash work ran;
no converter/path/helper/worker/model/configure/compiler/build/test execution.
Other consumers, locale/error/resource details and full compiler/link/runtime
parity remain unverified.

## UF JSON surrogate strings and identity bytes

Shared JSON escape decoding/ASCII formatting follows Python 3.10 policies,
including standalone surrogate retention and escaped surrogate-pair joining.
Primary [CPython _json.c](https://github.com/python/cpython/blob/v3.10.20/Modules/_json.c)
was read in the browser without saving implementation; installed encoder/decoder
and nlohmann serializer declarations were read only. The retained
[Python license](CPython-3.10.20.txt) applies to the existing policy adaptation;
nlohmann headers retain their MIT notice. Prepared identity bytes follow
maintained prepared.py; dashboard formatting follows dashboard.py. No extra
implementation source was copied or downloaded. No parser/formatter/hash/model/
worker/build/test ran; native WTF-8 consumers, errors/resources and complete
compiler/link/runtime parity remain unverified.

## UF evaluation, training, report and preset numeric JSON

JSON file/control-message migration follows maintained datasets, common.py,
test_video.py, check_uf_int16.py, pareto.py and presets.py through existing native
JSON/integer/UTF-8 helpers. Ordered formatting and integer index policies reuse
existing CPython adaptations covered by the retained
[Python license](CPython-3.10.20.txt). Custom .Nf nonfinite spelling was checked
against [CPython 3.10.20 pystrtod.c](https://github.com/python/cpython/blob/v3.10.20/Python/pystrtod.c),
read in the browser without saving source; nlohmann cross-JSON/binary conversion
declarations retain their existing MIT notice. No additional implementation was
copied or downloaded. No parser/dataset/training/report/worker/model/build/test
ran; native bounds, JSON/format/error/resource and compiler/runtime parity remain
unverified.

## UF remote numeric selection and worker JSON

Format filtering, ranking, coercion and publication timestamps follow maintained
streaming.py; existing-info ID comparison follows workflow.py. Mixed numeric
ordering follows existing CPython integer/float policies; date matching follows
installed Python 3.10 _strptime.py and existing Unicode tables. The retained
[Python license](CPython-3.10.20.txt) applies. Worker/event JSON reuses the existing
native formatter/parser and installed nlohmann interfaces with their MIT notice.
No additional implementation source was copied or downloaded. No metadata,
worker, parser, model, build or test ran; representation/error/resource behavior
and compiler/link/full runtime parity remain unverified.

## UF media scalar conversion and probe fallback

Pareto report timing, failed-row construction and loading/completion print
boundaries follow maintained archive/pareto.py and its inherited archive codec
under the existing project notice. Native code reuses declared CodecStatistics,
standard monotonic clocks, JSON and stream interfaces; the CLI delegates path
reporting. No external implementation was copied or downloaded and no new
dependency was added. No search/codec/helper/library/model/build/test execution;
full progress/timing/exception/compiler/runtime parity remains unverified.

Pareto resume file scopes, initial-sample capacity/exhaustion placement and
first-equal threshold representation additionally follow maintained pareto.py
under the existing project notice. Native code reuses owned JSON-file reads,
existing standard algorithms and bounded count interfaces, retaining the
borrowed stream reader. No implementation was copied or downloaded, no new
dependency was added, and no search/file/helper/library/model/build/test ran.
Full file/exception/sampling/compiler/runtime parity remains unverified.
Pareto total-time sum, resumed elapsed lookup, status string comparison and
left-to-right monotonic elapsed arithmetic additionally follow maintained
pareto.py under the existing project notice. Native code reuses the existing
integer/JSON helpers and standard clocks, without copied external code or a new
dependency. No arithmetic/search/helper/library/model/build/test execution;
full clock/type/exception/numeric/compiler/runtime parity remains unverified.

Input reader close/stop/join order follows maintained media.FrameReader,
workflow encode attempts and pareto source loading, plus the UF-used shared
Jetson bridge, under their existing project notices. Native shared ownership,
condition variables, threads and checked POSIX descriptors implement these
boundaries in existing units. Installed GCC declarations were read only;
no implementation was copied or downloaded and no new dependency was added.
No reader/process/worker/helper/library/model/build/test execution. Full
cleanup/timeout/compiler/runtime parity remains unverified.

Shared scalar conversion follows retained CPython integer/float policies with
the existing native arithmetic/charconv helpers. Probe conversion, truth/dict
fallback and Jetson error groups follow maintained media/storage and UF-used
RT jetson_decode.py. Installed Fraction and nlohmann empty/binary declarations
were inspected only; the retained [Python license](CPython-3.10.20.txt) and
existing library notices apply. No additional implementation source was copied
or downloaded. No probe/claim/process/model/build/test ran; conversion/error/
text/resource/fallback and compiler/runtime parity remain unverified.

## UF shared numeric JSON and exact metadata timestamps

Number parsing/formatting follows installed Python 3.10 JSON decoder/scanner
policies using existing native integer/float helpers. Exact source timestamps
follow maintained storage.identity and workflow/thumbnail checks. The retained
[Python license](CPython-3.10.20.txt) applies. Typed binary integers and numeric
token protection use installed nlohmann interfaces, whose installed headers
retain the MIT notice; no additional implementation was copied or
downloaded. Compiler/link, caller conversions, strings/NaN identities/error order/
resources and full runtime parity are unverified; no parser/model/build/test ran.

## UF original identity and cleanup records

The shared exact size/mtime identity and cleanup sequence follow maintained
storage.py, workflow.py and thumbnails.py. Nanosecond arithmetic reuses the
existing native PythonInteger helper. Atomic JSON writing now lives in the
existing JSON module and follows storage.atomic_json's owned-context/replace/
finally order for publication journals and cleanup records. The retained
[Python license](CPython-3.10.20.txt) applies to the existing Python policy
adaptations. Installed POSIX/libc declarations were inspected only; no new
implementation source was copied or downloaded. No helper/deletion/model/build/
test ran; temp-name/buffering/JSON/errors/races and compiler/runtime parity
remain unverified.

## UF flat video manifest suffix matching

The archive suffix parser follows the maintained storage.py fullmatch pattern
and reuses the existing Python 3.10.20 / Unicode 13.0.0 decimal table in
src/python_unicode_data.hpp through the native numeric helper. The retained
[Python license](CPython-3.10.20.txt) and existing table provenance apply. No
additional implementation source was copied or downloaded. The matcher was
not executed; compiler/link, filename/errors and full runtime parity are
unverified.

## UF text JSON file reads and archive validation

The shared readers in src/python_json.cpp follow installed CPython 3.10
pathlib read_text/open context order and JSON text BOM/full-document rules,
with native POSIX descriptor/error handling. Archive set iteration/equality and
manifest suffix handling follow represented Python values and Path policies.
The retained [Python license](CPython-3.10.20.txt) applies. Hashing follows the
maintained UF-used RT helper through existing OpenSSL. No installed implementation
source was copied into these helpers and no new reference was downloaded.
Native JSON representation/error/iteration and compiler/runtime parity remain
unverified; file/hash/path provenance is in PROVENANCE.md.
The pretty formatter also follows installed json.encoder's sorted dictionary,
indent/separator, ASCII-escape and float policies for represented native JSON.
Flat publication/recovery follows the maintained UF storage/workflow sequence;
atomic descriptor output retains the Python context/finally ordering. Remote
wildcard snapshots follow pathlib's directory-list-before-yield and permission
boundaries. The same retained Python notice applies to those policies. Installed
libc/nlohmann declarations were read through their interfaces; implementation
source was not copied or executed. Full formatter/publication/runtime parity
remains unverified.

## UF image header JSON and staging sequence

python_sorted_json in the existing JSON module follows installed json.encoder
default comma/colon spacing, sorted dictionary and ASCII/float policies for
represented native values. Image staging follows maintained thumbnails.py's
source-check/payload-release/publication/finally-cleanup sequence using the shared
path remover and checked native sync. The retained
[Python license](CPython-3.10.20.txt) applies. No installed implementation was
copied or executed during this pass. Stream copying, TemporaryDirectory cleanup,
native JSON/numerical/error/memory and compiler/runtime parity remain unverified.
The descriptor container writer also follows maintained output/data context
order and installed shutil.copyfileobj's Linux 64 KiB binary read size. Sync
precedes reader/writer close; completed image/video copying shares the native
archive module and follows maintained 1 MiB chunks and heartbeat placement.
Direct image decode/structure inspection now reopen the payload at its recorded
offset after chunked verification, following read_image/FrameSource ownership.
Indexed source constructor failure and explicit close now follow the maintained
FrameSource stream-close-before-codec/cache-release boundary. Shared RT gallery
finally and video worker close paths provide the caller ordering. Existing
standard streams and native source/viewer units implement checked closure,
without copied external code or a new dependency. No source/viewer/helper/
decoder/model/build/test execution; full parity is unverified.
Video publication follows maintained link/unlink without additional sync calls.
FFmpeg FPS and frame-duration arguments reuse Python str/repr and filesystem
encoding policies; installed fstream close declarations were read only.
The writer inherits stdout and checks input-pipe closure before waiting,
following maintained subprocess stdin-close/completion order. Explicit finally
cleanup follows workflow.run_decode child poll/kill/wait, stdin/log close and
missing-tolerant stage unlink order. Installed subprocess signal/poll policies
were inspected only; native descriptor helpers and process linkage are reused.
Reservation cleanup and strict publication unlink now propagate failures;
diagnostic retries check interruption. Archive decode now follows maintained
codec/output-before-packet ordering without an extra structure scan. Recorded
width/height/FPS argument text retains Python str policies; exact dimension
conversion remains bounded by the native UInt32 interface. Shared model loading
follows make_codec arithmetic-version rejection for image/video/viewer paths.
Existing archive/media/model units are reused. No helper or process was executed.
The retained Python notice applies. Native buffer/flush/initialization/timestamp/
error/resource/close/signal/race and compiler/runtime parity remain unverified;
the writer/copier was not executed.

## UF completed-artifact disk copy and RAM-stage lifetime

The copier in src/workflow.cpp follows maintained completed_disk_stage and
installed Python binary I/O/context policies for filled reads and reverse
reader/writer cleanup. Reader directory/fstat initialization follows primary
[CPython 3.10.20 FileIO](https://raw.githubusercontent.com/python/cpython/v3.10.20/Modules/_io/fileio.c),
read in the browser without saving a source file. Explicit stage lifetime follows
the maintained workflow and installed shutil source. The retained
[Python license](CPython-3.10.20.txt) applies. Native rmtree race handling, buffered
I/O/flush/errors/allocation/memory and compiler/runtime parity remain unverified.

## UF staging directory removal

python_path.cpp adapts installed shutil.rmtree's descriptor-based directory
snapshot, stat identity, parent-relative deletion and error boundaries. Its
scandir descriptor duplication/rewind/close policy follows primary
[CPython 3.10.20 POSIX source](https://raw.githubusercontent.com/python/cpython/v3.10.20/Modules/posixmodule.c),
read in the browser without saving a source file. The retained
[Python license](CPython-3.10.20.txt) applies. Native owners do not retry released
descriptors after failed close; destructor/resource/name/error/race and compiler/
runtime parity remain unverified. No remover or library was executed.

## UF copy2 metadata policy

The private RAM-stage copier in `src/workflow.cpp` adapts CPython 3.10's
shutil.copy2/copystat/_copyxattr ordering and Linux xattr buffer/error policies.
The retained [Python license](CPython-3.10.20.txt) applies; the source references
that notice. Installed libc/kernel declarations were read and are used through
their interfaces; implementation source was not copied. Native copying,
filesystem/error and compiler/runtime parity remain unverified.

## UF POSIX path utilities

`src/python_path.cpp` and `include/uf/python_path.hpp` adapt CPython 3.10's POSIX
Path construction, absolute/relative/containment, exists/file/directory/symlink/
stat-size and realpath/Path.resolve(strict=False) component, symlink/cache/loop
policies. Video output, shared-work channel IDs, discovery, lifecycle, original
cleanup, artifact checks and filesize reuse those helpers through uf_archive.
The private lifecycle basename traversal also adapts pathlib's directory
snapshots, match/recursion distinction and permission-error boundaries.
The retained [Python license](CPython-3.10.20.txt) applies; the source retains
that notice. Native path representation, filesystem/recursion/error handling
and compiler/runtime parity remain unverified. Python is not invoked.

## UF JSON byte decoding

The existing `include/uf/json_compat.hpp` also adapts maintained UF dictionary
get, string membership/suffix and scalar-key/equality rules. Native exceptions
carry the represented Python type/method distinction; parser, representation,
diagnostic and compiler/runtime parity remain unverified. The retained Python
license applies to this adaptation.

`decode_python_json_bytes` in `src/utf8.cpp` adapts CPython 3.10's
json.detect_encoding and UTF-8/16/32 surrogatepass rules. Installed Python
JSON/codec source and primary
[codec error handling](https://raw.githubusercontent.com/python/cpython/v3.10.20/Python/codecs.c)
were read only; no external source file was downloaded. The retained
[Python license](CPython-3.10.20.txt) applies. Shared JSON now retains lone
surrogates; native representation boundaries and compiler/runtime parity remain unverified.

## UF audio process policy

The private audio runner in `src/media.cpp` follows CPython 3.10's
subprocess.run check/wait and inherited-stream policies through the existing
native child owner. The retained [Python license](CPython-3.10.20.txt) applies.
No Python runtime is embedded or invoked; native exception presentation and
runtime behavior remain unverified.

## UF float text

`src/python_json.cpp` adapts CPython 3.10's float str/repr notation policies:
fixed/scientific thresholds, integer `.0`, exponent padding and lowercase
nonfinite tokens. JSON retains the Python encoder's uppercase special tokens.
The retained [Python license](CPython-3.10.20.txt) applies. Shortest digits use
the C++ standard library; its implementation source was not copied. Source
inspection does not establish formatter, compiler/link or runtime parity.

## UF JSON value comparison

`include/uf/json_compat.hpp` adapts CPython 3.10's boolean/integer/float and
list/dictionary equality policies for the native JSON representation, including
container element identity. The retained [Python license](CPython-3.10.20.txt)
applies. Installed nlohmann declarations were inspected; its implementation
source was not copied. Native numeric limits, parsing/identity/recursion/errors
and compiler/runtime behavior remain unverified.

## UF fraction parsing and conversion

`src/python_fraction.cpp` adapts CPython 3.10.20's `fractions.Fraction` string
grammar/normalization, `numbers.Rational.__float__`, `long_true_divide`
rounding policy and `float.as_integer_ratio` binary decomposition. The retained
[Python license](CPython-3.10.20.txt) applies.
The C++ implementation uses the already linked OpenSSL BIGNUM interfaces and
does not embed or invoke Python. OpenSSL implementation source was not copied.
Source review and installed declaration/export inspection do not prove numeric
results, compiler/link acceptance or runtime behavior.

## UF executable and Deno discovery

`src/process.cpp` adapts CPython 3.10's POSIX shutil.which for shared GStreamer,
yt-dlp and Deno discovery. `src/remote.cpp` adapts expanduser/home and pathlib
component/file-check policies for UF streaming. The changes use typed
C++ strings/paths, POSIX stat/access/confstr and reentrant account lookup, with
native error and buffer handling. The retained
[Python license](CPython-3.10.20.txt) applies. Installed libc declarations were
read; no libc implementation source was copied or executed.

## UF stream helper

`src/bitstream.cpp` and `include/uf/bitstream.hpp` adapt UF's
`src/utils/stream_helper.py`, including its SPS helper, under InterDigital's
Apache-2.0 notice. The native changes add typed wire fields, C++ stream handling
and shared record ownership. The copyright/license comments and
[Apache-2.0 license](Apache-2.0.txt) are retained. Installed C++ library headers
were read for shared-pointer and byte-stream behavior; no standard-library
source was copied.
The file-size helper follows CPython 3.10's pathlib POSIX component/root
normalization and is_file error policy. The retained
[Python license](CPython-3.10.20.txt) is also referenced in the source.
Installed libc stat/type/errno declarations were read; no libc source was
copied into the port.

## Original DCVC family

`src/dcvc2021_model.cpp`, the video layer/estimator and evaluation workflow source and their interfaces
adapt the Microsoft project under its retained MIT license.
`src/dcvc2021_entropy.cpp`, `src/dcvc2021_stream.cpp` and the masked
convolution also adapt the original family's InterDigital CDF/rANS interface,
stream helper and layer code. Their Apache-2.0 copyright notices are retained
in the source and [Apache-2.0 license](Apache-2.0.txt).
The image model, entropy, GDN, residual, reparametrizer and lower-bound sources
in `dcvc2021_image_*.cpp` and their interfaces also adapt the family's embedded
InterDigital CompressAI code under that retained Apache-2.0 notice.

The underlying 64-bit rANS arithmetic is Fabian Giesen's public-domain/CC0
implementation. Its [retained notice](RygRans.txt) was copied from the separate
`ryg-rans-legacy` checkout at the revision specified by the Python project.
No Python headers or bindings are used by the native library.

The TCM video layers, model, video entropy and evaluation source also adapt
Microsoft MIT code. The TCM image layers, priors/entropy model, stream wrapper
and buffered entropy interface adapt InterDigital Apache-2.0 source and reuse
the same pinned rANS arithmetic. Original source hashes are recorded in
`ryg-rans-legacy/dcvc_tcm_sources.json`.

HEM model/layer/evaluation source also adapts Microsoft MIT code. Its stream
helpers and buffered entropy extension adapt InterDigital Apache-2.0 source
and reuse the pinned rANS arithmetic. HEM shares the retained pytorch-msssim
0.2.0 metric notice and installed PyTorch distribution references.
Original/native source hashes are in `ryg-rans-legacy/dcvc_hem_sources.json`.

HEM's log interpolation adapts NumPy's BSD-3-Clause linspace order.
The installed NumPy 1.26.1 notice is retained in [NumPy.txt](NumPy.txt);
local source/hash receipts are in `numpy-reference/dcvc-hem`.
No NumPy code is imported or executed by the native port.

DC model, common-prior, video network and entropy partitioning source adapts
Microsoft MIT code. DC layer/stream helpers and rANS interfaces adapt the
InterDigital Apache-2.0 source and share the pinned rANS arithmetic.
Fine-scale interpolation follows the retained NumPy BSD notice.
Original/native hashes are in `ryg-rans-legacy/dcvc_dc_sources.json`.

## SciPy Cephes quantile and spline interpolation

`src/dcvc2021_ndtri.hpp` adapts the inverse-normal and polynomial routines from
SciPy 1.15.3's Cephes source. The retained SciPy BSD license and Stephen Moshier
notice are in [SciPy.txt](SciPy.txt). Downloaded source URL/hash receipts and
installed version/license metadata are under
`/path/to/dependencies/scipy-reference`. No SciPy runtime is used by this code.
`src/dcvc_dc_splines.cpp` and the zoom traversal in `dcvc_dc_transforms.cpp`
adapt SciPy 1.15.3 ndimage source, including Peter J. Verveer's retained BSD
notice. Those source files and URL/hash receipts are retained under
`scipy-reference/dcvc-dc`. DC reductions also adapt NumPy 1.26.1 under
`NumPy.txt`, with references under `numpy-reference/dcvc-dc`.

## PyTorch-MSSSIM metric

`src/dcvc2021_evaluation_metrics.cpp` adapts the default RGB MS-SSIM path from
pytorch-msssim 0.2.1 by Gongfan Fang, Zhejiang University. Its retained MIT
notice is [PyTorch-MSSSIM.txt](PyTorch-MSSSIM.txt). The published source archive,
copied metric source, 1.0.0 comparison source, release metadata and URL/hash
receipts are under `/path/to/dependencies/pytorch-msssim-reference`. The source
was read without installing, importing or executing the package. The upstream
license copy comes from the same project's official 1.0.0 tag.
`src/dcvc_tcm_msssim.cpp` follows TCM/HEM/DC's pinned pytorch-msssim 0.2.0 source,
including its different separable convolution order. That published archive,
copied source and `SOURCE-0.2.0.json` URL/hash receipt are retained in the same
separate reference folder under the retained MIT notice.

## Matplotlib axes, layout, text and legends

`src/rd_axis.hpp` adapts installed Matplotlib 3.10.9's linear nonsingular
limits, AutoLocator/MaxNLocator default staircase, scale/offset selection,
edge tolerance and visible-tick filtering into C++ standard-library helpers.
Default ScalarFormatter offset/scientific/precision selection, zero suppression
and annotation strings are also adapted. Precision refinement reuses NumPy's
1.26.1 power table/repeated multiplication and rint/divide boundary under the
retained [NumPy license](NumPy.txt); its primary `calculation.c` source was read.
`src/rd_plot.cpp` supplies native Cairo drawing with default separate subplot
fractions, title padding, major marks and Unicode minus signs.
`src/rd_layout.hpp` adapts default GridSpec positions and one-row tight_layout
margin/gap calculation, default rect edge complements, title-width collapse
and failed-adjustment fallback.
The Cairo drawing source measures initial decoration bounds at 100 dpi and
recomputes final tick/text placements at 300 dpi. Plain-text placement adapts
"lp" minimum ascent/descent, logical widths, top/center-baseline alignment and
1.2 multiline spacing; glyph metrics/advances are supplied by Cairo. The UF
one-column legend adapts TextArea baseline adjustment, default handle/row/border
packing and lower-right anchoring, including empty frames. Default rounded
frame quadratics are converted to Cairo cubic paths with a gray border.
Drawing adapts default artist zorders, stable method order, per-axes background,
tick grid/mark/label order and clipped grids/collections/lines, with separate
projecting-capped spine segments. Scatter markers retain Path.unit_circle's
eight cubic Bezier control sets and default size. Empty legend diagnostics use
the installed message and native default repetition per UF call site; custom
Python warning filters/source context are not implemented.
Python objects, NumPy arrays, generalized Matplotlib renderer/artist/packer
classes and runtime imports are removed. Only the configuration used by the
maintained UF comparison script is implemented.
Its setup/report boundaries additionally follow compare_bd_rate.py and
common.create_folder under the existing project MIT notice: TXT redirection,
pair/anchor checks, directory exists handling, complete-line formatting and
checked messages/close. Existing C++ strings, streams, filesystem and parser
interfaces implement these changes. No external code was copied or downloaded;
no new dependency or comparison/plot/helper/model/build/test execution.
The positive-divisor float-divmod helper follows the retained CPython source.
Character conversion, scalar promotion/diagnostics, degenerate layout, mathtext,
font/fallback/hinting, snapping and full renderer parity remain
incomplete/unverified; no helper or plot
has been executed. Python-float decimal rounding uses a C++ fixed-decimal
format/parse boundary; installed NumPy scalar-round/division/dtype-discovery
symbols were inspected statically to select the native rounding path.

Copyright (c) 2012- Matplotlib Development Team; All Rights Reserved.
The installed distribution's complete, unchanged [license](Matplotlib.txt) is
retained. Sources were read from installed `ticker.py`, `transforms.py`,
`axis.py`, `axes/_base.py`, `_tight_layout.py`, `layout_engine.py`, `gridspec.py`,
`spines.py`, `text.py`, `legend.py`, `legend_handler.py`, `offsetbox.py`,
`patches.py`, `path.py`, `markers.py`, `collections.py`, `lines.py`,
`axes/_axes.py`, `backend_bases.py`, `backends/backend_agg.py`, `backends/backend_cairo.py` and
`mpl-data/matplotlibrc`; source/hash provenance is
recorded in `PROVENANCE.md`.

## CPython algorithms

`src/pareto_random.cpp` and `src/pareto_random.hpp` adapt CPython 3.10.20's
integer-seeded MT19937 core, `random`, `getrandbits`, rejection sampling,
shuffle, cached Gaussian distribution, SipHash-2-4, hash-secret initialization
and the string sets used for mutation parameter names and comparison sequences.

`src/training_loader.cpp` reuses the adapted generator and rejection sampling
for independent worker augmentation streams seeded with `base_seed + worker_id`.
This does not link or invoke CPython.

`src/training_checkpoint.cpp` follows the binary64 POSIX stat-time conversion
used by `os.path.getmtime` and the unlink dispatch used by `os.remove`.
The official `v3.10.20` `Modules/posixmodule.c` was read through the browser;
it was not downloaded into a reference folder or used as runtime code.
Installed `genericpath.py` and native POSIX declarations were read only.
The native source uses its existing system C library and the retained Python
license; source provenance is recorded in `PROVENANCE.md`.

`src/evaluation_workers.cpp` follows the installed Python 3.10 process
executor's policy of retaining workers after call exceptions, ordered Future
consumption in the UF script and draining submitted calls during shutdown.
The native implementation uses its existing Unix socket/process transport;
Python Futures, exception objects, queues and interpreter code are not used.
The scheduler also adapts one-process-per-submission adjustment and the
completed-call semaphore, including background replies while the UF CLI
continues enumeration. Initialization documents are lazy; enumeration-error
unwinding drains submitted calls. Native mutex/thread/eventfd transport owns
these states, with no Python runtime dependency. Evaluation setup now follows
maintained test_video.main availability/threshold/mask/worker/config/GPU-count/
executor/QP/report/root/class order, using owned options moved at dispatch.
Existing spacing/list-length checks replace the extra eager rate guard.
Usage handling is now limited to parser errors; typed-width conversion follows
CUDA preflight and raw CLI storage is released before evaluation. Shared helpers
retain one availability check per entry. Overall elapsed time follows maintained
wall-clock sampling before parsing, using the existing standard chrono interface.
Per-point worker timing follows the same clock policy. Maintained evaluator and
raw reader/writer source were read for explicit close and phase ordering. Native
JSON output uses the shared declared report writer on both per-point and
aggregate paths; bitstream file bodies keep private checked-close handling.
Rate-point metadata/override order additionally follows maintained main: fields
remain required and overwritten numeric values are not converted. Native
dimension/count bounds and full error ordering remain unverified. Output transfers
in bounded blocks and scoped stream swaps release retained storage. Installed
GCC C++17 stringstream/string declarations were read only, with no implementation
copied and no new unit, dependency, download or install. Memory/file/error/clock/
compiler/runtime fidelity remains unverified; no helper, model or library ran.
Process behavior is unverified.
The timing wrapper follows maintained test_compress_time.py's ignored os.system
status, report collection/sum ordering and separate encoding/decoding print
boundaries. It uses the existing native process transport, scalar integer/JSON
helpers and OpenSSL rational implementation, with no Python numeric runtime.
No new dependency/source/package/download or install; process, numeric, error,
formatting and compiler/runtime behavior remain unverified. No helper or model ran.
Installed `concurrent/futures/process.py`, `_base.py` and `threading.py` were
read only; their hashes are recorded in `PROVENANCE.md` under the
retained Python license. No process scheduler or model was executed.

The adaptation removes Python objects, headers and bindings, owns the state
in C++ classes, retains arbitrary-size signed decimal Pareto seeds,
uses OpenSSL for a shared randomized process hash secret, and handles the fixed
five ASCII mutation names in a native table. Comparison sets additionally
decode valid UTF-8 into native-endian 1/2/4-byte Unicode hash storage, insert
incrementally, remove duplicates and retain resize/iteration order.
CPython 3.10 on the reference 64-bit
Conda installation uses SipHash-2-4 with the small-string cutoff disabled.

`src/python_integer.cpp` and `src/python_numeric.cpp` implement the decimal
seed boundary and the reference numeric string rules. The native Unicode data
in `src/python_unicode_data.hpp` was exported from the installed CPython 3.10.20
Unicode 13 database. The generator and provenance remain in the separate
`cpython-reference` folder. Runtime conversion requires no Python headers,
bindings or interpreter. Decimal input follows the 4300-digit default and the
`PYTHONINTMAXSTRDIGITS` environment policy; native training seed APIs remain int64.
The integer helper also supplies native signed decimal addition/multiplication
for comparison count/statistic arithmetic. Its integer-to-binary64 conversion
uses leading bits, a guard/sticky remainder and ties-to-even rounding, retaining
the `_PyLong_Frexp`/`PyLong_AsDouble` conversion boundary without Python objects.
Arithmetic results bypass the decimal-input digit limit, as in Python.

Retained upstream licenses:

- [Python license](CPython-3.10.20.txt).
- [MT19937 copyright, conditions and disclaimer](MT19937.txt).
- [SipHash implementation's MIT license](SipHash.txt).

Source references are under `/path/to/dependencies/cpython-reference`.

Upstream files at tag `v3.10.20`:

- [Modules/_randommodule.c](https://github.com/python/cpython/blob/v3.10.20/Modules/_randommodule.c).
- [Lib/random.py](https://github.com/python/cpython/blob/v3.10.20/Lib/random.py).
- [Objects/setobject.c](https://github.com/python/cpython/blob/v3.10.20/Objects/setobject.c).
- [Python/pyhash.c](https://github.com/python/cpython/blob/v3.10.20/Python/pyhash.c).
- [Python/bootstrap_hash.c](https://github.com/python/cpython/blob/v3.10.20/Python/bootstrap_hash.c).
- [Objects/unicodeobject.c](https://github.com/python/cpython/blob/v3.10.20/Objects/unicodeobject.c).
- [Objects/longobject.c](https://github.com/python/cpython/blob/v3.10.20/Objects/longobject.c).
- [Objects/floatobject.c](https://github.com/python/cpython/blob/v3.10.20/Objects/floatobject.c).
- [Python/initconfig.c](https://github.com/python/cpython/blob/v3.10.20/Python/initconfig.c).
- [Python/preconfig.c](https://github.com/python/cpython/blob/v3.10.20/Python/preconfig.c).

## Pillow image ingestion

The PNG text and EXIF metadata handling in `src/image_pixels.cpp` follows the
installed Pillow 12.1.1 `PngImagePlugin.py`, `Image.py`, `ImageOps.py` and
`ImageFile.py` sources. It uses native C++/zlib and does not invoke Pillow.
BMP/DIB source uses the same installation's `BmpImagePlugin.py`, plus the
Pillow 12.1.1 `Unpack.c`, `RawDecode.c`, `Palette.c` and `_imaging.c`
references. It adapts header/mask/palette selection and RLE stream behavior.
Downloaded source provenance is in the reference folder's `bmp_sources.json`.
APNG first-frame selection and canvas placement use the same PNG plugin
reference and Pillow 12.1.1 `Storage.c`. Its downloaded reference provenance
is in `apng_sources.json`; the native implementation uses existing libpng.
The upstream copyright and permission notice is retained in
[Pillow.txt](Pillow.txt). Local source copies and original paths/SHA-256 hashes
are under `/path/to/dependencies/pillow-reference/installed`.

## PyTorch algorithms and checkpoint formats

Default model loading in `src/training_common.cpp` follows the installed
`torch/nn/modules/module.py`: local parameters/buffers before children,
historical scalar selection, in-place copies and retained compatible copies
before accumulated errors. The native tensor-only UF interface removes Python
objects, hooks/metadata, assign/swap paths and diagnostic formatting.
The installed source and LibTorch Module declarations were read only; hashes
are recorded in `PROVENANCE.md` under the retained PyTorch BSD notice.

`src/training_adamw.cpp` adapts the installed PyTorch
2.5.0a0+872d972e41.nv24.08 `torch/optim/adamw.py` update algorithms and
`torch/optim/optimizer.py` implementation selection, grad-mode and state rules.
The native source uses existing LibTorch/ATen operators without invoking Python.
Optimizer restoration/packing in `src/training_checkpoint.cpp` also follows
those installed optimizer/AdamW sources: recursive state tensor policy,
retained unmapped state, repeated-ID mapping, positional IDs and scalar counter
normalization after state/group commitment. Python objects/hooks/custom
iterables and generalized deepcopy/key semantics are outside this adaptation.
Sources and native IValue/Dict/Tensor declarations were read only; the current
path/hash receipt is in `PROVENANCE.md`. No optimizer was executed.
`src/checkpoint_storage_copy.hpp` and the optimizer group copier also follow
installed `_tensor.py`, `storage.py` and CPython `copy.py`: plain container/
tensor identity memos, full untyped-storage copying with retained view geometry,
leaf/grad handling and math-bit materialization. Model export follows Module's
detached state views. Writer normalization keeps repeated objects/shared
strided storage and rejects cycles unsupported by JIT Pickler. Installed native
declarations and upstream v2.5.0 IValue/Pickler source were read without saving
new browser source files; the upstream tag is a comparison rather than proof
of the installed NVIDIA fork. Python classes/hooks/attributes and generalized
backend behavior are outside the adaptation; runtime alias parity is unverified.
The plain container deepcopy rules retain the CPython notice in
[CPython-3.10.20.txt](CPython-3.10.20.txt).
Writer source storage/dtype checks in `src/torch_pickle.cpp` follow installed
`serialization.py` under the retained PyTorch BSD notice. The native descriptor
writer handles direct checkpoint/status and staged prepared output. Explicit finalization on error
was compared with official v2.5.0 `export.cpp`/`inline_container.cc`; those browser
sources were not saved to the filesystem. Installed declarations, selected
writer exports and AArch64 destructor/unwind bytes were read only. Source/header/
binary hash provenance is in `PROVENANCE.md`; no library or writer
was loaded or executed, and complete failure/interoperability parity is unverified.
The original DCVC entropy source also follows the installed
`torch/distributions/laplace.py` CDF expression. Its local reference and
path/hash provenance are under `pytorch-reference/dcvc2021`.
The image modules' device/dtype conversion also follows the installed
`torch/nn/modules/module.py`; its local copy and receipt are in that folder.
The native checkpoint reader follows installed `serialization.py`, `storage.py`,
`_utils.py` and `_tensor.py`, with the matching native headers and upstream
v2.5.0 unpickler/raw-storage source comparisons. These references and their
path/URL/hash receipt are in `pytorch-reference/serialization`. The v2.5.0
comparison tag is not claimed to be the installed NVIDIA fork. Its local
CPython pickle source copy uses the retained CPython notice.

The installed PyTorch/Caffe2 copyright notices, BSD conditions and disclaimer
are retained in [PyTorch.txt](PyTorch.txt). Local source copies, the complete
installed license, version metadata, original paths and SHA-256 hashes are in
`/path/to/dependencies/pytorch-reference`. No additional package was downloaded,
installed, imported or executed to obtain these references.

Comparison positivity gates reuse the native Python JSON ordering helper.
BD trapezoidal integration now uses the retained NumPy pairwise-sum adaptation
with its +0 reduction identity. Installed bd_metric, NumPy trapz/polyfit/vander
and retained NumPy arithmetic-loop sources were read only; the native polynomial
inputs now retain the source +0.0 boundary. These source changes do not establish
numerical or compiler/runtime parity. Existing Microsoft, NumPy, CPython and
bd-metric notices remain; no package, model, build or test was executed.

The UF streaming BufferPolicy adaptation now preserves a shared critical
section for claim/capacity/observe/release, using a native process-shared
recursive pthread mutex rather than independent field atomics. The maintained
stream_pipeline.py/workflow.py and installed multiprocessing/synchronize.py
RLock source were read only. Installed pthread declarations and existing Threads
linkage were inspected; no Python mutex object or runtime dependency is used.
Progress record parsing now retains complete lines and the source conversion/
finished-marker order. Native lock/thread/process/memory/ABI/compiler and runtime
parity remain unverified; no worker, model, library, build or test was executed.
