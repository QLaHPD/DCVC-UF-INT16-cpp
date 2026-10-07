#!/usr/bin/env python3
"""Sequential, one-core archive interoperability test and FPS measurements."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

import argparse
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--binary', type=Path, required=True)
parser.add_argument('--reference-root', type=Path, required=True)
parser.add_argument('--reference-python', type=Path, required=True)
parser.add_argument('--model-i', type=Path, required=True)
parser.add_argument('--model-p', type=Path, required=True)
parser.add_argument('--prepared-i', type=Path, required=True)
parser.add_argument('--prepared-p', type=Path, required=True)
parser.add_argument('--output-dir', type=Path)
parser.add_argument('--cpu', type=int, default=0)
parser.add_argument('--state-parity', action='store_true', help='Also compare native/Python seeded states and CPU/CUDA parity')
args = parser.parse_args()
ROOT = Path(__file__).resolve().parents[1]
REFERENCE = args.reference_root.resolve()
for name in ('binary', 'reference_python', 'model_i', 'model_p', 'prepared_i', 'prepared_p'):
    path = getattr(args, name).resolve()
    if not path.is_file():
        parser.error(f'{name} does not exist: {path}')
    setattr(args, name, path)
if not (REFERENCE / 'main.py').is_file():
    parser.error('reference-root must contain the UF Python main.py')
os.sched_setaffinity(0, {args.cpu})
if args.output_dir:
    OUT = args.output_dir.resolve()
    if OUT.exists():
        parser.error('output-dir must not already exist; generated files are preserved')
    OUT.mkdir(parents=True)
else:
    (ROOT / 'build').mkdir(exist_ok=True)
    OUT = Path(tempfile.mkdtemp(prefix='cross-language-', dir=ROOT / 'build'))
print(f'Artifacts: {OUT}', flush=True)
env = os.environ.copy()
for key in list(env):
    if key in ('CUDA_LAUNCH_BLOCKING', 'PYTHONPATH', 'PYTHONHOME', 'DCVC_USE_INT16') or key.startswith('DCVC_INT16_'):
        env.pop(key, None)
env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', MAX_JOBS='1', PYTHONNOUSERSITE='1')
cache = REFERENCE / '.local/cache'
for key, directory in [('TORCH_EXTENSIONS_DIR', cache / 'torch_extensions'), ('CUDA_CACHE_PATH', cache / 'cuda'), ('TRITON_CACHE_DIR', cache / 'triton')]:
    directory.mkdir(parents=True, exist_ok=True)
    env[key] = str(directory)
cpp = [str(args.binary)]
python = [str(args.reference_python), str(REFERENCE / 'main.py')]
models = ['--model_path_i', str(args.model_i), '--model_path_p', str(args.model_p),
          '--prepared_i', str(args.prepared_i), '--prepared_p', str(args.prepared_p)]
results = {'artifact_directory': str(OUT), 'cpu_affinity': sorted(os.sched_getaffinity(0)),
           'cuda_launch_blocking': False, 'frames': 120, 'variant': 'htl', 'runtime': 'int16',
           'resolution': [626, 352], 'source_fps': 1, 'qp_i': 0, 'qp_p': 20,
           'reset_interval': 8, 'intra_period': -1, 'skip_threshold': 0.2,
           'audio': 'Opus mono 6k, 60ms', 'commands': {}, 'cpp_encodes': [], 'decodes': []}

def save():
    (OUT / 'report.json').write_text(json.dumps(results, indent=2) + '\n')

def run(name, command):
    print(f'Starting {name}', flush=True)
    started = time.perf_counter()
    with (OUT / f'{name}.log').open('w') as log:
        status = subprocess.run(command, cwd=REFERENCE if command[:1] == python[:1] else ROOT,
                                env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    wall = time.perf_counter() - started
    results['commands'][name] = {'argv': command, 'exit_code': status, 'wall_seconds': wall}
    save()
    assert status == 0, f'{name} failed: see {OUT / (name + ".log")}'
    print(f'Finished {name}: {wall:.3f}s command wall time', flush=True)
    return wall

def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

if args.state_parity:
    native_report = OUT / 'native-state.json'
    python_report = OUT / 'python-state.json'
    common = ['--variant', 'htl', '--size', '32', '--cuda_idx', '0', '--cpu-reference',
              '--prepared_i', str(args.prepared_i), '--prepared_p', str(args.prepared_p)]
    run('native-state-parity', cpp + ['check-int16'] + common + ['--output', str(native_report)])
    run('python-state-parity', [str(args.reference_python), str(REFERENCE / 'scripts/check_uf_int16.py')]
        + common + ['--compare', str(native_report), '--output', str(python_report)])
    results['seeded_state_and_cpu_cuda_parity_pass'] = True
    save()

source = OUT / 'input/channel/fixture.mkv'
source.parent.mkdir(parents=True, exist_ok=True)
if not source.exists():
    run('generate', ['ffmpeg', '-v', 'error', '-nostdin', '-f', 'lavfi', '-i', 'testsrc2=size=640x360:rate=1',
                     '-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=16000', '-t', '120',
                     '-threads', '1', '-filter_threads', '1', '-c:v', 'ffv1', '-c:a', 'pcm_s16le', str(source)])
settings = ['--base_root', str(OUT / 'input'), '--channel_ids', 'channel', '--runtime', 'int16',
            '--model_structure', 'htl', '--resolution', '352', '--fps', '1', '--procs', '1', '--cuda_idx', '0',
            '--audio', 'opus', '--opus_channels', 'mono', '--opus_bitrate', '6k', '--opus_frame_ms', '60',
            '--qp_i', '0', '--qp_p', '20', '--reset_interval', '8', '--intra_period', '-1',
            '--skip_thres', '0.2', '--ui', 'plain', '--keep-originals']
manifests = {}
for encoder, prefix, count in [('cpp', cpp, 3), ('python', python, 1)]:
    for repeat in range(1, count + 1):
        name = f'{encoder}-encode-{repeat}'
        output = OUT / name
        wall = run(name, prefix + ['encode'] + settings + ['--output_root', str(output)] + models)
        found = list(output.rglob('*.uf.json'))
        assert len(found) == 1, found
        manifest = found[0]
        data = json.loads(manifest.read_text())
        assert data['video']['frames'] == 120, data['video']
        assert (data['video']['width'], data['video']['height']) == (626, 352), data['video']
        video_path = manifest.parent / data['files']['video.bin']
        record = {'repeat': repeat, 'manifest': str(manifest), **data['video'],
                  'fps': 120 / data['video']['encode_seconds'], 'command_wall_seconds': wall,
                  'command_fps': 120 / wall, 'bitstream_sha256': sha(video_path)}
        if encoder == 'cpp':
            results['cpp_encodes'].append(record)
        else:
            results['python_encode'] = record
        if repeat == 1:
            manifests[encoder] = manifest
        save()

hashes = {row['bitstream_sha256'] for row in results['cpp_encodes']}
hashes.add(results['python_encode']['bitstream_sha256'])
assert len(hashes) == 1, 'C++ and Python compressed video bytes differ'
results['all_encoded_video_bitstreams_identical'] = True
save()
for encoder, manifest in manifests.items():
    run(f'{encoder}-archive-structure', cpp + ['check-archive', str(manifest)])
    for decoder, prefix in [('cpp', cpp), ('python', python)]:
        name = f'{encoder}-to-{decoder}-decode'
        target = OUT / f'{name}.yuv'
        wall = run(name, prefix + ['decode', '--input', str(manifest), '--output_file', str(target),
                                   '--cuda_idx', '0'] + models)
        events = []
        for line in (OUT / f'{name}.log').read_text().splitlines():
            try:
                item = json.loads(line)
                if item.get('event') == 'decode_completed':
                    events.append(item)
            except (ValueError, AttributeError):
                pass
        assert len(events) == 1, events
        event = events[0]
        assert event['decoded_frames'] == 120, event
        assert target.stat().st_size == 120 * 626 * 352 * 3 // 2, target.stat().st_size
        digest = sha(target)
        assert digest == event['decoded_yuv_sha256'], 'Written pixels differ from decoder hash'
        results['decodes'].append({'encoder': encoder, 'decoder': decoder, **event,
                                  'fps': 120 / event['decode_seconds'], 'raw_yuv_sha256': digest,
                                  'command_wall_seconds': wall, 'command_fps': 120 / wall})
        save()
assert len({row['raw_yuv_sha256'] for row in results['decodes']}) == 1, 'Cross-language reconstructed pixels differ'
results['both_cross_language_directions_pass'] = True
results['all_four_decoded_outputs_byte_identical'] = True
results['cpp_encode_weighted_fps'] = 360 / sum(row['encode_seconds'] for row in results['cpp_encodes'])
results['cpp_decode_weighted_fps'] = 240 / sum(row['decode_seconds'] for row in results['decodes'] if row['decoder'] == 'cpp')
results['source_retained'] = source.exists()
assert results['cpu_affinity'] == [args.cpu] and results['source_retained']
save()
print('PASS: both cross-language directions, identical encoded video bytes and all four reconstructed raw videos', flush=True)
print(f'C++ encode {results["cpp_encode_weighted_fps"]:.3f} FPS; C++ decode {results["cpp_decode_weighted_fps"]:.3f} FPS', flush=True)
