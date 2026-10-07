#!/usr/bin/env python3
"""Audit tracked publication content without printing credential values."""
import fnmatch
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote

root = Path(__file__).resolve().parents[1]
names = subprocess.check_output(['git', 'ls-files', '-z'], cwd=root).decode().split('\0')
names = [name for name in names if name]
errors = []
forbidden_dirs = ('build/', '.local/', '.release-private/', 'models/', 'checkpoints/',
                  'data/', 'datasets/', 'archives/', 'restored/', 'runs/', 'reports/', 'benchmarks/')
forbidden_patterns = ('*.pt', '*.pth', '*.pth.tar', '*.safetensors', '*.onnx', '*.ckpt', '*.bin',
                      '*.uf.json', '*.dcvci', '*.mp4', '*.mkv', '*.webm', '*.yuv', '*.opus',
                      '*.m4a', '*.wav', '*.flac', '*.png', '*.jpg', '*.jpeg', '*.webp', '*.tif',
                      '*.tiff', '*.avif', '*.csv', '*.info.json', '*.log', '*.o', '*.a', '*.so', '*.so.*', '*.pyc',
                      '*cookies*.txt*', '*.pem', '*.key', '.env', '.env.*')
patterns = {
    'private machine path': re.compile(r'/(?:home/[A-Za-z0-9_-]+|mnt/to_storage)(?:/|\b)'),
    'private key': re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    'GitHub token': re.compile(r'gh[pousr]_[A-Za-z0-9]{30,}'),
    'GitHub fine-grained token': re.compile(r'github_pat_[A-Za-z0-9_]{40,}'),
    'Hugging Face token': re.compile(r'hf_[A-Za-z0-9]{30,}'),
    'AWS access key': re.compile(r'AKIA[0-9A-Z]{16}'),
}
required = ('README.md', 'LICENSE.txt', 'NOTICE.txt', 'PROVENANCE.md', 'CMakeLists.txt',
            'TESTING.md', 'tests/int16_cross_language.py', 'scripts/build_one_core.sh')
for name in required:
    if name not in names:
        errors.append(f'Required public file not tracked: {name}')
for name in names:
    path = root / name
    if name.startswith(forbidden_dirs) or any(fnmatch.fnmatch(path.name.lower(), pat) for pat in forbidden_patterns):
        errors.append(f'Generated/private artifact tracked: {name}')
    if name in ('UF_SOURCE_INVENTORY.json', 'SOURCE_COVERAGE.md', 'CMakeUserPresets.json'):
        errors.append(f'Local development receipt/config tracked: {name}')
    if path.is_symlink():
        errors.append(f'Symlink requires explicit publication review: {name}')
    if path.stat().st_size > 10 * 1024 * 1024:
        errors.append(f'Unexpected large source file: {name}')
    try:
        content = path.read_text()
    except UnicodeDecodeError:
        errors.append(f'Unexpected binary content: {name}')
        continue
    for label, pattern in patterns.items():
        if pattern.search(content):
            errors.append(f'{label} detected in {name}; value withheld')
    if path.suffix == '.md':
        for target in re.findall(r'\]\(([^)]+)\)', content):
            target = target.strip('<>').split('#', 1)[0]
            if not target or '://' in target or target.startswith('mailto:'):
                continue
            destination = (path.parent / unquote(target)).resolve()
            if not destination.exists():
                errors.append(f'Missing local documentation link in {name}: {target}')
            elif not destination.is_relative_to(root) or str(destination.relative_to(root)) not in names:
                errors.append(f'Documentation links to unpublished file in {name}: {target}')
if errors:
    print('\n'.join(errors), file=sys.stderr)
    sys.exit(1)
print(f'PASS: {len(names)} tracked text/source files; required notices, no detected credentials/private paths, no model/media/build artifacts, local documentation links resolve')
