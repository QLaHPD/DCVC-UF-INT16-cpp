#!/usr/bin/env python3
"""CLI smoke checks: no model/checkpoint operations, sequential child processes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--binary", type=Path, default=root / "build/uf/dcvc-uf")
parser.add_argument("--report", type=Path, default=root / "build/cli-smoke-results.json")
configuration = parser.parse_args()
binary = configuration.binary.resolve()
commands = ["check-setup", "check-int16", "encode", "stream", "decode", "verify", "view", "cleanup", "manage", "prepare-int16", "pareto-search", "encode-raw", "encode-image", "train-image", "train-video", "test-video", "test-compress-time", "compare-bd-rate"]
cases = [("global help", ["--help"], 0, "Usage:"), ("no arguments", [], 2, "Usage:"), ("unknown command", ["not-a-uf-command"], 2, "Usage:")]
cases += [(name + " help", [name, "--help"], 0, "Options:") for name in commands]
cases += [("missing encode output", ["encode"], 2, "output_root"), ("bad choice before help", ["encode", "--runtime", "invalid", "--help"], 2, "error"), ("missing training arguments", ["train-image"], 2, "required"), ("invalid integer before help", ["encode", "--procs", "bad", "--help"], 2, "integer"), ("setup readiness", ["check-setup", "--require-native", "--require-int16-native"], 0, "UF INT16 CUDA: OK")]
results = []
environment = dict(os.environ)
for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    environment[key] = "1"
for name, args, expected, text in cases:
    started = time.monotonic()
    try:
        result = subprocess.run([str(binary), *args], cwd=root, env=environment, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=40)
        combined = result.stdout + result.stderr
        # "error" is a marker for a diagnostic; root diagnostics use dcvc-uf.
        marker = (text in combined) or (text == "error" and "dcvc-uf:" in combined)
        ok = result.returncode == expected and marker
        row = dict(name=name, passed=ok, returncode=result.returncode, expected=expected, elapsed=time.monotonic()-started)
        if not ok:
            row.update(stdout=result.stdout[-6000:], stderr=result.stderr[-6000:])
    except Exception as error:
        row = dict(name=name, passed=False, error=str(error), elapsed=time.monotonic()-started)
    results.append(row)
    print(("PASS " if row["passed"] else "FAIL ") + name, flush=True)
    if not row["passed"]:
        print(json.dumps(row, ensure_ascii=False), flush=True)
configuration.report.parent.mkdir(parents=True, exist_ok=True)
configuration.report.write_text(json.dumps(results, indent=2) + "\n")
print(f"{sum(x['passed'] for x in results)}/{len(results)} CLI checks passed", flush=True)
sys.exit(0 if all(x["passed"] for x in results) else 1)
