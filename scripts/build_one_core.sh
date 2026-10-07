#!/usr/bin/env bash
# Configure/build one core; forward arguments to the UF CMake preset.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
export CMAKE_BUILD_PARALLEL_LEVEL=1 MAKEFLAGS=-j1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
taskset -c "${UF_CPU_CORE:-0}" cmake --preset uf '-DCMAKE_CUDA_FLAGS=--threads=1' "$@"
taskset -c "${UF_CPU_CORE:-0}" cmake --build --preset uf --parallel 1
