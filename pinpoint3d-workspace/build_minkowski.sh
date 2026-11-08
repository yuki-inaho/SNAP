#!/usr/bin/env bash
# Build MinkowskiEngine (NVIDIA official repo, commit 02fc608) for PinPoint3D.
# Isolated from the SNAP pixi env: torch 2.3.1 + CUDA 11.8, gcc 11, sm75 only.
#
# BLAS comes from the conda env ($CONDA_PREFIX/include|cblas.h) and the compiler
# uses the conda gcc sysroot: mixing /usr/include/x86_64-linux-gnu into a CUDA
# 11.8 nvcc build pulls in glibc 2.39 math headers, which nvcc 11.8 cannot parse.
set -euo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFIX="$WS/env/.pixi/envs/default"
ME_DIR="$WS/MinkowskiEngine"
export PATH="$PREFIX/bin:$PATH"

cd "$ME_DIR"

export CUDA_HOME="$PREFIX"
export CC=x86_64-conda-linux-gnu-gcc
export CXX=x86_64-conda-linux-gnu-g++
export MAX_JOBS=${MAX_JOBS:-6}
export TORCH_CUDA_ARCH_LIST=7.5

# config-settings are not forwarded to setup.py's custom --blas_* options, so
# invoke the official setup.py directly, as in the upstream README.
python setup.py bdist_wheel --blas=openblas \
  --blas_include_dirs="$PREFIX/include" \
  --blas_library_dirs="$PREFIX/lib"
python -m pip install --no-deps --force-reinstall dist/*.whl
