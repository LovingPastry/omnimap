#!/usr/bin/env bash
set -euo pipefail
WORK_DIR="/home/fuyx/lanzc/omnimap"
ENV_PY="/home/fuyx/anaconda3/envs/FisherField/bin/python"
PKG_DIR="/home/fuyx/lanzc/omnimap/thirdparty/modified-diff-gaussian-rasterization"
SITE_PACKAGES="/home/fuyx/anaconda3/envs/FisherField/lib/python3.11/site-packages"

"$ENV_PY" -m pip uninstall -y diff-gaussian-rasterization diff_gaussian_rasterization || true
rm -f "$SITE_PACKAGES/diff_gaussian_rasterization/_C"*.so

cd "$PKG_DIR"
rm -rf build dist *.egg-info
export CXXFLAGS="${CXXFLAGS:-} -g"
export NVCC_FLAGS="-lineinfo -G"
export TORCH_USE_CUDA_DSA=1
export TORCH_SHOW_CPP_STACKTRACES=1
"$ENV_PY" -m pip install --no-build-isolation .
cd "$WORK_DIR"