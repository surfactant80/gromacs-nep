#!/usr/bin/env bash
# Replace src/external/gpumd with a fresh GPUMD tree and reconfigure/build.
# Usage: scripts/update-gpumd-vendor.sh /path/to/GPUMD-master [build-dir]
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
NEW_TREE=${1:?usage: update-gpumd-vendor.sh /path/to/GPUMD-master [build-dir]}
BUILD_DIR=${2:-"$HERE/build-nep"}

[[ -f "$NEW_TREE/LICENSES/LGPL-3.0.txt" && -f "$NEW_TREE/src/force/nep.cu" ]] || {
    echo "Not a GPUMD source tree (missing LICENSES/ or src/force/nep.cu): $NEW_TREE" >&2
    exit 2
}

rm -rf "$HERE/src/external/gpumd"
cp -a "$NEW_TREE" "$HERE/src/external/gpumd"
echo "Replaced src/external/gpumd with $NEW_TREE"

cmake -S "$HERE" -B "$BUILD_DIR" -DGMX_NNPOT=NEP -DGMX_GPU=CUDA
cmake --build "$BUILD_DIR" --parallel "$(nproc)"
echo "Build finished: $BUILD_DIR/bin/gmx"
