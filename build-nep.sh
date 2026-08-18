#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BUILD_DIR=${BUILD_DIR:-"$SOURCE_DIR/build-nep"}
INSTALL_PREFIX=${INSTALL_PREFIX:-"$SOURCE_DIR/install-nep"}
BUILD_JOBS=${BUILD_JOBS:-$(nproc)}
NATIVE_CPU=${NATIVE_CPU:-0}
CUDA_ARCH=${CUDA_ARCH:-}
CLEAN_BUILD=${CLEAN_BUILD:-0}
RUN_TESTS=${RUN_TESTS:-1}
DO_INSTALL=${DO_INSTALL:-0}

if [[ -z "$CUDA_ARCH" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    compute_cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d ' .\r')
    [[ "$compute_cap" =~ ^[0-9]+$ ]] && CUDA_ARCH=$compute_cap
fi
if [[ -z "$CUDA_ARCH" ]]; then
    echo "Could not detect the NVIDIA GPU compute capability." >&2
    echo "Set CUDA_ARCH explicitly, for example CUDA_ARCH=89 for Ada GPUs." >&2
    exit 2
fi

[[ -f "$SOURCE_DIR/src/external/gpumd/LICENCE" ]] || {
    echo "Missing vendored GPUMD tree: $SOURCE_DIR/src/external/gpumd" >&2
    exit 2
}

if [[ "$CLEAN_BUILD" == 1 ]]; then
    rm -rf -- "$BUILD_DIR"
fi

native_cpu_args=()
if [[ "$NATIVE_CPU" == 1 ]]; then
    native_cpu_args+=("-DGMX_SIMD=AVX2_256")
fi

cmake -S "$SOURCE_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
    -DGMX_GPU=CUDA \
    -DGMX_NNPOT=NEP \
    -DGMX_MPI=OFF \
    -DGMX_THREAD_MPI=ON \
    -DGMX_OPENMP=ON \
    -DGMX_BUILD_OWN_FFTW=ON \
    -DBUILD_TESTING=ON \
    -DGMX_BUILD_UNITTESTS=ON \
    -DREGRESSIONTEST_DOWNLOAD=OFF \
    "${native_cpu_args[@]}"

cmake --build "$BUILD_DIR" --parallel "$BUILD_JOBS"

if [[ "$RUN_TESTS" == 1 ]]; then
    cmake --build "$BUILD_DIR" --parallel "$BUILD_JOBS" --target nnpot_applied_forces-test
    ctest --test-dir "$BUILD_DIR" --output-on-failure -R '^NNPotAppliedForcesUnitTest$'
fi

if [[ "$DO_INSTALL" == 1 ]]; then
    cmake --install "$BUILD_DIR"
    echo "Installed to: $INSTALL_PREFIX"
fi

echo "Build complete: $BUILD_DIR/bin/gmx"
echo "Recommended single-GPU full-system NEP launch:"
echo "  $BUILD_DIR/bin/gmx mdrun -s system.tpr -ntmpi 1 -ntomp 1"
echo "NEP-MM smoke test:"
echo "  MODEL=/path/to/nep.txt $SOURCE_DIR/examples/nep-mm-methane-water/run-test.sh"
