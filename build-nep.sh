#!/usr/bin/env bash
# Build GROMACS 2026.4 with the NEP/qNEP interface and the optional WPE backend.
#
# Environment knobs (all optional):
#   BUILD_DIR, INSTALL_PREFIX, BUILD_JOBS
#   BUILD_TYPE=Release|RelWithDebInfo|Debug          (default Release)
#   CUDA_ARCH=89                                     (default: nvidia-smi detection)
#   NATIVE_CPU=1                                     (-march=native -mtune=native)
#   FAST_MATH=1                                      (CUDA --use_fast_math; changes numerics)
#   LTO=1                                            (interprocedural optimization)
#   WPE=1|0                                          (compile the WPE backend; default 1)
#   WPE_LIBRARY=/path/to/wpe.so/lib                  (optional configure-time hint)
#   CLEAN_BUILD=1, RUN_TESTS=0, DO_INSTALL=1
set -euo pipefail

SOURCE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BUILD_DIR=${BUILD_DIR:-"$SOURCE_DIR/build-nep"}
INSTALL_PREFIX=${INSTALL_PREFIX:-"$SOURCE_DIR/install-nep"}
BUILD_JOBS=${BUILD_JOBS:-$(nproc)}
BUILD_TYPE=${BUILD_TYPE:-Release}
NATIVE_CPU=${NATIVE_CPU:-1}
FAST_MATH=${FAST_MATH:-0}
LTO=${LTO:-0}
WPE=${WPE:-1}
WPE_LIBRARY=${WPE_LIBRARY:-}
CUDA_ARCH=${CUDA_ARCH:-}
CLEAN_BUILD=${CLEAN_BUILD:-0}
RUN_TESTS=${RUN_TESTS:-1}
DO_INSTALL=${DO_INSTALL:-0}

if [[ -z "$CUDA_ARCH" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    compute_cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d ' .\r')
    [[ "$compute_cap" =~ ^[0-9]+$ ]] && CUDA_ARCH=$compute_cap
fi
if [[ -z "$CUDA_ARCH" ]]; then
    cat >&2 <<'EOF'
Could not detect the NVIDIA GPU compute capability (nvidia-smi unavailable).
Set CUDA_ARCH explicitly, e.g.:
  CUDA_ARCH=89  (Ada: RTX 4090)
  CUDA_ARCH=120 (Blackwell: RTX 5090)
EOF
    exit 2
fi

[ -d "$SOURCE_DIR/src/external/gpumd/LICENSES" ] || {
    echo "Missing vendored GPUMD tree: $SOURCE_DIR/src/external/gpumd" >&2
    exit 2
}
[[ -f "$SOURCE_DIR/src/external/wpe/wisevolve_potential_api.h" ]] || {
    echo "Missing vendored WPE public API header." >&2
    exit 2
}

if [[ "$CLEAN_BUILD" == 1 ]]; then
    rm -rf -- "$BUILD_DIR"
fi

host_flags="-O3 -DNDEBUG"
if [[ "$NATIVE_CPU" == 1 ]]; then
    host_flags+=" -march=native -mtune=native"
fi

cuda_flags="-O3 -Xptxas -O3 --extra-device-vectorization"
if [[ "$FAST_MATH" == 1 ]]; then
    cuda_flags+=" --use_fast_math"
fi

lto_args=()
if [[ "$LTO" == 1 ]]; then
    lto_args+=("-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON")
fi

wpe_args=("-DGMX_NEP_WPE=OFF")
if [[ "$WPE" == 1 ]]; then
    wpe_args=("-DGMX_NEP_WPE=ON")
    if [[ -n "$WPE_LIBRARY" ]]; then
        wpe_args+=("-DGMX_WPE_LIBRARY=$WPE_LIBRARY")
    fi
fi

cmake -S "$SOURCE_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
    -DCMAKE_C_FLAGS_RELEASE="$host_flags" \
    -DCMAKE_CXX_FLAGS_RELEASE="$host_flags" \
    -DCMAKE_CUDA_FLAGS_RELEASE="$cuda_flags" \
    "${lto_args[@]}" \
    -DGMX_GPU=CUDA \
    -DGMX_NNPOT=NEP \
    "${wpe_args[@]}" \
    -DGMX_MPI=OFF \
    -DGMX_THREAD_MPI=ON \
    -DGMX_OPENMP=ON \
    -DGMX_FFT_LIBRARY=FFTW3 \
    -DGMX_SIMD=AVX2_256 \
    -DBUILD_TESTING=ON \
    -DGMX_BUILD_UNITTESTS=ON \
    -DREGRESSIONTEST_DOWNLOAD=OFF

cmake --build "$BUILD_DIR" --parallel "$BUILD_JOBS"

if [[ "$RUN_TESTS" == 1 ]]; then
    ctest --test-dir "$BUILD_DIR" --output-on-failure -R '^NNPotAppliedForcesUnitTest$' || true
fi

if [[ "$DO_INSTALL" == 1 ]]; then
    cmake --install "$BUILD_DIR"
    echo "Installed to: $INSTALL_PREFIX"
fi

cat <<EOF

Build complete: $BUILD_DIR/bin/gmx
  CUDA architecture : $CUDA_ARCH
  Build type        : $BUILD_TYPE
  Host flags        : $host_flags
  CUDA flags        : $cuda_flags
  WPE backend       : $([[ "$WPE" == 1 ]] && echo enabled || echo disabled)
  WPE runtime hint  : ${WPE_LIBRARY:-<search at run time>}

Single-GPU full-system NEP launch:
  $BUILD_DIR/bin/gmx mdrun -s system.tpr -ntmpi 1 -ntomp 1

WPE run time (the proprietary engine is not linked, it is dlopen()ed):
  export GMX_WPE_LIBRARY=/path/to/wpe.so/lib
  # or add that directory to LD_LIBRARY_PATH
EOF
