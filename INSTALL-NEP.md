# Portable source installation

## Requirements

- 64-bit Linux
- NVIDIA GPU and working NVIDIA driver
- CUDA toolkit with `nvcc`, CMake, a C/C++ compiler, Python 3, and standard build tools
- Enough disk space and memory to build GROMACS and the vendored GPUMD CUDA sources

## Build in the source tree

```bash
./build-nep.sh
```

The script detects the current NVIDIA GPU compute capability and creates
`build-nep/bin/gmx`. It also runs the focused NNPot unit test.

For a clean rebuild or installation prefix:

```bash
CLEAN_BUILD=1 BUILD_JOBS=16 ./build-nep.sh
DO_INSTALL=1 INSTALL_PREFIX=$HOME/gromacs-nep ./build-nep.sh
```

Use `NATIVE_CPU=1` only when the destination CPU supports AVX2. The portable
default lets GROMACS/CMake choose a compatible SIMD setting. If GPU detection
is unavailable, specify the CUDA architecture explicitly, e.g. `CUDA_ARCH=89`.

After installation, initialize the GROMACS environment if needed:

```bash
source $HOME/gromacs-nep/bin/GMXRC
```

## Updating GPUMD

Replace the complete `src/external/gpumd` directory with a clean GPUMD source
tree, then run a clean rebuild:

```bash
rm -rf src/external/gpumd
cp -a /path/to/new/GPUMD src/external/gpumd
CLEAN_BUILD=1 ./build-nep.sh
```

Do not merge files into the old directory. The adapter currently expects the
same GPUMD classes and source paths used by this package. A future GPUMD update
that changes the NEP/qNEP API, filenames, model format, CUDA dependencies, or
kernel behavior can require corresponding adapter/CMake updates. Always rebuild
and run the supplied NEP, NEP-MM, and qNEP tests after replacement.
