# GROMACS 2026.4 + NEP/qNEP + WPE — Introduction, Installation, and Usage

This document describes the current release of the program: **GROMACS 2026.4 with the
latest GPUMD NEP/qNEP potentials and the Wisevolve Potential Engine (WPE) 1.0.0**.
It covers the feature set, architecture, installation, configuration, usage, validation,
performance, limitations, and troubleshooting. The Chinese version is available at
`docs/程序介绍-安装-使用-中文.md`.

---

## 1. Overview

This is a molecular-dynamics program built on **GROMACS 2026.4** that integrates the
**NEP/qNEP potentials from the latest GPUMD master** and the **WPE 1.0.0** high-performance
inference engine. It keeps the full GROMACS feature set — integrators, thermostats and
barostats, constraints, bonded interactions, Lennard-Jones, GPU PME, trajectory and energy
output — and exposes NEP/qNEP as a GROMACS force provider.

Key properties:

- **Two NEP engines**: the open-source CUDA kernels shipped with GPUMD (vendored,
  unmodified) and the proprietary WPE engine (loaded at runtime with `dlopen`, never linked).
- **Three electrostatic paths**: qNEP-native PPPM/Ewald, qNEP with GROMACS GPU PME
  (chain-rule forces), and plain NEP combined with classical MM.
- **NEP–MM hybrid systems**: NEP and classical MM regions coexist in one GROMACS run.
- **GPU-free `grompp`**: generate a `.tpr` on a CPU-only machine and run it later on a GPU.
- **Zero modifications to GPUMD**: `src/external/gpumd` is byte-identical to upstream
  GPUMD master.

---

## 2. Feature Summary

| Feature | Status | Notes |
|---|---|---|
| Plain NEP4 (`nep4` / `nep4_zbl`, legacy and new formats) | ✅ | Both GPUMD and WPE backends |
| qNEP charge1/2/3 (`nep4_charge*`, qnep1/2/3) | ✅ | GPUMD-native PPPM/Ewald; WPE uses its own PPPM/Ewald |
| qNEP with GROMACS-owned electrostatics (GPU PME + chain rule) | ✅ | `nnpot-gromacs-charges = yes`, GPUMD backend only |
| NEP–MM (NEP region + classical MM) | ✅ | Device and host paths validated |
| WPE 1.0.0 backend | ✅ | Proprietary binary, runtime-loaded, not linked |
| vdW-NEP (`nep_vdw` / `qnep3_vdw`) | ✅ GPUMD / ❌ WPE | WPE 1.0.0 does not implement the vdW family |
| D3 dispersion correction (`nnpot-dftd3`) | ✅ | Applied when supported by both model and backend |
| Standard GROMACS forces (PME, LJ, bonded) | ✅ | Original GROMACS paths unchanged |
| GPU-free `grompp` | ✅ | CPU-only model parsing, backend selection, `.tpr` generation |
| Device-side event ordering (asynchronous) | ✅ | Enabled by default; `GMX_NEP_NO_ASYNC=1` reverts |
| NEP5 | ❌ | Removed from the MD potential classes in latest GPUMD; `grompp` reports it |
| Temperature-dependent NEP4 | ❌ | Not driven by the adapter |
| `nep_dipole` / `polarizability` / `spin` | ❌ | Not ordinary energy/force models; `grompp` reports it |
| Multi-GPU / domain decomposition | ❌ | NEP/WPE device paths currently require a single rank |

---

## 3. Architecture and Data Flow

```text
GROMACS MD step
   │
   ├─ coordinates / box / types (device or host)
   ▼
src/gromacs/applied_forces/nnpot/
   ├─ NepModel            unified adapter (parsing, backend selection, buffers)
   ├─ nepbackendsettings  auto / gpumd / wpe
   ├─ wpeloader           dlopen + ABI validation (first evaluation in mdrun only)
   ├─ wpepotential        WPE ::Potential adapter
   └─ nepmodel_qnep_gromacs   qNEP-GROMACS chain-rule forces
   │
   ├─ GPUMD backend: Potential / NEP / NEP_Charge (vendored CUDA kernels)
   └─ WPE backend: libwisevolve_potential.so (C ABI, proprietary)
   ▼
GROMACS force reduction / integrator / output
```

Design notes:

- **WPE is not linked.** The source tree contains only the public BSD-2 header. `gmx`
  loads the proprietary engine with `dlopen()`; `ldd gmx` never shows
  `libwisevolve_potential.so`.
- **`grompp` and `mdrun` are separated.** `grompp` performs CPU-only model parsing,
  validation, and backend selection. It does not initialize CUDA, does not open
  `/dev/nvidia*`, and does not `dlopen` WPE. WPE is loaded on the first force evaluation
  in `mdrun`.
- **qNEP–GROMACS chain rule.** qNEP predicts charges → charges are written to GROMACS
  charge buffers → GROMACS non-bonded + GPU PME computes the Coulomb/reciprocal potential
  → chain-rule forces `(dE/dq)(dq/dR)` are added back to the NEP forces.
- **Asynchronous execution.** NEP/WPE kernels and the GROMACS force reduction are ordered
  with device-side events; no per-step host synchronization is needed on non-energy steps.

---

## 4. Requirements

| Item | Requirement |
|---|---|
| Operating system | Linux (x86-64) |
| Compiler | GCC 15.x (or compatible) |
| CMake | ≥ 3.28 |
| CUDA | CUDA Toolkit 13.x (tested with 13.4, nvcc 13.4.92) |
| GPU | NVIDIA, sm_80 or newer (tested on RTX 4090 D / sm_89, 24 GB) |
| CPU | AVX2 (default `GMX_SIMD=AVX2_256`) |
| FFT | System FFTW3 |
| MPI | Not required; GROMACS thread-MPI is used |
| WPE runtime | Optional; WPE 1.0.0 is a proprietary CUDA-12 `.so` that needs `libcufft.so.11` |

> The WPE runtime is not distributed with the source package. It is not needed when only
> plain NEP/qNEP with the GPUMD backend is used.

---

## 5. Installation

### 5.1 Extract

```bash
tar -xzf gromacs-2026.4-nep-source-20261104.tar.gz
cd gromacs-2026.4-nep
```

### 5.2 One-command build (recommended)

```bash
# Default: Release + -march=native + WPE enabled + auto-detected CUDA arch + unit tests
./build-nep.sh

# RTX 4090 (sm_89)
CUDA_ARCH=89 ./build-nep.sh

# RTX 5090 (sm_120)
CUDA_ARCH=120 ./build-nep.sh

# Without the WPE backend
WPE=0 CUDA_ARCH=89 ./build-nep.sh

# Build and install into install-nep/
DO_INSTALL=1 CUDA_ARCH=89 ./build-nep.sh

# Optional: configure-time WPE library hint (runtime loading still uses the search path)
WPE_LIBRARY=/path/to/wpe.so/lib CUDA_ARCH=89 ./build-nep.sh

# Optional: LTO, clean rebuild
LTO=1 CUDA_ARCH=89 ./build-nep.sh
CLEAN_BUILD=1 CUDA_ARCH=89 ./build-nep.sh
```

Outputs:

- `build-nep/bin/gmx`
- `install-nep/` (only with `DO_INSTALL=1`)

Environment variables understood by `build-nep.sh`:

| Variable | Default | Meaning |
|---|---|---|
| `BUILD_DIR` | `build-nep` | Build directory |
| `INSTALL_PREFIX` | `install-nep` | Install prefix |
| `BUILD_JOBS` | `nproc` | Parallel build jobs |
| `BUILD_TYPE` | `Release` | CMake build type |
| `CUDA_ARCH` | Auto-detected | For example `89`, `120` |
| `NATIVE_CPU` | `1` | `-march=native -mtune=native` |
| `FAST_MATH` | `0` | Adds `--use_fast_math` (GROMACS CUDA builds enable it by default) |
| `LTO` | `0` | Interprocedural optimization |
| `WPE` | `1` | Build the WPE backend |
| `WPE_LIBRARY` | Empty | Configure-time WPE path hint |
| `RUN_TESTS` | `1` | Run the NNPot unit test after building |
| `CLEAN_BUILD` | `0` | Remove the build directory first |
| `DO_INSTALL` | `0` | Run `cmake --install` |

### 5.3 Manual CMake (equivalent)

```bash
cmake -S . -B build-nep \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DCMAKE_C_FLAGS_RELEASE="-O3 -DNDEBUG -march=native -mtune=native" \
  -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG -march=native -mtune=native" \
  -DCMAKE_CUDA_FLAGS_RELEASE="-O3 -Xptxas -O3 --extra-device-vectorization" \
  -DGMX_GPU=CUDA \
  -DGMX_NNPOT=NEP \
  -DGMX_NEP_WPE=ON \
  -DGMX_MPI=OFF \
  -DGMX_THREAD_MPI=ON \
  -DGMX_OPENMP=ON \
  -DGMX_FFT_LIBRARY=FFTW3 \
  -DGMX_SIMD=AVX2_256 \
  -DBUILD_TESTING=ON \
  -DGMX_BUILD_UNITTESTS=ON \
  -DREGRESSIONTEST_DOWNLOAD=OFF

cmake --build build-nep --parallel "$(nproc)"
```

Set `-DGMX_NEP_WPE=OFF` if the WPE backend is not needed.

### 5.4 WPE runtime setup

WPE is a proprietary runtime library obtained separately. Make it visible to `gmx`
using either method:

```bash
# Option 1: environment variables
export LD_LIBRARY_PATH=/path/to/cuda12-cufft/nvidia/cufft/lib:$LD_LIBRARY_PATH
export GMX_WPE_LIBRARY=/path/to/wpe.so/lib
```

```text
# Option 2: mdp
nnpot-wpe-library = /path/to/wpe.so/lib
```

`nnpot-wpe-library` may be either the `.so` file or a directory containing
`libwisevolve_potential.so`. If loading fails, check:

1. `libwisevolve_potential.so` exists;
2. `libcufft.so.11` is available (WPE 1.0.0 is built against CUDA 12);
3. the ABI matches `src/external/wpe/wisevolve_potential_api.h`.

### 5.5 Post-install checks

```bash
export PATH=$PWD/build-nep/bin:$PATH

gmx --version | grep -E 'CUDA|SIMD'

# Unit tests (model parsing / WPE shape checks)
ctest --test-dir build-nep -R NNPotAppliedForcesUnitTest --output-on-failure

# CPU end-to-end checks (16 checks, no GPU required)
GMX=$PWD/build-nep/bin/gmx WPE_LIB=/path/to/wpe.so/lib \
  test/2026.4-validation/run_cpu_validation.sh
```

---

## 6. Quick Start

The examples below assume the `.tpr` has already been produced by `gmx grompp`.

### 6.1 Plain NEP (GPUMD backend)

`run.mdp`:

```text
integrator      = md
dt              = 0.0005
nsteps          = 2000
nstenergy       = 100
nstlog          = 1000

nnpot-active    = yes
nnpot-modelfile = nep.txt
nnpot-backend   = gpumd
```

```bash
gmx grompp -f run.mdp -c conf.gro -p topol.top -o run.tpr
gmx mdrun -s run.tpr -ntmpi 1 -ntomp 1
```

### 6.2 qNEP with native electrostatics (GPUMD or WPE)

```text
nnpot-active          = yes
nnpot-modelfile       = qnep_model1.txt
nnpot-backend         = gpumd          ; or wpe
nnpot-gromacs-charges = no
nnpot-gpumd-kspace    = pppm           ; or ewald (GPUMD backend)
nnpot-wpe-kspace      = pppm           ; or ewald (WPE backend)
```

### 6.3 qNEP with GROMACS-owned electrostatics (GPUMD backend only)

```text
nnpot-active           = yes
nnpot-modelfile        = qnep_model1.txt
nnpot-backend          = gpumd
nnpot-gromacs-charges  = yes

coulombtype            = PME
rcoulomb               = 0.8
rlist                  = 0.8
pme-order              = 4
```

Requirements:

- single rank, full-system (no domain decomposition);
- `pme-order = 4`, `rcoulomb = rlist`;
- box length ≥ 2.5 × `rcoulomb`;
- ZBL, charge3, FEP, and MTS are not supported on this path.

### 6.4 WPE backend

```bash
export LD_LIBRARY_PATH=/path/to/cuda12-cufft/nvidia/cufft/lib:$LD_LIBRARY_PATH
export GMX_WPE_LIBRARY=/path/to/wpe.so/lib
```

```text
nnpot-active           = yes
nnpot-modelfile        = qnep_model1.txt
nnpot-backend          = wpe
nnpot-wpe-kspace       = pppm           ; or ewald
nnpot-wpe-pppm-spacing = 1.0            ; Å
```

```bash
gmx grompp -f run.mdp -c conf.gro -p topol.top -o wpe.tpr
gmx mdrun -s wpe.tpr -ntmpi 1 -ntomp 1
```

### 6.5 NEP–MM

See `examples/nep-mm-methane-water/` for `nep-mm.mdp`, topology files, and a run script.

### 6.6 CPU-generated `.tpr`, GPU execution

```bash
CUDA_VISIBLE_DEVICES="" gmx grompp -f run.mdp -c conf.gro -p topol.top -o cpu.tpr
# No /dev/nvidia* access, no WPE dlopen; copy the .tpr to a GPU host and run mdrun
```

---

## 7. `.mdp` Options

### 7.1 New options

| Option | Default | Values / notes |
|---|---|---|
| `nnpot-backend` | `auto` | `auto` / `gpumd` / `wpe` |
| `nnpot-wpe-library` | Empty | WPE library file or directory; falls back to `GMX_WPE_LIBRARY` / `LD_LIBRARY_PATH` |
| `nnpot-wpe-kspace` | `pppm` | `pppm` / `ewald` |
| `nnpot-wpe-pppm-spacing` | `1.0` | Å, 0.2–2.0 |
| `nnpot-gpumd-kspace` | `pppm` | `pppm` / `ewald` (GPUMD-native qNEP) |
| `nnpot-gpumd-pppm-spacing` | `0` | Å; 0 means the model default |
| `nnpot-dftd3` | Empty | For example `"pbe 12.0 12.0"`; applied when supported |

### 7.2 Backend selection

- `nnpot-backend = gpumd`: force the GPUMD backend; unsupported models produce an error.
- `nnpot-backend = wpe`: force WPE; errors if WPE was not built, the library cannot be
  found, or the model violates WPE shape constraints.
- `nnpot-backend = auto` (default): choose WPE when it is built, the model satisfies the
  WPE shape constraints, and the runtime library is found; otherwise choose GPUMD and
  report the reason in `grompp`/`md.log`.

WPE shape constraints (checked by `grompp`):

- `l_max_3body = 4`;
- ANN hidden-layer width ≤ 256.

### 7.3 Runtime environment variables

| Variable | Purpose |
|---|---|
| `GMX_WPE_LIBRARY` | WPE library search path |
| `GMX_NEP_NO_ASYNC=1` | Revert to the legacy per-step host-synchronized path (debugging) |
| `GMX_WPE_ATOM_VIRIAL=1` | Ask WPE for per-atom virials (debugging; the default system virial is faster) |
| `GMX_QNEP_DEBUG` | Print qNEP adapter diagnostics |
| `GMX_CUDA_GRAPH` | Try GROMACS MD GPU Graph (external WPE may not be capture-compatible) |

---

## 8. Supported Models

| Model | GPUMD backend | WPE backend | Notes |
|---|---|---|---|
| `nep4` / `nep4_zbl` (legacy format) | ✅ | ✅ | Legacy first-line tag |
| `nep <N> ...` (new format) | ✅ | ✅ | Converted automatically to MD input |
| `nep4_charge1/2/3`, `_zbl_` | ✅ native | ✅ qNEP | WPE uses its own PPPM/Ewald |
| qnep1/2/3 (new format) | ✅ | ✅ | |
| `nep_vdw` / `qnep3_vdw` | ✅ | ❌ | WPE 1.0.0 does not implement the vdW family |
| `nep5` / `nep5_zbl` | ❌ | ❌ | Removed from the MD potential classes in latest GPUMD |
| Temperature-dependent NEP4 | ❌ | ❌ | Not driven by the adapter |
| `nep_dipole` / `polarizability` / `spin` | ❌ | ❌ | Not ordinary energy/force models |

The model file is selected with `nnpot-modelfile`. `grompp` parses it and prints the model
type, descriptor orders, ANN width, and related information. Unsupported models fail during
`grompp`, never silently at runtime.

---

## 9. Running and Output

### 9.1 Running

```bash
gmx mdrun -s run.tpr -ntmpi 1 -ntomp 1
```

- NEP/WPE device paths require **a single rank** (`-ntmpi 1`); do not use domain
  decomposition.
- `-ntomp 1` is sufficient; additional host threads barely help the GPU path.
- WPE uses one visible GPU only.

### 9.2 Output

- `md.log`: model information, backend selection, and the `Performance:` line;
- `ener.edr`: energies, temperature, pressure; use `gmx energy` and select
  `NN-Potential` (or `Potential` when GROMACS electrostatics are enabled);
- standard GROMACS outputs such as `traj_comp.xtc`, `confout.gro`, and `mdout.mdp`.

The `Performance:` line has four columns:

```text
Performance:  <ns/day>  <hour/ns>  <ms/step>  <Matom·steps/s>
```

`ms/step` is the **fourth** column.

---

## 10. Accuracy and Performance Summary

> Test environment: RTX 4090 D (sm_89, 24 GB), driver 595.99.02, CUDA 13.4; GROMACS
> 2026.4 mixed precision / thread-MPI / CUDA / AVX2; GPUMD and WPE use the same
> source/runtime. Full data: `docs/GPU-TESTS-2026.4.md`.

### 10.1 Single-point accuracy (vs GPUMD)

| Path | ΔE (eV/atom) | Force RMS (eV/Å) |
|---|---|---|
| Plain NEP (CH4) | 1.7e-5 | 5.1e-5 (0.004%) |
| qNEP native (water) | ~5e-6 | 3.5–3.9e-5 (0.0025–0.0028%) |
| qNEP + GROMACS PME (water) | -6.8e-6 | 1.9e-4 (0.0135%) |
| WPE PPPM (water qNEP) | -2.8e-8 | 1.0e-5 (0.0007%) |
| WPE Ewald (water qNEP) | -8.8e-8 | 1.0e-5 (0.0007%) |
| WPE (CH4 plain) | -3.2e-8 | 4.5e-5 (0.0035%) |

### 10.2 Other validation

- **Finite differences** (qNEP–GROMACS, 6 atoms, ±2e-3 nm): maximum relative error
  **0.95%**, pass.
- **NVE drift** (water qNEP, 0.5 fs, 2 ps after NVT equilibration): **-3.1e-6
  eV/(atom·ps)** (asynchronous path: -2.8e-6).
- **NEP–MM**: identical NEP energy; NEP-region force RMSE **2.1e-6 kJ/mol/nm**.
- **NPT/NPH** (WPE 200 ps / GPUMD backend 100 ps): GROMACS–GPUMD pressure difference
  **1.0–7.5 bar (≤0.8σ)**. Density and potential energy still differ by ~0.5–0.6% and
  ~1e-3–1.3e-2 eV/atom (largest in native NPH); densities are still drifting over
  100–200 ps, so these are sampling/equilibration differences, not force errors.
- **Automated tests**: unit test **1/1**, CPU end-to-end **16/16** pass.

### 10.3 Performance (single GPU, NVE, 0.5 fs)

| System | Engine | ms/step | ns/day | Relative to GPUMD |
|---|---|---|---|---|
| 1,500 | GROMACS + WPE (qNEP) | 0.387 | 111.8 | 2.57× |
| 1,500 | GPUMD + WPE | 0.362 | 119.3 | 2.75× |
| 1,500 | GROMACS + GPUMD backend (qNEP) | 0.988 | 43.7 | 1.006× |
| 1,500 | GPUMD (qNEP PPPM) | 0.994 | 43.5 | 1.00× |
| 1,500 | GROMACS PME (qNEP–GROMACS) | 1.191 | 36.3 | 0.83× |
| 1,500 | GROMACS + WPE (CH4 plain) | 0.324 | 133.3 | 2.09× |
| 18,000 | GROMACS + WPE | 0.632 | 68.4 | 2.36× |
| 18,000 | GPUMD + WPE | 0.577 | 74.9 | 2.58× |
| 18,000 | GROMACS + GPUMD backend | 1.529 | 28.3 | 0.97× |
| 18,000 | GPUMD (qNEP PPPM) | 1.488 | 29.0 | 1.00× |
| 18,000 | GROMACS PME (qNEP–GROMACS) | 2.572 | 16.8 | 0.58× |

Key takeaways:

- WPE vs the GPUMD-native backend: **2.09× (plain NEP) / 2.57× (qNEP)** at 1,500 atoms;
  **2.36× (GROMACS) / 2.58× (GPUMD)** at 18,000 atoms.
- Same-engine comparison: GROMACS + WPE is **7% slower (1,500)** and **9.6% slower
  (18,000)** than GPUMD + WPE.
- qNEP–GROMACS (PME) remains slower than the engine-internal PPPM at 18,000 atoms
  (2.57 vs 1.53 ms/step); no PME crossover was observed at this scale.

---

## 11. Known Limitations

1. **Single rank, no domain decomposition** for the NEP/WPE device paths. Multi-GPU DD is
   not supported; WPE uses one visible GPU.
2. **WPE is incompatible with GROMACS-owned electrostatics.** WPE performs qNEP
   electrostatics internally and cannot provide GPUMD's `D_real` chain-rule potential.
   `grompp` reports this combination; use `nnpot-backend = gpumd` instead.
3. **qNEP–GROMACS constraints**: single rank, full-system, `pme-order=4`,
   `rcoulomb=rlist`, box length ≥ 2.5·`rcoulomb`; no ZBL/charge3/FEP/MTS.
4. **WPE runtime dependency**: WPE 1.0.0 is built against CUDA 12 and needs
   `libcufft.so.11`; CUDA-13-only systems need the CUDA 12 cuFFT runtime or a CUDA 13
   WPE build from Wisevolve.
5. **WPE does not support the vdW family**; WPE shape constraints are `l_max_3body=4`
   and ANN width ≤ 256.
6. **NEP5, temperature-dependent NEP4, dipole/polarizability/spin models are not
   supported**; `grompp` reports the reason.
7. **`.qnc` charge dumps**: with full-system NEP, GROMACS disables classical
   non-bonded/PME work and therefore has no device coordinate buffer, so the binary
   charge output from `nnpot-charges-nst` is not triggered (the `.tpr` still contains
   the `QNC1` header).
8. **Largest validated system: 18,000 atoms.** Larger systems, multi-GPU, and DD are
   untested.
9. **Performance numbers use a 0.5 fs timestep.** If HMR/SETTLE is used to increase
   `dt`, revalidate the dynamics and thermodynamic observables.

---

## 12. Troubleshooting

### 12.1 `mdrun` cannot find `libwisevolve_potential.so`

- Check `nnpot-wpe-library` / `GMX_WPE_LIBRARY` / `LD_LIBRARY_PATH`.
- If the error mentions `libcufft.so.11`, add the CUDA 12 cuFFT directory to
  `LD_LIBRARY_PATH`.
- `grompp` only checks that the file exists; it does not `dlopen`. Loading failures occur
  at the first force evaluation in `mdrun`.

### 12.2 WPE is installed, but GPUMD was selected

With `nnpot-backend = auto`, the program falls back to GPUMD when:

- it was built with `GMX_NEP_WPE=OFF`;
- the model violates WPE shape constraints (`l_max_3body=4`, ANN ≤ 256);
- the WPE runtime library cannot be found.

`grompp`/`md.log` reports the reason. Set `nnpot-backend = wpe` to force WPE and expose
the error explicitly.

### 12.3 Does `gmx grompp` need a GPU?

No:

```bash
CUDA_VISIBLE_DEVICES="" gmx grompp -f run.mdp -c conf.gro -p topol.top -o cpu.tpr
```

The resulting `.tpr` can be copied to a GPU host and run directly with `mdrun`.

### 12.4 Why does the pressure differ from GPUMD by tens of bar?

Check the error bars first: at 50 ps the pressure SEM alone is 15–24 bar; at 200 ps the
GROMACS–GPUMD difference is 1.0–7.5 bar (≤0.8σ), within statistical uncertainty. Also note
that the GPUMD pressure sign is `+(sxx+syy+szz)/3`.

### 12.5 How to revert the asynchronous path?

```bash
GMX_NEP_NO_ASYNC=1 gmx mdrun -s run.tpr -ntmpi 1 -ntomp 1
```

### 12.6 NEP5 / spin / dipole models fail

These models are not supported in this release; `grompp` reports the reason. Use a
NEP4/qNEP model instead.

---

## 13. Validation Scripts

```bash
# Unit tests
ctest --test-dir build-nep -R NNPotAppliedForcesUnitTest --output-on-failure

# CPU end-to-end (16 checks)
GMX=$PWD/build-nep/bin/gmx WPE_LIB=/path/to/wpe.so/lib \
  test/2026.4-validation/run_cpu_validation.sh

# GPU validation (single point / finite difference / NVE / ensembles / performance)
test/2026.4-validation/gpu/run_gpu_validation.sh

# WPE ABI probe
python3 test/2026.4-validation/wpe_abi_probe.py \
  /path/to/wpe.so/lib/libwisevolve_potential.so

# Examples
examples/qnep-and-gromacs-pme/run-tests.sh
examples/nep-mm-methane-water/run-test.sh
```

---

## 14. License and Citation

- GROMACS: https://www.gromacs.org (LGPL-2.1+)
- GPUMD: https://github.com/brucefan1983/GPUMD (LGPL-3.0+; see
  `src/external/gpumd/LICENSES`)
- WPE: public interface under BSD-2-Clause; the engine itself is proprietary Wisevolve
  software. See `src/external/wpe/LICENSE-WPE` and `NOTICE`.
  Citation: Hekai Bu and Zheyong Fan, arXiv:2610.17366 (2026).

The source package and its checksum are listed in `SHA256SUMS-2026.4.txt`.
