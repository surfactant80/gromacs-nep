> **历史文档（2026.3 版本）**：本报告描述的是基于 GROMACS 2026.3 的上一轮集成。
> 2026.4 + 最新 GPUMD + WPE 的迁移见 `docs/迁移记录-2026.4.md`，使用说明见
> `README-NEP.md` / `INSTALL-NEP.md`。

# GROMACS 2026.3 + GPUMD NEP integration

Date: 2026-08-12  
Test GPU: NVIDIA GeForce RTX 4090 D, CUDA architecture 8.9

## Overview

This source tree adds the unmodified GPUMD NEP evaluator to the existing
GROMACS `nnpot` force-provider module. GROMACS still owns integration,
constraints, thermostats, barostats, PME, bonded interactions, trajectory I/O,
checkpointing, and the classical MM part of hybrid NEP/MM systems.

The complete GPUMD source is vendored at `src/external/gpumd`. Integration and
performance changes are outside that directory, so a future GPUMD update can be
performed by replacing the vendor directory and rebuilding.

Supported model headers are `nep4`, `nep4_zbl`, `nep5`, and `nep5_zbl`.
Mechanical NEP/MM embedding and the existing GROMACS link-atom force
redistribution are supported.

## Performance work completed

1. Removed the quadratic full-system classical exclusion path. For a
   full-system NEP simulation, classical charges and non-bonded type parameters
   are disabled during topology preprocessing instead.
2. Cached the static atomic-number to NEP-type mapping and uploads it only when
   the atom input changes.
3. Moved coordinate layout conversion and nm-to-Angstrom conversion to CUDA.
4. Added page-locked staging buffers and queued CUDA copies.
5. Moved total-energy and nine-component virial reduction to CUDA.
6. Converted NEP forces directly to GROMACS precision/layout on CUDA.
7. Reduced per-step device-to-host output from approximately `13*N` doubles
   (potential, forces, virial) to `3*N` GROMACS reals plus ten doubles.
8. Preserved the original GPUMD source tree without local patches.

The remaining difference from native GPUMD is mainly architectural: the current
GROMACS force-provider interface supplies host coordinates and consumes host
forces, while GROMACS GPU update keeps its own private device buffers. Removing
that final round trip safely would require a broader device-aware force-provider
API rather than a NEP-only shortcut.

## Build

From the source root:

```bash
./build-nep.sh
```

Useful environment overrides:

```bash
CLEAN_BUILD=1 BUILD_JOBS=16 CUDA_ARCH=89 ./build-nep.sh
DO_INSTALL=1 INSTALL_PREFIX=/opt/gromacs-nep ./build-nep.sh
```

The generated executable is `build-nep/bin/gmx` by default. The build script
also runs `NNPotAppliedForcesUnitTest` unless `RUN_TESTS=0` is set.

## Minimal MDP configuration

```text
nnpot-active         = yes
nnpot-modelfile      = nep.txt
nnpot-input-group    = System
nnpot-embedding      = mechanical
```

For the tested single-GPU full-system case, the fastest stable launch was:

```bash
build-nep/bin/gmx mdrun -s system.tpr -ntmpi 1 -ntomp 1
```

More OpenMP threads did not improve this GPU-bound workload. Keep GROMACS task
offload on automatic selection; forcing update/non-bonded/PME to CPU was slower.

## Validation procedure

The reproducible workflow is:

```bash
scripts/nep-validation/run_hydrate_validation.sh \
    build-nep \
    /path/to/native/gpumd \
    /path/to/test-inputs \
    /path/to/validation-work
```

It performs:

- a native-GPUMD versus GROMACS single-point energy, all-force, and virial check;
- 20,000-step NVE and NVT runs;
- a 50,000-step NPT run;
- energy-drift, thermostat/barostat, finite-coordinate, displacement, and bond
  distribution checks;
- separate GROMACS and native-GPUMD performance logging.

The validation generator was corrected so each MDP parameter is defined once.
For independently thermostatted GROMACS and GPUMD trajectories, each mean
temperature is checked against the target; their finite-window difference is
reported but is not used as a correctness criterion because the random streams
and thermostat implementations differ.

## Results

### Single point against native GPUMD

- Energy absolute error: `2.02e-5 meV/atom`
- Force RMSE: `6.01e-6 eV/A`
- Maximum force absolute error: `1.62e-4 eV/A`
- Relative force RMSE: `9.78e-6`
- Maximum virial absolute error: `0.01555 kJ/mol`

### Dynamics

- GROMACS NVE drift: `0.000723 meV/(atom ps)`
- GROMACS NVE total-energy span: `0.08366 meV/atom`
- NPT temperature means: GROMACS `277.68 K`, GPUMD `276.67 K`
- NPT pressure-mean difference: `431.96 bar`
- NPT final-volume relative difference: `0.954%`
- NVE/NVT/NPT coordinates remained finite and O-H/C-H bond distributions
  remained within the validation limits.

Full machine-readable results are in:

`/root/autodl-tmp/test/validation-perfopt/validation-report.json`

### Current performance, 8010 atoms, 20000 steps

Retested on 2026-08-12 after the device-coordinate and device-force integration.
Both implementations used the same coordinates, NEP model, 0.5 fs timestep,
NVE ensemble, one RTX 4090 D, and no trajectory or periodic thermodynamic
output. Each implementation was warmed up first. Five production runs were
performed in alternating order to reduce clock and thermal bias.

| Implementation | Mean time/step | Mean throughput | Standard deviation | Relative to native GPUMD |
|---|---:|---:|---:|---:|
| Native GPUMD | 1.466 ms | 5.465 Matom-step/s | 0.031 Matom-step/s | 100% |
| Current GROMACS-NEP | 1.492 ms | 5.368 Matom-step/s | 0.013 Matom-step/s | **98.22%** |

Individual GROMACS measurements were `5.382`, `5.357`, `5.373`, `5.351`, and
`5.376 Matom-step/s`. Native GPUMD measurements were `5.488`, `5.472`, `5.410`,
`5.480`, and `5.475 Matom-step/s`. Thus the current adapter is only about
`1.78%` slower than native GPUMD for this full-system benchmark.

The previously reported `82.85%` result was measured before the final direct
GROMACS device-coordinate/device-force path and is retained only as historical
context: the earlier optimized adapter copied coordinates and forces through
host memory and achieved `4.449 Matom-step/s`.

Machine-readable benchmark data are in:
`/root/autodl-tmp/test/nep-benchmark-current-20260812/result.json`.

## Important files

- `src/gromacs/applied_forces/nnpot/nepmodel.cu`: GPUMD NEP adapter and CUDA
  transfer/reduction path.
- `src/gromacs/applied_forces/nnpot/nnpotoptions.cpp`: topology handling,
  including the scalable full-system NEP path.
- `src/external/gpumd`: unmodified GPUMD source tree.
- `scripts/nep-validation`: end-to-end validation and analysis.
- `build-nep.sh`: one-command build script.

## 2026-08-12 NEP-MM device-path update

The single-GPU direct path now supports link-free partial NEP regions. A cached
local-index map gathers only NEP atoms from the GROMACS coordinate buffer on the
GPU. NEP forces are scattered into a zeroed full-system device buffer and added
by the existing GROMACS GPU force reduction, while MM forces, PME, constraints,
and the remaining GROMACS machinery stay active.

The host fallback remains available with `GMX_NEP_DISABLE_DEVICE_PATH=1` and is
used automatically for link atoms or domain decomposition. The five-atom
methane-in-MM-water smoke test gave identical NEP energy and a NEP-region force
RMSE of `2.6e-7 kJ mol^-1 nm^-1` between device and host paths. A three-run
post-change full-system benchmark measured `5.402`, `5.376`, and `5.307`
Matom-step/s (mean `5.362`), confirming that the generalized gather/scatter
kernel did not regress the contiguous full-system path.

## 2026-08-12 qNEP and long-range update

- Added native support for `nep4[_zbl]_charge1` and
  `nep4[_zbl]_charge2`, compiling the unmodified GPUMD qNEP, PPPM, and Ewald
  sources.
- Added `nnpot-use-gromacs-electrostatics`. For a full-system plain residual
  NEP it preserves topology charges and GROMACS PME/Coulomb while disabling
  classical LJ.
- Native qNEP plus this option is rejected because dynamic qNEP charges require
  coordinate-response chain-rule forces and would otherwise double count
  electrostatics.

Against native GPUMD on the 63-atom water fixtures:

| model | energy error | force RMSE | maximum force error |
|---|---:|---:|---:|
| qNEP charge1 | `2.96e-6 eV` | `8.31e-6 eV/A` | `4.01e-5 eV/A` |
| qNEP charge2 | `2.08e-6 eV` | `7.71e-6 eV/A` | `3.60e-5 eV/A` |

A fixed-charge residual-NEP smoke test confirmed that GROMACS GPU PME remains
active and contributes separate Coulomb short-range and reciprocal energies.

## 2026-08-12 energy-zero calibration and adaptive A/B coupling

Added `nnpot-energy-offsets` for constant per-element energy-zero calibration.
The correction changes reported NEP energy only and leaves forces/virial
unchanged. On the 8010-atom hydrate fixture, offsets `H:1 C:2 O:3 kJ/mol`
shifted the reported NEP energy by exactly `-12510 kJ/mol`, matching the
composition count.

Added two single-rank adaptive modes:

- `cross-only`: A-A and B-B remain MM; nearby complete B molecules receive the
  strict delta interaction `DeltaE(A+B)-DeltaE(A)-DeltaE(B)`.
- `a-nep-cross`: A internal interactions use the existing static NEP path; B-B
  remains MM and the same adaptive A-B delta correction is added.

B selections are expanded to complete topology molecules. A quintic switching
function and its force/virial derivative are applied between user-defined inner
and outer cutoffs. Adaptive models must be plain delta-NEP models trained on
reference-minus-MM labels; total-energy NEP and qNEP adaptive models are
rejected to prevent double counting and discontinuous dynamic charges.

Smoke tests using one methane A molecule and one complete water B molecule
completed for both modes. These tests validate integration and topology
behavior, not physical accuracy, because the available total-energy NEP was
only used as a placeholder delta model. Static NEP-MM regression remained
unchanged with a device/host force RMSE of `1.29e-7 kJ mol^-1 nm^-1`.
