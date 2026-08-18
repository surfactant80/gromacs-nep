# GPUMD NEP backend

This tree adds a CUDA NEP backend to the existing GROMACS `nnpot` MD module.
The GROMACS integrator, constraints, thermostats, barostats, PME and native
GPU non-bonded kernels remain unchanged.

## Model and units

The model file is a GPUMD `nep4`, `nep4_zbl`, `nep5` or `nep5_zbl` text model.
GROMACS coordinates are converted from nm to Angstrom before NEP evaluation.
NEP energies and forces are converted from eV/eV-Angstrom to kJ/mol and
kJ/mol/nm on return.

## NNP/MM

Use the existing `nnpot` input group to select the NEP region. The remaining
atoms stay in the normal GROMACS force field and interact with the NEP region
through the ordinary GROMACS MM terms. Link atoms are supported using the
existing GROMACS frontier-atom force redistribution.

The current NEP backend implements mechanical embedding. The
`electrostatic-model` option is intentionally rejected because a standard
GPUMD NEP model does not accept external MM charges.

## MDP example

```text
nnpot-active         = yes
nnpot-modelfile      = nep.txt
nnpot-input-group    = NEP
nnpot-embedding      = mechanical
nnpot-link-type      = H
nnpot-link-distance  = 0.1
```

No `nnpot-model-input*` values are required for NEP; Cartesian positions are
selected automatically. The `NEP` group must be present in the index file.

## GPU behavior

`GMX_NNPOT=NEP` requires a CUDA GROMACS build. The complete, unmodified GPUMD
repository is stored at `src/external/gpumd`. The adapter compiles only the
original GPUMD source files needed by the `NEP` evaluator. To update GPUMD,
replace that directory with a new GPUMD source tree and rebuild GROMACS; no
GPUMD source patch is required.

Native GROMACS GPU acceleration remains available for the classical part of
the system. Because the current GROMACS force-provider API hands the external
provider host-side coordinates and forces, each NEP step contains host/device
transfers around the NEP call; the classical GROMACS GPU path is not disabled.

When configured with `GMX_NNPOT=OFF`, no GPUMD file is compiled or linked and
the build follows the original GROMACS NNPot stub path.

## Validation

The reproducible hydrate validation is in `scripts/nep-validation`. It compares
the complete atomic force array, total energy and virial tensor against a
native GPUMD executable, then runs matched NVE, NVT and NPT simulations and
checks energy conservation, thermostat/barostat response, finite trajectories,
displacement scales and intramolecular distance distributions.

For a full-system NEP calculation, the topology preprocessor disables
classical charges and non-bonded atom-type parameters directly instead of
constructing a global intermolecular exclusion group. The latter scales
quadratically and can dominate the GROMACS neighbor search for large ML-only
systems. Partial NNP/MM systems retain the original exclusion-group path.

## Optimized adapter path (2026-08-12)

The adapter caches the static species mapping, converts coordinate layout and
units on CUDA, reduces energy/virial on CUDA, and returns only native-layout
GROMACS forces plus ten scalar values through page-locked staging buffers. On
the 8010-atom hydrate benchmark this increased throughput from 3.326 to 4.449
Matom-step/s (three-run mean), or 82.85% of native GPUMD. See
`NEP-INTEGRATION-REPORT.md` for the complete procedure and results.

## NEP-MM mechanical embedding and GPU fast path

For a partial `nnpot-input-group`, GROMACS keeps all classical interactions
except NEP-NEP non-bonded terms, which are replaced by NEP. Thus NEP-MM and
MM-MM electrostatics/Lennard-Jones, PME, bonded terms crossing the boundary,
constraints, thermostats, and barostats remain GROMACS responsibilities.

On a single rank with no link atoms, the adapter now gathers the selected NEP
coordinates directly from the GROMACS device coordinate buffer. It evaluates
GPUMD NEP and scatters the resulting forces into a zeroed full-size device
force buffer, which the normal GROMACS GPU force reduction adds to classical
forces. This removes the per-step coordinate D2H, NEP coordinate H2D, NEP force
D2H, and force H2D round trips for link-free NEP-MM simulations.

Set `GMX_NEP_DISABLE_DEVICE_PATH=1` to force the portable host path. Link-atom
boundaries and domain decomposition intentionally remain on that path until a
safe device implementation of link-force redistribution and distributed NEP
inference is available.

A runnable example is in `examples/nep-mm-methane-water`.

## qNEP and long-range electrostatics

The adapter accepts GPUMD `nep4[_zbl]_charge1` and
`nep4[_zbl]_charge2` models. Native qNEP uses the unmodified GPUMD charge
implementation, including charge neutrality, PPPM/Ewald, Born effective
charges, and coordinate-dependent charge chain-rule forces.

A qNEP charge cannot be passed directly to ordinary GROMACS PME as if it were a
fixed topology charge. Since `q_i(r)` depends on coordinates, the complete
force contains `-(dE_elec/dq)(dq/dr)` in addition to the ordinary `qE` force.
Omitting it makes force and energy inconsistent. For this reason,
`nnpot-use-gromacs-electrostatics = yes` is rejected for native qNEP files.

For maximum GROMACS performance and parallel scalability, the recommended
production architecture is instead a fixed-charge long-range baseline plus a
plain short-range residual NEP. Train the NEP after subtracting the exact same
fixed-charge electrostatic energy and forces used by the GROMACS topology, then
set `nnpot-use-gromacs-electrostatics = yes`. GROMACS retains charges, GPU PME,
and Coulomb interactions while classical LJ is disabled for a full-system NEP.

See `examples/qnep-and-gromacs-pme`.

## Energy zero and adaptive A/B coupling

`nnpot-energy-offsets` applies constant per-element energy corrections to the
reported NEP energy. It does not modify forces or virial. Fit offsets with
`scripts/nep-tools/fit_energy_offsets.py` using composition-balanced reference
structures.

Adaptive modes are configured with:

```text
nnpot-adaptive-mode              = cross-only   ; or a-nep-cross
nnpot-adaptive-group-b           = B
nnpot-adaptive-modelfile         = delta-nep.txt
nnpot-adaptive-model-is-delta    = yes
nnpot-adaptive-inner-cutoff      = 0.4
nnpot-adaptive-outer-cutoff      = 0.6
```

`cross-only` keeps A-A and B-B as MM and adds a smooth, molecule-wise A-B
correction. `a-nep-cross` keeps A internal interactions on the static NEP path
and adds the same adaptive A-B correction. The B group is expanded to complete
molecules; a molecule enters the correction when its closest A-B atom distance
is within the outer cutoff. A quintic smoothstep is used between inner and outer
cutoffs, including the switching-force derivative.

The adaptive model must be a Delta-NEP trained with ordinary GPUMD tools from
reference-minus-MM energies/forces/virials. A total-energy NEP cannot be added
to an unchanged MM baseline without double counting; `grompp` rejects that
configuration.

The first implementation supports one PP rank and no domain decomposition. It
is intended for development and validation before distributed adaptive coupling.
