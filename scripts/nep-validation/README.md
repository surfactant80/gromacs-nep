# GPUMD NEP validation

This directory contains the reproducible validation used for the vendored
GPUMD NEP backend.

## Required input files

The input directory must contain:

- `em.gro`
- `CH4.top`
- `CH4.itp`
- `SOL.itp`
- `nep.txt`
- `model.xyz`

The default analysis assumes the hydrate test system has 8010 atoms,
2070 water molecules, and 360 methane molecules.

## Run

```bash
scripts/nep-validation/run_hydrate_validation.sh \
    build-nep \
    /path/to/gpumd \
    /path/to/test-inputs \
    /path/to/validation-work
```

The script checks:

- single-point total energy, all atomic forces, and the full virial tensor;
- NVE total-energy conservation;
- NVT temperature and potential-energy statistics;
- NPT temperature, pressure, and volume response;
- finite trajectories, displacement scales, and O-H/C-H bond distributions;
- GROMACS and native GPUMD performance logs.

It writes `validation-report.json` and exits nonzero if a numerical or
trajectory threshold is violated.
