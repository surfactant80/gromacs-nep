# NEP-MM methane-in-water smoke test

This case treats one complete methane molecule (global atoms 6211-6215) with
GPUMD NEP and leaves the other 8005 atoms to the GROMACS classical force field.
The coupling is **mechanical embedding**:

- GROMACS excludes classical non-bonded interactions only within the NEP group.
- NEP supplies the intraregion energy, forces, and virial.
- GROMACS retains NEP-MM and MM-MM Lennard-Jones/electrostatic interactions,
  PME, constraints, integration, and output.

The bundled topology is intentionally a smoke-test fixture. Use a physically
validated MM force field and a NEP model trained for the selected region in
production.

Run after `./build-nep.sh`:

```bash
MODEL=/absolute/path/to/nep.txt examples/nep-mm-methane-water/run-test.sh
```

The script runs the optimized device path and the host fallback, then compares
NEP energy and the 15 force components of the five-atom NEP methane. Set
`GMX_NEP_DISABLE_DEVICE_PATH=1` for debugging or reference calculations.

The optimized path requires one PP rank and no link atoms. Domain decomposition
and link-atom boundaries continue to use the established host fallback.
