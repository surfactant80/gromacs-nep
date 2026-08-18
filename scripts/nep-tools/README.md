# NEP energy-zero calibration and Delta-NEP labels

## Energy zero

Prepare a CSV with `reference_energy`, `nep_energy`, and composition columns
such as `count_1`, `count_6`, `count_8`. Fit per-element offsets:

```bash
python3 scripts/nep-tools/fit_energy_offsets.py energies.csv
```

The printed string can be placed in the MDP file:

```text
nnpot-energy-offsets = 1:-0.12 6:3.4 8:-1.8
```

Values are kJ/mol per atom. This changes reported NEP energy only; it does not
change forces or virial. Use a composition-balanced set of reference structures.

## Delta-NEP labels

For adaptive cross interactions, generate training labels externally:

```text
E_delta = E_reference - E_GROMACS_MM
F_delta = F_reference - F_GROMACS_MM
V_delta = V_reference - V_GROMACS_MM
```

Write these labels into an ordinary GPUMD `train.xyz` file and train it with the
normal GPUMD `nep` executable. No GPUMD source modification or special trainer is
needed. The resulting `nep.txt` is a normal NEP file, but its energy/force
meaning is a correction to the exact MM baseline used in production.

Generate a normal GPUMD training file from matched reference and GROMACS-MM
extxyz files:

```bash
python3 scripts/nep-tools/make_delta_train.py reference.xyz mm.xyz train_delta.xyz
```

The inputs must have identical frames, atom ordering, property layout and units.
The output is a normal GPUMD extxyz file; train it with the standard `nep`
executable. No GPUMD source modification is required.
