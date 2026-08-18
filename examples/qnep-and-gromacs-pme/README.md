# qNEP and GROMACS long-range electrostatics

Two physically distinct modes are supported.

## 1. Native qNEP

Headers `nep4_charge1`, `nep4_zbl_charge1`, `nep4_charge2`, and
`nep4_zbl_charge2` are accepted. The unmodified GPUMD qNEP implementation
computes dynamic charges, PPPM/Ewald electrostatics, and the required
charge-response chain-rule forces.

```text
nnpot-active       = yes
nnpot-modelfile    = qnep.txt
nnpot-input-group  = System
nnpot-embedding    = mechanical
```

## 2. Recommended GROMACS-PME decomposition

Use a **plain NEP**, trained as a short-range residual after subtracting a
chosen fixed-charge electrostatic model from every training energy and force.
Put exactly the same fixed charges in the GROMACS topology and enable:

```text
coulombtype                       = PME
nnpot-active                      = yes
nnpot-modelfile                   = residual-nep.txt
nnpot-input-group                 = System
nnpot-embedding                   = mechanical
nnpot-use-gromacs-electrostatics  = yes
```

This preserves topology charges and GROMACS PME but disables classical LJ for
the full-system NEP region. It is suitable only when the model was trained with
the same electrostatic baseline removed. It must not be enabled for native
qNEP files; `grompp` rejects that double-counted combination.

Run the native qNEP smoke test:

```bash
examples/qnep-and-gromacs-pme/run-tests.sh
```

Run both modes with a user-supplied residual model:

```bash
RESIDUAL_MODEL=/path/to/residual-nep.txt \
examples/qnep-and-gromacs-pme/run-tests.sh
```
