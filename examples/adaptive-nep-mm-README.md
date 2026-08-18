# Adaptive NEP/MM configuration template

Define disjoint index groups A and B. B is expanded to complete molecules.

```text
nnpot-active                    = yes
nnpot-modelfile                 = internal-a-nep.txt
nnpot-input-group               = A
nnpot-embedding                 = mechanical
nnpot-adaptive-mode             = cross-only
nnpot-adaptive-group-b          = B
nnpot-adaptive-modelfile        = delta-ab-nep.txt
nnpot-adaptive-model-is-delta   = yes
nnpot-adaptive-inner-cutoff     = 0.4
nnpot-adaptive-outer-cutoff     = 0.6
```

Use `cross-only` when A-A and B-B must remain MM. Use `a-nep-cross` when A-A
must use `nnpot-modelfile`, B-B remains MM, and A-B receives the adaptive delta
correction. The first implementation supports one PP rank only.
