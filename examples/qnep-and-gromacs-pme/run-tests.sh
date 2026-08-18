#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
GMX=${GMX:-"$HERE/../../build-nep/bin/gmx"}
QNEP_MODEL=${QNEP_MODEL:-"$HERE/native-qnep-water63/qnep.txt"}
RESIDUAL_MODEL=${RESIDUAL_MODEL:-}

run_case() {
    local source=$1 work=$2 model=$3 model_name=$4
    rm -rf "$work"; mkdir -p "$work"
    cp "$source/conf.g96" "$source/system.itp" "$source/topol.top" "$source/run.mdp" "$work/"
    cp "$model" "$work/$model_name"
    (cd "$work" && "$GMX" grompp -f run.mdp -c conf.g96 -p topol.top -o run.tpr -po mdout.mdp -maxwarn 3 \
        && "$GMX" mdrun -s run.tpr -deffnm run -ntmpi 1 -ntomp 1)
}

run_case "$HERE/native-qnep-water63" "$HERE/work-native-qnep" "$QNEP_MODEL" qnep.txt

if [[ -n "$RESIDUAL_MODEL" ]]; then
    run_case "$HERE/fixed-charge-residual-nep" "$HERE/work-residual-pme" "$RESIDUAL_MODEL" nep.txt
else
    echo "Skipped residual-NEP + GROMACS PME case. Set RESIDUAL_MODEL to a plain NEP"
    echo "trained on energies/forces after subtracting the same fixed-charge electrostatics."
fi
