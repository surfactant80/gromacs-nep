#!/usr/bin/env bash
# CPU-only validation for the GROMACS 2026.4 + GPUMD NEP + WPE integration.
#
# These checks do not need a GPU: they exercise model parsing, backend
# selection, grompp, the error paths, and a plain-MM mdrun smoke test.
#
#   GMX=/path/to/build-nep/bin/gmx ./run_cpu_validation.sh
#
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SOURCE=$(cd "$HERE/../.." && pwd)
GMX=${GMX:-"$SOURCE/build-nep/bin/gmx"}
WORK=${WORK:-"$HERE/work"}
WPE_LIB=${WPE_LIB:-}
PASS=0
FAIL=0

note() { printf '\n=== %s\n' "$*"; }
ok()   { PASS=$((PASS+1)); printf 'PASS: %s\n' "$*"; }
bad()  { FAIL=$((FAIL+1)); printf 'FAIL: %s\n' "$*"; }

run_case() { # name expected_exit -- command...
    local name=$1 expected=$2; shift 2
    "$@" >"$WORK/$name.log" 2>&1
    local rc=$?
    if [[ $rc -eq $expected ]]; then ok "$name (exit $rc)"; else bad "$name (exit $rc, expected $expected)"; fi
}

# NEP model paths in the .mdp are relative to the process working directory.
run_case_in() { # dir name expected_exit -- command...
    local dir=$1 name=$2 expected=$3; shift 3
    ( cd "$dir" && "$@" ) >"$WORK/$name.log" 2>&1
    local rc=$?
    if [[ $rc -eq $expected ]]; then ok "$name (exit $rc)"; else bad "$name (exit $rc, expected $expected)"; fi
}

expect_grep() { # name pattern file
    if grep -qE "$2" "$3"; then ok "$1"; else bad "$1 (pattern '$2' not found)"; fi
}

[[ -x "$GMX" ]] || { echo "gmx not found: $GMX" >&2; exit 2; }
rm -rf "$WORK"; mkdir -p "$WORK"

note "CLI smoke"
run_case version 0 "$GMX" --version
run_case help 0 "$GMX" help
run_case grompp-help 0 "$GMX" grompp -h
run_case mdrun-help 0 "$GMX" mdrun -h

EX="$SOURCE/examples/qnep-and-gromacs-pme/native-qnep-water63"
WPE_ARGS=()
if [[ -n "$WPE_LIB" ]]; then WPE_ARGS=("nnpot-wpe-library = $WPE_LIB"); fi

note "qNEP legacy model, native electrostatics"
mkdir -p "$WORK/qnep" && cp "$EX"/{conf.g96,topol.top,system.itp,qnep.txt,run.mdp} "$WORK/qnep/"
run_case_in "$WORK/qnep" qnep-grompp 0 "$GMX" grompp -f run.mdp -c conf.g96 \
    -p topol.top -o qnep.tpr -maxwarn 1
expect_grep qnep-parsed 'NEP model: qNEP .*ANN hidden = 30' "$WORK/qnep-grompp.log"

note "plain NEP model (legacy), WPE shape compatible"
mkdir -p "$WORK/nep" && cp "$SOURCE/examples/qnep-and-gromacs-pme/native-qnep-water63"/{conf.g96,topol.top,system.itp} "$WORK/nep/"
cp "$HERE/models/water-hcno-sp-nep4.txt" "$WORK/nep/nep.txt"
if [[ -s "$WORK/nep/nep.txt" ]]; then
    sed -e 's|nnpot-modelfile = qnep.txt|nnpot-modelfile = nep.txt|' \
        -e '/nnpot-gromacs-charges/d' "$EX/run.mdp" > "$WORK/nep/run.mdp"
    run_case_in "$WORK/nep" nep-grompp 0 "$GMX" grompp -f run.mdp -c conf.g96 \
        -p topol.top -o nep.tpr -maxwarn 1
    expect_grep nep-parsed 'NEP model: NEP4 .*l_max_3body = 4' "$WORK/nep-grompp.log"
fi

note "backend selection with an explicit WPE library"
if [[ -n "$WPE_LIB" ]]; then
    cp "$WORK/nep/run.mdp" "$WORK/nep/run-wpe.mdp"
    printf '\nnnpot-wpe-library = %s\n' "$WPE_LIB" >> "$WORK/nep/run-wpe.mdp"
    run_case_in "$WORK/nep" nep-wpe-select 0 "$GMX" grompp -f run-wpe.mdp -c conf.g96 \
        -p topol.top -o nep-wpe.tpr -maxwarn 1
    expect_grep wpe-selected 'Selected NEP backend: wpe backend' "$WORK/nep-wpe-select.log"
fi

note "error paths"
printf '\nnnpot-backend = wpe\nnnpot-wpe-library = /nonexistent\n' >> "$WORK/nep/run.mdp"
run_case_in "$WORK/nep" wpe-missing-lib 1 "$GMX" grompp -f run.mdp -c conf.g96 \
    -p topol.top -o nolib.tpr -maxwarn 1
expect_grep wpe-missing-lib-message 'libwisevolve_potential.so was not found' "$WORK/wpe-missing-lib.log"

note "grompp without any visible GPU"
if [[ -d "$WORK/qnep" ]]; then
    ( cd "$WORK/qnep" && CUDA_VISIBLE_DEVICES="" "$GMX" grompp -f run.mdp -c conf.g96 \
        -p topol.top -o no-gpu.tpr -maxwarn 1 ) >"$WORK/no-gpu-grompp.log" 2>&1
    if [[ $? -eq 0 && -f "$WORK/qnep/no-gpu.tpr" ]]; then
        ok "grompp with CUDA_VISIBLE_DEVICES empty"
    else
        bad "grompp with CUDA_VISIBLE_DEVICES empty"
    fi
    expect_grep no-gpu-tpr 'NEP model:' "$WORK/no-gpu-grompp.log"
fi

note "plain MM regression (no NEP)"
mkdir -p "$WORK/mm" && cp "$SOURCE/share/top/spc216.gro" "$WORK/mm/" && cp "$SOURCE/share/top/spc.itp" "$WORK/mm/" 2>/dev/null || true
cat > "$WORK/mm/topol.top" <<'TOP'
#include "oplsaa.ff/forcefield.itp"
#include "oplsaa.ff/spc.itp"
[ system ]
SPC water
[ molecules ]
SOL 216
TOP
cat > "$WORK/mm/mm.mdp" <<'MDP'
integrator = md
dt = 0.002
nsteps = 20
cutoff-scheme = Verlet
rlist = 0.9
coulombtype = PME
rcoulomb = 0.9
vdwtype = Cut-off
rvdw = 0.9
pme-order = 4
constraints = none
nstxout = 0
nstvout = 0
nstxout-compressed = 0
nstlog = 10
nstenergy = 10
tcoupl = no
pcoupl = no
gen-vel = yes
gen-temp = 300
MDP
run_case_in "$WORK/mm" mm-grompp 0 "$GMX" grompp -f mm.mdp -c spc216.gro \
    -p topol.top -o mm.tpr
run_case_in "$WORK/mm" mm-mdrun 0 "$GMX" mdrun -s mm.tpr -ntmpi 1 -ntomp 2 -nb cpu -pme cpu -deffnm mmrun

printf '\n%d passed, %d failed\n' "$PASS" "$FAIL"
[[ $FAIL -eq 0 ]]
