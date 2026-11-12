#!/usr/bin/env bash
# GPU correctness validation: GROMACS+NEP vs GPUMD itself.
#
#   GMX=<build-nep/bin/gmx> \
#   GPUMD=<gpumd executable> \
#   TEST_DIR=<workspace>/test \
#   ./run_gpu_validation.sh
#
# Requires a visible CUDA GPU. Uses test/test2 (CH4, 900 atoms) and
# test/test3 (water, 1500 atoms) from the project test directory.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
GMX=${GMX:-"$HERE/../../../build-nep/bin/gmx"}
GPUMD=${GPUMD:-"$HERE/../../../src/external/gpumd/src/gpumd"}
TEST_DIR=${TEST_DIR:-"$HERE/../../../../test"}
WORK=${WORK:-"$HERE/work"}
mkdir -p "$WORK"

fail=0
run() { "$@" >/dev/null 2>&1 || { echo "FAILED: $*"; fail=1; }; }

# ---------- helpers ----------
make_xyz() { python3 "$HERE/tools.py" gro2xyz "$1" "$2"; }
last_energy() { tail -1 "$1" | awk '{print $2}'; }

single_point_gpumd() { # dir model run.in
    ( cd "$1" && timeout 300 "$GPUMD" > gpumd.log 2>&1 )
}

# ---------- case 1: plain NEP, CH4 ----------
case_plain() {
    local d="$WORK/plain"; mkdir -p "$d"
    cp "$TEST_DIR/test2/system.gro" "$TEST_DIR/test2/system.top" "$TEST_DIR/test2/nep.txt" "$d/"
    ( cd "$d" && make_xyz system.gro model.xyz
      cat > run.in <<'RUN'
potential    nep.txt
velocity     300
ensemble     nve
time_step    0.000001
dump_xyz     1 gpumd.xyz force potential precision double
run          1
RUN
      single_point_gpumd "$d"
      cp "$TEST_DIR/test2/single-test/single.mdp" .
      "$GMX" grompp -f single.mdp -c system.gro -p system.top -o single.tpr -maxwarn 1
      "$GMX" mdrun -s single.tpr -rerun system.gro -ntmpi 1 -ntomp 1 -deffnm rerun
      "$GMX" energy -f rerun.edr -o nnpot.xvg -xvg none <<< "3" >/dev/null 2>&1
      printf '0\n' | "$GMX" traj -f rerun.trr -s single.tpr -of gmx-force.xvg >/dev/null 2>&1 )
    local e=$(last_energy "$d/nnpot.xvg")
    echo "== plain NEP (CH4, 900 atoms)"
    python3 "$HERE/compare.py" --gmx-energy "$e" --gmx-force "$d/gmx-force.xvg" --gpumd "$d/gpumd.xyz"
}

# ---------- case 2: qNEP native, water ----------
case_qnep() { # model tag
    local model=$1 tag=$2
    local d="$WORK/$tag"; mkdir -p "$d"
    cp "$TEST_DIR/test3/system.gro" "$TEST_DIR/test3/system.top" "$TEST_DIR/test3/$model" "$d/"
    ( cd "$d" && make_xyz system.gro model.xyz
      cat > run.in <<RUN
potential    $model
velocity     300
ensemble     nve
time_step    0.000001
dump_xyz     1 gpumd.xyz force potential charge precision double
run          1
RUN
      single_point_gpumd "$d"
      sed -e 's/nsteps = 200000/nsteps = 0/' -e 's/gen-vel = yes/gen-vel = no/' \
          -e 's/nstfout = 0/nstfout = 1/' -e 's/nstenergy = 1000/nstenergy = 1/' \
          -e 's/nstcalcenergy = 1000/nstcalcenergy = 1/' \
          -e "s/qnep_model1.txt/$model/" "$TEST_DIR/test3/nvt-native.mdp" > single.mdp
      "$GMX" grompp -f single.mdp -c system.gro -p system.top -o single.tpr -maxwarn 1
      "$GMX" mdrun -s single.tpr -rerun system.gro -ntmpi 1 -ntomp 1 -deffnm rerun
      "$GMX" energy -f rerun.edr -o nnpot.xvg -xvg none <<< "3" >/dev/null 2>&1
      printf '0\n' | "$GMX" traj -f rerun.trr -s single.tpr -of gmx-force.xvg >/dev/null 2>&1 )
    local e=$(last_energy "$d/nnpot.xvg")
    echo "== $tag (water, 1500 atoms)"
    python3 "$HERE/compare.py" --gmx-energy "$e" --gmx-force "$d/gmx-force.xvg" --gpumd "$d/gpumd.xyz" || true
}

echo "GMX   = $GMX"
echo "GPUMD = $GPUMD"
echo "tests = $TEST_DIR"
case_plain || fail=1
case_qnep qnep_model1.txt qnep-charge1 || fail=1
case_qnep qnep_model2.txt qnep-charge2 || fail=1

echo
echo "Notes: force files are .trr dumps; compare.py converts kJ/mol/nm <-> eV/A."
echo "For qNEP-GROMACS (PME) and finite-difference see docs/GPU-TESTS-2026.4.md."
[[ $fail -eq 0 ]] && echo "GPU validation completed." || echo "GPU validation had failures."
exit $fail
