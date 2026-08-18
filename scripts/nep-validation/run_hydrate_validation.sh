#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
    echo "Usage: $0 <gromacs-build-dir> <gpumd-executable> <input-dir> [work-dir]" >&2
    exit 2
fi

GMX_BUILD=$(realpath "$1")
GPUMD=$(realpath "$2")
INPUT=$(realpath "$3")
WORK=${4:-"$PWD/nep-validation-work"}
WORK=$(mkdir -p "$WORK" && realpath "$WORK")
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
GMX="$GMX_BUILD/bin/gmx"

for file in em.gro CH4.top CH4.itp SOL.itp nep.txt model.xyz; do
    test -f "$INPUT/$file" || { echo "Missing input: $INPUT/$file" >&2; exit 2; }
done
test -x "$GMX" || { echo "Missing GROMACS executable: $GMX" >&2; exit 2; }
test -x "$GPUMD" || { echo "Missing GPUMD executable: $GPUMD" >&2; exit 2; }

rm -rf "$WORK"
mkdir -p "$WORK"
cp "$INPUT"/{em.gro,CH4.top,CH4.itp,SOL.itp,nep.txt,model.xyz} "$WORK/"
cd "$WORK"

cat > common.mdp <<'EOF'
integrator               = md
dt                       = 0.0005
comm-mode                = Linear
nstcomm                  = 1000
cutoff-scheme            = Verlet
nstlist                  = 100
rlist                    = 0.8
coulombtype              = Cut-off
rcoulomb                 = 0.8
vdwtype                  = Cut-off
rvdw                     = 0.8
pbc                      = xyz
periodic-molecules       = yes
constraints              = none
nstxout                  = 0
nstvout                  = 0
nstxout-compressed       = 0
nstlog                   = 1000
nnpot-active             = yes
nnpot-modelfile          = nep.txt
nnpot-input-group        = System
nnpot-embedding          = mechanical
EOF

cat common.mdp > nve.mdp
cat >> nve.mdp <<'EOF'
nsteps                   = 20000
continuation             = no
gen-vel                  = yes
gen-temp                 = 277
gen-seed                 = 20260812
nstfout                  = 0
nstcalcenergy            = 100
nstenergy                = 100
tcoupl                   = no
pcoupl                   = no
EOF

cat common.mdp > nvt.mdp
cat >> nvt.mdp <<'EOF'
nsteps                   = 20000
continuation             = no
gen-vel                  = yes
gen-temp                 = 277
gen-seed                 = 20260812
nstfout                  = 0
nstcalcenergy            = 100
nstenergy                = 100
tcoupl                   = V-rescale
tc-grps                  = System
tau-t                    = 2.0
ref-t                    = 277
nsttcouple               = 1
pcoupl                   = no
EOF

cat common.mdp > npt.mdp
cat >> npt.mdp <<'EOF'
nsteps                   = 50000
continuation             = no
gen-vel                  = yes
gen-temp                 = 277
gen-seed                 = 20260812
nstfout                  = 0
nstcalcenergy            = 100
nstenergy                = 100
tcoupl                   = V-rescale
tc-grps                  = System
tau-t                    = 2.0
ref-t                    = 277
nsttcouple               = 1
pcoupl                   = C-rescale
pcoupltype               = Isotropic
tau-p                    = 200
compressibility          = 4.5e-5
ref-p                    = 70
nstpcouple               = 1
EOF

cat common.mdp > single.mdp
cat >> single.mdp <<'EOF'
nsteps                   = 0
continuation             = yes
gen-vel                  = no
tcoupl                   = no
pcoupl                   = no
nstcalcenergy            = 1
nstenergy                = 1
nstfout                  = 1
EOF

mkdir single
cp nep.txt single/
"$GMX" grompp -f single.mdp -c em.gro -p CH4.top -o single/system.tpr \
    -po single/mdout.mdp > single/grompp.stdout 2>&1
(cd single && "$GMX" mdrun -s system.tpr -deffnm run -ntmpi 1 -ntomp 1 \
    > mdrun.stdout 2>&1)
"$GMX" dump -f single/run.trr > single/gmx.dump 2>/dev/null
printf '3 9 10 11 12 13 14 15 16 17\n0\n' |
    "$GMX" energy -f single/run.edr -o single/gmx-energy.xvg \
    > single/gmx-energy.stdout 2>&1
python3 "$SCRIPT_DIR/analyze_hydrate.py" prepare-single model.xyz single/model.xyz
cat > single/run.in <<'EOF'
potential nep.txt
velocity 1
ensemble nve
time_step 0
dump_xyz 1 gpumd.xyz force potential virial precision double
run 1
EOF
(cd single && "$GPUMD" > gpumd.stdout 2>&1)

for ensemble in nve nvt npt; do
    mkdir "gmx_${ensemble}" "gpumd_${ensemble}"
    cp nep.txt "gmx_${ensemble}/"
    cp model.xyz nep.txt "gpumd_${ensemble}/"
    "$GMX" grompp -f "${ensemble}.mdp" -c em.gro -p CH4.top \
        -o "gmx_${ensemble}/system.tpr" -po "gmx_${ensemble}/mdout.mdp" \
        > "gmx_${ensemble}/grompp.stdout" 2>&1
    (cd "gmx_${ensemble}" && "$GMX" mdrun -s system.tpr -deffnm run \
        -ntmpi 1 -ntomp 1 > mdrun.stdout 2>&1)
done

cat > gpumd_nve.in <<'EOF'
potential nep.txt
velocity 277
time_step 0.5
ensemble nve
dump_thermo 100
dump_xyz 20000 final.xyz precision double
run 20000
EOF
cat > gpumd_nvt.in <<'EOF'
potential nep.txt
velocity 277
time_step 0.5
ensemble nvt_bdp 277 277 4000
dump_thermo 100
dump_xyz 20000 final.xyz precision double
run 20000
EOF
cat > gpumd_npt.in <<'EOF'
potential nep.txt
velocity 277
time_step 0.5
ensemble npt_scr 277 277 4000 0.007 2.22222 400000
dump_thermo 100
dump_xyz 50000 final.xyz precision double
run 50000
EOF

for ensemble in nve nvt npt; do
    cp "gpumd_${ensemble}.in" "gpumd_${ensemble}/run.in"
    (cd "gpumd_${ensemble}" && "$GPUMD" > gpumd.stdout 2>&1)
done

printf '4 5 6 7 8\n0\n' |
    "$GMX" energy -f gmx_nve/run.edr -o gmx_nve/energy.xvg \
    > gmx_nve/energy.stdout 2>&1
printf '4 5 6 7 8 9\n0\n' |
    "$GMX" energy -f gmx_nvt/run.edr -o gmx_nvt/energy.xvg \
    > gmx_nvt/energy.stdout 2>&1
printf '4 5 6 7 8 9 10 11 12 13\n0\n' |
    "$GMX" energy -f gmx_npt/run.edr -o gmx_npt/energy.xvg \
    > gmx_npt/energy.stdout 2>&1

python3 "$SCRIPT_DIR/analyze_hydrate.py" analyze "$WORK"
echo "Validation passed. Report: $WORK/validation-report.json"
