#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
GMX=${GMX:-"$HERE/../../build-nep/bin/gmx"}
MODEL=${MODEL:-"$HERE/nep.txt"}
WORK=${WORK:-"$HERE/work"}
mkdir -p "$WORK"
cp "$HERE/conf.gro" "$HERE/topol.top" "$HERE/CH4.itp" "$HERE/SOL.itp" \
   "$HERE/index.ndx" "$HERE/nep-mm.mdp" "$MODEL" "$WORK/"
cd "$WORK"
model_name=$(basename "$MODEL")
if [[ "$model_name" != nep.txt ]]; then mv "$model_name" nep.txt; fi
"$GMX" grompp -f nep-mm.mdp -c conf.gro -p topol.top -n index.ndx \
    -o nep-mm.tpr -po mdout.mdp -maxwarn 2
"$GMX" mdrun -s nep-mm.tpr -deffnm device -ntmpi 1 -ntomp 1 -nb gpu
GMX_NEP_DISABLE_DEVICE_PATH=1 "$GMX" mdrun -s nep-mm.tpr -deffnm host-reference \
    -ntmpi 1 -ntomp 1 -nb gpu
printf '1 2 3 4 5\n0\n' | "$GMX" energy -f device.edr -o device-energy.xvg >/dev/null
printf '1 2 3 4 5\n0\n' | "$GMX" energy -f host-reference.edr -o host-energy.xvg >/dev/null
printf 'System\n' | "$GMX" traj -f device.trr -s nep-mm.tpr -of device-force.xvg -fp >/dev/null
printf 'System\n' | "$GMX" traj -f host-reference.trr -s nep-mm.tpr -of host-force.xvg -fp >/dev/null
python3 - <<'PY'
from pathlib import Path
import math

def last_row(path):
    rows = [line for line in Path(path).read_text().splitlines()
            if line and line[0] not in '#@']
    return [float(value) for value in rows[-1].split()[1:]]

device_energy = last_row('device-energy.xvg')
host_energy = last_row('host-energy.xvg')
device_force = last_row('device-force.xvg')
host_force = last_row('host-force.xvg')
nep_force_delta = [a-b for a,b in zip(device_force[3*6210:3*6215],
                                      host_force[3*6210:3*6215])]
print('NEP energy device/host (kJ/mol):', device_energy[3], host_energy[3])
print('NEP-region force RMSE (kJ/mol/nm):',
      math.sqrt(sum(x*x for x in nep_force_delta)/len(nep_force_delta)))
if abs(device_energy[3] - host_energy[3]) > 1e-4:
    raise SystemExit('NEP energy mismatch')
if math.sqrt(sum(x*x for x in nep_force_delta)/len(nep_force_delta)) > 1e-3:
    raise SystemExit('NEP force mismatch')
PY
echo "NEP-MM device-path smoke test passed: $WORK"
