#!/usr/bin/env python3
"""Compare GROMACS NEP energies/forces with GPUMD output.

Usage: compare.py --gmx-energy kJmol --gmx-force force.xvg --gpumd gpumd.xyz
"""
import argparse, math, sys

EV_TO_KJMOL = 96.48533212331002
FORCE_SCALE = EV_TO_KJMOL * 10.0  # kJ/mol/nm per eV/A

def parse_properties(comment):
    import re
    m = re.search(r'Properties=([^ ]+)', comment)
    names = []
    if m:
        parts = m.group(1).split(':')
        for k in range(0, len(parts) - 2, 3):
            name, _kind, count = parts[k], parts[k+1], int(parts[k+2])
            names.extend([name] * count)
    return names

def read_gpumd_last(path):
    lines = open(path).read().splitlines()
    i = 0; frames = []
    while i < len(lines):
        if not lines[i].strip(): i += 1; continue
        n = int(lines[i].split()[0]); comment = lines[i+1]
        props = parse_properties(comment)
        atoms = []
        for j in range(n):
            f = lines[i+2+j].split()
            species = f[0]
            values = [float(v) for v in f[1:]]
            entry = {'species': species}
            valueProps = [name for name in props if name != 'species']
            for k, name in enumerate(valueProps):
                entry.setdefault(name, []).append(values[k] if k < len(values) else 0.0)
            atoms.append(entry)
        frames.append((comment, atoms)); i = 2 + n
    return frames[-1], props

def parse_energy(comment):
    for tok in comment.split():
        if tok.startswith('energy='):
            return float(tok.split('=')[1])
    return None

def read_gmx_force(path):
    row = None
    for line in open(path):
        s = line.strip()
        if not s or s[0] in '@#': continue
        vals = [float(v) for v in s.split()]
        if len(vals) > 1: row = vals
    return row[1:]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gmx-energy', type=float, required=True)
    ap.add_argument('--gmx-force', required=True)
    ap.add_argument('--gpumd', required=True)
    ap.add_argument('--gmx-charge')
    a = ap.parse_args()
    (comment, atoms), props = read_gpumd_last(a.gpumd)
    e_gpumd = parse_energy(comment)
    e_gmx = a.gmx_energy / EV_TO_KJMOL
    n = len(atoms)
    fg = read_gmx_force(a.gmx_force)
    assert len(fg) == 3*n, (len(fg), 3*n)
    f_gmx = [v / FORCE_SCALE for v in fg]
    f_gpumd = [c for atom in atoms for c in atom['forces']]
    diffs = [x - y for x, y in zip(f_gmx, f_gpumd)]
    max_abs = max(abs(d) for d in diffs)
    rms = math.sqrt(sum(d*d for d in diffs) / len(diffs))
    ref_rms = math.sqrt(sum(x*x for x in f_gpumd) / len(f_gpumd))
    # per-atom vector norm errors
    atom_err = []
    for k in range(n):
        dv = diffs[3*k:3*k+3]
        atom_err.append(math.sqrt(sum(d*d for d in dv)))
    print(f"atoms                : {n}")
    print(f"E(GPUMD)             : {e_gpumd:.9f} eV")
    print(f"E(GROMACS)           : {e_gmx:.9f} eV  ({a.gmx_energy:.6f} kJ/mol)")
    print(f"dE                   : {e_gmx - e_gpumd:.9e} eV  ({(e_gmx-e_gpumd)/n:.3e} eV/atom)")
    print(f"F max|dF|            : {max_abs:.6e} eV/A")
    print(f"F RMS|dF|            : {rms:.6e} eV/A")
    print(f"F RMS(ref)           : {ref_rms:.6e} eV/A")
    print(f"F relative RMS       : {100*rms/ref_rms:.4f} %")
    print(f"F max atom |dF|      : {max(atom_err):.6e} eV/A")
    print(f"|dF| > 1e-3 atoms    : {sum(1 for e in atom_err if e > 1e-3)}")
    if a.gmx_charge is not None and 'charge' in props:
        qg = [float(v) for v in open(a.gmx_charge).read().splitlines()[-1].split()[1:]]
        qp = [atom['charge'][0] for atom in atoms]
        if len(qg) == len(qp):
            qd = [x - y for x, y in zip(qg, qp)]
            print(f"Q max|dq| (latent)   : {max(abs(d) for d in qd):.6e} e")
            print(f"Q RMS|dq|            : {math.sqrt(sum(d*d for d in qd)/len(qd)):.6e} e")

if __name__ == '__main__':
    main()
