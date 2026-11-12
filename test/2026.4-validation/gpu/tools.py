#!/usr/bin/env python3
"""Small coordinate/format helpers for GROMACS <-> GPUMD comparisons."""
import sys

def read_gro(path):
    lines = open(path).read().splitlines()
    n = int(lines[1].strip())
    atoms = []
    for line in lines[2:2+n]:
        resnum = line[:5].strip(); resname = line[5:10].strip(); name = line[10:15].strip()
        idx = int(line[15:20]); x = float(line[20:28]); y = float(line[28:36]); z = float(line[36:44])
        atoms.append((resnum, resname, name, idx, x, y, z))
    box = [float(v) for v in lines[2+n].split()[:3]]
    return atoms, box

def element_from_name(name):
    return ''.join(c for c in name if c.isalpha())[:2].capitalize()

def read_xyz_last_frame(path):
    lines = open(path).read().splitlines()
    i = 0; frames = []
    while i < len(lines):
        if not lines[i].strip(): i += 1; continue
        n = int(lines[i].split()[0]); comment = lines[i+1]
        atoms = []
        for j in range(n):
            f = lines[i+2+j].split()
            atoms.append((f[0], [float(f[1]), float(f[2]), float(f[3])], f[4:]))
        frames.append((comment, atoms)); i += 2 + n
    return frames

def lattice_from_comment(comment):
    import re
    m = re.search(r'Lattice="([^"]+)"', comment)
    vals = [float(v) for v in m.group(1).split()] if m else []
    return vals

if __name__ == '__main__':
    cmd, inp, out = sys.argv[1], sys.argv[2], sys.argv[3]
    if cmd == 'gro2xyz':
        atoms, box = read_gro(inp)
        with open(out, 'w') as f:
            f.write(f"{len(atoms)}\n")
            lat = [box[0]*10, 0.0, 0.0, 0.0, box[1]*10, 0.0, 0.0, 0.0, box[2]*10]
            f.write('Lattice="' + ' '.join(f"{v:.6f}" for v in lat) + '" Properties=species:S:1:pos:R:3\n')
            for resnum, resname, name, idx, x, y, z in atoms:
                f.write(f"{element_from_name(name)} {x*10:.8f} {y*10:.8f} {z*10:.8f}\n")
    elif cmd == 'xyz2gro':
        template, xyz = inp, out
        t_atoms, t_box = read_gro(template)
        frames = read_xyz_last_frame(xyz)
        comment, atoms = frames[-1]
        box = lattice_from_comment(comment)
        bx = [box[0], box[4], box[8]] if len(box) == 9 else ([v*10 for v in t_box])
        with open(xyz + '.gro', 'w') as f:
            f.write('GPUMD single point\n')
            f.write(f"{len(atoms)}\n")
            for k, ((resnum, resname, name, idx, *_ ), (el, pos, _)) in enumerate(zip(t_atoms, atoms), 1):
                f.write(f"{resnum:>5}{resname:<5}{name:>5}{k:5d}{pos[0]/10:8.3f}{pos[1]/10:8.3f}{pos[2]/10:8.3f}\n")
            f.write(f"{bx[0]/10:10.5f}{bx[1]/10:10.5f}{bx[2]/10:10.5f}\n")
