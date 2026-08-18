#!/usr/bin/env python3
"""Fit per-element NEP energy offsets from a CSV table.

Required columns: reference_energy, nep_energy, and one integer count column per
atomic number, named count_Z (for example count_1 for H, count_6 for C).
The fitted convention is nep - reference = sum_Z count_Z * offset_Z.
"""
import argparse, csv, sys
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('csv_file')
    ap.add_argument('--output', default='nnpot-energy-offsets.txt')
    args=ap.parse_args()
    rows=list(csv.DictReader(Path(args.csv_file).open(newline='')))
    if not rows: raise SystemExit('empty CSV')
    zcols=sorted([k for k in rows[0] if k.startswith('count_')], key=lambda x:int(x[6:]))
    if not zcols: raise SystemExit('need count_Z columns')
    A=[[float(r.get(z,0.0)) for z in zcols] for r in rows]
    y=[float(r['nep_energy'])-float(r['reference_energy']) for r in rows]
    # Normal equations with Gaussian elimination; this tool is for small offset fits.
    n=len(zcols); M=[[sum(A[k][i]*A[k][j] for k in range(len(rows))) for j in range(n)] +
                       [sum(A[k][i]*y[k] for k in range(len(rows)))] for i in range(n)]
    for col in range(n):
        pivot=max(range(col,n), key=lambda r:abs(M[r][col]))
        if abs(M[pivot][col]) < 1e-14: raise SystemExit('rank-deficient composition table')
        M[col],M[pivot]=M[pivot],M[col]
        scale=M[col][col]; M[col]=[x/scale for x in M[col]]
        for r in range(n):
            if r==col: continue
            scale=M[r][col]
            M[r]=[a-scale*b for a,b in zip(M[r],M[col])]
    x=[M[i][-1] for i in range(n)]
    residual=[sum(A[k][i]*x[i] for i in range(n))-y[k] for k in range(len(rows))]
    rms=(sum(v*v for v in residual)/len(residual))**0.5
    with Path(args.output).open('w') as f:
        for z,v in zip(zcols,x): f.write(f'{z[6:]} {v:.16g}\n')
    print('nnpot-energy-offsets entries (atomic-number:value, kJ/mol/atom):')
    print(' '.join(f'{z[6:]}:{v:.16g}' for z,v in zip(zcols,x)))
    print(f'fit_rms_kj_mol={rms:.9g}')
if __name__=='__main__': main()
