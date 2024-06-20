# MIT License
# 
# Copyright (c) 2019-2024 The PyQMC Developers
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# what should the structure for orbital evaluation look like?
import numpy as np
import scipy.special
from numba import njit
from pbcgto import PeriodicAtomicOrbitalEvaluator
from time_gto_mol import printout
from pyscf.pbc import gto

precision = 1e-2

def generate_mol():
    L = 4.168
    pos = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 0, 1],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
    ]) * 0.5 * L
    mol = gto.M(
        atom=[("Ni", r) for r in pos[:4]] + [("O", r) for r in pos[4:]],
        a=np.eye(3)*L,
        ecp="ccecp", 
        basis="ccecp-ccpvtz", 
        unit="A", 
        spin=0,
        precision=precision,
    )
    return mol


def compare_pbc_pyscf():
    # Test
    from pyqmc.coord import PeriodicConfigs
    from pyqmc.pbc import enforce_pbc
    import pandas as pd
    import time
    def timecall(f, *args):
        t0 = time.perf_counter()
        f(*args)
        return time.perf_counter() - t0

    mol = generate_mol()
    lvecs = mol.lattice_vectors()
    n = 500
    coords, _ = enforce_pbc(lvecs, np.random.randn(n, 3)*2 + np.array([0, 0, 0.5]))
    orbitals = PeriodicAtomicOrbitalEvaluator(mol, kpts=np.zeros((1, 3)), eval_gto_precision=precision)
    #f1 = lambda c: np.asarray(gto.eval_gto.eval_gto(mol, "PBCGTOval_sph_deriv1", c, kpts=np.zeros((1, 3)), Ls=orbitals.Ls))
    def f1(c): 
        x = np.asarray(gto.eval_gto.eval_gto(mol, "PBCGTOval_sph_deriv2", c, kpts=np.zeros((1, 3)), Ls=orbitals.Ls))
        lap = np.sum(x[0, [4, 7, 9]], axis=0, keepdims=True)
        return np.concatenate([x[0, :4], lap], axis=0)
            
    # compile functions
    ao1 = orbitals.eval_gto_lap(PeriodicConfigs(coords, mol.lattice_vectors()))[0]
    ao0 = f1(coords)#[0]
    print("compiled", flush=True)

    # timing
    N = 15
    dat = dict(pyscf=np.zeros(N), new=np.zeros(N))
    for i in range(N):
        coords, _ = enforce_pbc(lvecs, np.random.randn(n, 3)*2 + np.array([0, 0, 0.5]))
        configs = PeriodicConfigs(coords, mol.lattice_vectors())
        tpys = timecall(f1, coords)
        tpyq = timecall(orbitals.eval_gto_lap, configs)
        dat["pyscf"][i] = tpys
        dat["new"][i] = tpyq
        print("rep ", i)
        
    printout(mol, ao0, ao1, dat)


if __name__ == "__main__":
    compare_pbc_pyscf()


# How to accumulate variance
#    d = dict(pyscf=0., new=0.)
#    v = dict(pyscf=0., new=0.)
#        delta_pys = (tpys - d["pyscf"]) / (i+1)
#        delta_pyq = (tpyq - d["new"]) / (i+1)
#        d["pyscf"] += delta_pys
#        d["new"] += delta_pyq
#        v["pyscf"] += delta_pys**2 * i - v["pyscf"]/(i+1)
#        v["new"] += delta_pyq**2 * i - v["new"]/(i+1)
