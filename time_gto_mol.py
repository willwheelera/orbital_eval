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
from gto import AtomicOrbitalEvaluator
import pandas as pd


def compare_pyscf():
    # Test
    from pyscf import gto
    from pyqmc.coord import OpenConfigs
    import time
    def timecall(f, *args):
        t0 = time.perf_counter()
        f(*args)
        return time.perf_counter() - t0


    mol = gto.M(atom="Mn 0. 0. 0.; N 0. 0. 2.5", ecp="ccecp", basis="ccecp-ccpvtz", unit="B", spin=0)
    n = 1500
    coords = np.random.randn(n, 3) + np.array([0, 0, 0.5])
    orbitals = AtomicOrbitalEvaluator(mol)

    f1 = lambda c: mol.eval_gto("GTOval_sph_deriv1", c)
    # compile functions
    ao1 = orbitals.eval_gto_grad(OpenConfigs(coords))
    ao0 = f1(coords)

    N = 20
    dat = dict(pyscf=np.zeros(N), new=np.zeros(N))
    for i in range(5):
        coords = np.random.randn(n, 3)*2 + np.array([0, 0, 0.5])
        configs = OpenConfigs(coords)
        tpys = timecall(f1, coords)
        tpyq = timecall(orbitals.eval_gto_grad, configs)
        dat["pyscf"][i] = tpys
        dat["new"][i] = tpyq

    printout(mol, ao0, ao1, dat)

def printout(mol, ao0, ao1, dat):
    print("pyscf", ao0.shape)
    print("python", ao1.shape)
    df = {"label": mol.ao_labels()}
    #args = (np.argmax(np.abs(ao1-ao0), axis=0), range(ao0.shape[1]))
    #j = 0
    #df["ao0"] = ao0[args]
    #df["ao1"] = ao1[args]
    #df["ao1/ao0"] = ao1[args] / ao0[args]
    #df["ao0/ao1"] = ao0[args] / ao1[args]
    #df["diff"] = ao1[args] - ao0[args]
    #df = pd.DataFrame(df)
    #print(df[:50])
    #print(df[50:])
    diff_norm = np.linalg.norm(ao0-ao1)
    print("max diff", np.amax(np.abs(ao1-ao0)))
    eps = 0.01
    print(f"greater than {eps}", np.count_nonzero(np.abs(ao1-ao0)>eps))
    print("diff_norm", diff_norm)
    #assert diff_norm < 1e-2, df

    print("time")
    for k, v in dat.items():
        print(k, np.around(v.mean(), 6), np.around(v.std(ddof=1), 6))
    print("ratio", dat["new"].mean() / dat["pyscf"].mean())


if __name__ == "__main__":
    compare_pyscf()

