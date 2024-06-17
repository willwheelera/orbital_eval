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
import pandas as pd
import scipy.special
from numba import njit
import simple_test as st

def run():
    # Test
    from pyscf.pbc import gto
    from pyqmc.coord import PeriodicConfigs
    from pyqmc.pbc import enforce_pbc
    import pandas as pd
    import time
    def timecall(f, *args):
        t0 = time.perf_counter()
        f(*args)
        return time.perf_counter() - t0


    mol = gto.M(
        atom="Fe 0. 0. 0.; O 1. 1. 1.", 
        a=np.eye(3)*4,
        ecp="ccecp", 
        basis="ccecp-ccpvtz", 
        unit="B", 
        spin=0,
        precision=1e-2,
    )

    #for k, basis_ in mol._basis.items():
    #    for bas in basis_:
    #        print(k, bas[0])
    #        for b in bas[1:]:
    #            print(b)
    #quit()
        

    lvecs = mol.lattice_vectors()
    n = 500
    coords, _ = enforce_pbc(lvecs, np.random.randn(n, 3)*2 + np.array([0, 0, 0.5]))
    funcs = [
        st.eval_gto_loopL_loope_nocutoff,
        st.eval_gto_vecL_loope_nocutoff,
        st.eval_gto_loopL_vece_nocutoff,
        st.eval_gto_vecL_vece_nocutoff,
        st.eval_gto_loopL_loope_cutoff,
        st.eval_gto_vecL_loope_cutoff,
        st.eval_gto_loopL_vece_cutoff,
        st.eval_gto_vecL_vece_cutoff,
    ]
    labels = [
        "loopL_loope_nocutoff",
        "vecL_loope_nocutoff",
        "loopL_vece_nocutoff",
        "vecL_vece_nocutoff",
        "loopL_loope_cutoff",
        "vecL_loope_cutoff",
        "loopL_vece_cutoff",
        "vecL_vece_cutoff",
    ]
    orbitals = [st.SimplePeriodicAtomicOrbitalEvaluator(mol, f) for f in funcs]
            
    # compile functions
    configs = PeriodicConfigs(coords, mol.lattice_vectors())
    #aos, nmask = list(zip(*[orb.eval_gto(configs) for orb in orbitals]))
    aos = [orb.eval_gto(configs) for orb in orbitals]

    # timing
    N = 20
    dat = dict(pyscf=np.zeros(N), new=np.zeros(N))
    dat = np.zeros((8, N))
    for i in range(N):
        coords, _ = enforce_pbc(lvecs, np.random.randn(n, 3)*2 + np.array([0, 0, 0.5]))
        configs = PeriodicConfigs(coords, mol.lattice_vectors())
        for o, orb in enumerate(orbitals):
            dat[o, i] = timecall(orb.eval_gto, configs)
        
    printout(mol, aos, dat)

looplabel = ["loop", "vec"]
noyes = ["no", "yes"]

def printout(mol, aos, dat):
    df = []
    for c in [0, 1]:
        for e in [0, 1]:
            for L in [0, 1]:
                d = dict(L=looplabel[L], elec=looplabel[e], cutoff=noyes[c])
                i = 4*c + 2*e + L
                d["diff_norm"] = np.linalg.norm(np.abs(aos[i]-aos[4*c]))
                d["time"] = dat[i].mean()
                d["stddev"] = dat[i].std()
                #d["nmasked"] = nmask[i]
                df.append(d)
    df = pd.DataFrame(df)
    print(df)

if __name__ == "__main__":
    run()


# How to accumulate variance
#    d = dict(pyscf=0., new=0.)
#    v = dict(pyscf=0., new=0.)
#        delta_pys = (tpys - d["pyscf"]) / (i+1)
#        delta_pyq = (tpyq - d["new"]) / (i+1)
#        d["pyscf"] += delta_pys
#        d["new"] += delta_pyq
#        v["pyscf"] += delta_pys**2 * i - v["pyscf"]/(i+1)
#        v["new"] += delta_pyq**2 * i - v["new"]/(i+1)
