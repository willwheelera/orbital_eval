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
from pyqmc.orbitals import _estimate_rcut
from pyscf.pbc.gto.cell import estimate_rcut, _extract_pgto_params
from pyscf.pbc.gto.cell import _estimate_rcut as _estimate_rcut_pyscf
from pyscf.gto import mole
#import hardcoded_spherical_harmonics as hsh
import modified_spherical_harmonics as hsh
import gto
#from gto import sph2, sph3, sph4, sph5


"""
Wrappers for hsh.SPHn: evaluate spherical harmonics through l=2
v: (3,) vector to evaluate
out: ((n+1)**2,) output array
"""
@njit(cache=False, fastmath=True)
def sph2(v, out):
    hsh.SPH2(v[0], v[1], v[2], v[0]**2, v[1]**2, v[2]**2, out)

@njit(cache=False, fastmath=True)
def sph3(v, out):
    hsh.SPH3(v[0], v[1], v[2], v[0]**2, v[1]**2, v[2]**2, out)

@njit(cache=False, fastmath=True)
def sph4(v, out):
    hsh.SPH4(v[0], v[1], v[2], v[0]**2, v[1]**2, v[2]**2, out)

@njit(cache=False, fastmath=True)
def sph5(v, out):
    hsh.SPH5(v[0], v[1], v[2], v[0]**2, v[1]**2, v[2]**2, out)

@njit(cache=False, fastmath=True)
def sph2_grad(v, out):
    a, b, c, d = out
    hsh.SPH2_GRAD(v[0], v[1], v[2], v[0]**2, v[1]**2, v[2]**2, a, b, c, d)

@njit(cache=False, fastmath=True)
def sph3_grad(v, out):
    a, b, c, d = out
    hsh.SPH3_GRAD(v[0], v[1], v[2], v[0]**2, v[1]**2, v[2]**2, a, b, c, d)

@njit(cache=False, fastmath=True)
def sph4_grad(v, out):
    a, b, c, d = out
    hsh.SPH4_GRAD(v[0], v[1], v[2], v[0]**2, v[1]**2, v[2]**2, a, b, c, d)

@njit(cache=False, fastmath=True)
def sph5_grad(v, out):
    a, b, c, d = out
    hsh.SPH5_GRAD(v[0], v[1], v[2], v[0]**2, v[1]**2, v[2]**2, a, b, c, d)


@njit(fastmath=True)
def _pbc_eval_gto(all_rvec, basis_ls, basis_arrays, max_l, splits, l_splits, Ls, num_Ls, r2_l_cutoff, r2_cutoff, phases):
    """
    all_rvec: (natom, nelec, 3) atom-electron distances
    basis_ls: (ncontractions,) l value for every Gaussian contraction (concatenated together)
    basis_arrays: (ngaussians, 2) contraction coefficients for all Gaussian contractions (concatenated together)
    max_l: (natom,) max angular momentum l for each atom
    splits: (ncontractions+1,) indexing for basis_arrays
    l_splits: (natom+1,) indexing for basis_ls
    Ls: (nL, 3) list of (sorted) lattice points to sum over
    num_Ls: (natom,) number of Ls to check for each atom
    r2_l_cutoff: (ncontractions,) distance cutoff for each contraction
    r2_cutoff: (natoms,) distance cutoff for each atom
    """
    natom, nelec = all_rvec.shape[:2]
    nbas_tot = np.sum(2 * basis_ls + 1)
    ao = np.zeros((1, all_rvec.shape[1], nbas_tot), dtype=phases.dtype)

    atom_start = np.zeros(natom+1, dtype=np.int32)
    basis_ = []
    bstart = 0
    for a in range(natom):
        basis_ls_a = basis_ls[l_splits[a]:l_splits[a+1]]
        tmp_bas = []
        atom_start[a+1] = atom_start[a]
        for l_ind, l in enumerate(basis_ls_a):
            split = bstart + l_ind
            tmp_bas.append(basis_arrays[splits[split]:splits[split+1]])
            atom_start[a+1] += 2 * l + 1
        basis_.append(tmp_bas)
        bstart += len(basis_ls_a)

    for a, rvec in enumerate(all_rvec):
        basis_ls_a = basis_ls[l_splits[a]:l_splits[a+1]]
        _single_atom(
            ao, 
            rvec, 
            basis_ls_a, 
            basis_[a], 
            l_splits[a], 
            Ls[:num_Ls[a]], 
            r2_l_cutoff, 
            r2_cutoff[a], 
            atom_start[a], 
            max_l[a],
            phases[:num_Ls[a]],
        )
    return ao


@njit(fastmath=True)
def _single_atom(ao, rvec, basis_ls_a, basis_a, l_split_a, Ls_a, r2_l_cutoff, cut, astart, max_l, phases):
    """
    Calculate basis functions for one atom

    ao: (1, nelec, nao) output array
    rvec: (nelec, 3) atom-electron distances
    basis_ls_a: (ncontractions,) l value for every Gaussian contraction in this atom's basis
    basis_arrays: (ngaussians, 2) contraction coefficients for all Gaussian contractions in this atom's basis
    l_split_a: (int) starting index for r_l_cutoff
    Ls_a: (nL, 3) list of (sorted) lattice points to check for this atom
    r2_l_cutoff: (ncontractions,) distance cutoff for each contraction
    cut: (float) distance cutoff for this atom
    astart: (int) starting index for ao
    max_l: (int) max angular momentum l for this atom
    """
    if max_l == 2: sph_func = sph2#hsh.SPH2
    elif max_l == 3: sph_func = sph3#hsh.SPH3
    elif max_l == 4: sph_func = sph4#hsh.SPH4
    else: sph_func = sph5#hsh.SPH5

    rvec_L = np.zeros(3)
    spherical = np.zeros((max_l+1)**2)
    nbas = np.sum(basis_ls_a * 2 + 1)
    for e, v in enumerate(rvec):
        for j, L in enumerate(Ls_a):
            r2 = 0
            for i in range(3):
                rvec_L[i] = v[i] - L[i]
                r2 += rvec_L[i]**2
            if r2 > cut: continue

            phases_j = phases[j]
            sph_func(rvec_L, spherical)
            # for some reason numba doesn't accept this
            #sph_func(rvec_L[0], rvec_L[1], rvec_L[2], rvec_L[0]**2, rvec_L[1]**2, rvec_L[2]**2, spherical)
            #spherical[1:4] = spherical[np.array([3, 1, 2])]

            # this loops over all basis functions for the atom
            bstart=astart
            for l_ind, l in enumerate(basis_ls_a):#[l_splits[a]:l_splits[a+1]]):
                if r2 < r2_l_cutoff[l_split_a+l_ind]: 
                    rad = gto.single_radial_gto(r2, basis_a[l_ind])
                    for k, phase in enumerate(phases_j):
                        for b in range(2*l+1):
                            ao[k, e, bstart+b] += spherical[l*l+b] * rad * phase
                bstart += 2*l+1


@njit(fastmath=True)
def _pbc_eval_gto_grad(all_rvec, basis_ls, basis_arrays, max_l, splits, l_splits, Ls, num_Ls, r2_l_cutoff, r2_cutoff, phases):
    """
    all_rvec: (natom, nelec, 3) atom-electron distances
    basis_ls: (ncontractions,) l value for every Gaussian contraction (concatenated together)
    basis_arrays: (ngaussians, 2) contraction coefficients for all Gaussian contractions (concatenated together)
    max_l: (natom,) max angular momentum l for each atom
    splits: (ncontractions+1,) indexing for basis_arrays
    l_splits: (natom+1,) indexing for basis_ls
    Ls: (nL, 3) list of (sorted) lattice points to sum over
    num_Ls: (natom,) number of Ls to check for each atom
    r2_l_cutoff: (ncontractions,) distance cutoff for each contraction
    r2_cutoff: (natoms,) distance cutoff for each atom
    """
    natom, nelec = all_rvec.shape[:2]
    nbas_tot = np.sum(2 * basis_ls + 1)
    ao = np.zeros((1, all_rvec.shape[1], nbas_tot, 4), dtype=phases.dtype)

    atom_start = np.zeros(natom+1, dtype=np.int32)
    basis_ = []
    bstart = 0
    for a in range(natom):
        basis_ls_a = basis_ls[l_splits[a]:l_splits[a+1]]
        tmp_bas = []
        atom_start[a+1] = atom_start[a]
        for l_ind, l in enumerate(basis_ls_a):
            split = bstart + l_ind
            tmp_bas.append(basis_arrays[splits[split]:splits[split+1]])
            atom_start[a+1] += 2 * l + 1
        basis_.append(tmp_bas)
        bstart += len(basis_ls_a)

    for a, rvec in enumerate(all_rvec):
        basis_ls_a = basis_ls[l_splits[a]:l_splits[a+1]]
        _single_atom_grad(
            ao, 
            rvec, 
            basis_ls_a, 
            basis_[a], 
            l_splits[a], 
            Ls[:num_Ls[a]], 
            r2_l_cutoff, 
            r2_cutoff[a], 
            atom_start[a], 
            max_l[a],
            phases[:num_Ls[a]],
        )
    return np.transpose(ao, (0, 3, 1, 2))


@njit(fastmath=True)
def _single_atom_grad(ao, rvec, basis_ls_a, basis_a, l_split_a, Ls_a, r2_l_cutoff, cut, astart, max_l, phases):
    """
    Calculate basis functions for one atom

    ao: (1, nelec, nao) output array
    rvec: (nelec, 3) atom-electron distances
    basis_ls_a: (ncontractions,) l value for every Gaussian contraction in this atom's basis
    basis_arrays: (ngaussians, 2) contraction coefficients for all Gaussian contractions in this atom's basis
    l_split_a: (int) starting index for r_l_cutoff
    Ls_a: (nL, 3) list of (sorted) lattice points to check for this atom
    r2_l_cutoff: (ncontractions,) distance cutoff for each contraction
    cut: (float) distance cutoff for this atom
    astart: (int) starting index for ao
    max_l: (int) max angular momentum l for this atom
    """
    if max_l == 2: sph_func = sph2_grad#hsh.SPH2
    elif max_l == 3: sph_func = sph3_grad#hsh.SPH3
    elif max_l == 4: sph_func = sph4_grad#hsh.SPH4
    else: sph_func = sph5_grad#hsh.SPH5

    rvec_L = np.zeros(3)
    spherical = np.zeros((4, (max_l+1)**2))
    nbas = np.sum(basis_ls_a * 2 + 1)
    for e, v in enumerate(rvec):
        for j, L in enumerate(Ls_a):
            r2 = 0
            for i in range(3):
                rvec_L[i] = v[i] - L[i]
                r2 += rvec_L[i]**2
            if r2 > cut: continue

            phases_j = phases[j]
            sph_func(rvec_L, spherical)
            # for some reason numba doesn't accept this
            #sph_func(rvec_L[0], rvec_L[1], rvec_L[2], rvec_L[0]**2, rvec_L[1]**2, rvec_L[2]**2, spherical)
            #spherical[1:4] = spherical[np.array([3, 1, 2])]

            # this loops over all basis functions for the atom
            b_ind = 0
            for l_ind, l in enumerate(basis_ls_a):#[l_splits[a]:l_splits[a+1]]):
                if r2 > r2_l_cutoff[l_split_a+l_ind]: 
                    b_ind += 2*l+1
                    continue
                rad = gto.single_radial_gto_grad(r2, rvec_L, basis_a[l_ind])
                for b in range(2*l+1):
                    for k, phase in enumerate(phases_j):
                        ao[k, e, astart+b_ind, 0] += spherical[0, l*l+b] * rad[0] * phase
                        for i in range(1, 4):
                            ao[k, e, astart+b_ind, i] += (spherical[i, l*l+b] * rad[0] + spherical[0, l*l+b] * rad[i]) * phase
                    b_ind += 1


@njit(fastmath=True)
def _pbc_eval_gto_lap(all_rvec, basis_ls, basis_arrays, max_l, splits, l_splits, Ls, num_Ls, r2_l_cutoff, r2_cutoff, phases):
    """
    all_rvec: (natom, nelec, 3) atom-electron distances
    basis_ls: (ncontractions,) l value for every Gaussian contraction (concatenated together)
    basis_arrays: (ngaussians, 2) contraction coefficients for all Gaussian contractions (concatenated together)
    max_l: (natom,) max angular momentum l for each atom
    splits: (ncontractions+1,) indexing for basis_arrays
    l_splits: (natom+1,) indexing for basis_ls
    Ls: (nL, 3) list of (sorted) lattice points to sum over
    num_Ls: (natom,) number of Ls to check for each atom
    r2_l_cutoff: (ncontractions,) distance cutoff for each contraction
    r2_cutoff: (natoms,) distance cutoff for each atom
    """
    natom, nelec = all_rvec.shape[:2]
    nbas_tot = np.sum(2 * basis_ls + 1)
    ao = np.zeros((1, all_rvec.shape[1], nbas_tot, 5), dtype=phases.dtype)

    atom_start = np.zeros(natom+1, dtype=np.int32)
    basis_ = []
    bstart = 0
    for a in range(natom):
        basis_ls_a = basis_ls[l_splits[a]:l_splits[a+1]]
        tmp_bas = []
        atom_start[a+1] = atom_start[a]
        for l_ind, l in enumerate(basis_ls_a):
            split = bstart + l_ind
            tmp_bas.append(basis_arrays[splits[split]:splits[split+1]])
            atom_start[a+1] += 2 * l + 1
        basis_.append(tmp_bas)
        bstart += len(basis_ls_a)

    for a, rvec in enumerate(all_rvec):
        basis_ls_a = basis_ls[l_splits[a]:l_splits[a+1]]
        _single_atom_lap(
            ao, 
            rvec, 
            basis_ls_a, 
            basis_[a], 
            l_splits[a], 
            Ls[:num_Ls[a]], 
            r2_l_cutoff, 
            r2_cutoff[a], 
            atom_start[a], 
            max_l[a],
            phases[:num_Ls[a]],
        )
    return np.transpose(ao, (0, 3, 1, 2))


@njit(fastmath=True)
def _single_atom_lap(ao, rvec, basis_ls_a, basis_a, l_split_a, Ls_a, r2_l_cutoff, cut, astart, max_l, phases):
    """
    Calculate basis functions for one atom

    ao: (1, nelec, nao) output array
    rvec: (nelec, 3) atom-electron distances
    basis_ls_a: (ncontractions,) l value for every Gaussian contraction in this atom's basis
    basis_arrays: (ngaussians, 2) contraction coefficients for all Gaussian contractions in this atom's basis
    l_split_a: (int) starting index for r_l_cutoff
    Ls_a: (nL, 3) list of (sorted) lattice points to check for this atom
    r2_l_cutoff: (ncontractions,) distance cutoff for each contraction
    cut: (float) distance cutoff for this atom
    astart: (int) starting index for ao
    max_l: (int) max angular momentum l for this atom
    """
    if max_l == 2: sph_func = sph2_grad#hsh.SPH2
    elif max_l == 3: sph_func = sph3_grad#hsh.SPH3
    elif max_l == 4: sph_func = sph4_grad#hsh.SPH4
    else: sph_func = sph5_grad#hsh.SPH5

    rvec_L = np.zeros(3)
    spherical = np.zeros((4, (max_l+1)**2))
    nbas = np.sum(basis_ls_a * 2 + 1)
    for e, v in enumerate(rvec):
        for j, L in enumerate(Ls_a):
            r2 = 0
            for i in range(3):
                rvec_L[i] = v[i] - L[i]
                r2 += rvec_L[i]**2
            if r2 > cut: continue

            phases_j = phases[j]
            sph_func(rvec_L, spherical)
            # for some reason numba doesn't accept this
            #sph_func(rvec_L[0], rvec_L[1], rvec_L[2], rvec_L[0]**2, rvec_L[1]**2, rvec_L[2]**2, spherical)
            #spherical[1:4] = spherical[np.array([3, 1, 2])]

            # this loops over all basis functions for the atom
            b_ind = 0
            for l_ind, l in enumerate(basis_ls_a):#[l_splits[a]:l_splits[a+1]]):
                if r2 > r2_l_cutoff[l_split_a+l_ind]: 
                    b_ind += 2*l+1
                    continue
                rad = gto.single_radial_gto_lap(r2, rvec_L, basis_a[l_ind])
                for b in range(2*l+1):
                    for k, phase in enumerate(phases_j):
                        ao[k, e, astart+b_ind, 0] += spherical[0, l*l+b] * rad[0] * phase
                        tmp_lap = spherical[0, l*l+b] * rad[4] * phase
                        for i in range(1, 4):
                            ao[k, e, astart+b_ind, i] += (spherical[i, l*l+b] * rad[0] + spherical[0, l*l+b] * rad[i]) * phase
                            tmp_lap += 2 * spherical[i, l*l+b] * rad[i] * phase
                        ao[k, e, astart+b_ind, 4] += tmp_lap * phase
                    b_ind += 1

