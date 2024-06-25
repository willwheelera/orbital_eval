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
import hardcoded_spherical_harmonics as hsh
from gto import radial_gto, single_radial_gto
from pbcgto import PeriodicAtomicOrbitalEvaluator


@njit(fastmath=True)
def eval_gto_loopL_loope_nocutoff(all_rvec, basis_ls, basis_arrays, max_l, splits, l_splits, Ls, num_Ls, r2_l_cutoff, r2_cutoff, kpts=None):
    ao = np.zeros((1, *all_rvec.shape[:2]))
    bas = np.array([[0.1, 1.]])
    rvec_L = np.zeros(3)

    for a, rvec in enumerate(all_rvec):
        for e, v in enumerate(rvec):
            tmp = 0.
            for L in Ls:
                r2 = 0
                for i in range(3):
                    rvec_L[i] = v[i] - L[i]
                    r2 += rvec_L[i]**2
                tmp += single_radial_gto(r2, bas)
            ao[0, a, e] = tmp
    return ao

@njit(fastmath=True)
def eval_gto_vecL_loope_nocutoff(all_rvec, basis_ls, basis_arrays, max_l, splits, l_splits, Ls, num_Ls, r2_l_cutoff, r2_cutoff, kpts=None):
    ao = np.zeros((1, *all_rvec.shape[:2]))
    bas = np.array([[0.1, 1.]])
    r2 = np.zeros(len(Ls))
    rvec_L = np.zeros((len(Ls), 3))

    for a, rvec in enumerate(all_rvec):
        for e, v in enumerate(rvec):
            r2[:] = 0
            for i in range(3):
                rvec_L[:, i] = v[i] - Ls[:, i]
                r2 += rvec_L[:, i]**2
            ao[0, a, e] = radial_gto(r2, bas).sum()
    return ao


@njit(fastmath=True)
def eval_gto_loopL_vece_nocutoff(all_rvec, basis_ls, basis_arrays, max_l, splits, l_splits, Ls, num_Ls, r2_l_cutoff, r2_cutoff, kpts=None):
    ao = np.zeros((1, *all_rvec.shape[:2]))
    bas = np.array([[0.1, 1.]])

    for a, rvec in enumerate(all_rvec):
        for L in Ls:
            rvec_L = rvec - L
            r2 = np.sum(rvec_L**2, axis=-1)
            ao[0, a] += radial_gto(r2, bas)
    return ao


@njit(fastmath=True)
def eval_gto_vecL_vece_nocutoff(all_rvec, basis_ls, basis_arrays, max_l, splits, l_splits, Ls, num_Ls, r2_l_cutoff, r2_cutoff, kpts=None):
    ao = np.zeros((1, *all_rvec.shape[:2]))
    bas = np.array([[0.1, 1.]])

    for a, rvec in enumerate(all_rvec):
        rvec_L = rvec[:, np.newaxis] - Ls
        r2 = np.sum(rvec_L**2, axis=-1)
        ao[0, a] = radial_gto(r2, bas).sum(axis=-1)
    return ao


@njit(fastmath=True)
def eval_gto_loopL_loope_cutoff(all_rvec, basis_ls, basis_arrays, max_l, splits, l_splits, Ls, num_Ls, r2_l_cutoff, r2_cutoff, kpts=None):
    natom, nelec = all_rvec.shape[:2]
    ao = np.zeros((1, *all_rvec.shape[:2]))
    bas = np.array([[0.1, 1.]])
    nmasked = 0
    rvec_L = np.zeros(3)

    for a, rvec in enumerate(all_rvec):
        cut = r2_cutoff[a]
        for e, v in enumerate(rvec):
            tmp = 0.
            for L in Ls:
                r2 = 0
                for i in range(3):
                    rvec_L[i] = v[i] - L[i]
                    r2 += rvec_L[i]**2
                if r2 > cut: 
                    nmasked += 1
                    continue
                tmp += single_radial_gto(r2, bas)
            ao[0, a, e] = tmp
    return ao


@njit(fastmath=True)
def eval_gto_vecL_loope_cutoff(all_rvec, basis_ls, basis_arrays, max_l, splits, l_splits, Ls, num_Ls, r2_l_cutoff, r2_cutoff, kpts=None):
    natom, nelec = all_rvec.shape[:2]
    ao = np.zeros((1, *all_rvec.shape[:2]))
    bas = np.array([[0.1, 1.]])
    nmasked = 0
    rvec_L = np.zeros((len(Ls), 3))
    r2 = np.zeros((len(Ls)))
    mask = np.zeros((len(Ls))) < 1

    for a, rvec in enumerate(all_rvec):
        cut = r2_cutoff[a]
        for e, v in enumerate(rvec):
            for j, L in enumerate(Ls):
                tmp = 0.
                for i in range(3):
                    rvec_L[j, i] = v[i] - L[i]
                    tmp += rvec_L[j, i]**2
                r2[j] = tmp
                mask[j] = r2[j] < cut
            nmasked += (1 - mask).sum()
            ao[0, a, e] = radial_gto(r2[mask], bas).sum()
    return ao


@njit(fastmath=True)
def eval_gto_loopL_vece_cutoff(all_rvec, basis_ls, basis_arrays, max_l, splits, l_splits, Ls, num_Ls, r2_l_cutoff, r2_cutoff, kpts=None):
    natom, nelec = all_rvec.shape[:2]
    ao = np.zeros((1, *all_rvec.shape[:2]))
    bas = np.array([[0.1, 1.]])
    nmasked = 0

    for a, rvec in enumerate(all_rvec):
        cut = r2_cutoff[a]
        for L in Ls:
            rvec_L = rvec - L
            r2 = np.sum(rvec_L**2, axis=-1)
            mask = r2 < cut
            nmasked += (1 - mask).sum()
            ao[0, a, mask] += radial_gto(r2[mask], bas)
    return ao


@njit(fastmath=True)
def eval_gto_vecL_vece_cutoff(all_rvec, basis_ls, basis_arrays, max_l, splits, l_splits, Ls, num_Ls, r2_l_cutoff, r2_cutoff, kpts=None):
    natom, nelec = all_rvec.shape[:2]
    ao = np.zeros((1, natom, nelec))
    bas = np.array([[0.1, 1.]])
    nmasked = 0

    for a, rvec in enumerate(all_rvec):
        rvec_L = rvec[:, np.newaxis] - Ls
        r2 = np.sum(rvec_L**2, axis=-1)
        mask = np.where(r2.ravel() < r2_cutoff[a])[0]
        r2mask = r2.ravel()[mask]
        nmasked += r2.size - len(mask)

        rad = radial_gto(r2mask, bas)
        for j, m in enumerate(mask):
            ao[0, a, m//len(Ls)] += rad[j]
    return ao


class SimplePeriodicAtomicOrbitalEvaluator(PeriodicAtomicOrbitalEvaluator):
    def __init__(self, cell, eval_func):
        super().__init__(cell, np.zeros((1, 3)))
        self._gto_func = eval_func

