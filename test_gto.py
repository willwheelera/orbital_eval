import pyscf.gto
from pyqmc.coord import OpenConfigs, PeriodicConfigs
import numpy as np

def test_gradient(fval, fgrad, delta=1e-5):
    rvec = np.asarray(np.random.randn(500, 3)) 
    grad = fgrad(rvec)
    numeric = np.zeros(grad.shape)
    for d in range(3):
        pos = rvec.copy()
        pos[..., d] += delta
        plusval = fval(pos)
        pos[..., d] -= 2 * delta
        minuval = fval(pos)
        numeric[d, ...] = (plusval - minuval) / (2 * delta)
    maxerror = np.max(np.abs(grad - numeric))
    ind = np.where(np.abs(grad - numeric) > 0.01)
    print(ind)
    print("electrons")
    for e in np.unique(ind[1]):
        print(e, rvec[e])
    diff = (numeric - grad) / grad
    #print(diff[:, :, 0].T)
    return (maxerror)


def test_laplacian(fgrad, flap, delta=1e-5):
    rvec = np.asarray(np.random.randn(5, 3)) 
    lap = flap(rvec)
    numeric = np.zeros(lap.shape)
    for d in range(3):
        pos = rvec.copy()
        pos[..., d] += delta
        plusgrad = fgrad(pos)
        pos[..., d] -= 2 * delta
        minugrad = fgrad(pos)
        numeric += (plusgrad[d] - minugrad[d]) / (2 * delta)
    maxerror = np.max(np.abs(lap - numeric))
    ind = np.where(np.abs(lap - numeric) > 0.01)
    print(ind)
    #print("electrons")
    #for e in np.unique(ind):
    #    print(e, rvec[e])
    diff = (numeric - lap) / lap
    #print(diff[:, :, 0].T)
    return (maxerror)


def run_tests(bc="molecule"):
    print("boundary conditions", bc)
    coeffs = []
    coeffs.append(np.array([[0.2, 1.]]))
    coeffs.append(np.array([[1.7, 1.]]))
    coeffs.append(np.array([
        [84.322332, 2e-06],
        [44.203528, 0.004103],
        [23.288963, -0.04684],
        [13.385163, 0.052833],
        [7.518052, 0.218094],
        [4.101835, -0.044999],
        [2.253571, -0.287386],
        [1.134924, -0.71322],
        [0.56155, 0.249174],
        [0.201961, 1.299872],
        [0.108698, 0.192119],
        [0.053619, -0.658616],
        [0.025823, -0.521047],
    ]))
    coeffs.append(np.array([
        [152.736742, 5.1e-05],
        [50.772485, 0.008769],
        [26.253589, 0.047921],
        [12.137022, 0.132475],
        [5.853719, 0.297279],
        [2.856224, 0.275018],
        [1.386132, -0.222842],
        [0.670802, -0.619067],
        [0.33028, -0.07629],
        [0.170907, 0.256056],
        [0.086794, 0.541482],
    ]))
    if bc == "molecule":
        import gto
        sph_funcs = [lambda x: gto.eval_spherical(max_l, x).sum(axis=0) for max_l in [2, 3, 4, 5]]
        sph_grads = [lambda x: gto.eval_spherical_grad(max_l, x).sum(axis=0)[1:] for max_l in [2, 3, 4, 5]]
        sph_laps = [lambda x: 0*gto.eval_spherical(2, x)[0] for max_l in [2, 3, 4, 5]]
        rad_funcs = [lambda x: gto.radial_gto(np.sum(x**2, axis=-1), c) for c in coeffs]
        rad_grads = [lambda x: gto.radial_gto_grad(np.sum(x**2, axis=-1), x, c)[1:] for c in coeffs]
        rad_laps = [lambda x: gto.radial_gto_lap(np.sum(x**2, axis=-1), x, c)[4] for c in coeffs]
    else:
        import pbcgto as gto
        sph_funcs = [gto.sph2, gto.sph3, gto.sph4, gto.sph5]
        sph_grads = [gto.sph2_grad, gto.sph3_grad, gto.sph4_grad, gto.sph5_grad]
        rad_funcs = [lambda x: gto.single_radial_gto(np.sum(x**2, axis=-1), c) for c in coeffs]
        rad_grads = [lambda x: gto.single_radial_gto_grad(np.sum(x**2, axis=-1), x, c)[1:] for c in coeffs]

    #if bc == "molecule":
    #    print("spherical")
    #    for sval, sgrad, slap in zip(sph_funcs, sph_grads, sph_laps):
    #        err = np.amax([test_gradient(sval, sgrad, d) for d in (1e-5, 1e-6, 1e-7)])
    #        print(err)
    #        err = np.amax([test_laplacian(sgrad, slap, d) for d in (1e-5, 1e-6, 1e-7)])
    #        print(err)
    #    print("radial")
    #    for rval, rgrad, rlap in zip(rad_funcs, rad_grads, rad_laps):
    #        err = np.amax([test_gradient(rval, rgrad, d) for d in (1e-5, 1e-6, 1e-7)])
    #        print(err)
    #        err = np.amax([test_laplacian(rgrad, rlap, d) for d in (1e-5, 1e-6, 1e-7)])
    #        print(err)
    
    if bc == "molecule":
        mol = pyscf.gto.M(atom="Mn 0. 0. 0.; N 0. 0. 2.5", ecp="ccecp", basis="ccecp-ccpvtz", unit="B", spin=0)
        orbitals = gto.AtomicOrbitalEvaluator(mol)
        configs = lambda x: OpenConfigs(x)
        orbval = lambda x: orbitals.eval_gto(configs(x))
        orbgrad = lambda x: orbitals.eval_gto_grad(configs(x))[1:]
        orblap = lambda x: orbitals.eval_gto_lap(configs(x))[4]
    else:
        from time_gto_pbc import generate_mol
        mol = generate_mol()
        orbitals = gto.PeriodicAtomicOrbitalEvaluator(mol)
        configs = lambda x: PeriodicConfigs(x, mol.lattice_vectors())
        orbval = lambda x: orbitals.eval_gto(configs(x))[0]
        orbgrad = lambda x: orbitals.eval_gto_grad(configs(x))[0, 1:]
        orblap = lambda x: orbitals.eval_gto_lap(configs(x))[0, 4]
    print("orbitals")
    #print(test_gradient(orbval, orbgrad))
    print(test_laplacian(orbgrad, orblap))

    #labels = mol.ao_labels()
    #for i in [224, 225, 226, 227, 228]:
    #    print(i, labels[i])
    #err = np.amax([test_gradient(orbval, orbgrad, d) for d in (1e-5, 1e-6, 1e-7)])
    #print(err)


if __name__ == "__main__":
    run_tests("pbc")
