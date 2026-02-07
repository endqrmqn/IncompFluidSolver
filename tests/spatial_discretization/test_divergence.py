import numpy as xp
import ibfs
import pytest
from .. import pytest_utils as pyut


def test_divergence(domain_and_Reynolds):
    r"""
    Test for :func:`ibfs.SpatialOperators.evaluate_divergence_integral`.
    Check that the spatial discretization is second-order.
    """

    xvec, yvec, spacings, Re = domain_and_Reynolds
    dxs, dys = spacings
    dxs /= 5
    dys /= 5

    niter = 5
    error = xp.zeros(niter)
    spacings = xp.zeros(niter)

    for iter in range(niter):
        dxs /= 2
        dys /= 2
        mesh = ibfs.Mesh(
            xvec, dxs, yvec, dys, mirror_y=True, check_equal_min_spacing=True
        )
        bcs = pyut.instantiate_boundary_conditions(mesh)
        nsop = ibfs.SpatialOperators(Re, mesh, bcs, True)
        _, torch_mesh = ibfs.generate_meshgrids(mesh, True)
        Xu, Yu, Xv, Yv, Xp, Yp = torch_mesh

        ufun, vfun, _ = pyut.analytical_functions()
        u, _, _, _, _ = pyut.evaluate_fun_and_derivatives(Xu, Yu, ufun, xp)
        v, _, _, _, _ = pyut.evaluate_fun_and_derivatives(Xv, Yv, vfun, xp)
        mesh.u_ext = u.copy()
        mesh.v_ext = v.copy()
        _, du_dx, _, _, _ = pyut.evaluate_fun_and_derivatives(Xp, Yp, ufun, xp)
        _, _, dv_dy, _, _ = pyut.evaluate_fun_and_derivatives(Xp, Yp, vfun, xp)

        vec = nsop.Mp.dot((du_dx + dv_dy).reshape(-1))
        vech = nsop.evaluate_divergence_integral().reshape(-1)
        error[iter] = xp.max(xp.abs(vec - vech))
        spacings[iter] = xp.min(dxs)

    order, _ = xp.polyfit(xp.log(spacings), xp.log(error), 1)
    assert xp.abs(order - 4) / 4 < 1e-2 and error[-1] < 1e-7
