import numpy as xp
import scipy as sp
import ibfs
import pytest
from .. import pytest_utils as pyut


def test_gradient(domain_and_Reynolds, grid_sizes):
    r"""
    Test for :func:`ibfs.SpatialOperators.evaluate_gradient_integral`.
    Check that the spatial discretization is second-order.
    """

    xvec, yvec, spacings, Re = domain_and_Reynolds
    dxs, dys = spacings
    dxs /= 10
    dys /= 10

    niter = 3
    error = xp.zeros(niter)
    spacings = error.copy()

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
        Xu = Xu[1:-1, 1:-1]
        Yu = Yu[1:-1, 1:-1]
        Xv = Xv[1:-1, 1:-1]
        Yv = Yv[1:-1, 1:-1]

        _, _, pfun = pyut.analytical_functions()
        _, dp_dx, _, _, _ = pyut.evaluate_fun_and_derivatives(Xu, Yu, pfun, xp)
        _, _, dp_dy, _, _ = pyut.evaluate_fun_and_derivatives(Xv, Yv, pfun, xp)
        p, _, _, _, _ = pyut.evaluate_fun_and_derivatives(Xp, Yp, pfun, xp)
        mesh.p = p.copy()

        dpdx_h, dpdy_h = nsop.evaluate_pressure_integral()
        grad_h = xp.concatenate((dpdx_h.reshape(-1), dpdy_h.reshape(-1)))
        grad = nsop.M.dot(
            xp.concatenate((dp_dx.reshape(-1), dp_dy.reshape(-1)))
        )

        error[iter] = xp.max(xp.abs(grad_h - grad))
        spacings[iter] = xp.min(dxs)

    order, _ = xp.polyfit(xp.log(spacings), xp.log(error), 1)
    print(f"Order = {order}")
    assert xp.abs(order - 4) / 4 < 1e-1 and error[-1] < 1e-8
