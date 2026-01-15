import numpy as xp
import ibfs
import pytest
from .. import pytest_utils as pyut


def test_gradient(domain_and_Reynolds, grid_sizes):
    r"""
    Test for :func:`ibfs.SpatialOperators.evaluate_gradient`.
    Check that the spatial discretization is second-order.
    """
    nxs, nys = grid_sizes
    x0, x1, y0, y1, Re = domain_and_Reynolds
    error_x = xp.zeros(len(nxs))
    error_y = xp.zeros(len(nxs))

    iter = 0
    for nx, ny in zip(nxs, nys):
        mesh = ibfs.Mesh(x0, x1, nx, y0, y1, ny)
        bcs = pyut.instantiate_boundary_conditions(mesh)
        nsop = ibfs.SpatialOperators(Re, mesh, bcs, True)
        _, torch_mesh = mesh.generate_meshgrids(output_torch=True)
        Xu, Yu, Xv, Yv, Xp, Yp = torch_mesh

        _, _, pfun = pyut.analytical_functions()
        _, dp_dx, _, _, _ = pyut.evaluate_fun_and_derivatives(Xu, Yu, pfun, xp)
        _, _, dp_dy, _, _ = pyut.evaluate_fun_and_derivatives(Xv, Yv, pfun, xp)
        p, _, _, _, _ = pyut.evaluate_fun_and_derivatives(Xp, Yp, pfun, xp)
        mesh.p = p.copy()
        dpdx_h, dpdy_h = nsop.evaluate_pressure_gradient()
        dpdx = dp_dx[1:-1, 1:-1]
        dpdy = dp_dy[1:-1, 1:-1]

        error_x[iter] = xp.max(xp.abs(dpdx_h - dpdx))
        error_y[iter] = xp.max(xp.abs(dpdy_h - dpdy))
        iter += 1

    order_x, _ = xp.polyfit(xp.log((x1 - x0) / nxs), xp.log(error_x), 1)
    order_y, _ = xp.polyfit(xp.log((x1 - x0) / nxs), xp.log(error_y), 1)
    assert (
        xp.abs(order_x - 2) < 1e-2
        and xp.abs(order_y - 2) < 1e-2
        and error_x[-1] < 1e-3
        and error_y[-1] < 1e-3
    )
