import numpy as xp
import ibfs
import pytest
from .. import pytest_utils as pyut


def test_divergence(domain_and_Reynolds, grid_sizes):
    r"""
    Test for :func:`ibfs.SpatialOperators.evaluate_divergence`.
    Check that the spatial discretization is second-order.
    """
    nxs, nys = grid_sizes
    x0, x1, y0, y1, Re = domain_and_Reynolds
    error = xp.zeros(len(nxs))

    iter = 0
    for nx, ny in zip(nxs, nys):
        mesh = ibfs.Mesh(x0, x1, nx, y0, y1, ny)
        bcs = pyut.instantiate_boundary_conditions(mesh)
        nsop = ibfs.SpatialOperators(Re, mesh, bcs, True)
        _, torch_mesh = mesh.generate_meshgrids(output_torch=True)
        Xu, Yu, Xv, Yv, Xp, Yp = torch_mesh

        ufun, vfun, _ = pyut.analytical_functions()
        u, _, _, _, _ = pyut.evaluate_fun_and_derivatives(Xu, Yu, ufun, xp)
        v, _, _, _, _ = pyut.evaluate_fun_and_derivatives(Xv, Yv, vfun, xp)
        mesh.u_ext = u.copy()
        mesh.v_ext = v.copy()
        _, du_dx, _, _, _ = pyut.evaluate_fun_and_derivatives(Xp, Yp, ufun, xp)
        _, _, dv_dy, _, _ = pyut.evaluate_fun_and_derivatives(Xp, Yp, vfun, xp)

        div_h = nsop.evaluate_divergence()
        div = du_dx + dv_dy

        error[iter] = xp.max(xp.abs(div - div_h))
        iter += 1

    order, _ = xp.polyfit(xp.log((x1 - x0) / nxs), xp.log(error), 1)
    assert xp.abs(order - 2) < 1e-2 and error[-1] < 1e-3
