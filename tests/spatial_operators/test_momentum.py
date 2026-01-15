import numpy as xp
import ibfs
import pytest
from .. import pytest_utils as pyut


def test_momentum(domain_and_Reynolds, grid_sizes):
    r"""
    Test for :func:`ibfs.SpatialOperators.evaluate_momentum`.
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

        ufun, vfun, _ = pyut.analytical_functions()
        # X momentum forcing term
        u, du_dx, du_dy, d2u_dx2, d2u_dy2 = pyut.evaluate_fun_and_derivatives(
            Xu, Yu, ufun, xp
        )
        v, dv_dx, dv_dy, _, _ = pyut.evaluate_fun_and_derivatives(Xu, Yu, vfun, xp)
        fx_int = -2 * u * du_dx - v * du_dy - u * dv_dy + (d2u_dx2 + d2u_dy2) / Re
        mesh.fx_int[:, :] = -fx_int[1:-1, 1:-1]
        # Y momentum forcing term
        u, du_dx, du_dy, _, _ = pyut.evaluate_fun_and_derivatives(Xv, Yv, ufun, xp)
        v, dv_dx, dv_dy, d2v_dx2, d2v_dy2 = pyut.evaluate_fun_and_derivatives(
            Xv, Yv, vfun, xp
        )
        fy_int = -2 * v * dv_dy - u * dv_dx - v * du_dx + (d2v_dx2 + d2v_dy2) / Re
        mesh.fy_int[:, :] = -fy_int[1:-1, 1:-1]
        
        u, _, _, _, _ = pyut.evaluate_fun_and_derivatives(Xu, Yu, ufun, xp)
        v, _, _, _, _ = pyut.evaluate_fun_and_derivatives(Xv, Yv, vfun, xp)
        mesh.u_ext[:, :] = u
        mesh.v_ext[:, :] = v
        x_mom, y_mom = nsop.evaluate_momentum_equation()

        error_x[iter] = xp.max(xp.abs(x_mom.reshape(-1)))
        error_y[iter] = xp.max(xp.abs(y_mom.reshape(-1)))
        iter += 1

    order_x, _ = xp.polyfit(xp.log((x1 - x0) / nxs), xp.log(error_x), 1)
    order_y, _ = xp.polyfit(xp.log((x1 - x0) / nxs), xp.log(error_y), 1)
    assert (
        xp.abs(order_x - 2) < 1e-2
        and xp.abs(order_y - 2) < 1e-2
        and error_x[-1] < 1e-3
        and error_y[-1] < 1e-3
    )
