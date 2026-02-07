import numpy as xp
import ibfs
import pytest
from .. import pytest_utils as pyut

import matplotlib.pyplot as plt


def test_momentum(domain_and_Reynolds):
    r"""
    Test for :func:`ibfs.SpatialOperators.evaluate_momentum`.
    Check that the spatial discretization is second-order.
    """
    xvec, yvec, spacings, Re = domain_and_Reynolds
    dxs, dys = spacings
    dxs /= 2
    dys /= 2

    niter = 5
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
        Xu, Yu, Xv, Yv, _, _ = torch_mesh

        ufun, vfun, _ = pyut.analytical_functions()
        # X momentum forcing term
        u, du_dx, du_dy, d2u_dx2, d2u_dy2 = pyut.evaluate_fun_and_derivatives(
            Xu, Yu, ufun, xp
        )
        v, dv_dx, dv_dy, _, _ = pyut.evaluate_fun_and_derivatives(
            Xu, Yu, vfun, xp
        )
        fx_int = (
            (-2 * u * du_dx +  (-v * du_dy - u * dv_dy))
            + (d2u_dx2 + d2u_dy2) / Re
        )
        mesh.fx_int[:, :] = -fx_int[1:-1, 1:-1]
        # Y momentum forcing term
        u, du_dx, du_dy, _, _ = pyut.evaluate_fun_and_derivatives(
            Xv, Yv, ufun, xp
        )
        v, dv_dx, dv_dy, d2v_dx2, d2v_dy2 = pyut.evaluate_fun_and_derivatives(
            Xv, Yv, vfun, xp
        )
        fy_int = (
            (-2 * v * dv_dy - (u * dv_dx + v * du_dx))
            + (d2v_dx2 + d2v_dy2) / Re
        )
        mesh.fy_int[:, :] = -fy_int[1:-1, 1:-1]

        u, _, _, _, _ = pyut.evaluate_fun_and_derivatives(Xu, Yu, ufun, xp)
        v, _, _, _, _ = pyut.evaluate_fun_and_derivatives(Xv, Yv, vfun, xp)
        mesh.u_ext[:, :] = u
        mesh.v_ext[:, :] = v
        x_mom, y_mom = nsop.evaluate_momentum_equation()
        x_mom = x_mom[3:-3, 3:-3]
        y_mom = y_mom[3:-3, 3:-3]
        # vec_h = x_mom.reshape(-1)
        vec_h = xp.concatenate((x_mom.reshape(-1), y_mom.reshape(-1)))

        error[iter] = xp.max(xp.abs(vec_h))
        spacings[iter] = xp.min(dxs)
        iter += 1

    plt.figure()
    plt.plot(spacings, error, "o-")
    ax = plt.gca()
    ax.set_yscale("log")
    ax.set_xscale("log")
    plt.tight_layout()
    plt.show()
    print(error, spacings)
    order, _ = xp.polyfit(xp.log(spacings), xp.log(error), 1)
    print(f"Order = {order}")
    assert xp.abs(order - 4) < 1e-1 and error[-1] < 1e-3
