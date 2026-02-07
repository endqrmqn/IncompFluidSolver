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
    dxs /= 5
    dys /= 5

    niter = 1
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


        idx_r = xp.argmin(xp.abs(Yu[:, 0].detach().numpy() - 0.25))
        u_ = u[idx_r -2:idx_r+3,]
        val = ibfs.compute_limited_face_values(u_, mesh.xu, mesh.xc, u_[:, :-1])
        val_, _ = ibfs.quadratic_interpolation(u_, mesh.xu, mesh.xc, False)
        
        plt.figure()
        plt.plot(mesh.xu, u_[0, ], 'g', label=r'$u(x)$')
        plt.plot(mesh.xc, val[0,], '--', label=r'$u(x)$ limited')
        # plt.plot(mesh.xc, val_[0,], 'r--', label=r'$u(x)$ quad recon.')
        plt.legend()
        plt.tight_layout()
        plt.show()

        # plt.figure()
        # plt.contourf(Xu.detach().numpy(), Yu.detach().numpy(), mesh.u_ext, levels=200)
        # plt.show()