import numpy as np
import torch
import ibfs
import pytest_utils as pyut
import matplotlib.pyplot as plt

Re = 100
nsop = ibfs.SpatialOperators(Re)


x0 = -2
x1 = 2
y0 = -1
y1 = 1

nys = np.asarray([100, 200, 400, 800, 1000, 1600])
nxs = 2 * nys
error_x = []
error_y = []

iter = 0
for nx, ny in zip(nxs, nys):
    mesh = ibfs.Mesh(x0, x1, nx, y0, y1, ny)
    _, torch_mesh = mesh.generate_meshgrids(output_torch=True)
    Xu, Yu, Xv, Yv, Xp, Yp = torch_mesh
    ufun, vfun, pfun = pyut.analytical_functions()

    # X momentum forcing term
    u, du_dx, du_dy, d2u_dx2, d2u_dy2 = pyut.evaluate_fun_and_derivatives(
        Xu, Yu, ufun, np
    )
    v, dv_dx, dv_dy, _, _ = pyut.evaluate_fun_and_derivatives(Xu, Yu, vfun, np)
    fx_int = -2 * u * du_dx - v * du_dy - u * dv_dy + (d2u_dx2 + d2u_dy2) / Re
    mesh.fx_int *= 0.0
    mesh.fx_int -= fx_int[1:-1, 1:-1]

    # Y momentum forcing term
    u, du_dx, du_dy, _, _ = pyut.evaluate_fun_and_derivatives(Xv, Yv, ufun, np)
    v, dv_dx, dv_dy, d2v_dx2, d2v_dy2 = pyut.evaluate_fun_and_derivatives(
        Xv, Yv, vfun, np
    )
    fy_int = -2 * v * dv_dy - u * dv_dx - v * du_dx + (d2v_dx2 + d2v_dy2) / Re
    mesh.fy_int *= 0.0
    mesh.fy_int -= fy_int[1:-1, 1:-1]

    u, _, _, _, _ = pyut.evaluate_fun_and_derivatives(Xu, Yu, ufun, np)
    v, _, _, _, _ = pyut.evaluate_fun_and_derivatives(Xv, Yv, vfun, np)
    mesh.u_ext *= 0.0
    mesh.v_ext *= 0.0
    mesh.u_ext += u
    mesh.v_ext += v
    x_mom, y_mom = nsop.evaluate_momentum_equation(mesh)

    error_x.append(np.max(np.abs(x_mom.reshape(-1))))
    error_y.append(np.max(np.abs(y_mom.reshape(-1))))
    
print(error_x)
print(error_y)

plt.figure()
plt.plot((x1 - x0) / nxs, error_x, "o-", label=r"x momentum")
plt.plot((x1 - x0) / nxs, error_y, "x-", label=r"y momentum")
ax = plt.gca()
ax.set_yscale("log")
ax.set_xscale("log")
ax.set_xlabel(r"$\Delta x$")
ax.set_ylabel(r"Error")
plt.legend()
plt.tight_layout()
plt.show()

# plt.figure()
# plt.contourf(Xu[1:-1,1:-1].detach().numpy(), Yu[1:-1,1:-1].detach().numpy(), x_mom)
# plt.colorbar()
# plt.tight_layout()
# plt.show()
