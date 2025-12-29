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

iter = 0
for nx, ny in zip(nxs, nys):
    mesh = ibfs.Mesh(x0, x1, nx, y0, y1, ny)
    _, torch_mesh = mesh.generate_meshgrids(output_torch=True)
    Xu, Yu, Xv, Yv, Xp, Yp = torch_mesh

    ufun, vfun, pfun = pyut.analytical_functions()
    u, _, _, _, _ = pyut.evaluate_fun_and_derivatives(Xu, Yu, ufun, np)
    v, _, _, _, _ = pyut.evaluate_fun_and_derivatives(Xv, Yv, vfun, np)
    mesh.u_ext = u.copy()
    mesh.v_ext = v.copy()
    _, du_dx, _, _, _ = pyut.evaluate_fun_and_derivatives(Xp, Yp, ufun, np)
    _, _, dv_dy, _, _ = pyut.evaluate_fun_and_derivatives(Xp, Yp, vfun, np)

    t1 = nsop.evaluate_divergence(mesh)
    truth_1 = du_dx + dv_dy

    error_x.append(np.max(np.abs(t1 - truth_1)))


plt.figure()
plt.plot((x1 - x0) / nxs, error_x, "o-", label=r"divergence")
ax = plt.gca()
ax.set_yscale("log")
ax.set_xscale("log")
ax.set_xlabel(r"$\Delta x$")
ax.set_ylabel(r"Error")
plt.legend()
plt.tight_layout()
plt.show()
