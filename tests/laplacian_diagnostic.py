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
    u, du_dx, du_dy, d2u_dx2, d2u_dy2 = pyut.evaluate_fun_and_derivatives(
        Xu, Yu, ufun, np
    )
    v, dv_dx, dv_dy, d2v_dx2, d2v_dy2 = pyut.evaluate_fun_and_derivatives(
        Xv, Yv, vfun, np
    )

    t1 = nsop.evaluate_laplacian(u, mesh.d)
    t2 = nsop.evaluate_laplacian(v, mesh.d)
    truth_1 = (d2u_dx2 + d2u_dy2)[1:-1, 1:-1]
    truth_2 = (d2v_dx2 + d2v_dy2)[1:-1, 1:-1]

    error_x.append(np.max(np.abs(t1 - truth_1)))
    error_y.append(np.max(np.abs(t2 - truth_2)))


plt.figure()
plt.plot((x1 - x0) / nxs, error_x, "o-", label=r"x mom.")
plt.plot((x1 - x0) / nxs, error_y, "x-", label=r"y mom.")
ax = plt.gca()
ax.set_yscale("log")
ax.set_xscale("log")
ax.set_xlabel(r"$\Delta x$")
ax.set_ylabel(r"Error")
plt.legend()
plt.tight_layout()
plt.show()
