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
    _, dp_dx, _, _, _ = pyut.evaluate_fun_and_derivatives(Xu, Yu, pfun, np)
    _, _, dp_dy, _, _ = pyut.evaluate_fun_and_derivatives(Xv, Yv, pfun, np)
    p, _, _, _, _ = pyut.evaluate_fun_and_derivatives(Xp, Yp, pfun, np)
    mesh.p = p.copy()
    t1, t2 = nsop.evaluate_pressure_gradient(mesh)
    truth_1 = dp_dx.copy()[1:-1, 1:-1]
    truth_2 = dp_dy.copy()[1:-1, 1:-1]

    error_x.append(np.max(np.abs(t1 - truth_1)))
    error_y.append(np.max(np.abs(t2 - truth_2)))


plt.figure()
plt.plot((x1 - x0) / nxs, error_x, "o-", label=r"$dp/dx$")
plt.plot((x1 - x0) / nxs, error_y, "x-", label=r"$dp/dy$")
ax = plt.gca()
ax.set_yscale("log")
ax.set_xscale("log")
ax.set_xlabel(r"$\Delta x$")
ax.set_ylabel(r"Error")
plt.legend()
plt.tight_layout()
plt.show()
