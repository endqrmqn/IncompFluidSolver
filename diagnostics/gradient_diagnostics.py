import numpy as xp
import scipy as sp
import ibfs
import pytest_utils as pyut
import matplotlib.pyplot as plt


xvec = xp.asarray([-2, -1, -0.5, 0.5, 1, 2])
yvec = xp.asarray([-1.75, -0.5, 0])
dxs = xp.asarray([0.2, 0.15, 0.1, 0.18, 0.21])
dys = xp.asarray([0.2, 0.1])

# xvec = xp.array([-1, 1])
# yvec = xp.array([-1, 0])
# dxs = xp.array([0.1])
# dys = xp.array([0.1])

dxs *= 0.25
dys *= 0.25

niter = 5
error = xp.zeros(niter)
spacings = xp.zeros(niter)

for iter in range(niter):
    dxs /= 2
    dys /= 2
    mesh = ibfs.Mesh_(
        xvec, dxs, yvec, dys, mirror_y=True, check_equal_min_spacing=True
    )
    bcs = pyut.instantiate_boundary_conditions(mesh)
    nsop = ibfs.SpatialOperators(100, mesh, bcs, True)
    _, torch_mesh = ibfs.generate_meshgrids(mesh, True)
    Xu, Yu, Xv, Yv, Xp, Yp = torch_mesh

    _, _, pfun = pyut.analytical_functions()
    _, dp_dx, _, _, _ = pyut.evaluate_fun_and_derivatives(Xu, Yu, pfun, xp)
    fun_x = sp.interpolate.interp1d(Xu.detach().numpy()[0,], dp_dx, axis=-1)
    dp_dx = fun_x(0.5 * (mesh.xc[1:] + mesh.xc[:-1]))[1:-1, :]

    _, _, dp_dy, _, _ = pyut.evaluate_fun_and_derivatives(Xv, Yv, pfun, xp)
    fun_y = sp.interpolate.interp1d(Yv.detach().numpy()[:, 0], dp_dy, axis=0)
    dp_dy = fun_y(0.5 * (mesh.yc[1:] + mesh.yc[:-1]))[:, 1:-1]

    p, _, _, _, _ = pyut.evaluate_fun_and_derivatives(Xp, Yp, pfun, xp)
    mesh.p = p.copy()

    dpdx_h, dpdy_h = nsop.evaluate_pressure_integral()
    grad_h = xp.concatenate((dpdx_h.reshape(-1), dpdy_h.reshape(-1)))
    grad = nsop.M.dot(xp.concatenate((dp_dx.reshape(-1), dp_dy.reshape(-1))))

    error[iter] = xp.max(xp.abs(grad_h - grad))
    spacings[iter] = xp.min(dxs)

order, _ = xp.polyfit(xp.log(spacings), xp.log(error), 1)
print(order)

plt.figure()
plt.plot(spacings, error)
ax = plt.gca()
ax.set_yscale("log")
ax.set_xscale("log")
plt.show()
