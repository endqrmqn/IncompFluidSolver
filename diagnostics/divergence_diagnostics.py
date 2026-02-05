import numpy as xp
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


dxs *= 0.5
dys *= 0.5

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

    ufun, vfun, _ = pyut.analytical_functions()
    u, _, _, _, _ = pyut.evaluate_fun_and_derivatives(Xu, Yu, ufun, xp)
    v, _, _, _, _ = pyut.evaluate_fun_and_derivatives(Xv, Yv, vfun, xp)
    mesh.u_ext = u.copy()
    mesh.v_ext = v.copy()
    _, du_dx, _, _, _ = pyut.evaluate_fun_and_derivatives(Xp, Yp, ufun, xp)
    _, _, dv_dy, _, _ = pyut.evaluate_fun_and_derivatives(Xp, Yp, vfun, xp)

    vec = nsop.Mp.dot((du_dx + dv_dy).reshape(-1))
    vech = nsop.evaluate_divergence_integral().reshape(-1)
    error[iter] = xp.max(xp.abs(vec - vech))
    spacings[iter] = xp.min(dxs)


order, _ = xp.polyfit(xp.log(spacings), xp.log(error), 1)
print(order)

plt.figure()
plt.plot(spacings, error)
ax = plt.gca()
ax.set_yscale("log")
ax.set_xscale("log")
ax.set_xlabel(r"$\Delta x$")
ax.set_ylabel(r"Error")
plt.tight_layout()
plt.show()
