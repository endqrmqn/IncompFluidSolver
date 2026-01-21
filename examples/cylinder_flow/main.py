import numpy as xp
import ibfs
import matplotlib.pyplot as plt


def make_circle(D, d):
    th = xp.arange(0, 2 * xp.pi, 2 * xp.pi / (2 * xp.pi // d))
    return 0.5 * D * xp.cos(th), 0.5 * D * xp.sin(th)


Re = 100

x0, x1 = -3, 17
y0, y1 = -5, 5
nx, ny = 1000, 500

mesh = ibfs.Mesh(x0, x1, nx, y0, y1, ny)

# U velocity boundary conditons
ones_l_u = xp.ones(mesh.u_int.shape[0])
zeros_l_u = xp.ones(mesh.u_int.shape[0])
ones_b_u = xp.ones(mesh.u_int.shape[-1])
bcuvel = ibfs.BoundaryConditions(
    "u",
    "dirichlet",
    "neumann",
    "neumann",
    "neumann",
    lambda t: ones_l_u,
    None,
    None,
    None,
)

# V velocity boundary conditions
zeros_l_v = xp.zeros(mesh.v_int.shape[0])
zeros_b_v = xp.zeros(mesh.v_int.shape[-1])
bcvvel = ibfs.BoundaryConditions(
    "v",
    "dirichlet",
    "neumann",
    "neumann",
    "neumann",
    lambda t: zeros_l_v,
    None,
    None,
    None,
)


bcs = [bcuvel, bcvvel]
spops = ibfs.SpatialOperators(Re, mesh, bcs, False)

# %%

# spops.assemble_divergence_matrix()
# spops.assemble_gradient_matrix()
# spops.assemble_laplacian_matrix()

# G = spops.G.todense()
# G = spops.D.T.todense()

# import scipy as sp
# u, s, vh = sp.linalg.svd(G)
# v = vh.T

# v1 = v[:, -1].reshape(mesh.p.shape)
# v2 = v[:, -2].reshape(mesh.p.shape)

# sz = xp.prod(mesh.u_int.shape)
# u1u = u[:sz, -1].reshape(mesh.u_int.shape)
# u1v = u[sz:, -1].reshape(mesh.v_int.shape)

vec = xp.random.randn(spops.D.shape[-1])
q = spops.D.dot(vec)
ibfs.vector_to_fields(0.0, vec, mesh, bcs)
q_ = spops.evaluate_divergence().reshape(-1)

error = (q_ - q).reshape(mesh.p.shape)
print(xp.linalg.norm(error))

vec = xp.random.randn(spops.G.shape[-1])
q = spops.G.dot(vec)
mesh.p = vec.reshape(mesh.p.shape)
dpdx, dpdy = spops.evaluate_pressure_gradient()
q_ = xp.concatenate((dpdx.reshape(-1), dpdy.reshape(-1)))

error_ = q_ - q
print(xp.linalg.norm(error_))

# %%
ib = ibfs.ImmersedBody(*make_circle(1.0, mesh.d), spops)

dt = 1e-2
tstep = ibfs.TimeStepper(dt, spops, ib, scheme="RK2")

# %%
q0 = xp.ones(xp.prod(mesh.u_int.shape) + xp.prod(mesh.v_int.shape))
q0[xp.prod(mesh.u_int.shape) :] = 0.0
Q, tsave = tstep.solve(0.0, 1000 * dt, 10, Q[:, -1])

# %%
ibfs.vector_to_fields(0.0, Q[:, -1], mesh, bcs)
Xu, Yu, Xv, Yv, _, _ = mesh.generate_meshgrids(False)

plt.figure()
plt.contourf(Xu, Yu, mesh.u_ext, cmap="bwr", levels=200)
plt.fill(ib.xi, ib.eta, color="k")
ax = plt.gca()
ax.set_aspect("equal")
plt.colorbar()

plt.figure()
plt.contourf(Xv, Yv, mesh.v_ext, cmap="bwr", levels=200)
plt.fill(ib.xi, ib.eta, color="k")
ax = plt.gca()
ax.set_aspect("equal")
plt.colorbar()
