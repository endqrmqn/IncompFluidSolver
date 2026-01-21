import numpy as xp
import ibfs
import matplotlib.pyplot as plt

Re = 1000

x0, x1 = -0.5, 0.5
y0, y1 = -0.5, 0.5
nx, ny = 200, 200

mesh = ibfs.Mesh(x0, x1, nx, y0, y1, ny)

# U velocity boundary conditons
zeros_lr_u = xp.zeros(mesh.u_int.shape[0])
zeros_tb_u = xp.zeros(mesh.u_int.shape[-1])
ones_tb_u = xp.ones_like(zeros_tb_u)
fun_lr_u = lambda t: zeros_lr_u
fun_b_u = lambda t: zeros_tb_u
fun_t_u = lambda t: ones_tb_u
bcuvel = ibfs.BoundaryConditions(
    "u",
    "dirichlet",
    "dirichlet",
    "dirichlet",
    "dirichlet",
    fun_lr_u,
    fun_lr_u,
    fun_b_u,
    fun_t_u,
)

# V velocity boundary conditions
zeros_lr_v = xp.zeros(mesh.v_int.shape[0])
zeros_tb_v = xp.zeros(mesh.v_int.shape[-1])
fun_lr_v = lambda t: zeros_lr_v
fun_tb_v = lambda t: zeros_tb_v
bcvvel = ibfs.BoundaryConditions(
    "v",
    "dirichlet",
    "dirichlet",
    "dirichlet",
    "dirichlet",
    fun_lr_v,
    fun_lr_v,
    fun_tb_v,
    fun_tb_v,
)


bcs = [bcuvel, bcvvel]
spops = ibfs.SpatialOperators(Re, mesh, bcs)

dt = 5e-3
tstep = ibfs.TimeStepper(dt, spops, None, scheme="RK2")

# %%
q0 = xp.zeros(xp.prod(mesh.u_int.shape) + xp.prod(mesh.v_int.shape))
Q, tsave = tstep.solve(0.0, 100, 10, q0)

# %%
energy = xp.linalg.norm(Q, axis=0)
plt.figure()
plt.plot(tsave, energy)

# %%
ibfs.vector_to_fields(0.0, Q[:, -1], mesh, bcs)
Xu, Yu, Xv, Yv, _, _ = mesh.generate_meshgrids(False)

plt.figure()
plt.contourf(Xu, Yu, mesh.u_ext, cmap="bwr", levels=200)
ax = plt.gca()
ax.set_aspect("equal")

plt.figure()
plt.contourf(Xv, Yv, mesh.v_ext, cmap="bwr", levels=200)
ax = plt.gca()
ax.set_aspect("equal")

# %%
idx = xp.argmin(xp.abs(Xu[0,]))
uvel = mesh.u_ext[:, idx]

data_ghia = xp.loadtxt("ghia_data.txt")[:, 1:]
Re_ghia = xp.loadtxt("ghia_Re.txt")
idx = xp.argmin(xp.abs(Re_ghia - Re))
uvel_ghia = data_ghia[:, idx + 1]

plt.figure()
plt.plot(uvel[1:-1], Yu[1:-1, 0] + y1, "k", label="Present")
plt.plot(uvel_ghia, data_ghia[:, 0], "ro", label="Ghia et al., (1982)")
ax = plt.gca()
ax.set_xlabel(r"$u / u_{\infty}$ velocity")
ax.set_ylabel(r"$y / L$")
ax.set_aspect("equal")
plt.legend()

# %%
