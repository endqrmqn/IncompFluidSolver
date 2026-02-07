import numpy as xp
import ibfs
import os
import matplotlib.pyplot as plt

Re = 1000

# Define the spatial domain (a box of size 1 x 1)
# discretized with 200 cells in the x and y directions
xvec = xp.array([-0.5, 0])
dxvec = xp.array([0.005])
# xvec = xp.array([-0.5, 0.5])
# dxvec = xp.array([0.005])

mesh = ibfs.Mesh(xvec, dxvec, xvec, dxvec, True, True)

# Define the boundary conditions. Zero velocity boundary conditions
# on all sides, except the u = 1 at the top wall

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


# Plot
save_path = "data/"
q = xp.load(save_path + "snapshot.npy")

ibfs.vector_to_fields(0.0, q, mesh, bcs)
Xu, Yu, Xv, Yv, _, _ = ibfs.generate_meshgrids(mesh, False)


fig, ax = plt.subplots(nrows=1, ncols=2)
ax[0].contourf(Xu, Yu, mesh.u_ext, cmap="inferno", levels=200)
ax[0].set_aspect("equal")
ax[0].set_xlabel(r"$x/L$")
ax[0].set_ylabel(r"$y/L$")
ax[0].set_title(r"$u$ velocity")

ax[1].contourf(Xv, Yv, mesh.v_ext, cmap="bwr", levels=200)
ax[1].set_aspect("equal")
ax[1].set_xlabel(r"$x/L$")
ax[1].set_yticks([])
ax[1].set_title(r"$v$ velocity")

plt.tight_layout()

plt.show()

idx = xp.argmin(xp.abs(Xu[0,]))
uvel = mesh.u_ext[:, idx]

data_ghia = xp.loadtxt("ghia_data.txt")[:, 1:]
Re_ghia = xp.loadtxt("ghia_Re.txt")
idx = xp.argmin(xp.abs(Re_ghia - Re))
uvel_ghia = data_ghia[:, idx + 1]

plt.figure()
plt.plot(uvel[1:-1], Yu[1:-1, 0] + 0.5, "k", label="Present")
plt.plot(uvel_ghia, data_ghia[:, 0], "ro", label="Ghia et al., (1982)")
ax = plt.gca()
ax.set_xlabel(r"$u / u_{\infty}$ velocity")
ax.set_ylabel(r"$y / L$")
ax.set_aspect("equal")
plt.legend()
plt.tight_layout()
plt.show()
