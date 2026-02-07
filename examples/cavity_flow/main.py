import numpy as xp
import ibfs
import os

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


# Instantiate the SpatialOperator class to evaluate the
# right-hand side of the Navier-Stokes equation
spops = ibfs.SpatialOperators(Re, mesh, bcs)


# Instantiate the TimeStepper class to evolve the system
# We integrate with a fixed delta t = 5e-3
dt = 2e-3
tstep = ibfs.TimeStepper(dt, spops, None, scheme="RK2")

# Run the time stepper from t = 0 to t = 100 with
# initial condition q0 = 0
q0 = xp.zeros(xp.prod(mesh.u_int.shape) + xp.prod(mesh.v_int.shape))
Q, tsave = tstep.solve(0.0, 20, 500, Q[:, -1])

save_path = "data/"
os.makedirs(save_path, exist_ok=True)
xp.save(save_path + "snapshot.npy", Q[:, -1])

import matplotlib.pyplot as plt

Xu, Yu, Xv, Yv, _, _ = ibfs.generate_meshgrids(mesh, False)
# %%

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
