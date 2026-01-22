import numpy as xp
import ibfs
import matplotlib.pyplot as plt


def make_circle(D, d):
    th = xp.arange(0, 2 * xp.pi, 2 * xp.pi / (2 * xp.pi // d))
    return 0.5 * D * xp.cos(th), 0.5 * D * xp.sin(th)


Re = 100

# Define the spatial domain
# discretized with 500 cells in the x direction and
# 250 in the y direction
x0, x1 = -3, 15
y0, y1 = -4.5, 4.5
nx, ny = 500, 250
mesh = ibfs.Mesh(x0, x1, nx, y0, y1, ny)


# Define the boundary conditions. Zero neumann everywhere,
# except u = 1 and v = 0 at the left boundary (inflow)

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

# Instantiate the SpatialOperator class to evaluate the
# right-hand side of the Navier-Stokes equation
spops = ibfs.SpatialOperators(Re, mesh, bcs, False)


# Instantiate the ImmersedBody class to account for the
# presence of a cylinder with diameter = 1 and center
# at (0, 0)
ib = ibfs.ImmersedBody(*make_circle(1.0, mesh.d), spops)


save_path = 'data/'
q = xp.load(save_path + 'snapshot.npy')

ibfs.vector_to_fields(0.0, q, mesh, bcs)
Xu, Yu, Xv, Yv, _, _ = mesh.generate_meshgrids(False)

fig, ax = plt.subplots(nrows=1, ncols=2)
ax[0].contourf(Xu, Yu, mesh.u_ext, cmap="bwr", levels=200)
ax[0].fill(ib.xi, ib.eta, color="k")
ax[0].set_aspect("equal")
ax[0].set_xlabel(r'$x/L$')
ax[0].set_ylabel(r'$y/L$')
ax[0].set_title(r'$u$ velocity')

ax[1].contourf(Xv, Yv, mesh.v_ext, cmap="bwr", levels=200)
ax[1].fill(ib.xi, ib.eta, color="k")
ax[1].set_aspect("equal")
ax[1].set_xlabel(r'$x/L$')
ax[1].set_yticks([])
ax[1].set_title(r'$v$ velocity')

plt.show()