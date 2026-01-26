import numpy as xp
import ibfs
import matplotlib.pyplot as plt


def make_circle(D, d):
    th = xp.arange(0, 2 * xp.pi, 2 * xp.pi / (2 * xp.pi // (2 * d)))
    return 0.5 * D * xp.cos(th), 0.5 * D * xp.sin(th)


Re = 40

# Define the spatial domain
# discretized with 500 cells in the x direction and
# 250 in the y direction
x0, x1 = -5, 15
y0, y1 = -5, 5
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
    lambda t: ones_b_u,
    lambda t: ones_b_u,
)

# V velocity boundary conditions
zeros_l_v = xp.zeros(mesh.v_int.shape[0])
zeros_b_v = xp.zeros(mesh.v_int.shape[-1])
bcvvel = ibfs.BoundaryConditions(
    "v",
    "dirichlet",
    "neumann",
    "dirichlet",
    "dirichlet",
    lambda t: zeros_l_v,
    None,
    lambda t: zeros_b_v,
    lambda t: zeros_b_v,
)
bcs = [bcuvel, bcvvel]

# Instantiate the SpatialOperator class to evaluate the
# right-hand side of the Navier-Stokes equation
spops = ibfs.SpatialOperators(Re, mesh, bcs, False)

# Instantiate the ImmersedBody class to account for the
# presence of a cylinder with diameter = 1 and center
# at (0, 0)
ib = ibfs.ImmersedBody(*make_circle(1.0, mesh.d), spops)

save_path = "data/"
q = xp.load(save_path + "snapshot.npy")


omega, X, Y = ibfs.compute_vorticity(0.0, q, mesh, bcs)

plt.figure()
plt.contour(X, Y, omega, cmap="bwr", levels=xp.arange(-3, 3 + 0.4, 0.4))
plt.fill(ib.xi, ib.eta, color="k")
ax = plt.gca()
ax.set_ylim([-2.5, 2.5])
ax.set_xlim([-2, 4])
ax.set_xlabel(r"$x/D$")
ax.set_ylabel(r"$y/D$")
ax.set_title(r"Steady-state vorticity contours at $Re = 40$")
ax.set_aspect("equal")
plt.colorbar()
plt.tight_layout()
plt.show()
