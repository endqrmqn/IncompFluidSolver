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
x0, x1 = -10, 30
y0, y1 = -10, 10
nx, ny = 1000, 500
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


save_path = "data/"
q = xp.load(save_path + "snapshot.npy")

U, V, X, Y = ibfs.interpolate_to_nodes(0.0, q, mesh, bcs)

spx = xp.arange(0.5, 3, 3.5 / 10)
spy = xp.arange(-1.5, 1.5, 3 / 10)
spx, spy = xp.meshgrid(spx, spy)
spx = spx.reshape(-1).reshape(1, -1)
spy = spy.reshape(-1).reshape(1, -1)

start_points = xp.concatenate((spx, spy), axis=0)
plt.figure()
plt.streamplot(
    X[210:-210, 40:-600],
    Y[210:-210, 40:-600],
    U[210:-210, 40:-600],
    V[210:-210, 40:-600],
    start_points=start_points.T,
    maxlength=2000,
    density=9,
    minlength=0.0,
    integration_direction="both",
)
plt.plot(1.26, 0.3, "ro", linewidth=3.0)
plt.plot(2.83, 0.0, "ro", linewidth=3.0)
plt.fill(ib.xi, ib.eta, color="k")
ax = plt.gca()
ax.set_ylim([-2, 2])
ax.set_xlim([-2, 4])
ax.set_aspect("equal")
ax.set_xlabel(r"$x/D$")
ax.set_ylabel(r"$y/D$")
ax.text(0.5, 1.7, r"$a/D \approx 0.76$, $b/D \approx 0.6$, $l/D\approx 2.33$")
ax.text(0.5, 1.45, r"See Fig. 6 in Taira and Colonius, JCP, (2007)")
ax.set_title(r"Steady-state streamlines at $Re = 40$")
plt.tight_layout()
plt.show()


omega, _, _ = ibfs.compute_vorticity(0.0, q, mesh, bcs)

plt.figure()
plt.contour(X, Y, omega, cmap="bwr", levels=xp.arange(-3, 3 + 0.4, 0.4))
plt.fill(ib.xi, ib.eta, color="k")
ax = plt.gca()
ax.set_ylim([-2.5, 2.5])
ax.set_xlim([-2, 4])
ax.set_xlabel(r"$x/D$")
ax.set_ylabel(r"$y/D$")
ax.set_title(r'Steady-state vorticity contours at $Re = 40$')
ax.set_aspect("equal")
plt.colorbar()
plt.tight_layout()
plt.show()
