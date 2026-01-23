import numpy as xp
import ibfs
import os


def make_circle(D, d):
    th = xp.arange(0, 2 * xp.pi, 2 * xp.pi / (2 * xp.pi // (2 * d)))
    return 0.5 * D * xp.cos(th), 0.5 * D * xp.sin(th)


Re = 40

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
    "dirichlet",
    "dirichlet",
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


# Instantiate the TimeStepper class to evolve the system
# We integrate with a fixed delta t = 1e-2
dt = 1e-2
tstep = ibfs.TimeStepper(dt, spops, ib, scheme="RK2")

# Run the time stepper from t = 0 to t = 100 with
# initial condition q0 = 0
q0 = xp.ones(xp.prod(mesh.u_int.shape) + xp.prod(mesh.v_int.shape))
q0[xp.prod(mesh.u_int.shape) :] = 0.0
Q, tsave = tstep.solve(0.0, 10000 * dt, 100, q0)

save_path = "data/"
os.makedirs(save_path, exist_ok=True)
xp.save(save_path + "snapshot.npy", Q[:, -1])
