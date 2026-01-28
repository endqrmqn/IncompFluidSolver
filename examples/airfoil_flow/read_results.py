import numpy as xp
import ibfs
import matplotlib.pyplot as plt

Re = 200

# Define the spatial domain
# discretized with 500 cells in the x direction and
# 250 in the y direction
x0, x1 = -4, 12
y0, y1 = -2.5, 2.5
nx, ny = 800, 250
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
    lambda t: 0 * ones_l_u,
    None,
    lambda t: 0 * ones_b_u,
    lambda t: 0 * ones_b_u,
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
ib = ibfs.ImmersedBody(*ibfs.make_airfoil(0.12, 50, 20), spops)

#%%
# Read Resolvent4py Results

print("Reading res4py results...")

Nu = xp.prod(mesh.u_int.shape)
Nv = xp.prod(mesh.v_int.shape)
Np = xp.prod(mesh.p.shape) 
N =  Nu + Nv + Np + 2 * len(ib.xi) + 1
nfreqs = 4


S = xp.load('results/S.npy')

k = nfreqs + 3

j = 0
uj = xp.load('results/u_%02d.npy' % j).reshape((N, 2 * nfreqs + 1), order='F')[: Nu + Nv, k]
omegaj_, X, Y = ibfs.compute_vorticity(0.0, uj.real, mesh, bcs_pert)

vmin = xp.min(omegaj_)
vmax = -vmin

plt.figure()
plt.contourf(X, Y, omegaj_, cmap="bwr", levels=200, vmax=vmax, vmin=vmin)
plt.fill(ib.xi, ib.eta, color="k")
ax = plt.gca()
# ax.set_ylim([-2.5, 2.5])
# ax.set_xlim([-2, 4])
ax.set_xlabel(r"$x/D$")
ax.set_ylabel(r"$y/D$")
ax.set_aspect("equal")
plt.colorbar()
plt.tight_layout()
plt.show()
