import numpy as xp
import scipy as sp
import ibfs
import matplotlib.pyplot as plt


def make_circle(D, d):
    th = xp.arange(0, 2 * xp.pi, 2 * xp.pi / (2 * xp.pi // (2 * d)))
    return 0.5 * D * xp.cos(th), 0.5 * D * xp.sin(th)


Re = 40

# Define the spatial domain
# discretized with 500 cells in the x direction and
# 250 in the y direction
xvec = xp.array([-10, -5, -2.5, 2.5, 6, 12, 24, 48])
dxvec = xp.array([0.2, 0.1, 0.04, 0.06, 0.08, 0.1, 0.2])
yvec = xp.array([-30, -20, -10, -2.5, 0])
dyvec = xp.array([0.4, 0.1, 0.08, 0.04])
# xvec = xp.array([-10, 30])
# dxvec = xp.array([0.04])
# yvec = xp.array([-10, 10])
# dyvec = xp.array([0.04])


mesh = ibfs.Mesh(xvec, dxvec, yvec, dyvec, False, True)


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
ib = ibfs.ImmersedBody(*make_circle(1.0, xp.min(mesh.dx)), spops)

# %%
save_path = "data/"
q = xp.load(save_path + "snapshot.npy")


ibfs.vector_to_fields(0.0, Q[:, -1], mesh, bcs)
Xu, Yu, Xv, Yv, _, _ = ibfs.generate_meshgrids(mesh, False)


i = 8
fig, ax = plt.subplots(nrows=1, ncols=2)
ax[0].contourf(Xu, Yu, mesh.u_ext, cmap="inferno", levels=200)
ax[0].pcolormesh(Xu[::i, ::i], Yu[::i, ::i], xp.zeros_like(Xu[::i, ::i]), 
               edgecolors='k', 
               facecolor='none', 
               linewidth=0.5, 
               alpha=0.1,  # Keep it faint
               zorder=2)
ax[0].fill(ib.xi, ib.eta, "w")
ax[0].set_aspect("equal")
ax[0].set_xlabel(r"$x/L$")
ax[0].set_ylabel(r"$y/L$")
ax[0].set_title(r"$u$ velocity")
ax[0].set_xlim(-25, 25)
ax[0].set_ylim(-25, 25)

ax[1].contourf(Xv, Yv, mesh.v_ext, cmap="bwr", levels=200)
ax[1].pcolormesh(Xu[::i, ::i], Yu[::i, ::i], xp.zeros_like(Xu[::i, ::i]), 
               edgecolors='k', 
               facecolor='none', 
               linewidth=0.5, 
               alpha=0.1,  # Keep it faint
               zorder=2)
ax[1].fill(ib.xi, ib.eta, "k")
ax[1].set_aspect("equal")
ax[1].set_xlabel(r"$x/L$")
ax[1].set_yticks([])
ax[1].set_title(r"$v$ velocity")
ax[1].set_xlim(-25, 25)
ax[1].set_ylim(-25, 25)

plt.tight_layout()

plt.show()


# %%
def interp2d(X, Y, U, Xt, Yt):
    
    interp_ux = sp.interpolate.interp1d(X[0,], U, axis=-1, kind='cubic')
    U_ = interp_ux(Xt[0,])
    interp_uy = sp.interpolate.interp1d(Y[:, 0], U_, axis=0, kind='cubic')
    return interp_uy(Yt[:, 0])
    
U_, V_, X, Y = ibfs.interpolate_to_nodes(0.0, q, mesh, bcs)

xuni = xp.linspace(-2, 4, num=100)
yuni = xp.linspace(-2.5, 2.5, num=100)

Xt, Yt = xp.meshgrid(xuni, yuni)

U = interp2d(X, Y, U_, Xt, Yt)
V = interp2d(X, Y, V_, Xt, Yt)

#%%



spx = xp.arange(0.5, 3, 3.5 / 10)
spy = xp.arange(-1.5, 1.5, 3 / 10)
spx, spy = xp.meshgrid(spx, spy)
spx = spx.reshape(-1).reshape(1, -1)
spy = spy.reshape(-1).reshape(1, -1)

idx_ = 50
start_points = xp.concatenate((spx, spy), axis=0)
plt.figure()
plt.streamplot(
    Xt,
    Yt,
    U,
    V,
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

#%%

omega, _, _ = ibfs.compute_vorticity(0.0, q, mesh, bcs)

file = xp.loadtxt('data/taira_colonius_wake.csv', delimiter=',')

radius = 0.5
mask = (X**2 + Y**2) < radius**2
omega_masked = xp.where(mask, xp.nan, omega)

plt.figure()
plt.contour(X, Y, omega_masked, colors='gray', levels=xp.arange(-3, 3 + 0.4, 0.4), linestyles='solid')
plt.plot(file[:, 0], file[:, 1], 'ro', markersize=1.5, label=r'Taira & Colonius, JCP, 2007')
plt.fill(ib.xi, ib.eta, color="k")
ax = plt.gca()
ax.set_ylim([-2.5, 2.5])
ax.set_xlim([-2, 4])
ax.set_xlabel(r"$x/D$")
ax.set_ylabel(r"$y/D$")
ax.set_title(r"Steady-state vorticity contours at $Re = 40$")
ax.set_aspect("equal")
# plt.colorbar()
plt.legend()
plt.tight_layout()
plt.show()


#%%
