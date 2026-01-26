import numpy as xp
import ibfs
import os
import scipy as sp
import matplotlib.pyplot as plt


Re = 200

# Define the spatial domain
# discretized with 500 cells in the x direction and
# 250 in the y direction
x0, x1 = -4, 12
y0, y1 = -2.5, 2.5
nx, ny = 400, 125
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
ib = ibfs.ImmersedBody(*ibfs.make_airfoil(0.12, 25, 20), spops)

# Instantiate the TimeStepper class to evolve the system
# We integrate with a fixed delta t = 1e-2
dt = 2.64 / 300
tstep = ibfs.TimeStepper(dt, spops, ib, scheme="RK2")

# %%
# Run the time stepper from t = 0 to t = 100 with
# initial condition q0 = 0
save_path = "data/"

q0 = xp.ones(xp.prod(mesh.u_int.shape) + xp.prod(mesh.v_int.shape))
q0[xp.prod(mesh.u_int.shape) :] = 0.0
# q0 = xp.load(save_path + 'snapshot.npy')
Q, tsave = tstep.solve(0.0, 5000 * dt, 100, q0)


# os.makedirs(save_path, exist_ok=True)
# xp.save(save_path + "snapshot.npy", Q[:, -1])

# %%
# diff = Q[:, 1:] - Q[:, :-1]
plt.figure()
plt.plot(tsave, xp.linalg.norm(Q - xp.mean(Q, axis=-1).reshape(-1, 1), axis=0), 'o-')
# plt.plot(tsave[::10], xp.linalg.norm(Q[:, ::10] - xp.mean(Q[:, ::10], axis=-1).reshape(-1, 1), axis=0), 'r-')

# %%

Qhat = sp.fft.rfft(Q[:, ::30], axis=-1) / len(tsave[::30])
en = xp.linalg.norm(Qhat[:, :7], axis=0)
T = tsave[-1] + dt
freqs = 2 * xp.pi  / T * xp.arange(len(en))

plt.figure()
plt.stem(freqs, en, "o-")
ax = plt.gca()
ax.set_xticks(freqs)
ax.set_yscale("log")

# %%
omega, X, Y = ibfs.compute_vorticity(0.0, Q[:, -1], mesh, bcs)
omega_mean, _, _ = ibfs.compute_vorticity(0.0, xp.mean(Q, axis=-1), mesh, bcs)

plt.figure()
plt.contourf(X, Y, omega - omega_mean, cmap="bwr", levels=200)
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


# %%

Q_ = Q[:, ::30]
t_ = tsave[::30]

for i in range (Q_.shape[-1]):
    print("Evaluating jacobian %d / %d" % (i + 1, len(t_)))
    t = t_[i]
    q = Q_[:, i]
    rows, cols, data, sz = ibfs.extract_full_jacobian(mesh, spops, t, q, 1)
    
    if i == 0:
        Rows = rows.reshape(-1, 1)
        Cols = cols.reshape(-1, 1)
        Data = data.reshape(-1, 1)
    else:
        Rows = xp.concatenate((Rows, rows.reshape(-1, 1)), axis=-1)
        Cols = xp.concatenate((Cols, cols.reshape(-1, 1)), axis=-1)
        Data = xp.concatenate((Data, data.reshape(-1, 1)), axis=-1)
        

#%%
DataHat = sp.fft.rfft(Data, axis=-1) / Data.shape[-1]

#%%
path = 'jacobian/'
os.makedirs(path, exist_ok=True)

omega = 2 * xp.pi / T
nf = 3
xp.save(path + 'freqs.npy', omega * xp.arange(nf + 1))
xp.save(path + 'rows.npy', Rows[:, 0])
xp.save(path + 'cols.npy', Cols[:, 0])
for j in range (nf + 1):
    xp.save(path + 'vals_%02d.npy' % j, DataHat[:, j])

szvel = xp.prod(mesh.u_int.shape) + xp.prod(mesh.v_int.shape)
rows_id = xp.arange(szvel)
cols_id = rows_id.copy()
data_id = xp.ones(len(rows_id))

xp.save(path + 'rows_id.npy', rows_id)
xp.save(path + 'cols_id.npy', cols_id)
xp.save(path + 'vals_id.npy', data_id)
