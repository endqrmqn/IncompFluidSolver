import numpy as xp
import ibfs
import os
import scipy as sp
import matplotlib.pyplot as plt


Re = 200

# Define the spatial domain
# discretized with 500 cells in the x direction and
# 250 in the y direction
x0, x1 = -4, 16
y0, y1 = -5, 5
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
ib = ibfs.ImmersedBody(*ibfs.make_airfoil(0.12, 50, 20), spops)

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
Q, tsave = tstep.solve(0.0, 5000 * dt, 100, Q[:, -1])


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

# Xu, Yu, _, _, _, _ = mesh.generate_meshgrids(False)

# plt.figure()
# plt.contourf(Xu, Yu, mesh.u_ext, cmap="bwr", levels=200)
# plt.fill(ib.xi, ib.eta, color="k")
# ax = plt.gca()
# # ax.set_ylim([-2.5, 2.5])
# # ax.set_xlim([-2, 4])
# ax.set_xlabel(r"$x/D$")
# ax.set_ylabel(r"$y/D$")
# ax.set_aspect("equal")
# plt.colorbar()
# plt.tight_layout()
# plt.show()


field = omega - omega_mean
vmin = xp.min(field)
vmax = xp.max(field)
vmin = - vmax

plt.figure()
plt.contourf(X, Y, field, cmap="bwr", levels=50, vmin=vmin, vmax=vmax)
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

omega = 2 * xp.pi / T

Qhat = sp.fft.rfft(Q_, axis=-1) / len(t_)
omegas = omega * xp.arange(Qhat.shape[-1])
dQHat = 1j * omegas * Qhat

for i in range (Q_.shape[-1]):
    print("Evaluating jacobian %d / %d" % (i + 1, len(t_)))
    t = t_[i]
    q = Q_[:, i]
    rows, cols, data, sz = ibfs.extract_full_jacobian(mesh, spops, t, q, 1, ib)
    
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

Nu = xp.prod(mesh.u_int.shape)
Nv = xp.prod(mesh.v_int.shape)
Np = xp.prod(mesh.p.shape)
N = Nu + Nv + Np + 2 * len(ib.xi) + 1

nf = 3
xp.save(path + 'freqs.npy', omegas[:(nf + 1)])
xp.save(path + 'rows.npy', Rows[:, 0])
xp.save(path + 'cols.npy', Cols[:, 0])
for j in range (nf + 1):
    xp.save(path + 'vals_%02d.npy' % j, DataHat[:, j])
    
    vec = xp.zeros(N, dtype=xp.complex128)
    vec[: Nu + Nv] = Qhat[:, j]
    xp.save(path + 'Q_%02d.npy' % j, vec)
    
    vec = xp.zeros(N, dtype=xp.complex128)
    vec[: Nu + Nv] = dQHat[:, j]
    xp.save(path + 'dQ_%02d.npy' % j, vec)

szvel = xp.prod(mesh.u_int.shape) + xp.prod(mesh.v_int.shape)
rows_id = xp.arange(szvel)
cols_id = rows_id.copy()
data_id = xp.ones(len(rows_id))

xp.save(path + 'rows_id.npy', rows_id)
xp.save(path + 'cols_id.npy', cols_id)
xp.save(path + 'vals_id.npy', data_id)

#%%

rows, cols, data, _ = ibfs.extract_full_jacobian(mesh, spops, 0.0, Q_[:, 0], 1.0, ib)

#%%
M = ibfs.assemble_matrix(Rows[:, 0], Cols[:, 0], DataHat[:, 0].real)

#%%
Mlu = sp.sparse.linalg.splu(M)

#%%


#%%
y = xp.random.randn(M.shape[0])
x = M.dot(y)

path = 'jacobian/'
os.makedirs(path, exist_ok=True)
# xp.save(path + 'rows.npy', rows)
# xp.save(path + 'cols.npy', cols)
# xp.save(path + 'data.npy', data)

xp.save(path + 'y.npy', y)
xp.save(path + 'x.npy', x)

#%%

N = xp.prod(mesh.u_int.shape) + xp.prod(mesh.v_int.shape) + xp.prod(mesh.p.shape) + 2 * len(ib.xi) + 1
for j in range (nf + 1):
    vec = xp.zeros(N, dtype=xp.complex128)
    vec[:szvel] = 1j * j * omega * Qhat[:, j]
    xp.save(path + 'Qhat_%02d.npy' % j, vec)

#%%

bcuvel_pert = ibfs.BoundaryConditions(
    "u",
    "dirichlet",
    "neumann",
    "dirichlet",
    "dirichlet",
    lambda t: 0 * ones_l_u,
    None,
    lambda t: 0 * ones_b_u,
    lambda t: 0 * ones_b_u,
)

bcs_pert = [bcuvel_pert, bcvvel]


k = 3
bcs_plot = bcs if k == 0 else bcs_pert

omega, X, Y = ibfs.compute_vorticity(0.0, dQHat[:, k].real, mesh, bcs_plot)

plt.figure()
plt.contourf(X, Y, omega, cmap="bwr", levels=200)
plt.fill(ib.xi, ib.eta, color="k")
ax = plt.gca()
ax.set_xlabel(r"$x/D$")
ax.set_ylabel(r"$y/D$")
ax.set_aspect("equal")
plt.colorbar()
plt.tight_layout()
plt.show()

#%%

omega, _, _ = ibfs.compute_vorticity(0.0, Qhat[:, 0].real, mesh, bcs)
omega = omega.reshape(-1)
OmegasHat = xp.zeros((len(omega), Qhat.shape[-1]), dtype=xp.complex128)
OmegasHat[:, 0] = omega
for i in range (1, OmegasHat.shape[-1]):
    omegar, _, _ = ibfs.compute_vorticity(0.0, Qhat[:, i].real, mesh, bcs_pert)
    omegai, _, _ = ibfs.compute_vorticity(0.0, Qhat[:, i].imag, mesh, bcs_pert)
    omega = (omegar + 1j * omegai).reshape(-1)
    OmegasHat[:, i] = omega
    
en = xp.linalg.norm(OmegasHat, axis=0) ** 2


plt.figure()
plt.stem(freqs[:len(en)], en, "o-")
ax = plt.gca()
ax.set_xticks(freqs)
ax.set_yscale("log")

#%%

omega, _, _ = ibfs.compute_vorticity(0.0, Q_[:, 1] - xp.mean(Q_, axis=-1), mesh, bcs_pert)

vmin = xp.min(omega)
vmax = xp.max(omega)
vmin = - vmax

plt.figure()
plt.contourf(X, Y, omega, cmap="bwr", levels=25, vmin=vmin, vmax=vmax)
plt.fill(ib.xi, ib.eta, color="k")
ax = plt.gca()
ax.set_xlabel(r"$x/D$")
ax.set_ylabel(r"$y/D$")
ax.set_aspect("equal")
plt.colorbar()
plt.tight_layout()
plt.show()