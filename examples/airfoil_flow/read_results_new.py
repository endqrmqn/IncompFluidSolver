#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 15:21:34 2026

@author: albertopadovan
"""

import numpy as xp
import scipy as sp
import ibfs
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.sans-serif": ["Computer Modern"],
        "font.size": 20,
        "text.usetex": True,
    }
)
plt.rc("text.latex", preamble=r"\usepackage{amsmath}")


def check_complex_conjugacy(vec, N):
    nf = int(len(vec) // N)
    vec = vec.reshape((N, nf), order="F")
    error = vec[:, : nf // 2] - xp.fliplr(vec[:, nf // 2 + 1 :].conj())
    print(xp.linalg.norm(error))
    print(xp.linalg.norm(vec[:, nf // 2].imag))


def rotate_modes(u, v, N):
    nf = int(len(u) // N)
    u0 = u.reshape((N, nf), order="F")[:, nf // 2]

    idx = xp.argmax(xp.abs(u0.imag))
    angle = xp.angle(u0[idx].real / u0[idx].imag)
    u = u * xp.exp(-1j * angle)
    v = v * xp.exp(-1j * angle)

    return u, v


def checkerboard_plot(U, S, V, N):
    nf = int((V.shape[0] // N - 1) / 2)
    Svals = xp.zeros((2 * nf + 1, 2 * nf + 1))

    for i in range(2 * nf + 1):
        Ui = U[i * N : (i + 1) * N,]
        for j in range(2 * nf + 1):
            Vj = V[j * N : (j + 1) * N,]

            print("Computing SVD (%d, %d)" % (i, j))
            Zr = xp.random.randn(N, U.shape[-1])
            Zi = xp.random.randn(N, U.shape[-1])
            Z = Zr + 1j * Zi if j != nf + 1 else Zr
            Q = Ui @ xp.diag(S[: U.shape[-1]]) @ (Vj.conj().T @ Z)
            Q = sp.linalg.orth(Q)
            M = (Q.conj().T @ Ui) @ (xp.diag(S[: U.shape[-1]]) @ Vj.conj().T)

            _, s, _ = sp.linalg.svd(M, full_matrices=False)
            Svals[i, j] = s[0]

    return Svals


# %%

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

# %%
# Read Resolvent4py Results


Nu = xp.prod(mesh.u_int.shape)
Nv = xp.prod(mesh.v_int.shape)
Np = xp.prod(mesh.p.shape)
N = Nu + Nv + Np + 2 * len(ib.xi) + 1
nfreqs = 7


Snp = xp.load("results/S_noproj.npy")
Sp = xp.load("results/S_proj.npy")

plt.figure()
plt.plot(Snp, "ko")
plt.plot(Sp, "rx")
ax = plt.gca()
ax.set_yscale("log")


# %%

Unp = xp.zeros((N * (2 * nfreqs + 1), 5), dtype=xp.complex128)
Vnp = Unp.copy()
Up = Unp.copy()
Vp = Unp.copy()

for i in range(Unp.shape[-1]):
    print("Loading modes %d / %d" % (i + 1, Unp.shape[-1]))
    unp = xp.load("results/u_noproj_%02d.npy" % i)
    vnp = xp.load("results/v_noproj_%02d.npy" % i)
    Unp[:, i], Vnp[:, i] = rotate_modes(unp, vnp, N)

    up = xp.load("results/u_proj_%02d.npy" % i)
    vp = xp.load("results/v_proj_%02d.npy" % i)
    Up[:, i], Vp[:, i] = rotate_modes(up, vp, N)

# %%

Svals_np = checkerboard_plot(Unp, Snp, Vnp, N)
Svals_p = checkerboard_plot(Up, Sp, Vp, N)

# %%

ticks = xp.arange(0, 15, 2)
labs = [r"$%d\omega$" % int(j) for j in (ticks - nfreqs)]

Svals_np_ = Svals_np**2 / xp.sum(Snp**2)
plt.figure()
im = plt.imshow(
    Svals_np_, cmap="gray_r", norm=LogNorm(vmin=3e-5, vmax=xp.max(Svals_np_))
)
ax = plt.gca()
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(labs)
ax.set_yticklabels(labs)
ax.set_title(r"W/o projection")
ax.set_xlabel(r"$(k\omega)_{in}$")
ax.set_ylabel(r"$(j\omega)_{out}$")
plt.colorbar(im)
plt.tight_layout()


Svals_p_ = Svals_p**2 / xp.sum(Sp**2)
plt.figure()
im = plt.imshow(
    Svals_p_, cmap="gray_r", norm=LogNorm(vmin=3e-5, vmax=xp.max(Svals_p_))
)
ax = plt.gca()
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(labs)
ax.set_yticklabels(labs)
ax.set_title(r"W/ projection")
ax.set_xlabel(r"$(k\omega)_{in}$")
ax.set_ylabel(r"$(j\omega)_{out}$")
plt.colorbar(im)
plt.tight_layout()

# %%


k = nfreqs + 0

j = 0
uj = Up[:, j].reshape((N, 2 * nfreqs + 1), order="F")[: Nu + Nv, k]
omegaj_, X, Y = ibfs.compute_vorticity(0.0, uj.real, mesh, bcs)

# vmin = 0.5 * xp.min(omegaj_)
# vmax = -vmin
vmax = 0.95 * xp.max(omegaj_)
vmin = -vmax

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
