import resolvent4py as res4py
from petsc4py import PETSc
from slepc4py import SLEPc
import numpy as np

nx = 800
ny = 250
Nib = 98

Nu = (nx - 1) * ny
Nv = nx * (ny - 1)
Np = nx * ny + 1
n = Nu + Nv + Np + 2 * Nib

comm = PETSc.COMM_WORLD
res4py.petscprint(comm, "Started job...")

path = 'jacobian/'


# --- Load Matrices
bflow_freqs = np.load(path + "freqs.npy")
omega_f = bflow_freqs[1]
bflow_freqs = bflow_freqs[:2]
nfb = len(bflow_freqs) - 1
fnames_lst = [
    (
        path + "rows_%02d.dat" % j,
        path + "cols_%02d.dat" % j,
        path + "vals_%02d.dat" % j,
    )
    for j in range(nfb + 1)
]

nfp = nfb + 3
perts_freqs = np.arange(-nfp, nfp + 1) * omega_f
nblocks = 2 * nfp + 1

N = n * len(perts_freqs)
Nl = res4py.compute_local_size(N)
nl = res4py.compute_local_size(n)
res4py.petscprint(comm, "%d %d" % (N, n))
A = res4py.read_harmonic_balanced_matrix(
    fnames_lst,
    True,
    ((nl, n), (nl, n)),
    ((Nl, N), (Nl, N)),
)
A.scale(-1.0)
res4py.petscprint(comm, "Done reading Fourier coefficients of A(t).")
fnames_lst_mass = [
    (
        path + "rows_id.dat",
        path + "cols_id.dat",
        path + "vals_id.dat",
    )
]
Mass = res4py.read_harmonic_balanced_matrix(
    fnames_lst_mass,
    True,
    ((nl, n), (nl, n)),
    ((Nl, N), (Nl, N)),
)
res4py.petscprint(comm, "Done reading mass matrix.")
MassOp = res4py.linear_operators.MatrixLinearOperator(Mass, nblocks=nblocks)
T = res4py.assemble_harmonic_resolvent_generator(A, perts_freqs, Mass)
res4py.petscprint(comm, "Matrix size = %d"%N)

# Compute LU factorization
res4py.petscprint(comm, "Computing LU decomposition...")
s = 0.0
Tinv = Mass.copy()
Tinv.scale(s)
Tinv.axpy(-1.0, T)
ksp = res4py.create_mumps_solver(Tinv)
res4py.check_lu_factorization(Tinv, ksp)
TinvOp = res4py.linear_operators.MatrixLinearOperator(Tinv, ksp, nblocks=nblocks)
res4py.petscprint(comm, TinvOp.get_block_cc_flag())

# Read time-derivative of the base flow
fnames_lst = [(path + "Qhat_%02d.dat" % j) for j in range(len(bflow_freqs))]
dQ = res4py.read_harmonic_balanced_vector(fnames_lst, True, (nl, n), (Nl, N))
dQ.scale(1 / dQ.norm())
w = TinvOp.solve_hermitian_transpose(dQ)
Mw = MassOp.apply(w)
Mw.scale(1 / Mw.norm())

Phi = SLEPc.BV().create(comm)
Phi.setSizes(dQ.getSizes(), 1)
Phi.setType("mat")
Psi = Phi.copy()
Phi.insertVec(0, dQ)
Psi.insertVec(0, Mw)

Pd = res4py.linear_operators.ProjectionLinearOperator(Psi, Psi, True, nblocks)
Pr = res4py.linear_operators.ProjectionLinearOperator(Phi, Phi, True, nblocks)

lops = [Pr, MassOp, TinvOp, MassOp, Pd]
lops_actions = [Pr.apply, MassOp.apply, TinvOp.solve, MassOp.apply, Pd.apply]
Linop = res4py.linear_operators.ProductLinearOperator(
    lops, lops_actions, nblocks
)

U, S, V = res4py.randomized_svd(Linop, Linop.apply, 50, 2, 10, verbose=1)

res4py.write_to_file('results/U.dat', U)
res4py.write_to_file('results/V.dat', V)
if comm.getRank() == 0:
    np.save('results/S.npy', np.diag(S))

#x = res4py.read_vector('jacobian/x.dat', (Nl, N))
#y = res4py.read_vector('jacobian/y.dat', (Nl, N))
#y_ = TinvOp.solve(x, y)
#y.axpy(-1.0, y_)
#res4py.petscprint(comm, y_.norm())

# res4py.petscprint(comm, "Computing eigendecomposition...")
# lops = [MassOp, TinvOp, MassOp]
# lops_actions = [MassOp.apply, TinvOp.solve, MassOp.apply]
# Linop = res4py.linear_operators.ProductLinearOperator(
#     lops, lops_actions, nblocks
# )

# Dfwd, _ = res4py.linalg.eig(Linop, Linop.apply, 600, 100, lambda x: s - 1 / x)
# Dfwd = np.diag(Dfwd)

# np.save('results_petsc/floquet_exponents_%1.4f.npy' % np.imag(s), Dfwd)

# if comm.getRank() == 0:
#     Vnp = np.zeros((N, Vfwd.getSizes()[-1]), dtype=np.complex128)

# for j in range (Vfwd.getSizes()[-1]):
#     vj = Vfwd.getColumn(j)
#     vjseq = res4py.distributed_to_sequential_vector(vj)
#     if comm.getRank() == 0:
#         Vnp[:, j] = vjseq.getArray()
#     vjseq.destroy()
#     Vfwd.restoreColumn(j, vj)

# if comm.getRank() == 0: 
#     np.save('results_petsc/floquet_eigfunctions_nfp%02d.npy' % nfp, Vnp)


# plt.figure()
# plt.plot(Dfwd.real, Dfwd.imag, 'ko')
# ax = plt.gca()
# ax.axhline(y = np.imag(s), color='r', alpha=0.5)
# ax.axvline(x = 0.0, color='b', alpha=0.5)
# plt.tight_layout()
# plt.savefig("results_petsc/Floquet_exponents.png")
