from typing import Tuple, List, TYPE_CHECKING
import numpy as xp
import scipy.sparse as sps

if TYPE_CHECKING:
    from .mesh import Mesh


def gradient_sparsity_pattern(
    mesh: "Mesh"
) -> Tuple[xp.array, xp.array, xp.array, xp.array]:
    r"""
    Compute the sparsity pattern for the gradient operator in 
    COO format.

    :param mesh: instance of the :class:`Mesh` class
    :type mesh: Mesh
    :rtype: Tuple[xp.array, xp.array, xp.array, xp.array]
    """
    svec = xp.asarray([-1, 0, 1])
    s = len(svec)

    nyu, nxu = mesh.u_int.shape
    nyv, nxv = mesh.v_int.shape
    nyp, nxp = mesh.p.shape

    cols_mat_query, rows_mat_extract = [], []
    rows_mat, cols_mat = [], []

    for k in range(s):
        # Identify pressure nodes that can be perturbed simultaneosuly
        i_idces = xp.arange(nyp)
        j_idces = xp.arange(k, nxp, s)
        j, i = xp.meshgrid(j_idces, i_idces)
        j = j.reshape(-1)
        i = i.reshape(-1)
        # Identify u velocity nodes that are impacted by each perturbation
        cols_mat_k = i * nxp + j
        cols_mat_query.append(cols_mat_k)
        rows_mat_extract_k = []
        for l, c in enumerate(cols_mat_k):
            vec_jl = j[l] + svec
            vec_jl = vec_jl[(vec_jl > -1) & (vec_jl < nxu)]
            rows_mat_l = i[l] * nxu + vec_jl
            rows_mat.extend(rows_mat_l)
            cols_mat.extend(c * xp.ones(len(rows_mat_l), dtype=xp.int32))
            rows_mat_extract_k.extend(rows_mat_l)
        rows_mat_extract.append(rows_mat_extract_k)

        # Identify pressure nodes that can be perturbed simultaneosuly
        i_idces = xp.arange(k, nyp, s)
        j_idces = xp.arange(nxp)
        j, i = xp.meshgrid(j_idces, i_idces)
        j = j.reshape(-1)
        i = i.reshape(-1)
        # Identify v velocity nodes that are impacted by each perturbation
        cols_mat_k = i * nxp + j
        cols_mat_query.append(cols_mat_k)
        rows_mat_extract_k = []
        for l, c in enumerate(cols_mat_k):
            vec_il = i[l] + svec
            vec_il = vec_il[(vec_il > -1) & (vec_il < nyv)]
            rows_mat_l = vec_il * nxv + j[l] + nyu * nxu
            rows_mat.extend(rows_mat_l)
            cols_mat.extend(c * xp.ones(len(rows_mat_l), dtype=xp.int32))
            rows_mat_extract_k.extend(rows_mat_l)
        rows_mat_extract.append(rows_mat_extract_k)

    rows = xp.asarray(rows_mat)
    cols = xp.asarray(cols_mat)
    return (rows, cols, rows_mat_extract, cols_mat_query)

def divergence_sparsity_pattern(mesh: "Mesh"):
    r"""
    Compute the sparsity pattern for the divergence operator in 
    COO format. 

    :param mesh: instance of the :class:`Mesh` class
    :type mesh: Mesh
    :rtype: Tuple[xp.array, xp.array, xp.array, xp.array]
    """
    svec = xp.asarray([-1, 0, 1])
    s = len(svec)

    nyu, nxu = mesh.u_int.shape
    nyv, nxv = mesh.v_int.shape
    nyp, nxp = mesh.p.shape

    cols_mat_query, rows_mat_extract = [], []
    rows_mat, cols_mat = [], []

    # Extracting rows/columns corresponding to u velocity
    i_idces = xp.arange(nyu)
    for k in range(s):
        # Identify velocity nodes that can be perturbed simultaneosuly
        j_idces = xp.arange(k, nxu, s)
        j, i = xp.meshgrid(j_idces, i_idces)
        j = j.reshape(-1)
        i = i.reshape(-1)
        # Identify pressure nodes that are impacted by each perturbation
        cols_mat_k = i * nxu + j
        cols_mat_query.append(cols_mat_k)
        rows_mat_extract_k = []
        for l, c in enumerate(cols_mat_k):
            vec_jl = j[l] + svec
            vec_jl = vec_jl[(vec_jl > -1) & (vec_jl < nxp)]
            rows_mat_l = i[l] * nxp + vec_jl
            rows_mat.extend(rows_mat_l)
            cols_mat.extend(c * xp.ones(len(rows_mat_l), dtype=xp.int32))
            rows_mat_extract_k.extend(rows_mat_l)
        rows_mat_extract.append(rows_mat_extract_k)

    # Extracting rows/columns corresponding to v velocity
    j_idces = xp.arange(nxv)
    for k in range(s):
        # Identify velocity nodes that can be perturbed simultaneosuly
        i_idces = xp.arange(k, nyv, s)
        j, i = xp.meshgrid(j_idces, i_idces)
        j = j.reshape(-1)
        i = i.reshape(-1)
        # Identify pressure nodes that are impacted by each perturbation
        cols_mat_k = i * nxv + j + nyu * nxu
        cols_mat_query.append(cols_mat_k)
        rows_mat_extract_k = []
        for l, c in enumerate(cols_mat_k):
            vec_il = i[l] + svec
            vec_il = vec_il[(vec_il > -1) & (vec_il < nyp)]
            rows_mat_l = vec_il * nxp + j[l]
            rows_mat.extend(rows_mat_l)
            cols_mat.extend(c * xp.ones(len(rows_mat_l), dtype=xp.int32))
            rows_mat_extract_k.extend(rows_mat_l)
        rows_mat_extract.append(rows_mat_extract_k)

    rows = xp.asarray(rows_mat)
    cols = xp.asarray(cols_mat)

    return (rows, cols, rows_mat_extract, cols_mat_query)

def gradient_data(mesh, spatial_ops, rows_extract, cols_query, eps):
    def grad_fun(p):
        mesh.p[:, :] = p.reshape(*mesh.p.shape)
        dpdx, dpdy = spatial_ops.evaluate_pressure_gradient(mesh)
        dpdx = dpdx.reshape(-1)
        dpdy = dpdy.reshape(-1)
        return xp.concatenate((dpdx, dpdy))
    Q = xp.zeros(xp.prod(mesh.p.shape))
    return compute_matrix_data(
        rows_extract, cols_query, grad_fun, Q, eps
    )

def divergence_data(mesh, spatial_ops, rows_extract, cols_query, eps):
    def div_fun(vec):
        szu = xp.prod(mesh.u_int.shape)
        mesh.u_ext *= 0.0
        mesh.v_ext *= 0.0
        mesh.u_int[:, :] = vec[:szu].reshape(*mesh.u_int.shape)
        mesh.v_int[:, :] = vec[szu:].reshape(*mesh.v_int.shape)
        return spatial_ops.evaluate_divergence(mesh).reshape(-1)
    Q = xp.zeros(xp.prod(mesh.u_int.shape) + xp.prod(mesh.v_int.shape))
    return compute_matrix_data(
        rows_extract, cols_query, div_fun, Q, eps
    )

def compute_matrix_data(rows_extract, cols_query, fun, Q, eps):
    data = []
    for k in range(len(cols_query)):
        vec = xp.zeros(len(Q))
        vec[cols_query[k]] = 1.0
        q = (fun(Q + eps * vec) - fun(Q - eps * vec)) / (2 * eps)
        data.extend(q[rows_extract[k]])
    return data

def assemble_matrix(rows, cols, data):
    nr = xp.max(rows) + 1
    nc = xp.max(cols) + 1
    rows, cols, data = eliminate_zeros(rows, cols, data)
    return sps.csc_matrix((data, (rows, cols)), shape=(nr, nc))
    
def eliminate_zeros(rows, cols, data):
    idces = xp.argwhere(xp.abs(data) <= 1e-13)
    rows = xp.delete(rows, idces)
    cols = xp.delete(cols, idces)
    data = xp.delete(data, idces)
    return rows, cols, data
