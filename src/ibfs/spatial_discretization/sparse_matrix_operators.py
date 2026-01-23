from typing import Tuple, List, TYPE_CHECKING
import numpy as xp
import scipy.sparse as sps
from functools import partial

if TYPE_CHECKING:
    from .mesh import Mesh
    from .spatial_operators import SpatialOperators

from ..utils.helpers import (
    vector_to_fields,
    zero_out_fields,
)


def gradient_sparsity_pattern(
    mesh: "Mesh",
) -> Tuple[xp.array, xp.array, xp.array, xp.array]:
    r"""
    Compute the sparsity pattern for the gradient operator
    :math:`G p = \nabla p` in COO format.

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
    Compute the sparsity pattern for the divergence operator
    :math:`D \mathbf{u} = \nabla \cdot \mathbf{u}` in
    COO format.

    :param mesh: instance of the :class:`Mesh` class
    :type mesh: Mesh
    :rtype: Tuple[xp.array, xp.array, xp.array, xp.array]
    """
    svec = xp.asarray([-2, -1, 0, 1, 2])
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


def momentum_sparsity_pattern(mesh: "Mesh"):
    r"""
    Compute the sparsity pattern for the momentum operator
    :math:`N(\mathbf{u}) = -\mathbf{u}\cdot\nabla \mathbf{u} + Re^{-1}\Delta \mathbf{u}`
    in COO format.

    :param mesh: instance of the :class:`Mesh` class
    :type mesh: Mesh
    :rtype: Tuple[xp.array, xp.array, xp.array, xp.array]
    """
    nyu, nxu = mesh.u_int.shape
    nyv, nxv = mesh.v_int.shape
    szu = nyu * nxu

    cols_mat_query, rows_mat_extract = [], []
    rows_mat, cols_mat = [], []

    stenc = xp.asarray(xp.arange(-2, 3, 1))
    s = len(stenc)
    for flowvar_col in range(2):
        cols = nxu if flowvar_col == 0 else nxv
        rows = nyu if flowvar_col == 0 else nyv
        shift = 0 if flowvar_col == 0 else szu

        for k in range(s**2):
            idx_j = xp.mod(k, s) - 2
            idx_i = k // s - 2

            j = xp.arange(idx_j, cols, s)
            i = xp.arange(idx_i, rows, s)
            j = j[j > -1]
            i = i[i > -1]

            j, i = xp.meshgrid(j, i)
            j = j.reshape(-1)
            i = i.reshape(-1)

            cols_mat_k = i * cols + j + shift
            cols_mat_query.append(cols_mat_k)
            rows_mat_extract_k = []
            for flowvar_row in range(2):
                cols_ = nxu if flowvar_row == 0 else nxv
                rows_ = nyu if flowvar_row == 0 else nyv
                shift_ = 0 if flowvar_row == 0 else szu

                for l, c in enumerate(cols_mat_k):
                    il, jl = i[l], j[l]
                    rows_mat_l = (
                        _compute_row_indices(il, jl, rows_, cols_, stenc)
                        + shift_
                    )
                    rows_mat.extend(rows_mat_l)
                    cols_mat.extend(
                        c * xp.ones(len(rows_mat_l), dtype=xp.int32)
                    )
                    rows_mat_extract_k.extend(rows_mat_l)

            rows_mat_extract.append(rows_mat_extract_k)

    return (
        xp.asarray(rows_mat),
        xp.asarray(cols_mat),
        rows_mat_extract,
        cols_mat_query,
    )


def gradient_data(
    spatial_ops: "SpatialOperators", rows_extract, cols_query, eps
):
    def grad_fun(p):
        spatial_ops.mesh.p[:, :] = p.reshape(*spatial_ops.mesh.p.shape)
        dpdx, dpdy = spatial_ops.evaluate_pressure_gradient()
        spatial_ops.mesh.p[:, :] = 0.0
        return xp.concatenate((dpdx.reshape(-1), dpdy.reshape(-1)))

    Q = xp.zeros(xp.prod(spatial_ops.mesh.p.shape))
    return compute_matrix_data(rows_extract, cols_query, grad_fun, Q, eps)


def divergence_data(
    spatial_ops: "SpatialOperators", rows_extract, cols_query, eps
):
    def div_fun(vec):
        vector_to_fields(0.0, vec, spatial_ops.mesh, spatial_ops.bcs)
        div = spatial_ops.evaluate_divergence().reshape(-1)
        zero_out_fields(spatial_ops.mesh)
        return div

    Q = xp.zeros(
        xp.prod(spatial_ops.mesh.u_int.shape)
        + xp.prod(spatial_ops.mesh.v_int.shape)
    )
    return compute_matrix_data(rows_extract, cols_query, div_fun, Q, eps)


def momentum_data(
    spatial_ops: "SpatialOperators", rows_extract, cols_query, t, Qbflow, eps
):
    fun = lambda q: spatial_ops.evaluate_right_hand_side(t, q)
    return compute_matrix_data(rows_extract, cols_query, fun, Qbflow, eps)


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
    rows, cols, data = _eliminate_zeros(rows, cols, data)
    return sps.csc_matrix((data, (rows, cols)), shape=(nr, nc))


def _compute_row_indices(i, j, ny, nx, stencil):
    rows_i = i + stencil
    rows_i = rows_i[(rows_i > -1) & (rows_i < ny)]
    rows_j = j + stencil
    rows_j = rows_j[(rows_j > -1) & (rows_j < nx)]
    rows_j, rows_i = xp.meshgrid(rows_j, rows_i)
    return (rows_i * nx + rows_j).reshape(-1)


def _eliminate_zeros(rows, cols, data):
    idces = xp.argwhere(xp.abs(data) <= 1e-13)
    rows = xp.delete(rows, idces)
    cols = xp.delete(cols, idces)
    data = xp.delete(data, idces)
    return rows, cols, data
