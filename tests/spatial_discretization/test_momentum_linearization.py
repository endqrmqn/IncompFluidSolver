import numpy as xp
import ibfs
import pytest
from functools import partial
from .. import pytest_utils as pyut


def test_momentum_linearization(domain_and_Reynolds):
    r"""
    Test routines involved in the extraction
    of the linearized momentum equation matrix
    """
    xvec = [-3, -1, -0.5, 0.5, 1, 2]
    dxvec = [0.2, 0.1, 0.05, 0.075, 0.15]
    yvec = [-2, -1, 0.5, 0]
    dyvec = [0.2, 0.1, 0.05]
    mesh = ibfs.Mesh(xvec, dxvec, yvec, dyvec, False, True)
    bcs = pyut.instantiate_boundary_conditions(mesh)
    nsop = ibfs.SpatialOperators(100, mesh, bcs, False)

    eps = 1e-1
    szflow = xp.prod(mesh.u_int.shape) + xp.prod(mesh.v_int.shape)
    Qbflow = xp.random.randn(szflow)
    Qbflow /= xp.linalg.norm(Qbflow)
    rows, cols, rows_extract, cols_query = ibfs.momentum_sparsity_pattern(mesh)
    data = ibfs.momentum_data(nsop, rows_extract, cols_query, 0, Qbflow, eps)
    L = ibfs.assemble_matrix(rows, cols, data)

    f = lambda q: nsop.evaluate_right_hand_side(0, q)
    vec = xp.random.randn(szflow)
    vec /= xp.linalg.norm(vec)
    x = (f(Qbflow + eps * vec) - f(Qbflow - eps * vec)) / (2 * eps)
    error = 100 * xp.linalg.norm(x - L.dot(vec)) / xp.linalg.norm(x)
    print(error)
    assert error < 1e-3
