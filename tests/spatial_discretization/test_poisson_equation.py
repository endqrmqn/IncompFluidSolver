import numpy as xp
import ibfs
import pytest
from .. import pytest_utils as pyut


def test_poisson_equation():
    r"""
    Test for :func:`ibfs.SpatialOperators.enforce_divergence_free`.
    This implicitly tests all the routines involved in the extraction
    of the divergence and gradient matrices.
    """

    xvec = [-3, -1, -0.5, 0.5, 1, 2]
    dxvec = [0.2, 0.1, 0.05, 0.075, 0.15]
    yvec = [-2, -1, 0.5, 0]
    dyvec = [0.2, 0.1, 0.05]
    mesh = ibfs.Mesh(xvec, dxvec, yvec, dyvec, False, True)
    bcs = pyut.instantiate_boundary_conditions(mesh)
    nsop = ibfs.SpatialOperators(100, mesh, bcs, False)

    vec = xp.random.randn(
        xp.prod(mesh.u_int.shape) + xp.prod(mesh.v_int.shape)
    )
    vec /= xp.linalg.norm(vec)
    Pvec = nsop.enforce_divergence_free(0.0, vec)
    ibfs.vector_to_fields(0.0, Pvec, mesh, bcs)
    error = xp.linalg.norm(nsop.evaluate_divergence_integral().reshape(-1))
    assert error < 1e-10
