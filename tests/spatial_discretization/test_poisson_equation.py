import numpy as xp
import ibfs
import pytest
from .. import pytest_utils as pyut


def test_poisson_equation(domain_and_Reynolds):
    r"""
    Test for :func:`ibfs.SpatialOperators.enforce_divergence_free`.
    This implicitly tests all the routines involved in the extraction
    of the divergence and gradient matrices.
    """
    x0, x1, y0, y1, Re = domain_and_Reynolds
    ny, nx = 50, 100

    mesh = ibfs.Mesh(x0, x1, nx, y0, y1, ny)
    bcs = pyut.instantiate_boundary_conditions(mesh)
    nsop = ibfs.SpatialOperators(Re, mesh, bcs, False)

    vec = xp.random.randn(
        xp.prod(mesh.u_int.shape) + xp.prod(mesh.v_int.shape)
    )
    vec /= xp.linalg.norm(vec)
    Pvec = nsop.enforce_divergence_free(0.0, vec)
    ibfs.vector_to_fields(0.0, Pvec, mesh, bcs)
    error = xp.linalg.norm(nsop.evaluate_divergence().reshape(-1))
    assert error < 1e-10
