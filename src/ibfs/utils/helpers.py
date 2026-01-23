import numpy as xp
from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..spatial_discretization.mesh import Mesh
    from ..spatial_discretization.spatial_operators import SpatialOperators
    from ..spatial_discretization.boundary_conditions import BoundaryConditions


def vector_to_fields(
    t, q, mesh: "Mesh", bcs: Tuple["BoundaryConditions"]
) -> None:
    r"""
    Populate the fields :code:`mesh.u_int` and :code:`mesh.v_int` from
    a vector :math:`q = (u,v)\in\mathbb{R}^n`. Then, populate the boundary
    conditions in :code:`mesh.u_ext` and :code:`mesh.v_ext`.

    :param t: time
    :type t: float
    :param mesh: instance of :class:`Mesh`
    :type mesh: Mesh
    :param bcs: Tuple of instances of :class:`BoundaryConditions`
    :type bcs: Tuple[BoundaryConditions]
    """
    szu = xp.prod(mesh.u_int.shape)
    mesh.u_int[:, :] = q[:szu].reshape(mesh.u_int.shape)
    mesh.v_int[:, :] = q[szu:].reshape(mesh.v_int.shape)
    fields = [mesh.u_ext, mesh.v_ext]
    for i, f in enumerate(fields):
        bcs[i].impose_boundary_conditions(f, t)


def fields_to_vector(mesh: "Mesh") -> xp.array:
    r"""
    Inverse operation of the function :func:`vector_to_fields`.

    :rtype: xp.array
    """
    return xp.concatenate((mesh.u_int.reshape(-1), mesh.v_int.reshape(-1)))


def zero_out_fields(mesh: "Mesh") -> None:
    r"""
    Set :code:`mesh.u_ext` and :code:`v_ext` to zero.

    :param mesh: instance of :class:`Mesh`
    :type mesh: Mesh
    """
    fields = [mesh.u_ext, mesh.v_ext]
    for f in fields:
        f *= 0.0
