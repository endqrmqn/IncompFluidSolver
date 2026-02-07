import numpy as xp
from typing import Tuple, TYPE_CHECKING
from .helpers import vector_to_fields, quadratic_interpolation

if TYPE_CHECKING:
    from ..spatial_discretization.mesh import Mesh
    from ..spatial_discretization.spatial_operators import SpatialOperators
    from ..spatial_discretization.boundary_conditions import BoundaryConditions


def compute_vorticity(
    t, q, mesh: "Mesh", bcs: Tuple["BoundaryConditions"]
) -> Tuple[xp.array, xp.array, xp.array]:
    r"""
    Compute the vorticity field given a velocity vector
    a vector :math:`q = (u,v)\in\mathbb{R}^n`. The vorticity field
    is naturally stored at cell nodes. The function also returns
    the meshgrid fields :code:`Xw` and :code:`Yw` for plotting purposes.

    :param t: time
    :type t: float
    :param mesh: instance of :class:`Mesh`
    :type mesh: Mesh
    :param bcs: Tuple of instances of :class:`BoundaryConditions`
    :type bcs: Tuple[BoundaryConditions]

    :rtype: Tuple[xp.array, xp.array, xp.array]
    """
    vector_to_fields(t, q, mesh, bcs)

    u = mesh.u_ext
    v = mesh.v_ext
    _, dudy = quadratic_interpolation(u.T, mesh.yu, mesh.y, deriv=True)
    _, dvdx = quadratic_interpolation(v, mesh.xv, mesh.x, deriv=True)
    Xw, Yw = xp.meshgrid(mesh.x, mesh.y)

    return dvdx - dudy.T, Xw, Yw


def interpolate_to_nodes(
    t, q, mesh: "Mesh", bcs: Tuple["BoundaryConditions"]
) -> Tuple[xp.array, xp.array, xp.array]:
    r"""
    Interpolate the :math:`u` and :math:`v` velocity fields in
    :math:`q =(u, v)` onto the cell nodes. The function also returns
    the meshgrid fields :code:`Xw` and :code:`Yw` for plotting purposes.

    :param t: time
    :type t: float
    :param mesh: instance of :class:`Mesh`
    :type mesh: Mesh
    :param bcs: Tuple of instances of :class:`BoundaryConditions`
    :type bcs: Tuple[BoundaryConditions]

    :rtype: Tuple[xp.array, xp.array, xp.array]
    """
    vector_to_fields(t, q, mesh, bcs) 
    u = mesh.u_ext
    v = mesh.v_ext
    un, _ = quadratic_interpolation(u.T, mesh.yu, mesh.y, deriv=False)
    vn, _ = quadratic_interpolation(v, mesh.xv, mesh.x, deriv=False)
    Xw, Yw = xp.meshgrid(mesh.x, mesh.y)

    return un.T, vn, Xw, Yw
