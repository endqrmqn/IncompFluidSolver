import numpy as xp
import scipy.sparse as sps
from typing import Tuple, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .mesh import Mesh
    from .spatial_operators import SpatialOperators
    from .boundary_conditions import BoundaryConditions


def interpolate_1d(w: xp.array, axis: int) -> xp.array:
    r"""
    Perform 1d interpolation of the :math:`r`-dimensional array
    :math:`w` along the direction specified by :code:`axis`.

    :param w: two-dimensional array we wish to interpolate
    :type w: xp.array
    :param axis: direction of interpolation
    :type axis: int

    .. attention::

        This function returns an array of a different size than the
        original input array :math:`w`. If :math:`w` has size :math:`n\times m`,
        the dimension along :code:`axis` is reduced by :math:`1`.

    :rtype: xp.array
    """
    # For reference: slice(start, stop, step)
    slc0 = [slice(None)] * w.ndim
    slc1 = [slice(None)] * w.ndim
    slc0[axis] = slice(None, -1)  # Slice from 0 to -1 in axis direction
    slc1[axis] = slice(1, None)  # Slice from 1 to end in axis direction
    return 0.5 * (w[tuple(slc0)] + w[tuple(slc1)])


def evaluate_derivative(w: xp.array, d: float, order: int, axis: int):
    r"""
    Evaluate derivative of field :math:`w` in the direction specified by :code:`axis`.
    The order of the derivative (i.e., first or second) is specified by :code:`order`.
    We use a second-order central difference scheme.

    :param w: two-dimensional array
    :type w: xp.array
    :param d: grid spacing
    :type d: float
    :param order: order of the derivative
    :type order: int
    :param axis: direction along which to take the derivative
    :type axis: int

    :rtype: xp.array
    """
    slcm1 = [slice(None)] * w.ndim
    slcp1 = [slice(None)] * w.ndim
    slc0 = [slice(None)] * w.ndim
    slcm1[axis] = slice(None, -2)  # Slice from 0 to -1 in axis direction
    slcp1[axis] = slice(2, None)  # Slice from 1 to end in axis direction
    slc0[axis] = slice(1, -1, 1)

    return (
        (w[tuple(slcp1)] - w[tuple(slcm1)]) / (2 * d)
        if order == 1
        else (w[tuple(slcp1)] - 2 * w[tuple(slc0)] + w[tuple(slcm1)]) / (d**2)
    )


def evaluate_derivative_staggered(
    w: xp.array, d: float, axis: int
) -> xp.array:
    r"""
    Evaluate the first derivative of field :math:`w` in
    the direction specified by :code:`axis`.
    This function is used for the gradient and divergence evaluation
    in the Navier-Stokes equation.

    :param w: two-dimensional array
    :type w: xp.array
    :param d: grid spacing
    :type d: float
    :param axis: direction along which to take the derivative
    :type axis: int

    :rtype: xp.array
    """
    slcm = [slice(None)] * w.ndim
    slcp = [slice(None)] * w.ndim
    slcm[axis] = slice(None, -1)
    slcp[axis] = slice(1, None)
    return (w[tuple(slcp)] - w[tuple(slcm)]) / d


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
