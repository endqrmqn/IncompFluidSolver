import numpy as xp
import torch
from typing import Tuple, TYPE_CHECKING, Optional, List
from scipy.spatial import cKDTree

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


def generate_meshgrids(
    mesh: "Mesh", output_torch: Optional[bool] = False
) -> List[xp.array] | List[torch.tensor]:
    r"""
    Generate meshgrids for the :math:`u`, :math:`v`, and :math:`p` fields as a list
    :code:`[Xu, Yu, Xv, Yv, Xp, Yp]`.

    :param output_torch: outputs the meshgrids as torch tensors if :code:`True`,
        and as numpy/cupy arrays otherwise. (Torch is often used in tests/ for its
        autodiff capabilities.)
    :type output_torch: Optional[bool], defaults to :code:`False`

    :rtype: List[xp.array] | List[torch.tensor]
    """

    # Meshgrids for the u velocity field
    # xu = mesh.x
    # yub = 2 * mesh.yu[0] - mesh.yu[1]
    # yut = 2 * mesh.yu[-1] - mesh.yu[-2]
    # yu = xp.concatenate(([yub], mesh.yu, [yut]))
    # Xu, Yu = xp.meshgrid(xu, yu)
    Xu, Yu = xp.meshgrid(mesh.xu, mesh.yu)
    assert Xu.shape == mesh.u_ext.shape
    assert Yu.shape == mesh.u_ext.shape
    # Meshgrids for the v velocity field
    # xvl = 2 * mesh.xv[0] - mesh.xv[1]
    # xvr = 2 * mesh.xv[-1] - mesh.xv[-2]
    # xv = xp.concatenate(([xvl], mesh.xv, [xvr]))
    # yv = mesh.y
    # Xv, Yv = xp.meshgrid(xv, yv)
    Xv, Yv = xp.meshgrid(mesh.xv, mesh.yv)
    assert Xv.shape == mesh.v_ext.shape
    assert Yv.shape == mesh.v_ext.shape
    # Meshgrids for the pressure field
    Xp, Yp = xp.meshgrid(mesh.xc, mesh.yc)

    xp_tensors = [Xu, Yu, Xv, Yv, Xp, Yp]
    if output_torch:
        torch_tensors = [
            torch.tensor(t, requires_grad=True) for t in xp_tensors
        ]
        return xp_tensors, torch_tensors
    else:
        return xp_tensors


# class LinearReconstruction:

#     def __init__(self):

#         self.a = 0
#         self.b = 0
#         self.xc = 0
#         self.bounds = 0

#     def create_interpolant(self, xc, xf, u):

#         dx = xf[1:] - xf[:-1]
#         xf2, xf1, xf0 = xf[2:], xf[1:-1], xf[:-2]
#         lhs = (0.5 * (xf2 ** 2 - xf1 ** 2) - xc[:-1] * (xf2 - xf1)).reshape(1, -1)
#         rhs = u[:, 1:] * dx[1:].reshape(1, -1) - u[:, :-1] * (xf2 - xf1).reshape(1, -1)
#         a = rhs / lhs

#         xf2, xf1, xf0 = xf[-1], xf[-2], xf[-3]
#         lhs = 0.5 * (xf1 ** 2 - xf0 ** 2) - xc[-1] * (xf1 - xf0)
#         rhs = u[:, -2] * dx[-2] - u[:, -1] * (xf1 - xf0)

#         self.a = xp.concatenate((a, (rhs / lhs).reshape(-1, 1)), axis=0)
#         self.b = u.copy()
#         self.xc = xc.copy()
#         self.bounds = xf.copy()
#         self.tree = cKDTree(self.xc.reshape(-1, 1))

#     def interpolate(self, x):
#         _, indices = cKDTree.query(x.reshape(-1, 1))
#         return self.a[:, indices] * ((x - x[indices]).reshape(1, -1)) + self.b

#     def derivative(self, x):
#         _, indices = cKDTree.query(x.reshape(-1, 1))
#         return self.a[:, indices]


class FastLinearReconstruction:
    """
    Fast, vectorized 1D linear reconstruction for 2D arrays (n x m),
    using precomputed slopes for interpolation and derivative evaluation.
    """

    def __init__(self):
        self.a = None  # slopes
        self.b = None  # intercepts
        self.x = None  # grid points
        self.n = 0
        self.m = 0

    def create_interpolant(self, x, y):
        """
        Precompute slopes and intercepts for linear interpolation.

        Parameters
        ----------
        x : 1D array of length m, strictly increasing
        y : 2D array of shape (n, m) - function values at x
        """
        self.x = xp.asarray(x)
        y = xp.asarray(y)
        self.n, self.m = y.shape

        dx = self.x[1:] - self.x[:-1]  # length m-1
        self.a = (y[:, 1:] - y[:, :-1]) / dx[None, :]  # n x (m-1)
        self.b = y[:, :-1]  # n x (m-1)

    def interpolate(self, x_eval):
        """
        Evaluate linear interpolation at x_eval points.

        Parameters
        ----------
        x_eval : array-like of shape (k,)

        Returns
        -------
        y_eval : array of shape (n, k)
        """
        x_eval = xp.asarray(x_eval)
        # Find interval indices
        indices = xp.searchsorted(self.x, x_eval) - 1
        indices = xp.clip(indices, 0, self.m - 2)

        # Vectorized evaluation
        y_eval = (
            self.a[:, indices] * (x_eval - self.x[indices])[None, :]
            + self.b[:, indices]
        )
        return y_eval

    def derivative(self, x_eval):
        """
        Evaluate derivative (slope) at x_eval points.

        Parameters
        ----------
        x_eval : array-like of shape (k,)

        Returns
        -------
        dy_dx : array of shape (n, k)
        """
        x_eval = xp.asarray(x_eval)
        indices = xp.searchsorted(self.x, x_eval) - 1
        indices = xp.clip(indices, 0, self.m - 2)

        return self.a[:, indices]
