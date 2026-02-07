import numpy as xp
import torch
from typing import Tuple, TYPE_CHECKING, Optional, List
from scipy.spatial import cKDTree
import random as rand

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
        autodiff functionalities.)
    :type output_torch: Optional[bool], defaults to :code:`False`

    :rtype: List[xp.array] | List[torch.tensor]
    """

    # Meshgrids for the u velocity field
    Xu, Yu = xp.meshgrid(mesh.xu, mesh.yu)
    assert Xu.shape == mesh.u_ext.shape
    assert Yu.shape == mesh.u_ext.shape
    # Meshgrids for the v velocity field
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


def quadratic_interpolation(fc, xc, xcc, deriv=False):
    r"""
    Interpolate the field :math:`f` stored at locations :math:`x` to some target 
    coordinates :math:`\tilde{x}` using second-order Lagrange interpolation

    .. math::
        
        (\tilde{x}) = \sum_{i=j-1}^{j+1}f_i(x)
        \prod_{\substack{k=j-1\\k\neq i}}^{j+1}\frac{\tilde{x}-x_k}{x_j - x_k}

    :param fc: two-dimensional array containing the field :math:`f`. The interpolation
        is performed along the second axis of the array.
    :type fc: xp.array
    :param xc: :math:`x` coordinates at which the field is stored
    :type xc: xp.array
    :param xcc: target coordinates
    :type xcc: xp.array


    """

    def _interp_(fs, xs, x, deriv):
        r"""
        :param fs: :math:`\{f_{0}, f_{1}, f_{2}\}`. The first value is the
            "far upwind" one, the middle value is the "upwind" one, and the third value
            is the "downwind" one. So, for a face at :math:`x_{j+1/2}` with :math:`f_j > 0`
            we have :math:`\{f_{j-1}, f_{j}, f_{j+1}\}`. For face :math:`x_{j-1/2}` with
            :math:`f_j < 0`, we have :math:`\{f_{j+1}, f_{j}, f_{j-1}\}`.
        :param xs: :math:`\{x_{0}, x_{1}, x_{2}\}` (:math:`x`-coordinates of :math:`f_i`.)
        :param x: evaluation points
        """
        xjm1, xj, xjp1 = xs
        fjm1, fj, fjp1 = fs

        Ljm1 = (x - xj) * (x - xjp1) / ((xjm1 - xj) * (xjm1 - xjp1))
        Ljp1 = (x - xj) * (x - xjm1) / ((xjp1 - xj) * (xjp1 - xjm1))
        f = fj + (fjm1 - fj) * Ljm1[None, :] + (fjp1 - fj) * Ljp1[None, :]

        df = 0
        if deriv:
            dLjm1 = ((x - xj) + (x - xjp1)) / ((xjm1 - xj) * (xjm1 - xjp1))
            dLjp1 = ((x - xj) + (x - xjm1)) / ((xjp1 - xj) * (xjp1 - xjm1))
            df = (fjp1 - fj) * dLjp1[None, :] + (fjm1 - fj) * dLjm1[None, :]

        return (f, df)

    xjm1, xj, xjp1 = xc[:-2], xc[1:-1], xc[2:]
    fjm1, fj, fjp1 = fc[:, :-2], fc[:, 1:-1], fc[:, 2:]

    same_length = True if len(xcc) == len(xc) else False
    xtarget = xcc[1:-1] if same_length else xcc[1:]
    f_rgw, df_r = _interp_([fjm1, fj, fjp1], [xjm1, xj, xjp1], xtarget, deriv)

    if not same_length:
        f_lgw, df_l = _interp_(
            [fjp1[:, :1], fj[:, :1], fjm1[:, :1]],
            [xjp1[:1], xj[:1], xjm1[:1]],
            xcc[:1],
            deriv,
        )
        f = xp.concatenate((f_lgw, f_rgw), axis=-1)
        df = xp.concatenate((df_l, df_r), axis=-1) if deriv else 0
    else:
        f = xp.concatenate(
            (fc[:, 0][:, None], f_rgw, fc[:, -1][:, None]), axis=-1
        )
        df = 0
        if deriv:
            dx0 = xcc[1] - xcc[0]
            df0 = (-3 * f[:, 0] + 4 * f[:, 1] - f[:, 2]) / (2 * dx0)
            dx1 = xcc[-1] - xcc[-2]
            df1 = (f[:, -3] - 4 * f[:, -2] + f[:, -1]) / (2 * dx1)
            df = xp.concatenate((df0[:, None], df_r, df1[:, None]), axis=-1)

    return (f, df)


def compute_limited_face_values(fc, xc, xcc, f_upw):
    r"""
    Given the :math:`u` velocity stored at :math:`x`-staggered cell centroids, we compute
    the velocity :math:`u`at the faces of the :math:`x`-staggered cells (i.e., the grid
    cell centers). (Notice that
    for a uniform grid, the centroids are located at grid cell faces) The reconstruction is achieved with
    quadratic interpolation. In particular, at face :math:`x_{j+1/2}` we have

    .. math::

        u_{j+1/2} = \sum_{i=j-1}^{j+1}l_i(x)\prod_{\substack{k=j-1\\k\neq i}}^{j+1}\frac{x-x_k}{x_j - x_k}

    :param fc: array of size :math:`n_y \times n_x` corresponding to streamwise velocity
        values at :math:`x`-staggered-cell centroids
    :type fc: xp.array
    :param xc: :math:`x` coordinates of the staggered-cell centroids (size :math:`n_x`)
    :type xc: xp.array
    :param xcc: :math:`x` coordinates of the faces (size :math:`n_x - 1`)
    :type xcc: xp.array
    """

    def _interp(fs, xs, x):
        r"""
        :param fs: :math:`\{f_{0}, f_{1}, f_{2}\}`. The first value is the
            "far upwind" one, the middle value is the "upwind" one, and the third value
            is the "downwind" one. So, for a face at :math:`x_{j+1/2}` with :math:`f_j > 0`
            we have :math:`\{f_{j-1}, f_{j}, f_{j+1}\}`. For face :math:`x_{j-1/2}` with
            :math:`f_j < 0`, we have :math:`\{f_{j+1}, f_{j}, f_{j-1}\}`.
        :param xs: :math:`\{x_{0}, x_{1}, x_{2}\}` (:math:`x`-coordinates of :math:`f_i`.)
        :param x: evaluation points
        """
        xjm1, xj, xjp1 = xs
        fjm1, fj, fjp1 = fs

        Ljm1 = (x - xj) * (x - xjp1) / ((xjm1 - xj) * (xjm1 - xjp1))
        Ljp1 = (x - xj) * (x - xjm1) / ((xjp1 - xj) * (xjp1 - xjm1))
        s = (xjp1 - xj) / (xj - xjm1)
        num, den = (fj - fjm1), (fjp1 - fj)
        num_is_zero, den_is_zero = xp.abs(num) < 1e-15, xp.abs(den) < 1e-15
        r = xp.divide(
            num * s[None, :], den, out=xp.zeros_like(num), where=~den_is_zero
        )
        r[:, :] = xp.where(den_is_zero & ~num_is_zero, 1e10, r)
        phi = 2 * (Ljp1[None, :] - r * (Ljm1 / s)[None, :])

        twrb = 2 * r * ((x - xj) / (xjp1 - xj))[None, :]
        twos = 2 * xp.ones_like(phi)
        phi = xp.maximum(0 * twos, xp.minimum(twrb, phi, twos))

        return fj + 0.5 * phi * (fjp1 - fj)

    xjm1, xj, xjp1 = xc[:-2], xc[1:-1], xc[2:]
    fjm1, fj, fjp1 = fc[:, :-2], fc[:, 1:-1], fc[:, 2:]
    f_rgw = _interp([fjm1, fj, fjp1], [xjm1, xj, xjp1], xcc[1:])
    f_lgw = _interp([fjp1, fj, fjm1], [xjp1, xj, xjm1], xcc[:-1])

    f_rgw = xp.concatenate((fc[:, 0][:, None], f_rgw), axis=-1)
    f_lgw = xp.concatenate((f_lgw, fc[:, -1][:, None]), axis=-1)
    cp = xp.ones_like(f_rgw)
    cm = xp.ones_like(f_rgw)
    cp[f_upw <= 0] = 0.0
    cm[f_upw > 0] = 0.0

    return cp * f_rgw + cm * f_lgw
