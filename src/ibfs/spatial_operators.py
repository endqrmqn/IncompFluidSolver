import numpy as xp
import scipy as sp
from .helpers import (
    interpolate_1d,
    evaluate_derivative,
    evaluate_derivative_staggered,
)
from typing import Tuple


class SpatialOperators:
    r"""
    Class to evaluate the right-hand side of the Navier-Stokes equation.
    """

    def __init__(self, Re):
        self.Re = Re

    def evaluate_laplacian(self, w: xp.array, d: float) -> xp.array: #type: ignore
        r"""
        Compute

        .. math::

            Re^{-1}\nabla^2 w = Re^{-1}\left(\partial_x^2 w + \partial_y^2 w\right)

        where :math:`w` is either the :math:`x` or :math:`y` velocity field.

        .. attention::

            This function returns an array of a different size than the
            original input array :math:`w`. If :math:`w` has size :math:`n\times m`,
            the dimension of the output array will be :math:`(n-2)\times (m-2)`.

        :param w: two-dimensional array
        :type w: numpy/cupy array
        :param d: grid spacing
        :type d: float

        :rtype: cupy/numpy array
        """
        size = tuple([s - 2 for s in w.shape])
        dw = xp.zeros(size, dtype=w.dtype)
        axes = xp.arange(w.ndim, dtype=xp.int32)
        for axis in axes:
            slc = [slice(1, -1) if i != axis else slice(None) for i in axes]
            dw += evaluate_derivative(w, d, 2, axis)[tuple(slc)] # type: ignore
        return dw

    def evaluate_streamwise_advection(
        self, u: xp.array, v: xp.array, d: float #type: ignore
    ) -> xp.array: #type: ignore
        r"""
        Compute

        .. math::

            \int_{Y} u^2(x,y)\big\lvert_{x=x_{l}}^{x=x_r}\,dy +
            \int_{X}\left[u(x,y)v(x,y)\right]_{y={y_b}}^{y=y_{t}}\,dx

        on each finite volume (see Figure 2 in the docs/notes/cfd_solver_details.pdf).

        .. attention::

            This function returns an array of size :math:`n_y \times n_x - 1`,
            where :math:`n_y` and :math:`n_x` are the number of internal cell
            centers in the :math:`y` and :math:`x` directions, respectively.

        :param u: streamwise velocity field
        :type u: numpy/cupy array
        :param v: wall-normal velocity field
        :type v: numpy/cupy array
        :param d: grid spacing
        :type d: float

        :rtype: cupy/numpy array
        """
        # Compute u_interp_x = (u_{i,j+1} + u_{i,j})/2
        u_interp_x = interpolate_1d(u, axis=1)
        # Compute u_interp_y = (u_{i+1,j} + u_{i,j})/2
        u_interp_y = interpolate_1d(u, axis=0)
        # Compute v_interp_x = (v_{i,j+1} + v_{i,j})/2
        v_interp_x = interpolate_1d(v, axis=1)

        return (
            u_interp_x[1:-1, 1:] ** 2
            - u_interp_x[1:-1, :-1] ** 2
            + u_interp_y[1:, 1:-1] * v_interp_x[1:, 1:-1]
            - u_interp_y[:-1, 1:-1] * v_interp_x[:-1, 1:-1]
        ) * d

    def evaluate_wallnormal_advection(
        self, u: xp.array, v: xp.array, d: float        #type: ignore
    ) -> xp.array:                                      #type: ignore
        r"""
        Compute

        .. math::

            \int_{X} v^2(x,y)\big\lvert_{y=y_{b}}^{y=y_t}\,dx +
            \int_{Y}\left[u(x,y)v(x,y)\right]_{x={x_l}}^{x=x_{t}}\,dy

        on each finite volume (see Figure 3 in the docs/notes/cfd_solver_details.pdf).

        .. attention::

            This function returns an array of size :math:`n_y-1 \times n_x`,
            where :math:`n_y` and :math:`n_x` are the number of internal cell
            centers in the :math:`y` and :math:`x` directions, respectively.

        :param u: streamwise velocity field
        :type u: numpy/cupy array
        :param v: wall-normal velocity field
        :type v: numpy/cupy array
        :param d: grid spacing
        :type d: float

        :rtype: cupy/numpy array
        """
        # Compute u_interp_y = (u_{i+1,j} + u_{i,j})/2
        u_interp_y = interpolate_1d(u, axis=0)
        # Compute v_interp_y = (v_{i+1,j} + v_{i,j})/2
        v_interp_y = interpolate_1d(v, axis=0)
        # Compute v_interp_y = (v_{i,j+1} + v_{i,j})/2
        v_interp_x = interpolate_1d(v, axis=1)

        return (
            v_interp_y[1:, 1:-1] ** 2
            - v_interp_y[:-1, 1:-1] ** 2
            + v_interp_x[1:-1, 1:] * u_interp_y[1:-1, 1:]
            - v_interp_x[1:-1, :-1] * u_interp_y[1:-1, :-1]
        ) * d

    def evaluate_momentum_equation(self, mesh) -> Tuple[xp.array, xp.array]: #type: ignore
        r"""
        Evaluate the momentum equation (without pressure) in the
        incompressible Navier-Stokes equations.
        """
        mom_x = (
            self.evaluate_laplacian(mesh.u_ext, mesh.d) / self.Re
            - self.evaluate_streamwise_advection(
                mesh.u_ext, mesh.v_ext, mesh.d
            )
            / (mesh.d**2)
            + mesh.fx_int
        )
        mom_y = (
            self.evaluate_laplacian(mesh.v_ext, mesh.d) / self.Re
            - self.evaluate_wallnormal_advection(
                mesh.u_ext, mesh.v_ext, mesh.d
            )
            / (mesh.d**2)
            + mesh.fy_int
        )
        return (mom_x, mom_y)
    
    def evaluate_pressure_gradient(self, mesh):
        return tuple(
            [
                evaluate_derivative_staggered(mesh.p, mesh.d, ax)
                for ax in [1, 0]
            ]
        )

    def updated_evaluate_pressure_gradient(self, p: xp.array, d: float) -> Tuple[xp.array, xp.array]: #type: ignore
        #eval at cell faces corresponding to correct momentum eqn
        #dp/dx at u locations
        #dp/dy at v locations
        #return dp/dx, dp/dy as tuple
        return tuple(
            [
                evaluate_derivative_staggered(
                    p, d, ax
                )
                for ax in [1, 0]
            ]
        )

    def evaluate_divergence(self, mesh) -> xp.array: #type: ignore
        #eval at cell center
        fields = [mesh.u_ext, mesh.v_ext]
        axes = xp.flipud(xp.arange(mesh.p.ndim, dtype=xp.int32))
        div = xp.zeros_like(mesh.p)
        for i, f in enumerate(fields):
            slc = [slice(None) if j != axes[i] else slice(1, -1) for j in axes]
            div += evaluate_derivative_staggered(f, mesh.d, axis=axes[i])[
                tuple(slc)
            ]
        return div
    
    
    def updated_evaluate_divergence(self, mesh, u: xp.array, v: xp.array) -> xp.array: #type: ignore
        #eval at cell center
        fields = [mesh.u_ext, mesh.v_ext]
        axes = xp.flipud(xp.arange(mesh.p.ndim, dtype=xp.int32))
        div = xp.zeros_like(mesh.p)
        for i, f in enumerate(fields):
            slc = [slice(None) if j != axes[i] else slice(1, -1) for j in axes]
            div += evaluate_derivative_staggered(f, mesh.d, axis=axes[i])[
                tuple(slc)
            ]
        return div
    
