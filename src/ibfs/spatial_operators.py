import numpy as xp
import scipy as sp


class SpatialOperators:
    r"""
    Class to evaluate the right-hand side of the Navier-Stokes equation.
    """

    def __init__(self, Re):
        self.Re = Re

    def interpolate_1d(self, w: xp.array, axis: int) -> xp.array:
        r"""
        Perform 1d interpolation of the two-dimensional array :math:`w` along the
        direction specified by axis.

        :param w: two-dimensional array we wish to interpolate
        :type w: numpy/cupy array
        :param axis: direction of interpolation
        :type axis: int

        .. attention::

            This function returns an array of a different size than the
            original input array :math:`w`. If :math:`w` has size :math:`n\times m`,
            the dimension along :code:`axis` is reduced by :math:`1`.

        :rtype: numpy/cupy array
        """
        # For reference: slice(start, stop, step)
        slc0 = [slice(None)] * w.ndim
        slc1 = [slice(None)] * w.ndim
        slc0[axis] = slice(None, -1)  # Slice from 0 to -1 in axis direction
        slc1[axis] = slice(1, None)  # Slice from 1 to end in axis direction
        return 0.5 * (w[tuple(slc0)] + w[tuple(slc1)])

    def interpolate_2d(self, w: xp.array) -> xp.array:
        r"""
        Apply :func:`interpolate_1d` twice along both directions.

        :param w: two-dimensional array we wish to interpolate
        :type w: numpy/cupy array

        .. attention::

            This function returns an array of a different size than the
            original input array :math:`w`. If :math:`w` has size :math:`n\times m`,
            the dimension of the output array will be :math:`(n-1)\times (m-1)`.

        :rtype: numpy/cupy array
        """
        return self.interpolate_1d(self.interpolate_1d(w, 0), 1)

    def evaluate_derivative(self, mesh, w: xp.array, order: int, axis: int):
        r"""
        Evaluate derivative of field :math:`w` in the direction specified by :code:`axis`.
        The order of the derivative (i.e., first or second) is specified by :code:`order`.

        :param w: two-dimensional array
        :type w: numpy/cupy array
        :param order: order of the derivative
        :type order: int
        :param axis: direction along which to take the derivative
        :type axis: int

        :rtype: numpy/cupy array
        """
        slc0 = [slice(None)] * w.ndim
        slc1 = [slice(None)] * w.ndim
        slc0[axis] = slice(None, -1)  # Slice from 0 to -1 in axis direction
        slc1[axis] = slice(1, None)  # Slice from 1 to end in axis direction

        return (
            (w[tuple(slc1)] - w[tuple(slc0)]) / (2 * mesh.d)
            if order == 1
            else (w[tuple(slc1)] - 2 * w + w[tuple(slc0)]) / (mesh.d**2)
        )

    def evaluate_viscous_term(self, mesh, w: xp.array) -> xp.array:
        r"""
        Compute

        .. math::

            Re^{-1}\nabla^2 w = Re^{-1}\left(\partial_x^2 w + \partial_y^2 w\right)

        where :math:`w` is either the :math:`x` or :math:`y` velocity field.

        :param w: two-dimensional array
        :type w: numpy/cupy array

        .. attention::

            This function returns an array of a different size than the
            original input array :math:`w`. If :math:`w` has size :math:`n\times m`,
            the dimension of the output array will be :math:`(n-2)\times (m-2)`.

        :rtype: cupy/numpy array
        """
        size = w.shape
        size = (size[0] - 2, size[1] - 2)
        dw = xp.zeros(size, dtype=w.type)
        for axis in range(2):
            other_axis = xp.abs(axis - 1)
            slc = [slice(None)] * w.ndim
            slc[other_axis] = slice(1, -1)
            dw += self.evaluate_derivative(mesh, w, 2, axis)[tuple(slc)]
        return dw / self.Re

    def evaluate_advective_term_x(
        self, mesh, u: xp.array, v: xp.array
    ) -> xp.array:
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
        """
        # Compute u_interp_x = (u_{i,j+1} + u_{i,j})/2
        u_interp_x = self.interpolate_1d(u, axis=1)
        # Compute u_interp_y = (u_{i+1,j} + u_{i,j})/2
        u_interp_y = self.interpolate_1d(u, axis=0)
        # Compute v_interp_x = (v_{i,j+1} + v_{i,j})/2
        v_interp_x = self.interpolate_1d(v, axis=1)

        return (
            u_interp_x[1:-1, 1:] ** 2
            - u_interp_x[1:-1, :-1] ** 2
            + u_interp_y[1:, 1:-1] * v_interp_x[1:, 1:-1]
            - u_interp_y[:-1, 1:-1] * v_interp_x[:-1, 1:-1]
        ) * mesh.d

    def evaluate_advective_term_y(
        self, mesh, u: xp.array, v: xp.array
    ) -> xp.array:
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
        """
        # Compute u_interp_y = (u_{i+1,j} + u_{i,j})/2
        u_interp_y = self.interpolate_1d(u, axis=0)
        # Compute v_interp_y = (v_{i+1,j} + v_{i,j})/2
        v_interp_y = self.interpolate_1d(v, axis=0)
        # Compute v_interp_y = (v_{i,j+1} + v_{i,j})/2
        v_interp_x = self.interpolate_1d(v, axis=1)

        return (
            v_interp_y[1:, 1:-1] ** 2
            - v_interp_y[:-1, 1:-1] ** 2
            + v_interp_x[1:-1, 1:] * u_interp_y[1:-1, 1:]
            - v_interp_x[1:-1, :-1] * u_interp_y[1:-1, :-1]
        ) * mesh.d

    def evaluate_momentum_equation(self, mesh):
        mom_x = (
            self.evaluate_viscous_term(mesh, mesh.u_ext)
            + self.evaluate_advective_term_x(mesh, mesh.u_ext, mesh.v_ext)
            / mesh.d**2
        )
        mom_y = (
            self.evaluate_viscous_term(mesh, mesh.v)
            + self.evaluate_advective_term_y(mesh, mesh.u_ext, mesh.v_ext)
            / mesh.d**2
        )
        return (mom_x, mom_y)
