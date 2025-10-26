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
        slc0[axis] = slice(None, -1) # Slice from 0 to -1 in axis direction
        slc1[axis] = slice(1, None) # Slice from 1 to end in axis direction

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
        nx, ny = w.shape
        dw = xp.zeros((nx - 2, ny - 2), dtype=w.dtype)

        d2wdx2 = (w[2:, 1:-1] - 2 * w[1:-1, 1:-1] + w[:-2, 1:-1]) / (mesh.d ** 2)

        d2wdy2 = (w[1:-1, 2:] - 2 * w[1:-1, 1:-1] + w[1:-1, :-2]) / (mesh.d ** 2)

        dw = (d2wdx2 + d2wdy2) / self.Re
        return dw
    

