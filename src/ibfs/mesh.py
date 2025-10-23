import numpy as xp
import scipy as sp


class Mesh:
    r"""
    Holds information on a two-dimensional uniform mesh with :math:`\Delta = \Delta x = \Delta y`
    over the spatial domain

    .. math::

        D = \left[ x_0 - \frac{\Delta}{2}, x_1 + \frac{\Delta}{2} \right] \times
            \left[ y_0 - \frac{\Delta}{2}, y_1 + \frac{\Delta}{2} \right]

    The grid is staggered, so :math:`x`-velocities are stored at right/left cell faces,
    :math:`y`-velocities at top/bottom cell faces, and the pressure is stored
    at cell centers.
    This class also holds two-dimensional arrays to store the :math:`x` velocity field,
    :math:`y` velocity field and the pressure.

    :param x0: :math:`x` location of the left-most cell center
    :type x0: float
    :param x1: :math:`x` location of the right-most cell center
    :type x1: float
    :param nx: number of cells in the x direction
    :type nx: int
    :param y0: :math:`y` location of the bottom cell center
    :type y0: float
    :param y1: :math:`y` location of the top cell center
    :type y1: float
    :param ny: number of cells in the y direction
    :type ny: int
    """

    def __init__(
        self, x0: float, x1: float, nx: int, y0: float, y1: float, ny: int
    ):
        dx = (x1 - x0) / nx
        dy = (y1 - y0) / ny

        if xp.abs(dx - dy) > 1e-10:
            raise ValueError(
                f"dx and dy should be the same. Currently "
                f"dx = {dx} and dy = {dy}"
            )
        self.x = dx * xp.arange(nx) + x0
        self.y = dy * xp.arange(ny) + y0
        self.d = dx

        self.p = xp.zeros((len(self.y), len(self.x)))
        self.u_ext = xp.zeros((len(self.y) + 2, len(self.x) + 1))
        self.v_ext = xp.zeros((len(self.y) + 1, len(self.x) + 2))
        self.u_int = self.u_ext[1:-1, 1:-1]
        self.v_int = self.v_ext[1:-1, 1:-1]