import numpy as xp
import scipy as sp
import torch


class Mesh:
    r"""
    Holds information on a two-dimensional uniform mesh with :math:`\Delta = \Delta x = \Delta y`
    over the spatial domain

    .. math::

        \Omega = \left[ x_0, x_1 \right] \times \left[ y_0, y_1 \right].

    The attributes :code:`self.x` and :code:`self.y` contain the coordinates
    of the grid at cell centers.
    The grid is staggered, so :math:`x`-velocities are stored at right/left cell faces,
    :math:`y`-velocities at top/bottom cell faces, and the pressure is stored
    at cell centers.
    This class also holds two-dimensional arrays to store the :math:`x` velocity field,
    :math:`y` velocity field and the pressure.

    :param x0: :math:`x` location of the left boundary (inflow)
    :type x0: float
    :param x1: :math:`x` location of the right boundary (inflow)
    :type x1: float
    :param nx: number of cells in the x direction
    :type nx: int
    :param y0: :math:`y` location of the bottom boundary
    :type y0: float
    :param y1: :math:`y` location of the top boundary
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
        self.x = dx * xp.arange(nx) + x0 + dx / 2
        self.y = dy * xp.arange(ny) + y0 + dy / 2
        self.d = dx

        # Pressure field
        self.p = xp.zeros((len(self.y), len(self.x)))
        # Streamwise velocity field (including and excluding boundaries)
        self.u_ext = xp.zeros((len(self.y) + 2, len(self.x) + 1))
        self.u_int = self.u_ext[1:-1, 1:-1]
        # Wall-normal velocity field (including and excluding boundaries)
        self.v_ext = xp.zeros((len(self.y) + 1, len(self.x) + 2))
        self.v_int = self.v_ext[1:-1, 1:-1]
        # Streamwise and wall-normal volumetric forcing
        self.fx_int = self.u_int.copy()
        self.fy_int = self.v_int.copy()

    def info(self):
        r"""
        Print basic information about the mesh.
        """

        nx, ny = len(self.x), len(self.y)
        Lx = self.x[-1] - self.x[0] + self.d
        Ly = self.y[-1] - self.y[0] + self.d

        print("Mesh Information")
        print("-" * 60)
        print(
            f"x domain     : [{self.x[0] - self.d / 2:.4f}, {self.x[-1] + self.d / 2:.4f}]"
        )
        print(
            f"y domain     : [{self.y[0] - self.d / 2:.4f}, {self.y[-1] + self.d / 2:.4f}]"
        )
        print(f"Grid points  : nx = {nx}, ny = {ny}, nx * ny = {nx * ny}")
        print(f"Grid spacing : Δx = Δy = {self.d:.4e}")
        print("-" * 60)

    def generate_meshgrids(self, output_torch=False):
        x0 = self.x[0] - self.d / 2
        x1 = self.x[-1] + self.d / 2
        y0 = self.y[0] - self.d / 2
        y1 = self.y[-1] + self.d / 2

        xu = xp.arange(x0, x1 + self.d / 2, self.d)
        yu = xp.arange(y0 - self.d / 2, y1 + self.d, self.d)
        Xu, Yu = xp.meshgrid(xu, yu)
        assert Xu.shape == self.u_ext.shape
        assert Yu.shape == self.u_ext.shape

        xv = xp.arange(x0 - self.d / 2, x1 + self.d, self.d)
        yv = xp.arange(y0, y1 + self.d / 2, self.d)
        Xv, Yv = xp.meshgrid(xv, yv)
        assert Xv.shape == self.v_ext.shape
        assert Yv.shape == self.v_ext.shape

        Xp, Yp = xp.meshgrid(self.x, self.y)

        xp_tensors = [Xu, Yu, Xv, Yv, Xp, Yp]
        if output_torch:
            torch_tensors = [
                torch.tensor(t, requires_grad=True) for t in xp_tensors
            ]
            return xp_tensors, torch_tensors
        else:
            return xp_tensors