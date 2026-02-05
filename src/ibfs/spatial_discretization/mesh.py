import numpy as xp
import scipy as sp
import torch
from typing import List, Optional


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
    :param x1: :math:`x` location of the right boundary (outflow)
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

        if xp.abs(dx - dy) > 1e-12:
            raise ValueError(
                f"dx and dy should be the same. Currently "
                f"dx = {dx} and dy = {dy}"
            )
        # Vectors containing the coordinates of the interior cell centers
        self.x = dx * xp.arange(nx) + x0 + dx / 2
        self.y = dy * xp.arange(ny) + y0 + dy / 2
        self.d = dx

        # Vectors containing the coordinates of the interior
        # u and v velocity nodes
        self.xu = (self.x[1:] + self.x[:-1]) / 2
        self.yu = self.y.copy()
        self.xv = self.x.copy()
        self.yv = (self.y[1:] + self.y[:-1]) / 2

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

    def generate_meshgrids(
        self, output_torch: Optional[bool] = False
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
        # Identify the coordinates of the left, right, top, and bottom walls
        x0 = self.x[0] - self.d / 2
        x1 = self.x[-1] + self.d / 2
        y0 = self.y[0] - self.d / 2
        y1 = self.y[-1] + self.d / 2

        # Meshgrids for the u velocity field
        xu = xp.arange(x0, x1 + self.d / 2, self.d)
        yu = xp.arange(y0 - self.d / 2, y1 + self.d, self.d)
        Xu, Yu = xp.meshgrid(xu, yu)
        assert Xu.shape == self.u_ext.shape
        assert Yu.shape == self.u_ext.shape
        # Meshgrids for the v velocity field
        xv = xp.arange(x0 - self.d / 2, x1 + self.d, self.d)
        yv = xp.arange(y0, y1 + self.d / 2, self.d)
        Xv, Yv = xp.meshgrid(xv, yv)
        assert Xv.shape == self.v_ext.shape
        assert Yv.shape == self.v_ext.shape
        # Meshgrids for the pressure field
        Xp, Yp = xp.meshgrid(self.x, self.y)

        xp_tensors = [Xu, Yu, Xv, Yv, Xp, Yp]
        if output_torch:
            torch_tensors = [
                torch.tensor(t, requires_grad=True) for t in xp_tensors
            ]
            return xp_tensors, torch_tensors
        else:
            return xp_tensors


class Mesh_:
    def __init__(
        self,
        xvec,
        dxvec,
        yvec,
        dyvec,
        mirror_x=False,
        mirror_y=False,
        check_equal_min_spacing=False,
    ):
        # Assemble x mesh
        self.x = self.assemble_mesh(xvec, dxvec)
        self.x = (
            xp.concatenate((self.x, -xp.flipud(self.x[1:])))
            if mirror_x
            else self.x
        )
        self.dx = self.x[1:] - self.x[:-1]  # Grid spacings
        self.xc = (self.x[1:] + self.x[:-1]) / 2  # Coordinates of cell centers

        # Assemble y mesh
        self.y = self.assemble_mesh(yvec, dyvec)
        self.y = (
            xp.concatenate((self.y, -xp.flipud(self.y[:-1])))
            if mirror_y
            else self.y
        )
        self.dy = self.y[1:] - self.y[:-1]  # Grid spacings
        self.yc = (self.y[1:] + self.y[:-1]) / 2  # Coordinates of cell centers

        if check_equal_min_spacing:
            dxmin = self.dx.min()
            dymin = self.dy.min()
            if xp.abs(dxmin - dymin) > 1e-10:
                raise ValueError(
                    f"Minimum dx and dy spacing are not equal. Currently "
                    f"dxmin = {dxmin}, dymin = {dymin}, and abs(dxmin - dymin) = "
                    f"{xp.abs(dxmin - dymin)}"
                )

        # Pressure field
        self.p = xp.zeros((len(self.yc), len(self.xc)))

        # Streamwise velocity field (including and excluding boundaries)
        # and coordinates of x-staggered finite volume centroids
        self.u_ext = xp.zeros((len(self.yc) + 2, len(self.xc) + 1))
        self.u_int = self.u_ext[1:-1, 1:-1]
        self.xu = xp.concatenate(
            ([self.x[0]], 0.5 * (self.xc[1:] + self.xc[:-1]), [self.x[-1]])
        )  # x-momentum centroids (x)
        self.yu = xp.concatenate(
            (
                [2 * self.yc[0] - self.yc[1]],
                self.yc,
                [2 * self.yc[-1] - self.yc[-2]],
            )
        )  # x-momentum centroids (y)

        # Wall-normal velocity field (including and excluding boundaries)
        # and coordinates of top/bottom cell faces
        self.v_ext = xp.zeros((len(self.yc) + 1, len(self.xc) + 2))
        self.v_int = self.v_ext[1:-1, 1:-1]
        self.yv = xp.concatenate(
            ([self.y[0]], 0.5 * (self.yc[1:] + self.yc[:-1]), [self.y[-1]])
        )  # y-momentum centroids (y)
        self.xv = xp.concatenate(
            (
                [2 * self.xc[0] - self.xc[1]],
                self.xc,
                [2 * self.xc[-1] - self.xc[-2]],
            )
        )  # y-momentum centroids (x)

        # Streamwise and wall-normal volumetric forcing
        self.fx_int = self.u_int.copy()
        self.fy_int = self.v_int.copy()

    def assemble_mesh(self, xvec, dxvec):
        xs = []
        n_intervals = len(dxvec)
        for i in range(n_intervals):
            x0, x1 = xvec[i], xvec[i + 1]
            nx = int((x1 - x0) / dxvec[i])
            dx = (x1 - x0) / nx
            n_points = nx if i < n_intervals - 1 else nx + 1
            xs.append(xp.arange(n_points) * dx + x0)

        return xp.concatenate(xs)
