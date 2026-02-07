import numpy as xp
import scipy as sp
import torch
from typing import List, Optional


class Mesh:
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
        if (len(dxvec) + 1 != len(xvec)) or (len(dyvec) + 1 != len(yvec)):
            raise ValueError(
                f"The length of the spacings vector (i.e., dxvec or dyvec) should be "
                f"one less than the length of the coordinates vector (i.e., xvec or yvec). "
                f"Currently len(dxvec) = {len(dxvec)}, len(xvec) = {len(xvec)}, "
                f"len(dyvec) = {len(dyvec)},, and len(yvec) = {len(yvec)}."
            )

        # Assemble x mesh
        self.x = self.assemble_mesh(xvec, dxvec)
        self.x = (
            xp.concatenate((self.x, -xp.flipud(self.x[:-1])))
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
