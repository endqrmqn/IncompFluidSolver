import numpy as xp
import scipy.sparse as sps
from .helpers import (
    interpolate_1d,
    evaluate_derivative,
    evaluate_derivative_staggered,
)
from typing import Optional, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from .mesh import Mesh
    from .spatial_operators import SpatialOperators


class ImmersedBody:
    r"""
    Class to account for the effect of an immersed body on the surrounding fluid.

    :param xi: :math:`x` location of the immersed body points
    :type xi: xp.array
    :param eta: :math:`y` location of the immersed body points
    :type eta: xp.array
    :param spatial_operators: instance of the :class:`SpatialOperators` class
    :type spatial_operators: SpatialOperators
    """

    def __init__(self, xi: xp.array, eta: xp.array, spatial_operators: "SpatialOperators"):
        self.xi = xi
        self.eta = eta
        self.spatial_operators = spatial_operators

        dxis = xp.roll(self.xi, -1) - self.xi
        detas = xp.roll(self.eta, -1) - self.eta
        dsvec = xp.sqrt(dxis ** 2 + detas ** 2)
        self.Sinv = xp.concatenate((1 / dsvec, 1 / dsvec))

        self.assemble_interpolation_matrix()

    def discrete_delta(self, r: float, d: float) -> float:
        r"""
        Evaluate the discrete delta function :math:`d(r) \approx \delta(r)`.

        :param r: :math:`r = x - \xi` or :math:`r = y - \eta`
        :type r: float
        :param d: grid spacing :math:`\Delta = \Delta x = \Delta y`
        :type d: float

        :rtype: float
        """
        x = xp.abs(r) / d
        if xp.abs(r) <= 0.5 * d:
            d = (1 + xp.sqrt(- 3 * x ** 2 + 1)) / (3 * d)
        elif 0.5 * d < xp.abs(r) <= 1.5 * d:
            d = (5 - 3 * x  - xp.sqrt(-3 * (1 - x) ** 2 + 1)) / (6 * d)
        else:
            d = 0.0
        return d
    
    def assemble_interpolation_matrix(self):
        r"""
        Evaluate the interpolation matrix

        .. math::

            E_{i,k} = \Delta^2 d(x_i - \xi_k)d(y_i - \eta_k).

        The sparse matrix :math:`E` is stored as attribute :code:`self.E`.
        """
        d = self.spatial_operators.mesh.d
        szu = xp.prod(self.spatial_operators.mesh.u_int.shape)
        szv = xp.prod(self.spatial_operators.mesh.v_int.shape)
        xvecs = [self.xu, self.xv]
        yvecs = [self.xv, self.yv]
        shift = [0, szu]
        iter = 0
        rows, cols, data = [], [], []
        for (x, y) in zip(xvecs, yvecs):
            for l in range (len(self.xi)):
                xii, etai = self.xi[l], self.eta[l]
                # Identify which grid points are closest to the 
                # lth immersed body point
                j = xp.argmin(xp.abs(x - xii))
                i = xp.argmin(xp.abs(y - etai))
                jvec = xp.arange(j-3, j+4, 1)
                ivec = xp.arange(i-3, i+4, 1)
                jvec, ivec = xp.meshgrid(jvec, ivec)
                jvec = jvec.reshape(-1)
                ivec = ivec.reshape(-1)
                # Loop over the closest points and populate
                # the entries of the interpolation matrix
                for m in range (len(ivec)):
                    xm, ym = x[jvec[m]], y[ivec[m]]
                    d_x = self.discrete_delta(xm - xii, d)
                    d_y = self.discrete_delta(ym - etai, d)
                    val = (self.spatial_operators.mesh.d ** 2) * d_x * d_y
                    if xp.abs(val) > 0.0:
                        rows.append(l + iter * len(self.xi))
                        cols.append(ivec[m] * len(x) + jvec[m] + shift[iter])
                        data.append(val)
            iter += 1
        shape = (2 * len(self.xi), szu + szv)
        self.E = sps.csc_matrix((data, (rows, cols)), shape=shape)
    
    def recover_forces(self, f):
        return - (self.spatial_operators.mesh.d ** 4) * self.Sinv * f