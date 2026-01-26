import numpy as xp
import scipy.sparse as sps
from ..utils.helpers import (
    vector_to_fields,
)
from typing import Tuple, TYPE_CHECKING

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

    def __init__(
        self,
        xi: xp.array,
        eta: xp.array,
        spatial_operators: "SpatialOperators",
    ):
        self.xi = xi
        self.eta = eta
        self.vel_ib = xp.zeros(2 * len(self.xi))
        self.ftil = xp.zeros_like(self.vel_ib)
        self.spatial_operators = spatial_operators

        dxis = xp.roll(self.xi, -1) - self.xi
        detas = xp.roll(self.eta, -1) - self.eta
        dsvec = xp.sqrt(dxis**2 + detas**2)
        self.S = xp.concatenate((dsvec, dsvec))

        self.assemble_interpolation_matrix()
        self.assemble_constraint_matrices()
        self.LuRQa = sps.linalg.splu(self.RQa)

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
        if x <= 0.5:
            val = (1 + xp.sqrt(-3 * x**2 + 1)) / (3 * d)
        elif 0.5 < x <= 1.5:
            val = (5 - 3 * x - xp.sqrt(-3 * (1 - x) ** 2 + 1)) / (6 * d)
        else:
            val = 0.0
        return val

    def assemble_interpolation_matrix(self):
        r"""
        Evaluate the interpolation matrix

        .. math::

            E_{k,i} = \Delta^2 d(x_i - \xi_k)d(y_i - \eta_k).

        The sparse matrix :math:`E` is stored as attribute :code:`self.E`.
        """
        d = self.spatial_operators.mesh.d
        szu = xp.prod(self.spatial_operators.mesh.u_int.shape)
        szv = xp.prod(self.spatial_operators.mesh.v_int.shape)
        xvecs = [
            self.spatial_operators.mesh.xu,
            self.spatial_operators.mesh.xv,
        ]
        yvecs = [
            self.spatial_operators.mesh.yu,
            self.spatial_operators.mesh.yv,
        ]
        shift = [0, szu]
        iter = 0
        rows, cols, data = [], [], []
        # This outer loop executes two iterations: one for the
        # x momentum equation, and one for the y momentum
        for x, y in zip(xvecs, yvecs):
            for l in range(len(self.xi)):
                xii, etai = self.xi[l], self.eta[l]
                # Identify which grid points are closest to the
                # lth immersed body point
                j = xp.argmin(xp.abs(x - xii))
                i = xp.argmin(xp.abs(y - etai))
                jvec = xp.arange(j - 3, j + 4, 1)
                ivec = xp.arange(i - 3, i + 4, 1)
                jvec, ivec = xp.meshgrid(jvec, ivec)
                jvec = jvec.reshape(-1)
                ivec = ivec.reshape(-1)
                # Loop over the closest points and populate
                # the entries of the interpolation matrix
                for m in range(len(ivec)):
                    xm, ym = x[jvec[m]], y[ivec[m]]
                    d_x = self.discrete_delta(xm - xii, d)
                    d_y = self.discrete_delta(ym - etai, d)
                    val = (self.spatial_operators.mesh.d**2) * d_x * d_y
                    if xp.abs(val) > 0.0:
                        rows.append(l + iter * len(self.xi))
                        cols.append(ivec[m] * len(x) + jvec[m] + shift[iter])
                        data.append(val)
            iter += 1
        shape = (2 * len(self.xi), szu + szv)
        self.E = sps.csc_matrix((data, (rows, cols)), shape=shape)

    def assemble_constraint_matrices(self):
        r"""
        Assemble matrices

        .. math::

            Q = \begin{bmatrix}
                G, E^\top
            \end{bmatrix}\quad\text{and}\quad R = \begin{bmatrix}
                D \\ E
            \end{bmatrix},

        as well as the matrix :math:`RQ`.
        """
        self.Q = sps.bmat([[self.spatial_operators.G, self.E.T]], format="csc")
        self.R = sps.bmat([[self.spatial_operators.D], [self.E]], format="csc")
        self.RQ = self.R.dot(self.Q)

        v = self.spatial_operators.La[:-1, -1].toarray().reshape(-1)
        w = self.spatial_operators.La[-1, :-1].T.toarray().reshape(-1)
        v = xp.concatenate((v, xp.zeros(2 * len(self.xi)))).reshape(-1, 1)
        w = xp.concatenate((w, xp.zeros(2 * len(self.xi)))).reshape(-1, 1)
        self.RQa = sps.bmat([[self.RQ, v], [w.T, None]], format="csc")

    def enforce_constraints(self, t, q):
        r"""
        Compute :math:`q - Q \left(RQ\right)^{-1} R q`.

        :param t: time
        :type t: float
        :param q: vector containing the velocity vector at collocation points
        :type q: xp.array
        :rtype: xp.array
        """
        spops = self.spatial_operators
        vector_to_fields(t, q, spops.mesh, spops.bcs)
        rhsvec = xp.concatenate(
            (
                spops.D.dot(q) - spops.evaluate_divergence().reshape(-1),
                self.vel_ib,
            )
        )
        vec = self.LuRQa.solve(xp.concatenate((self.R.dot(q) - rhsvec, [0])))[
            :-1
        ]
        spops.mesh.p[:, :] = vec[: xp.prod(spops.mesh.p.shape)].reshape(
            spops.mesh.p.shape
        )
        self.ftil = vec[xp.prod(spops.mesh.p.shape) :]
        return q - self.Q.dot(vec)

    def recover_physical_forces(self, dt) -> xp.array:
        r"""
        Given the transformed forces :math:`\tilde{f}_j` computed by
        the code, the physical forces are recovered by the formula

        .. math::

            f_j = -\frac{\Delta^2}{\Delta t \Delta s_j}\tilde{f}_j,

        where :math:`\Delta` is the (uniform) grid spacing in the cartesian
        grid, :math:`\Delta s_j` is the grid spacing between the :math:`j`th
        and :math:`(j+1)`th Lagrangian points, and :math:`\Delta t` is the
        time step.

        :rtype: xp.array
        """
        return -(self.spatial_operators.mesh.d**2) * self.ftil / self.S / dt

    def compute_total_force_on_the_body(self, dt) -> Tuple[xp.array, xp.array]:
        r"""
        The total force exerted by the fluid on the body is given by

        .. math::

            \mathbf{F} = -\left(F_x, F_y\right) = \int_{\mathcal{S}}\mathbf{f}(\pmb{\xi}(s))\,ds.

        (Notice the minus sign, since :math:`\mathbf{f}`, recovered through
        :func:`recover_physical_forces`, is the force exerted by the body
        on the fluid.)

        :rtype: Tuple[xp.array, xp.array]
        """
        F = -self.recover_physical_forces(dt) * self.S
        Fx, Fy = xp.sum(F[: len(self.xi)]), xp.sum(F[len(self.xi) :])
        return Fx, Fy
