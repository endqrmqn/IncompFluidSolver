import numpy as xp
import scipy.sparse as sps
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
from typing import Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .mesh import Mesh
    from .boundary_conditions import BoundaryConditions
    from .immersed_body import ImmersedBody

from ..utils.helpers import (
    vector_to_fields,
    compute_limited_face_values,
    quadratic_interpolation,
)
from .sparse_matrix_operators import (
    divergence_sparsity_pattern,
    gradient_sparsity_pattern,
    divergence_data,
    gradient_data,
    assemble_matrix,
)


class SpatialOperators:
    r"""
    Class to evaluate the right-hand side of the Navier-Stokes equation.

    :param Re: Reynolds number
    :type Re: float
    :param mesh: an instance of the :class:`Mesh` class
    :type mesh: Mesh
    :param bcs: a tuple of instances of the :class:`BoundaryConditions` class
        (the length of the tuple is equal to the number of velocity components)
    :type bcs: Tuple[BoundaryConditions]
    """

    def __init__(
        self,
        Re: float,
        mesh: "Mesh",
        bcs: Tuple["BoundaryConditions"],
        test: Optional[bool] = False,
    ):
        self.Re = Re
        self.mesh = mesh
        self.bcs = bcs

        self.assemble_mass_matrix()

        if not test:
            self.assemble_divergence_matrix()
            self.assemble_gradient_matrix()
            self.assemble_laplacian_matrix()

            self.augment_pressure_laplacian()
            self.LuLa = sps.linalg.splu(self.La)

    def evaluate_momentum_integral(
        self, u, v, xu, yu, xv, yv, xc, yc, x, y, dx, dy, f
    ):
        
        momn = xp.zeros_like(u[1:-1, 1:-1])
        # Streamwise advection term (interpolate the u velocity to
        # the cell centers so that we can compute fluxes)
        uc = compute_limited_face_values(u, xu, xc, u[:, :-1])
        momn -= (uc[1:-1, 1:] ** 2 - uc[1:-1, :-1] ** 2) * dy[:, None]
        # Wall-normal advection term (interpolate u and v velocities to
        # the corners of the x-staggered control volumes)
        uc, ducx = quadratic_interpolation(u, xu, xc, deriv=True)
        vcf, _ = quadratic_interpolation(v.T, yv, y)
        vcf = vcf[1:-1, :].T
        ucf = compute_limited_face_values(uc.T, yu, y, vcf.T).T
        uv = ucf * vcf
        momn -= (
            0.5
            * (uv[1:, 1:] + uv[1:, :-1] - uv[:-1, 1:] - uv[:-1, :-1])
            * 0.5
            * (dx[1:] + dx[:-1])[None, :]
        )
        # Laplacian
        _, ducy = quadratic_interpolation(u.T, yu, y, deriv=True)
        ducy = ducy.T
        momn += ((ducx[1:-1, 1:] - ducx[1:-1, :-1]) / self.Re) * dy[:, None]
        momn += (
            (ducy[1:, 1:-1] - ducy[:-1, 1:-1])
            / self.Re
            * 0.5
            * (dx[1:] + dx[:-1])[None, :]
        )
        # External forcing term
        momn += f * 0.5 * (dx[1:] + dx[:-1])[None, :] * dy[:, None]

        return momn

    def evaluate_momentum_equation(self):
        mesh = self.mesh
        x_mom = self.evaluate_momentum_integral(
            mesh.u_ext,
            mesh.v_ext,
            mesh.xu,
            mesh.yu,
            mesh.xv,
            mesh.yv,
            mesh.xc,
            mesh.yc,
            mesh.x,
            mesh.y,
            mesh.dx,
            mesh.dy,
            mesh.fx_int,
        )
        y_mom = self.evaluate_momentum_integral(
            mesh.v_ext.T,
            mesh.u_ext.T,
            mesh.yv,
            mesh.xv,
            mesh.yu,
            mesh.xu,
            mesh.yc,
            mesh.xc,
            mesh.y,
            mesh.x,
            mesh.dy,
            mesh.dx,
            mesh.fy_int.T,
        )
        return (x_mom, y_mom.T)

    def evaluate_pressure_integral(self) -> Tuple[xp.array, xp.array]:
        r"""
        Compute 

        .. math::

            \int_{\mathcal{V}_{i,j}} \nabla p\,dV_{i,j} = 
            \begin{bmatrix}
                \left(p_{i,j+1} - p_{i,j}\right)\Delta y_{i} \\
                \left(p_{i+1,j} - p_{i,j}\right)\Delta x_{j}
            \end{bmatrix}

        :rtype: Tuple[xp.array, xp.array]
        """
        return (
            (self.mesh.p[:, 1:] - self.mesh.p[:, :-1]) * self.mesh.dy[:, None],
            (self.mesh.p[1:, :] - self.mesh.p[:-1, :]) * self.mesh.dx[None, :],
        )

    def evaluate_divergence_integral(self) -> xp.array:
        r"""
        Compute :math:`\int_{\mathcal{V}_{i,j}} \nabla\cdot\mathbf{u}\,d V_{i,j}`.

        :rtype: xp.array
        """
        uf, _ = quadratic_interpolation(
            self.mesh.u_ext, self.mesh.xu, self.mesh.x
        )
        vf, _ = quadratic_interpolation(
            self.mesh.v_ext.T, self.mesh.yv, self.mesh.y
        )
        vf = vf.T
        return (uf[1:-1, 1:] - uf[1:-1, :-1]) * self.mesh.dy.reshape(-1, 1) + (
            (vf[1:, 1:-1] - vf[:-1, 1:-1]) * self.mesh.dx.reshape(1, -1)
        )

    def assemble_mass_matrix(self):
        r"""
        Instantiate the attributes :code:`self.M` and :code:`self.Minv`, containing the
        mass matrix :math:`M_{i,i} = \Delta x_{i}\Delta y_{i}` and its inverse.
        """
        dx = self.mesh.dx
        dy = self.mesh.dy

        data_u = xp.outer(dy, 0.5 * (dx[1:] + dx[:-1])).reshape(-1)
        data_v = xp.outer(0.5 * (dy[1:] + dy[:-1]), dx).reshape(-1)
        data = xp.concatenate((data_u, data_v))
        rows = xp.arange(len(data))
        self.M = assemble_matrix(rows, rows, data)
        self.Minv = assemble_matrix(rows, rows, 1 / data)

        data = xp.outer(dy, dx).reshape(-1)
        rows = xp.arange(len(data))
        self.Mp = assemble_matrix(rows, rows, data)

    def assemble_gradient_matrix(self):
        r"""
        Instantiate the attribute :code:`self.G`, containing the matrix
        representation of the gradient operator defined in
        :func:`evaluate_gradient`.
        """
        rows, cols, rows_extract, cols_query = gradient_sparsity_pattern(
            self.mesh
        )
        data = gradient_data(self, rows_extract, cols_query, 1.0)
        self.G = self.Minv.dot(assemble_matrix(rows, cols, data))

    def assemble_divergence_matrix(self):
        r"""
        Instantiate the attribute :code:`self.D`, containing the matrix
        representation of the divergence operator defined in
        :func:`evaluate_divergence`.
        """
        rows, cols, rows_extract, cols_query = divergence_sparsity_pattern(
            self.mesh
        )
        data = divergence_data(self, rows_extract, cols_query, 1.0)
        self.D = assemble_matrix(rows, cols, data)

    def assemble_laplacian_matrix(self):
        r"""
        Instantiate the attribute :code:`self.L`, containing the matrix
        representation of the pressure laplacian operator
        :math:`L = D G`, where :math:`D` and :math:`G` are the divergence
        and gradient operators, respectively.
        """
        self.L = self.D.dot(self.G)

    def augment_pressure_laplacian(self):
        r"""
        The discrete laplacian :math:`L = DG` (where :math:`D` and :math:`G`
        are the discrete divergence and gradient operators)
        is singular with a one-dimensional right nullspace spanned
        by :math:`v = const`. The left nullspace :math:`w` is unknown,
        but can be computed easily following the approach in section
        3.3 of [Padovan2021]_.
        We then update :code:`self.L` to

        .. math::

            L_a = \begin{bmatrix}
                L & v \\ w^T & 0
            \end{bmatrix}

        which is invertible.

        References
        ----------
        .. [Padovan2021] Padovan and Rowley, *A computationally efficient
            approach for the removal of the phase shift singularity in
            harmonic resolvent analysis*, arXiv:2102.09678, 2021
        """
        v = xp.ones((xp.prod(self.mesh.p.shape), 1))
        v /= xp.linalg.norm(v)
        M = sps.bmat([[self.L.T, v], [v.T, [0]]], format="csc")
        rhs = xp.zeros(M.shape[-1])
        rhs[-1] = 1.0
        w = sps.linalg.spsolve(M, rhs)
        w = w[:-1] / xp.linalg.norm(w[:-1])
        self.La = sps.bmat(
            [[self.L, v], [w.reshape(1, -1), [0]]], format="csc"
        )

    def evaluate_right_hand_side(self, t, q):
        r"""
        Compute

        .. math::

            \mathbf{r}(\mathbf{u}) = -\mathbf{u}\cdot\nabla\mathbf{u} +
            Re^{-1}\Delta \mathbf{u} + \mathbf{f}(t)

        where :math:`\mathbf{u}` is the continuous-in-space velocity vector and
        :math:`q = (u_h, v_h) \in \mathbb{R}^n` contains the spatially-discretized
        velocity vector.

        :param t: time
        :type t: float
        :param q: vector containing the velocity vector at collocation points
        :type q: xp.array

        :rtype: xp.array
        """
        vector_to_fields(t, q, self.mesh, self.bcs)
        xmom, ymom = self.evaluate_momentum_equation()
        return self.Minv.dot(
            xp.concatenate((xmom.reshape(-1), ymom.reshape(-1)))
        )

    def enforce_divergence_free(self, t, q):
        r"""
        Compute :math:`q - G \Delta^{-1} D q`.

        :param t: time
        :type t: float
        :param q: vector containing the velocity vector at collocation points
        :type q: xp.array
        :rtype: xp.array
        """
        vector_to_fields(t, q, self.mesh, self.bcs)
        pvec = self.LuLa.solve(
            xp.concatenate(
                (self.evaluate_divergence_integral().reshape(-1), [0])
            )
        )[:-1]
        self.mesh.p[:, :] = pvec.reshape(self.mesh.p.shape)
        return q - self.G.dot(pvec)
