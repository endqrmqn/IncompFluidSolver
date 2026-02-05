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

    def interpolate_1d(self, w: xp.array, axis: int) -> xp.array:
        r"""
        Perform 1d interpolation of the :math:`r`-dimensional array
        :math:`w` along the direction specified by :code:`axis`.

        :param w: two-dimensional array we wish to interpolate
        :type w: xp.array
        :param axis: direction of interpolation
        :type axis: int

        .. attention::

            This function returns an array of a different size than the
            original input array :math:`w`. If :math:`w` has size :math:`n\times m`,
            the dimension along :code:`axis` is reduced by :math:`1`.

        :rtype: xp.array
        """
        # For reference: slice(start, stop, step)
        slc0 = [slice(None)] * w.ndim
        slc1 = [slice(None)] * w.ndim
        slc0[axis] = slice(None, -1)  # Slice from 0 to -1 in axis direction
        slc1[axis] = slice(1, None)  # Slice from 1 to end in axis direction
        return 0.5 * (w[tuple(slc0)] + w[tuple(slc1)])

    def evaluate_derivative(
        self, w: xp.array, d: float, order: int, axis: int
    ):
        r"""
        Evaluate derivative of field :math:`w` in the direction specified by :code:`axis`.
        The order of the derivative (i.e., first or second) is specified by :code:`order`.
        We use a second-order central difference scheme.

        :param w: two-dimensional array
        :type w: xp.array
        :param d: grid spacing
        :type d: float
        :param order: order of the derivative
        :type order: int
        :param axis: direction along which to take the derivative
        :type axis: int

        :rtype: xp.array
        """
        slcm1 = [slice(None)] * w.ndim
        slcp1 = [slice(None)] * w.ndim
        slc0 = [slice(None)] * w.ndim
        slcm1[axis] = slice(None, -2)  # Slice from 0 to -1 in axis direction
        slcp1[axis] = slice(2, None)  # Slice from 1 to end in axis direction
        slc0[axis] = slice(1, -1, 1)

        return (
            (w[tuple(slcp1)] - w[tuple(slcm1)]) / (2 * d)
            if order == 1
            else (w[tuple(slcp1)] - 2 * w[tuple(slc0)] + w[tuple(slcm1)])
            / (d**2)
        )

    def evaluate_derivative_staggered(
        self, w: xp.array, d: float, axis: int
    ) -> xp.array:
        r"""
        Evaluate the first derivative of field :math:`w` in
        the direction specified by :code:`axis`.
        This function is used for the gradient and divergence evaluation
        in the Navier-Stokes equation.

        :param w: two-dimensional array
        :type w: xp.array
        :param d: grid spacing
        :type d: float
        :param axis: direction along which to take the derivative
        :type axis: int

        :rtype: xp.array
        """
        slcm = [slice(None)] * w.ndim
        slcp = [slice(None)] * w.ndim
        slcm[axis] = slice(None, -1)
        slcp[axis] = slice(1, None)
        return (w[tuple(slcp)] - w[tuple(slcm)]) / d

    def evaluate_laplacian(self, w: xp.array, d: float) -> xp.array:
        r"""
        Compute :math:`\nabla^2 w = \left(\partial_x^2 w + \partial_y^2 w\right)`,
        where :math:`w` is either the :math:`x` or :math:`y` velocity field.

        .. attention::

            This function returns an array of a different size than the
            original input array :math:`w`. If :math:`w` has size :math:`n\times m`,
            the dimension of the output array will be :math:`(n-2)\times (m-2)`.

        :param w: two-dimensional array
        :type w: xp.array
        :param d: grid spacing
        :type d: float

        :rtype: cupy/numpy array
        """
        size = tuple([s - 2 for s in w.shape])
        dw = xp.zeros(size, dtype=w.dtype)
        axes = xp.arange(w.ndim, dtype=xp.int32)
        for axis in axes:
            slc = [slice(1, -1) if i != axis else slice(None) for i in axes]
            dw += self.evaluate_derivative(w, d, 2, axis)[tuple(slc)]
        return dw

    def evaluate_streamwise_advection(
        self, u: xp.array, v: xp.array, d: float
    ) -> xp.array:
        r"""
        Compute

        .. math::

            \int_{Y} u^2(x,y)\big\lvert_{x=x_{l}}^{x=x_r}\,dy +
            \int_{X}\left[u(x,y)v(x,y)\right]_{y={y_b}}^{y=y_{t}}\,dx

        on each finite volume (see Figure 2 in the docs/notes/cfd_solver_details.pdf).

        .. attention::

            This function returns an array of size :math:`n_y \times (n_x - 1)`,
            where :math:`n_y` and :math:`n_x` are the number of internal cell
            centers in the :math:`y` and :math:`x` directions, respectively.

        :param u: streamwise velocity field
        :type u: xp.array
        :param v: wall-normal velocity field
        :type v: xp.array
        :param d: grid spacing
        :type d: float

        :rtype: xp.array
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
        ) * d

    def evaluate_wallnormal_advection(
        self, u: xp.array, v: xp.array, d: float
    ) -> xp.array:
        r"""
        Compute

        .. math::

            \int_{X} v^2(x,y)\big\lvert_{y=y_{b}}^{y=y_t}\,dx +
            \int_{Y}\left[u(x,y)v(x,y)\right]_{x={x_l}}^{x=x_{t}}\,dy

        on each finite volume (see Figure 3 in the docs/notes/cfd_solver_details.pdf).

        .. attention::

            This function returns an array of size :math:`(n_y-1) \times n_x`,
            where :math:`n_y` and :math:`n_x` are the number of internal cell
            centers in the :math:`y` and :math:`x` directions, respectively.

        :param u: streamwise velocity field
        :type u: xp.array
        :param v: wall-normal velocity field
        :type v: xp.array
        :param d: grid spacing
        :type d: float

        :rtype: cupy/numpy array
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
        ) * d

    def evaluate_momentum_equation(self) -> Tuple[xp.array, xp.array]:
        r"""
        Evaluate the discretized momentum equation (without pressure) in the
        incompressible Navier-Stokes equations. Returns a 2-tuple
        containing the evaluation of the :math:`x`- and :math:`y`-momentum
        equations.

        :rtype: Tuple[xp.array, xp.array]
        """
        mom_x = (
            self.evaluate_laplacian(self.mesh.u_ext, self.mesh.d) / self.Re
            - self.evaluate_streamwise_advection(
                self.mesh.u_ext, self.mesh.v_ext, self.mesh.d
            )
            / (self.mesh.d**2)
            + self.mesh.fx_int
        )
        mom_y = (
            self.evaluate_laplacian(self.mesh.v_ext, self.mesh.d) / self.Re
            - self.evaluate_wallnormal_advection(
                self.mesh.u_ext, self.mesh.v_ext, self.mesh.d
            )
            / (self.mesh.d**2)
            + self.mesh.fy_int
        )
        return (mom_x, mom_y)

    # def evaluate_streamwise_momentum_integral(
    #     self, u, v, xu, yu, xv, yv, xc, yc, x, y, dx, dy, f
    # ):
    #     copy = False
    #     kind = "quadratic"
    #     extr = "extrapolate"

    #     momentum = xp.zeros_like(self.mesh.u_int)

    #     # Streamwise advection term (interpolate the u velocity to
    #     # the cell centers so that we can compute fluxes)
    #     uc = interp1d(xu, u, kind, -1, copy, fill_value=extr)(xc)
    #     momentum -= (uc[1:-1, 1:] ** 2 - uc[1:-1, :-1] ** 2) * dy.reshape(
    #         -1, 1
    #     )
    #     # Wall-normal advection term (interpolate u and v velocities to
    #     # the corners of the x-staggered control volumes)
    #     ucf = interp1d(yu, uc, kind, 0, copy, fill_value=extr)(y)
    #     vcf = interp1d(yv, v, kind, 0, copy, fill_value=extr)(y)[:, 1:-1]
    #     uv = ucf * vcf
    #     momentum -= (
    #         0.5
    #         * (uv[1:, 1:] + uv[1:, :-1] - uv[:-1, 1:] - uv[:-1, :-1])
    #         * 0.5
    #         * (dx[1:] + dx[:-1]).reshape(1, -1)
    #     )
    #     # Laplacian
    #     uf = interp1d(xu, u, kind, -1, copy, fill_value=extr)(x)
    #     dudx = (uf[1:-1, 1:] - uf[1:-1, :-1]) / dx
    #     # dudy = (u[1:, 1:-1] - u[:-1, 1:-1]) / (yv[1:] - yv[:-1]).reshape(-1, 1)
    #     dudy = InterpolatedUnivariateSpline()
    #     momentum += ((dudx[:, 1:] - dudx[:, :-1]) / self.Re) * dy.reshape(
    #         -1, 1
    #     )
    #     momentum += (
    #         ((dudy[1:, :] - dudy[:-1, :]) / self.Re)
    #         * 0.5
    #         * (dx[1:] + dx[:-1]).reshape(1, -1)
    #     )
    #     # External forcing term
    #     momentum += f

    #     return momentum

    def evaluate_streamwise_momentum_integral(
        self, u, v, xu, yu, xv, yv, xc, yc, x, y, dx, dy, f
    ):
        
        def centroids_to_faces(fc, xc, xf):
            # Interpolate field fc from centroids to cell faces
            idces = [-1, 0, 1]
            ff = xp.zeros_like(fc[:, 1:-1])
            for idx, i in enumerate(idces):
                xi = xp.roll(xc[1:-1], i)
                pi = xp.ones_like(xi)
                for j in xp.delete(idces, idx):
                    xj = xp.roll(xc[1:-1], j)
                    pi *= (xf[1:-1] - xj) / (xi - xj)
                ff += xp.roll(fc[:, 1:-1], i, axis=-1) * pi[None, :]
            return xp.concatenate(
                (fc[:, 0].reshape(-1, 1), ff, fc[:, -1].reshape(-1, 1)),
                axis=-1,
            )
        
        def centroids_to_cell_centers(fc, xc, xcc, deriv=False):
            # Interpolate fc from centroids to cell centers
            idces = [-1, 0, 1]
            ff = xp.zeros_like(fc[:, 2:-1])
            df = xp.zeros_like(ff) if deriv else 0
            for idx, i in enumerate(idces):
                xi = xp.roll(xc[2:-1], i)
                pi_ff = xp.ones_like(xi)
                pi_df = xp.zeros_like(xi) if deriv else 0
                for j in xp.delete(idces, idx):
                    xj = xp.roll(xc[2:-1], j)
                    ratio = (xcc[1:-1] - xj) / (xi - xj)
                    pi_df += ratio if deriv else 0
                    pi_ff *= ratio
                ff += xp.roll(fc[:, 2:-1], i, axis=-1) * pi_ff[None, :]
                df += xp.roll(fc[:, 2:-1], i, axis=-1) * pi_df[None, :] if deriv else 0

            if not deriv:
                return xp.concatenate(
                    (
                        0.5 * (fc[:, 0] + fc[:, 1])[:, None],
                        ff,
                        0.5 * (fc[:, -1] + fc[:, -2])[:, None],
                    ),
                    axis=-1,
                )
            else:
                return (
                    xp.concatenate(
                        (
                            0.5 * (fc[:, 0] + fc[:, 1])[:, None],
                            ff,
                            0.5 * (fc[:, -1] + fc[:, -2])[:, None],
                        ),
                        axis=-1,
                    ),
                    xp.concatenate(
                        (
                            ((fc[:, 1] - fc[:, 0]) / (xc[1] - xc[0]))[:, None],
                            df,
                            ((fc[:, -1] - fc[:, -2]) / (xc[-1] - xc[-2]))[
                                :, None
                            ],
                        ),
                        axis=-1,
                    ),
                )

        momentum = xp.zeros_like(self.mesh.u_int)

        # Streamwise advection term (interpolate the u velocity to
        # the cell centers so that we can compute fluxes)
        uc, ducx = centroids_to_cell_centers(u, xu, xc, deriv=True)
        momentum -= (uc[1:-1, 1:] ** 2 - uc[1:-1, :-1] ** 2) * dy[:, None]
        # # Wall-normal advection term (interpolate u and v velocities to
        # # the corners of the x-staggered control volumes)
        # ucf = centroids_to_cell_centers(uc.T, yu, y).T
        # vcf = (centroids_to_faces(v.T, yv, y)[1:-1, :]).T
        # uv = ucf * vcf
        # momentum -= (
        #     0.5
        #     * (uv[1:, 1:] + uv[1:, :-1] - uv[:-1, 1:] - uv[:-1, :-1])
        #     * 0.5
        #     * (dx[1:] + dx[:-1])[None, :]
        # )
        # # Laplacian
        # _, ducy = centroids_to_cell_centers(u.T, yu, y, deriv=True)
        # ducy = ducy.T
        # momentum += ((ducx[1:-1, 1:] - ducx[1:-1, :-1]) / self.Re) * dy[:, None]
        # momentum += (
        #     (ducy[1:, 1:-1] - ducy[:-1, 1:-1])
        #     / self.Re
        #     * 0.5
        #     * (dx[1:] + dx[:-1])[None, :]
        # )
        # # External forcing term
        # momentum += f * 0.5 * (dx[1:] + dx[:-1])[None, :] * dy[:, None]

        return momentum

    def evaluate_momentum_equation(self):
        mesh = self.mesh
        x_mom = self.evaluate_streamwise_momentum_integral(
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
        # y_mom = self.evaluate_streamwise_momentum_integral(
        #     mesh.v_ext.T,
        #     mesh.u_ext.T,
        #     mesh.yv,
        #     mesh.xv,
        #     mesh.yu,
        #     mesh.xu,
        #     mesh.yc,
        #     mesh.xc,
        #     mesh.y,
        #     mesh.x,
        #     mesh.dy,
        #     mesh.dx,
        #     mesh.fy_int.T,
        # )
        return x_mom

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
        Velocities are interpolated from cell centroids to cell faces according to the formula

        .. math::

            f(x) = \frac{\left(f_{j+1} - f_{j}\right)}{\left(x_{j+1} - x_{j}\right)}\left(x - x_j\right) + f_j,

        where :math:`f_j` and :math:`x_j` are the values and coordinates at the cell centroids.

        :rtype: xp.array
        """

        def centroids_to_faces(fc, xc, xf):
            # Interpolate field fc from centroids to cell faces
            idces = [-1, 0, 1]
            ff = xp.zeros_like(fc[:, 1:-1])
            for idx, i in enumerate(idces):
                xi = xp.roll(xc[1:-1], i)
                pi = xp.ones_like(xi)
                for j in xp.delete(idces, idx):
                    xj = xp.roll(xc[1:-1], j)
                    pi *= (xf[1:-1] - xj) / (xi - xj)
                ff += xp.roll(fc[:, 1:-1], i, axis=-1) * pi[None, :]
            return xp.concatenate(
                (fc[:, 0].reshape(-1, 1), ff, fc[:, -1].reshape(-1, 1)),
                axis=-1,
            )

        uf = centroids_to_faces(self.mesh.u_ext, self.mesh.xu, self.mesh.x)
        vf = centroids_to_faces(self.mesh.v_ext.T, self.mesh.yv, self.mesh.y).T
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
        self.G = assemble_matrix(rows, cols, data)

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
        return xp.concatenate((xmom.reshape(-1), ymom.reshape(-1)))

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
            xp.concatenate((self.evaluate_divergence().reshape(-1), [0]))
        )[:-1]
        self.mesh.p[:, :] = pvec.reshape(self.mesh.p.shape)
        return q - self.G.dot(pvec)
