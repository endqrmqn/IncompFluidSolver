import numpy as xp
import scipy.sparse as sps
from typing import Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .mesh import Mesh
    from .boundary_conditions import BoundaryConditions
    from .immersed_body import ImmersedBody

from .helpers import (
    interpolate_1d,
    evaluate_derivative,
    evaluate_derivative_staggered,
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

        if not test:
            self.assemble_divergence_matrix()
            self.assemble_gradient_matrix()
            self.assemble_laplacian_matrix()

            self.augment_pressure_laplacian()
            self.LuLa = sps.linalg.splu(self.La)

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
            dw += evaluate_derivative(w, d, 2, axis)[tuple(slc)]
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
        u_interp_x = interpolate_1d(u, axis=1)
        # Compute u_interp_y = (u_{i+1,j} + u_{i,j})/2
        u_interp_y = interpolate_1d(u, axis=0)
        # Compute v_interp_x = (v_{i,j+1} + v_{i,j})/2
        v_interp_x = interpolate_1d(v, axis=1)

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
        u_interp_y = interpolate_1d(u, axis=0)
        # Compute v_interp_y = (v_{i+1,j} + v_{i,j})/2
        v_interp_y = interpolate_1d(v, axis=0)
        # Compute v_interp_y = (v_{i,j+1} + v_{i,j})/2
        v_interp_x = interpolate_1d(v, axis=1)

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

    def evaluate_pressure_gradient(self) -> Tuple[xp.array, xp.array]:
        r"""
        Compute :math:`\left(\frac{\partial p}{\partial x}, \frac{\partial p}{\partial y}\right)`
        at cell faces.

        :rtype: Tuple[xp.array, xp.array]
        """
        return tuple(
            [
                evaluate_derivative_staggered(self.mesh.p, self.mesh.d, ax)
                for ax in [1, 0]
            ]
        )

    def evaluate_divergence(self) -> xp.array:
        r"""
        Compute :math:`\nabla\cdot \mathbf{u} = \partial_x u + \partial_y v`
        at cell centers.

        :rtype: xp.array
        """
        fields = [self.mesh.u_ext, self.mesh.v_ext]
        axes = xp.flipud(xp.arange(self.mesh.p.ndim, dtype=xp.int32))  # [1, 0]
        div = xp.zeros_like(self.mesh.p)
        for i, f in enumerate(fields):
            slc = [
                slice(None) if j == axes[i] else slice(1, -1)
                for j in range(len(axes))
            ]
            div += evaluate_derivative_staggered(f, self.mesh.d, axis=axes[i])[
                tuple(slc)
            ]
        return div

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
        return q - self.G.dot(
            self.LuLa.solve(
                xp.concatenate((self.evaluate_divergence().reshape(-1), [0]))
            )[:-1]
        )
