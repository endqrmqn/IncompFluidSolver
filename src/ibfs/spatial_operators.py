import numpy as xp
import scipy as sp
from .helpers import (
    interpolate_1d,
    evaluate_derivative,
    evaluate_derivative_staggered,
)
from .sparse_matrix_operators import (
    divergence_sparsity_pattern,
    gradient_sparsity_pattern,
    divergence_data,
    gradient_data,
    assemble_matrix,
)
from typing import Tuple, TYPE_CHECKING
if TYPE_CHECKING:
    from .mesh import Mesh


class SpatialOperators:
    r"""
    Class to evaluate the right-hand side of the Navier-Stokes equation.

    :param Re: Reynolds number
    :type Re: float
    :param mesh: an instance of the :class:`Mesh` class
    """

    def __init__(self, Re: float, mesh: "Mesh"):
        self.Re = Re
        self.mesh = mesh

        self.assemble_divergence_matrix()
        self.assemble_gradient_matrix()

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

    def evaluate_momentum_equation(
        self, mesh: "Mesh"
    ) -> Tuple[xp.array, xp.array]:
        r"""
        Evaluate the discretized momentum equation (without pressure) in the
        incompressible Navier-Stokes equations. Returns a 2-tuple
        containing the evaluation of the :math:`x`- and :math:`y`-momentum
        equations.

        :param mesh: instance of the class :class:`Mesh`
        :type mesh: Mesh
        :rtype: Tuple[xp.array, xp.array]
        """
        mom_x = (
            self.evaluate_laplacian(mesh.u_ext, mesh.d) / self.Re
            - self.evaluate_streamwise_advection(
                mesh.u_ext, mesh.v_ext, mesh.d
            )
            / (mesh.d**2)
            + mesh.fx_int
        )
        mom_y = (
            self.evaluate_laplacian(mesh.v_ext, mesh.d) / self.Re
            - self.evaluate_wallnormal_advection(
                mesh.u_ext, mesh.v_ext, mesh.d
            )
            / (mesh.d**2)
            + mesh.fy_int
        )
        return (mom_x, mom_y)

    def evaluate_pressure_gradient(
        self, mesh: "Mesh"
    ) -> Tuple[xp.array, xp.array]:
        r"""
        Compute :math:`\left(\frac{\partial p}{\partial x}, \frac{\partial p}{\partial y}\right)`
        at cell centers.

        :param mesh: instance of the :class:`Mesh` class
        :type mesh: Mesh
        :rtype: Tuple[xp.array, xp.array]
        """
        return tuple(
            [
                evaluate_derivative_staggered(mesh.p, mesh.d, ax)
                for ax in [1, 0]
            ]
        )

    def evaluate_divergence(self, mesh: "Mesh") -> xp.array:
        r"""
        Compute :math:`\nabla\cdot \mathbf{u} = \partial_x u + \partial_y v`
        at cell centers.

        :param mesh: instance of the :class:`Mesh` class
        :type mesh: Mesh
        :rtype: xp.array
        """
        fields = [mesh.u_ext, mesh.v_ext]
        axes = xp.flipud(xp.arange(mesh.p.ndim, dtype=xp.int32))  # [1, 0]
        div = xp.zeros_like(mesh.p)
        for i, f in enumerate(fields):
            slc = [
                slice(None) if j == axes[i] else slice(1, -1)
                for j in range(len(axes))
            ]
            div += evaluate_derivative_staggered(f, mesh.d, axis=axes[i])[
                tuple(slc)
            ]
        return div

    def assemble_gradient_matrix(self):
        r"""
        Instatiate the attribute :code:`self.G`, containing the matrix
        representation of the gradient operator defined in 
        :func:`evaluate_gradient`.
        """
        rows, cols, rows_extract, cols_query = gradient_sparsity_pattern(self.mesh)
        data = gradient_data(self.mesh, self, rows_extract, cols_query, 1.0)
        self.G = assemble_matrix(rows, cols, data)
    
    def assemble_divergence_matrix(self):
        r"""
        Instatiate the attribute :code:`self.D`, containing the matrix
        representation of the divergence operator defined in 
        :func:`evaluate_divergence`.
        """
        rows, cols, rows_extract, cols_query = divergence_sparsity_pattern(self.mesh)
        data = divergence_data(self.mesh, self, rows_extract, cols_query, 1.0)
        self.D = assemble_matrix(rows, cols, data)

