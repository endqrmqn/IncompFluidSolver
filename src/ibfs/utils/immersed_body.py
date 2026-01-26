import numpy as xp
import scipy as sp
import torch
from typing import Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..spatial_discretization.mesh import Mesh
    from ..spatial_discretization.spatial_operators import SpatialOperators
    from ..spatial_discretization.boundary_conditions import BoundaryConditions


def make_circle(D: float, d: float) -> Tuple[xp.array, xp.array]:
    r"""
    Create a circle of diameter :math:`D` sampled at Lagrangian immersed coordinates
    :math:`(\xi_i, \eta_i)` with spacing :math:`d`. The function returns
    a tuple :code:`(xi, eta)`.

    :param D: diameter
    :type D: float
    :param d: spacing between Lagrangian points
    :type d: float

    :rtype: Tuple[xp.array, xp.array]
    """
    th = xp.arange(0, 2 * xp.pi, 2 * xp.pi / (2 * xp.pi // (2 * d)))
    return 0.5 * D * xp.cos(th), 0.5 * D * xp.sin(th)


def make_airfoil(
    thickness: float,
    n_points: int,
    alpha: float,
    device: Optional[str] = None,
) -> Tuple[xp.array, xp.array]:
    r"""
    Assemble symmetric airfoil with a given thickness, angle of attach :math:`\alpha`
    and :code:`n_points` Lagrangian points. The function returns the coordinates
    :math:`(\xi, \eta)` of the Lagrangian points.

    :param thickness: airfoil thickness
    :type thickness: float
    :param n_points: number of Lagrangian points describing the top surface
    :type n_points: int
    :param alpha: angle of attack
    :type alpha: float
    :param device: one of :code:`'gpu'` or :code:`'cpu'`
    :type str: Optional[str], default is :code:`None`

    :rtype: Tuple[xp.array, xp.array]
    """

    def airfoil_surface(thickness):
        a0, a1, a2 = 0.2969, -0.126, -0.3516
        a3, a4 = 0.2843, -0.1036
        y = (
            lambda t: thickness
            * (a0 * t + a1 * t**2 + a2 * t**4 + a3 * t**6 + a4 * t**8)
            / 0.2
        )
        return y

    dtype = torch.float32
    y_of_t = airfoil_surface(thickness)
    # Fine grid evaluations
    t_fine = torch.linspace(
        0, 1, 5000, device=device, dtype=dtype, requires_grad=True
    )
    y_fine = y_of_t(t_fine)

    # Compute arclength
    dy_dt = torch.autograd.grad(
        y_fine, t_fine, grad_outputs=torch.ones_like(y_fine), create_graph=True
    )[0]
    dx_dt = 2 * t_fine
    ds_dt = torch.sqrt((dx_dt) ** 2 + (dy_dt) ** 2)  # integrand
    dt = t_fine[1:] - t_fine[:-1]
    s_fine = torch.zeros_like(t_fine)
    s_fine[1:] = torch.cumsum(0.5 * (ds_dt[1:] + ds_dt[:-1]) * dt, dim=0)
    arclength = s_fine[-1]

    # Uniform arclength samples
    s = torch.linspace(0.0, arclength, n_points, device=device, dtype=dtype)
    t_interp = sp.interpolate.interp1d(
        s_fine.detach().cpu().numpy(),
        t_fine.detach().cpu().numpy(),
        kind="cubic",
        fill_value="extrapolate",
    )
    t = t_interp(s.detach().cpu().numpy())
    x = t**2
    y = y_of_t(t)

    # Assemble (xi, eta) vectors
    xi = xp.concatenate((x, xp.flipud(x[1:-1]))) - 0.5
    eta = xp.concatenate((y, -xp.flipud(y[1:-1])))

    return rotate_body(xi, eta, alpha)


def rotate_body(
    xi: xp.array, eta: xp.array, alpha: float
) -> Tuple[xp.array, xp.array]:
    r"""
    Rotate immersed body by angle :math:`\alpha` (given in degrees).

    :rtype: Tuple[xp.array, xp.array]
    """
    th = xp.pi * alpha / 180
    R = xp.array([[xp.cos(th), xp.sin(th)], [-xp.sin(th), xp.cos(th)]])
    for i in range(len(xi)):
        vec = R.dot(xp.asarray([xi[i], eta[i]]))
        xi[i], eta[i] = vec
    return (xi, eta)
