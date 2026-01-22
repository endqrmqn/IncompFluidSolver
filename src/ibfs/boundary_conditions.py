import numpy as xp
from .helpers import (
    interpolate_1d,
    evaluate_derivative,
    evaluate_derivative_staggered,
)
from typing import Optional, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from .mesh import Mesh


class BoundaryConditions:
    r"""
    Class to define and impose boundary conditions on the velocity fields.
    The currently available boundary conditions are :code:`'dirichlet'` or :code:`'neumann'`.

    :param field_name: one of :code:`'u'` or :code:`'v'`
    :type field_name: str
    :param x0: type of boundary condition for the left/inflow boundary.
        One of :code:`'dirichlet'` or :code:`'neumann'`.
    :type x0: str
    :param x1: type of boundary condition for the right/outflow boundary.
    :type x1: str
    :param y0: type of boundary condition for the bottom boundary.
    :type y0: str
    :param y1: type of boundary condition for the top boundary.
    :type y1: str
    :param fun_x0: function :math:`f: \mathbb{R}\to\mathbb{R}^{n_{y,int}}: t\mapsto f(t)`
        to evaluate the boundary condition. Here, :math:`t` is time and :math:`n_{y,int}` is the
        number of interior points in the :math:`y` direction. This input is considered only if
        :code:`x0 = 'dirichlet'`.
    :type fun_x0: Optional[Callable[[float], [array]]], default is :code:`None`
    :param fun_x1: function :math:`f: \mathbb{R}\to\mathbb{R}^{n_{y,int}}: t\mapsto f(t)`
        to evaluate the boundary condition. Here, :math:`t` is time and :math:`n_{y,int}` is the
        number of interior points in the :math:`y` direction. This input is considered only if
        :code:`x1 = 'dirichlet'`.
    :type fun_x1: Optional[Callable[[float], [array]]], default is :code:`None`
    :param fun_y0: function :math:`f: \mathbb{R}\to\mathbb{R}^{n_{x,int}}: t\mapsto f(t)`
        to evaluate the boundary condition. Here, :math:`t` is time and :math:`n_{x,int}` is the
        number of interior points in the :math:`x` direction. This input is considered only if
        :code:`y0 = 'dirichlet'`.
    :type fun_y0: Optional[Callable[[float], [array]]], default is :code:`None`
    :param fun_y1: function :math:`f: \mathbb{R}\to\mathbb{R}^{n_{x,int}}: t\mapsto f(t)`
        to evaluate the boundary condition. Here, :math:`t` is time and :math:`n_{x,int}` is the
        number of interior points in the :math:`x` direction. This input is considered only if
        :code:`y1 = 'dirichlet'`.
    :type fun_y1: Optional[Callable[[float], [array]]], default is :code:`None`
    """

    def __init__(
        self,
        field_name: str,
        x0: str,
        x1: str,
        y0: str,
        y1: str,
        fun_x0: Optional[Callable[[float], xp.array]] = None,
        fun_x1: Optional[Callable[[float], xp.array]] = None,
        fun_y0: Optional[Callable[[float], xp.array]] = None,
        fun_y1: Optional[Callable[[float], xp.array]] = None,
    ):
        self.field_name = field_name
        self.bc_type = [x0, x1, y0, y1]
        self.locs = ["left", "right", "bottom", "top"]
        self.dirichlet_funs = [fun_x0, fun_x1, fun_y0, fun_y1]

        allowed_names = ["u", "v", "w"]
        if self.field_name not in allowed_names:
            raise ValueError(
                f"Field_name should be one of {allowed_names}. "
                f"Currently field_name = {field_name}."
            )

        allowed_x0 = ["dirichlet", "neumann", "periodic"]
        allowed_x1 = ["dirichlet", "neumann", "periodic"]
        allowed_y0 = ["dirichlet", "neumann"]
        allowed_y1 = ["dirichlet", "neumann"]

        names = ["x0", "x1", "y0", "y1"]
        allowed_lists = [allowed_x0, allowed_x1, allowed_y0, allowed_y1]
        for i, bc in enumerate(self.bc_type):
            if bc not in allowed_lists[i]:
                raise ValueError(
                    f"Boundary condition at {names[i]} should be one of {allowed_lists[i]}. "
                    f"Currently {names[i]} = {bc}"
                )
            if bc == "dirichlet" and self.dirichlet_funs[i] == None:
                raise ValueError(
                    f"Please provide a callable for dirichlet boundary condition at {names[i]}."
                )

        if (x0 == "periodic" or x1 == "periodic") and x0 != x1:
            raise ValueError(
                f"If x0 or x1 are periodic, they should both be! Currently "
                f"x0 = {self.x0} and x1 = {self.x1}. Please fix this."
            )

    def impose_boundary_conditions(self, field: xp.array, t: float) -> None:
        r"""
        Impose boundary conditions on the velocity field. This function modifies
        the input :code:`field` in place. This field is usually :code:`mesh.u_ext` or
        :code:`mesh.v_ext`, with :code:`mesh` an instance of the :class:`Mesh` class.

        :param field: 2D array contaning either the streamwise or wall-normal velocity field
            (Usually :code:`mesh.u_ext` or :code:`mesh.v_ext`, as indicated previously.)
        :type field: xp.array
        :param t: value of time at which to evaluate the dirichlet boundary conditions (if any)
        :type t: float
        """
        for i, bc in enumerate(self.bc_type):
            loc = self.locs[i]

            # Impose streamwise velocity boundary conditions
            if self.field_name == "u":
                if loc == "left":
                    if bc == "dirichlet":
                        field[1:-1, 0] = self.dirichlet_funs[i](t)
                    elif bc == "neumann":
                        field[1:-1, 0] = (
                            4 * field[1:-1, 1] - field[1:-1, 2]
                        ) / 3
                if loc == "right":
                    if bc == "dirichlet":
                        field[1:-1, -1] = self.dirichlet_funs[i](t)
                    elif bc == "neumann":
                        field[1:-1, -1] = (
                            4 * field[1:-1, -2] - field[1:-1, -3]
                        ) / 3
                if loc == "bottom":
                    if bc == "dirichlet":
                        field[0, 1:-1] = (
                            2 * self.dirichlet_funs[i](t) - field[1, 1:-1]
                        )
                    elif bc == "neumann":
                        field[0, 1:-1] = field[1, 1:-1]
                if loc == "top":
                    if bc == "dirichlet":
                        field[-1, 1:-1] = (
                            2 * self.dirichlet_funs[i](t) - field[-2, 1:-1]
                        )
                    elif bc == "neumann":
                        field[-1, 1:-1] = field[-2, 1:-1]

            # Impose wallnormal velocity boundary conditions
            if self.field_name == "v":
                if loc == "left":
                    if bc == "dirichlet":
                        field[1:-1, 0] = (
                            2 * self.dirichlet_funs[i](t) - field[1:-1, 1]
                        )
                    elif bc == "neumann":
                        field[1:-1, 0] = field[1:-1, 1]
                if loc == "right":
                    if bc == "dirichlet":
                        field[1:-1, -1] = (
                            2 * self.dirichlet_funs[i](t) - field[1:-1, -2]
                        )
                    elif bc == "neumann":
                        field[1:-1, -1] = field[1:-1, -2]
                if loc == "bottom":
                    if bc == "dirichlet":
                        field[0, 1:-1] = self.dirichlet_funs[i](t)
                    elif bc == "neumann":
                        field[0, 1:-1] = (
                            4 * field[1, 1:-1] - field[2, 1:-1]
                        ) / 3
                if loc == "top":
                    if bc == "dirichlet":
                        field[-1, 1:-1] = self.dirichlet_funs[i](t)
                    elif bc == "neumann":
                        field[-1, 1:-1] = (
                            4 * field[-2, 1:-1] - field[-3, 1:-1]
                        ) / 3

            # Handle corners (these values are set only for plotting reasons). They
            # are never actually accessed by the solver.
            field[[0, -1], -1] = field[[0, -1], -2] # Corners at right wall
            field[[0, -1], 0] = field[[0, -1], 1]   # Corners at left wall