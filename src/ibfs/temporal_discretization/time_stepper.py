import numpy as xp
import scipy.sparse as sps
import tqdm
from typing import Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..spatial_discretization.mesh import Mesh
    from ..spatial_discretization.spatial_operators import SpatialOperators
    from ..spatial_discretization.immersed_body import ImmersedBody

from ..spatial_discretization.sparse_matrix_operators import (
    divergence_sparsity_pattern,
    gradient_sparsity_pattern,
    divergence_data,
    gradient_data,
    assemble_matrix,
)


class TimeStepper:
    r"""
    Class to evolve the Navier-Stokes equations in time according to the
    fractional step method.

    :param dt: time step
    :type dt: float
    :param spatial_operators: an instance of the :class:`SpatialOperators` class
    :type spatial_operators: SpatialOperators
    :param immersed_body: an instance of the :class:`ImmersedBody` class
    :type immersed_body: Optional[ImmersedBody], default is :code:`None`
    :param scheme: numerical scheme
    :type scheme: Optional[str], default is :code:`'RK2'`
    """

    def __init__(
        self,
        dt,
        spatial_operators: "SpatialOperators",
        immersed_body: Optional["ImmersedBody"] = None,
        scheme: Optional[str] = "RK2",
    ):
        self.dt = dt
        self.spatial_operators = spatial_operators
        self.immersed_body = immersed_body
        self.scheme = scheme

        self.assemble_operators()

    def assemble_operators(self):
        if self.scheme == "RK2":
            self.evaluate_right_hand_side = (
                self.spatial_operators.evaluate_right_hand_side
            )
            if self.immersed_body == None:
                self.enforce_constraints = (
                    self.spatial_operators.enforce_divergence_free
                )
            else:
                self.enforce_constraints = (
                    self.immersed_body.enforce_constraints
                )

    def solve(
        self,
        t0: float,
        t1: float,
        mjump: int,
        q0: xp.array,
        verbose: Optional[bool] = True,
    ) -> Tuple[xp.array, xp.array]:
        r"""
        Evolve the Navier-Stokes equations. This function returns a tuple with the solution
        :math:`Q\in\mathbb{R}^{n\times n_t}` and the time instances :math:`t\in\mathbb{R}^{n_t}`
        at which the solution was saved.

        :param t0: initial time
        :type t0: float
        :param t1: final time
        :type t1: float
        :param mjump: save solution every :code:`mjump` time steps
        :type mjump: int
        :param q0: initial condition (dimenions of the interior nodes)
        :type q0: xp.array
        :param verbose: print advancement if :code:`True`, do not if :code:`False`
        :type verbose: Optional[bool], default is :code:`True`

        :rtype: Tuple[xp.array, xp.array]
        """
        timevec = xp.arange(t0, t1, self.dt)
        tsave = timevec[::mjump]
        q = q0.copy()
        Q = xp.zeros((len(q), len(tsave)))
        q[:] = self.enforce_constraints(timevec[0], q)
        Q[:, 0] = q
        if self.scheme == "RK2":
            qs1 = xp.zeros_like(q)
            rhs = xp.zeros_like(q)
            k = 0
            iterable = range(1, len(timevec))
            iterable = tqdm.tqdm(iterable) if verbose else iterable
            for i in iterable:
                # First stage
                t = timevec[i - 1]
                rhs[:] = self.spatial_operators.evaluate_right_hand_side(t, q)
                qs1[:] = q + self.dt / 2 * rhs
                qs1[:] = self.enforce_constraints(t, qs1)
                # Second stage
                t = (timevec[i - 1] + timevec[i]) / 2
                rhs[:] = self.spatial_operators.evaluate_right_hand_side(t, qs1)
                q += self.dt * rhs
                q[:] = self.enforce_constraints(t, q)

                # Save data
                if xp.mod(i, mjump) == 0:
                    if xp.isnan(q).any() or xp.max(xp.abs(q)) > 1e4:
                        raise ValueError(f"Blow up detected.")
                    k += 1
                    Q[:, k] = q if k < Q.shape[-1] else None

        return (Q, tsave)
