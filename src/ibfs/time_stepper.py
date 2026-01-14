import numpy as xp
import scipy.sparse as sps
from typing import Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .mesh import Mesh
    from .spatial_operators import SpatialOperators

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


class TimeStepper:
    r"""
    Class to evolve the Navier-Stokes equations in time according to the
    fractional step method.

    :param dt: time step
    :type dt: float
    :param spatial_operators: an instance of the :class:`SpatialOperators` class
    :type spatial_operators: SpatialOperators
    :param scheme: numerical scheme
    :type scheme: Optional[str], default is :code:`'RK2'`
    """

    def __init__(
        self,
        dt,
        spatial_operators: "SpatialOperators",
        scheme: Optional[str] = "RK2",
    ):
        self.dt = dt
        self.spatial_operators = spatial_operators
        self.scheme = scheme

    def solve(
        self, t0: float, t1: float, mjump: int, q0: xp.array, verbose: Optional[bool]=True,
    ) -> Tuple[xp.array, xp.array]:
        timevec = xp.arange(t0, t1, self.dt)
        tsave = timevec[::mjump]
        q = q0.copy()
        Q = xp.zeros((len(q), len(tsave)))
        Q[:, 0] = q
        if self.scheme == "RK2":
            qs1 = xp.zeros_like(q)
            k = 0
            for i in range(1, len(timevec)):
                if verbose:
                    print(
                        f"Time step {i} out of {len(timevec)}"
                    )
                # First stage
                t = timevec[i - 1]
                qs1[:] = q + (
                    self.dt / 2
                ) * self.spatial_operators.evaluate_right_hand_side(t, q)
                qs1[:] = self.spatial_operators.enforce_divergence_free(t, qs1)
                # Second stage
                t = (timevec[i - 1] + timevec[i]) / 2
                q += self.dt * self.spatial_operators.evaluate_right_hand_side(
                    t, qs1
                )
                q[:] = self.spatial_operators.enforce_divergence_free(t, q)

                # Save data
                if xp.mod(i, mjump) == 0:
                    k += 1
                    Q[:, k] = q if k < Q.shape[-1] else None

        return (Q, tsave)
