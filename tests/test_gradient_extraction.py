import jax
import jax.numpy as jnp
from jax.experimental import sparse
from functools import partial
from jax.experimental.sparse import BCOO
import time as tlib
import ibfs
from functools import partial
import numpy as xp


Re = 100

x0, x1 = -2, 2
y0, y1 = -1, 1
nx, ny = 100, 50

mesh = ibfs.Mesh(x0, x1, nx, y0, y1, ny)
SpOps = ibfs.SpatialOperators(Re, mesh)


def grad_fun(p):
    mesh.p[:, :] = p.reshape(*mesh.p.shape)
    dpdx, dpdy = SpOps.evaluate_pressure_gradient(mesh)
    dpdx = dpdx.reshape(-1)
    dpdy = dpdy.reshape(-1)
    return xp.concatenate((dpdx, dpdy))


vec = xp.random.randn(xp.prod(mesh.p.shape))
x = grad_fun(vec)
y = SpOps.G.dot(vec)
error = xp.linalg.norm(x - y) / xp.linalg.norm(x) * 100
print(error, xp.linalg.norm(x))
