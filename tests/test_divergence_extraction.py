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

def div_fun(vec):
    szu = xp.prod(mesh.u_int.shape)
    mesh.u_ext *= 0.0
    mesh.v_ext *= 0.0
    mesh.u_int[:, :] = vec[:szu].reshape(*mesh.u_int.shape)
    mesh.v_int[:, :] = vec[szu:].reshape(*mesh.v_int.shape)
    return SpOps.evaluate_divergence(mesh).reshape(-1)

vec = xp.random.randn(xp.prod(mesh.u_int.shape) + xp.prod(mesh.v_int.shape))
x = div_fun(vec)
y = SpOps.D.dot(vec)
error = xp.linalg.norm(x - y) / xp.linalg.norm(x) * 100
print(error)



