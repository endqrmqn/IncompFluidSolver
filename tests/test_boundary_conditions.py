import numpy as np
import torch
import ibfs
import pytest_utils as pyut
import matplotlib.pyplot as plt

Re = 100
nsop = ibfs.SpatialOperators(Re)


x0 = -2
x1 = 2
y0 = -1
y1 = 1

nx = 100
ny = 50

mesh = ibfs.Mesh(x0, x1, nx, y0, y1, ny)
ones_u = np.ones(mesh.u_int.shape[0])
zeros_v = np.zeros(mesh.v_int.shape[0])
uinflow_bc = lambda t: ones_u
vinflow_bc = lambda t: zeros_v
bcsu = ibfs.BoundaryConditions(
    "u", "dirichlet", "neumann", "neumann", "neumann", uinflow_bc
)
bcsv = ibfs.BoundaryConditions(
    "v", "dirichlet", "neumann", "neumann", "neumann", vinflow_bc
)


print(mesh.u_ext[:, 0])
bcsu.impose_boundary_conditions(mesh.u_ext, 0.0)
print(mesh.u_ext[:, 0])
