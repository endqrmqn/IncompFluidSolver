import numpy as np
import ibfs


x0 = -5
x1 = 10
nx = 150
y0 = -5
y1 = 5
ny = 100

mesh = ibfs.Mesh(x0, x1, nx, y0, y1, ny)

mesh.info()
print(mesh.x)
print(len(mesh.x))
print(mesh.y)
print(len(mesh.y))
