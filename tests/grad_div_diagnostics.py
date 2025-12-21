import numpy as np
import torch
import pytest_utils as pyut
import ibfs
import matplotlib.pyplot as plt

Re = 100
nsop = ibfs.SpatialOperators(Re)

x0 = -2
x1 = 2
y0 = -2
y1 = 2

nys = np.array([5])
nys = np.array([100, 200, 400, 800, 1000, 1600, 2000])
nxs = nys
p_error_x = []
p_error_y = []

div_error_x = []
div_error_y = []

iter = 0
for nx, ny in zip(nxs, nys):
    mesh = ibfs.Mesh(x0, x1, nx, y0, y1, ny)
    _, torch_mesh = mesh.generate_meshgrids(output_torch=True)
    Xu, Yu, Xv, Yv, Xp, Yp = torch_mesh

    ufun, vfun, pfun = pyut.analytical_functions()
    
    # pressure at       cell faces
    px, dp_dx, _, d2p_dx, _ = pyut.evaluate_fun_and_derivatives(
        Xu, Yu, pfun, np
    )
    py, _, dp_dy, _, d2p_dy = pyut.evaluate_fun_and_derivatives(
        Xv, Yv, pfun, np
    )
    print(px.dtype)
    xp, _ = nsop.evaluate_pressure_gradient(px, mesh.d)
    _, yp = nsop.evaluate_pressure_gradient(py, mesh.d)
    
    xp = xp[1:-1, :-1]
    yp = yp[:-1, 1:-1]

    dpx_trimmed = dp_dx[1:-1, 1:-1]
    dpy_trimmed = dp_dy[1:-1, 1:-1]
    
    p_error_x.append(np.max(np.abs(xp - dpx_trimmed)))
    p_error_y.append(np.max(np.abs(yp - dpy_trimmed)))
    
    
    # divergence at     cell centers
    
    
print(p_error_x)
print(p_error_y)

plt.figure(figsize=(8, 6))
plt.loglog(nxs, p_error_x, 'o-', label='p_error_x')
plt.loglog(nxs, p_error_y, 's-', label='p_error_y')
plt.xlabel('Grid Resolution (nx)')
plt.ylabel('Error')
plt.legend()
plt.grid(True, which='both', alpha=0.3)
plt.show()