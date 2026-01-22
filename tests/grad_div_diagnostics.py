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

nys = np.array([100, 200, 400, 800, 1000, 1600, 2000])
nxs = nys
p_error_x = []
p_error_y = []

div_error = []

for nx, ny in zip(nxs, nys):
    mesh = ibfs.Mesh(x0, x1, nx, y0, y1, ny)
    _, torch_mesh = mesh.generate_meshgrids(output_torch=True)
    Xu, Yu, Xv, Yv, Xp, Yp = torch_mesh

    ufun, vfun, pfun = pyut.analytical_functions()
    
    # pressure at       cell faces
    px, dp_dx, _, _, _ = pyut.evaluate_fun_and_derivatives(
        Xu, Yu, pfun, np
    )
    py, _, dp_dy, _, _ = pyut.evaluate_fun_and_derivatives(
        Xv, Yv, pfun, np
    )
    xp, _ = nsop.updated_evaluate_pressure_gradient(px, mesh.d)
    _, yp = nsop.updated_evaluate_pressure_gradient(py, mesh.d)
    
    xp = xp[1:-1, :-1]
    yp = yp[:-1, 1:-1]
    

    dpx_trimmed = dp_dx[1:-1, 1:-1]
    dpy_trimmed = dp_dy[1:-1, 1:-1]
    
    
    p_error_x.append(np.max(np.abs(xp - dpx_trimmed)))
    p_error_y.append(np.max(np.abs(yp - dpy_trimmed)))
    
    
    # divergence at     cell centers
    
    u, du_dx, _, _, _ = pyut.evaluate_fun_and_derivatives(
        Xp, Yp, ufun, np
    )
    v, _, du_dy, _, _ = pyut.evaluate_fun_and_derivatives(
        Xp, Yp, vfun, np
    )
    
    truth_div = du_dx + du_dy
            
    div = nsop.updated_evaluate_divergence(mesh, u, v)
    div_error.append(np.max(np.abs(div - truth_div)))        
    
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

print(div_error)
plt.figure(figsize=(8, 6))
plt.loglog(nxs, div_error, 'x-', label='div_error')
plt.xlabel('Grid Resolution (nx)')
plt.ylabel('Error')
plt.legend()
plt.grid(True, which='both', alpha=0.3)
plt.show()