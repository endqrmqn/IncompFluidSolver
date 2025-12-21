import numpy as np
import ibfs
import pytest_utils as pyut
import matplotlib.pyplot as plt

Re = 1
nsop = ibfs.SpatialOperators(Re)

x0 = -5
x1 = 5
y0 = -5
y1 = 5

nys = np.asarray([50, 100, 200, 400, 800, 1000])    
nxs = nys
error_x = []
error_y = []

iter = 0
for nx, ny in zip(nxs, nys):
    mesh = ibfs.Mesh(x0, x1, nx, y0, y1, ny)
    _, torch_mesh = mesh.generate_meshgrids(output_torch=True)
    Xu, Yu, Xv, Yv, Xp, Yp = torch_mesh
    ufun, vfun, pfun = pyut.analytical_functions()
    
    #momentum with torch
    u, du_dx, du_dy, d2u_dx2, d2u_dy2 = pyut.evaluate_fun_and_derivatives(
        Xu, Yu, ufun, np
    )
    v, dv_dx, dv_dy, _, _ = pyut.evaluate_fun_and_derivatives(
        Xu, Yu, vfun, np
    )
    #xrhs
    fx_int = -2 * u * du_dx - v * du_dy - u * dv_dy + (d2u_dx2 + d2u_dy2) / Re
    mesh.fx_int *= 0.0
    mesh.fx_int -= fx_int[1:-1, 1:-1]
    
    
    u, du_dx, du_dy, _, _ = pyut.evaluate_fun_and_derivatives(
        Xv, Yv, ufun, np
    )
    v, dv_dx, dv_dy, d2v_dx2, d2v_dy2 = pyut.evaluate_fun_and_derivatives(
        Xv, Yv, vfun, np
    )
    #yrhs
    fy_int = -2 * v * dv_dy - u * dv_dx - v * du_dx + (d2v_dx2 + d2v_dy2) / Re
    mesh.fy_int *= 0.0
    mesh.fy_int -= fy_int[1:-1, 1:-1]
    
    #momentum with our stuff
    u, _, _, _, _ = pyut.evaluate_fun_and_derivatives(Xu, Yu, ufun, np)
    v, _, _, _, _ = pyut.evaluate_fun_and_derivatives(Xv, Yv, vfun, np)
    #reset and update u & v
    mesh.u_ext *= 0.0
    mesh.v_ext *= 0.0
    mesh.u_ext += u
    mesh.v_ext += v
    x_mom, y_mom = nsop.evaluate_momentum_equation(mesh)
    
    error_x.append(np.max(np.abs(x_mom.reshape(-1))))
    error_y.append(np.max(np.abs(y_mom.reshape(-1))))
    
print(error_x)
print(error_y)

log_nys = np.log(nys)
log_error_x = np.log(np.array(error_x))
log_error_y = np.log(np.array(error_y))

slope_x = np.polyfit(log_nys, log_error_x, 1)[0]
slope_y = np.polyfit(log_nys, log_error_y, 1)[0]

print(f"Log-log slope (x-momentum): {slope_x}")
print(f"Log-log slope (y-momentum): {slope_y}")