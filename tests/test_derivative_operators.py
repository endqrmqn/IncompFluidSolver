import torch                                #type: ignore
import numpy as np                          #type: ignore
from ibfs import Mesh, SpatialOperators     #type: ignore

#print(torch.__version__)
#print(np.__version__)

"""
    Two tests are performed to validate the implementation of the derivative
    operators in the SpatialOperators class. The tests compare the numerical
    derivatives computed using finite difference schemes against the analytical
    derivatives obtained via PyTorch's automatic differentiation.
    
    1. First Derivative Test:
         - A set of smooth test functions are defined.
            - For each function, the first derivatives with respect to x and y
            are computed using PyTorch's autograd.
            - The same derivatives are approximated using central finite difference
            schemes on a mesh grid.
            - The mean absolute error between the two methods is calculated and
            reported. The test passes if the error is below a specified threshold.
            
    2. Second Derivative and Viscous Term Test:
            - A smooth test function is defined.
            - The second derivatives (Laplacian) are computed using PyTorch's
            autograd.
            - The viscous term is approximated using the evaluate_viscous_term
            method from the SpatialOperators class.
            - The mean absolute error between the two methods is calculated and
            reported. The test passes if the error is below a specified threshold.
    
"""

x0 = -5
x1 = 10
nx = 150
y0 = -5
y1 = 5
ny = 100


def test_first_derivative(funcs = [
    lambda X, Y: torch.sin(X) + torch.cos(Y),
    lambda X, Y: torch.sin(2 * X) * torch.cos(3 * Y),
    lambda X, Y: torch.exp(-0.1 * (X**2 + Y**2)),
    lambda X, Y: X**2 * torch.sin(Y),
    lambda X, Y: X*Y + 0.5*X**3 - 0.2*Y**4,
    lambda X, Y: torch.exp(-((X-1)**2 + (Y+2)**2)),
    lambda X, Y: (X + Y)**2,
    lambda X, Y: torch.sin(X*Y) * torch.cos(0.5*X),
    lambda X, Y: torch.tanh(X) * torch.exp(-Y**2),
    lambda X, Y: 1.0 / (1 + X**2 + Y**2),
    lambda X, Y: torch.exp(-0.1*(X**2 + Y**2)) * torch.sin(3*X),
    lambda X, Y: torch.nn.functional.softplus(X + 0.5*Y),
    lambda X, Y: torch.sin(X) * torch.sin(Y),
    ]):
    
    """
    First-derivative consistency test between:

        (1) Finite-difference forward derivative (IBFS implementation)
        (2) Autograd derivative evaluated at staggered face locations

    WHY WE SHIFT THE AUTOGRAD COORDINATES:
    --------------------------------------
    The FD operator for ∂w/∂x used by SpatialOperators is a FORWARD difference:

         dWdx_fd[i,j] = (W[i+1,j] - W[i,j]) / dx

    This derivative is NOT located at x_i — it is located at the midpoint:

         x_face = x_i + dx/2

    Similarly, the FD ∂w/∂y operator is located at:

         y_face = y_j + dy/2

    If we compare forward FD derivatives at face locations with autograd
    derivatives at CENTER locations, the error will be large.

    So we evaluate autograd at the face coordinates directly by shifting the grid.

    WHERE WE EXPECT THE DERIVATIVES TO LIVE:
    ----------------------------------------
        dWdx_fd_raw shape  = (nx-1, ny)     → x-faces
        dWdy_fd_raw shape  = (nx, ny-1)     → y-faces

    After trimming to remove ghost boundaries, both become:
        (nx-2, ny-2)

    This matches the interior region.
    """
    

    for i, field_fn in enumerate(funcs, 1):
        
        # ---------------------------------------
        # -1) initialize mesh and operators
        # ---------------------------------------
        mesh = Mesh(x0, x1, nx, y0, y1, ny)
        ops = SpatialOperators(Re=1.0)
        X, Y = np.meshgrid(mesh.x, mesh.y, indexing="ij")

        # ---------------------------------------
        # 0) compute field W
        # ---------------------------------------
        X_t = torch.tensor(X, dtype=torch.float64, requires_grad=True)
        Y_t = torch.tensor(Y, dtype=torch.float64, requires_grad=True)
        W_t = field_fn(X_t, Y_t)

        # simply for getting W as numpy array
        W = W_t.detach().numpy()

        # ---------------------------------------
        # 1) x-face autograd
        # ---------------------------------------
        X_dx = torch.tensor(X + mesh.dx/2,  dtype=torch.float64, requires_grad=True)
        Y_dx = torch.tensor(Y,              dtype=torch.float64, requires_grad=True)

        W_dx = field_fn(X_dx, Y_dx)
        dWdx_true_face, _ = torch.autograd.grad(
            W_dx, (X_dx, Y_dx),
            grad_outputs=torch.ones_like(W_dx)
        )

        # ---------------------------------------
        # 2) y-face autograd
        # ---------------------------------------
        X_dy = torch.tensor(X,             dtype=torch.float64, requires_grad=True)
        Y_dy = torch.tensor(Y + mesh.dy/2, dtype=torch.float64, requires_grad=True)

        W_dy = field_fn(X_dy, Y_dy)
        _, dWdy_true_face = torch.autograd.grad(
            W_dy, (X_dy, Y_dy),
            grad_outputs=torch.ones_like(W_dy)
        )

        # ---------------------------------------
        # 3) finite difference derivatives
        # ---------------------------------------
        dWdx_fd = ops.evaluate_derivative(mesh, W, order=1, axis=0)[1:, 1:-1]
        dWdy_fd = ops.evaluate_derivative(mesh, W, order=1, axis=1)[1:-1, 1:]

        # ---------------------------------------
        # 4) trim and test
        # ---------------------------------------
        dWdx_true = dWdx_true_face.detach().numpy()[1:-1, 1:-1]
        dWdy_true = dWdy_true_face.detach().numpy()[1:-1, 1:-1]

        err_x = np.mean(np.abs(dWdx_fd - dWdx_true))
        err_y = np.mean(np.abs(dWdy_fd - dWdy_true))

        print(f"Test {i}: mean |∂w/∂x| error = {err_x:.3e}, |∂w/∂y| error = {err_y:.3e}")
        if err_x > 5e-3 or err_y > 5e-3:
            print("derivative test failed\n")
        else:
            print("derivative test passed\n")
            
            
def test_second_derivative_and_viscous(funcs = [
    lambda X, Y: torch.sin(X) + torch.cos(Y),
    lambda X, Y: torch.sin(2 * X) * torch.cos(3 * Y),
    lambda X, Y: torch.exp(-0.1 * (X**2 + Y**2)),
    lambda X, Y: X**2 * torch.sin(Y),
    lambda X, Y: X*Y + 0.5*X**3 - 0.2*Y**4,
    lambda X, Y: torch.exp(-((X-1)**2 + (Y+2)**2)),
    lambda X, Y: (X + Y)**2,
    lambda X, Y: torch.sin(X*Y) * torch.cos(0.5*X),
    lambda X, Y: torch.tanh(X) * torch.exp(-Y**2),
    lambda X, Y: 1.0 / (1 + X**2 + Y**2),
    lambda X, Y: torch.exp(-0.1*(X**2 + Y**2)) * torch.sin(3*X),
    lambda X, Y: torch.nn.functional.softplus(X + 0.5*Y),
    lambda X, Y: torch.sin(X) * torch.sin(Y),
    ]):
    """
    Second-derivative / viscous-term consistency test between:

        (1) Laplacian computed via autograd:
                ∇²w = ∂²w/∂x² + ∂²w/∂y²
        (2) Finite-difference Laplacian inside SpatialOperators.evaluate_viscous_term

    evaluate_viscous_term implements:

        d2w_dx2 = (w[i+1,j] - 2w[i,j] + w[i-1,j]) / dx²
        d2w_dy2 = (w[i,j+1] - 2w[i,j] + w[i,j-1]) / dy²

        Re^{-1} ∇²w = (d2w_dx2 + d2w_dy2) / Re

    So the function returns Re^{-1} ∇²w on the interior:

        input w.shape        = (nx, ny)
        output viscous.shape = (nx-2, ny-2)

    We compare against autograd by:
        - building w(X, Y) on the full grid
        - computing d²w/dx² and d²w/dy² via separate autograd graphs
        - restricting ∇²w_true to [1:-1, 1:-1]
        - multiplying the FD viscous term by Re to get ∇²w_fd
        - comparing on the (nx-2, ny-2) interior.
    """

    Re = 2.0
    mesh = Mesh(-5, 5, 100, -5, 5, 100)
    ops = SpatialOperators(Re=Re)

    X_numpy, Y_numpy = np.meshgrid(mesh.x, mesh.y, indexing="ij")

    for i, field_fn in enumerate(funcs, 1):

        # make NEW graph for this one function
        X_t = torch.tensor(X_numpy, dtype=torch.float64, requires_grad=True)
        Y_t = torch.tensor(Y_numpy, dtype=torch.float64, requires_grad=True)

        W_t = field_fn(X_t, Y_t)

        # --------------------------
        # First derivatives
        # --------------------------
        dWdx, dWdy = torch.autograd.grad(
            W_t, (X_t, Y_t),
            grad_outputs=torch.ones_like(W_t),
            create_graph=True      # <--- REQUIRED
        )

        # --------------------------
        # Second derivatives
        # --------------------------
        d2Wdx2, = torch.autograd.grad(
            dWdx, X_t,
            grad_outputs=torch.ones_like(dWdx),
            retain_graph=True       # <--- keep graph ONLY for second call
        )

        d2Wdy2, = torch.autograd.grad(
            dWdy, Y_t,
            grad_outputs=torch.ones_like(dWdy)
            # no retain_graph needed now
        )

        laplace_true = (d2Wdx2 + d2Wdy2)[1:-1, 1:-1].detach().numpy()

        # --------------------------
        # FD Laplacian (scaled)
        # --------------------------
        W = W_t.detach().numpy()
        laplace_fd = ops.evaluate_viscous_term(mesh, W) * Re  # convert Re^{-1}∇²→∇²

        ny_fd, nx_fd = laplace_fd.shape
        laplace_true = laplace_true[:ny_fd, :nx_fd]

        err = np.mean(np.abs(laplace_fd - laplace_true))
        print(f"Laplacian test {i}: mean |∇²w| error = {err:.3e}")

        if err > 5e-2:
            print("viscous-term test failed\n")
        else:
            print("viscous-term test passed\n")


if __name__ == "__main__":
    print("Running derivative operator tests...\n")
    test_first_derivative()
    test_second_derivative_and_viscous()
