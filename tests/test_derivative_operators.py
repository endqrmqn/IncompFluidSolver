import torch                                #type: ignore
import numpy as np                          #type: ignore
from ibfs import Mesh, SpatialOperators     #type: ignore

#print(torch.__version__)
#print(np.__version__)

x0 = -5
x1 = 10
nx = 300
y0 = -5
y1 = 5
ny = 200


def test_first_derivative():
    funcs = [
        lambda X, Y: torch.sin(X) + torch.cos(Y),
        lambda X, Y: torch.sin(2 * X) * torch.cos(3 * Y),
        lambda X, Y: torch.exp(-0.1 * (X**2 + Y**2)),
    ]

    for i, field_fn in enumerate(funcs, 1):
        mesh = Mesh(x0, x1, nx, y0, y1, ny)
        ops = SpatialOperators(Re=1.0)
        X, Y = np.meshgrid(mesh.x, mesh.y, indexing="ij")

        X_t = torch.tensor(X, dtype=torch.float64, requires_grad=True)
        Y_t = torch.tensor(Y, dtype=torch.float64, requires_grad=True)
        W_t = field_fn(X_t, Y_t)

        grad_outputs = torch.ones_like(W_t)
        dWdx, dWdy = torch.autograd.grad(W_t, (X_t, Y_t), grad_outputs=grad_outputs)

        W = W_t.detach().numpy()
        dx, dy = mesh.dx, mesh.dy

        dWdx_fd = (W[2:, 1:-1] - W[:-2, 1:-1]) / (2 * dx)
        dWdx_true = dWdx.detach().numpy()[1:-1, 1:-1]

        dWdy_fd = (W[1:-1, 2:] - W[1:-1, :-2]) / (2 * dy)
        dWdy_true = dWdy.detach().numpy()[1:-1, 1:-1]

        err_x = np.mean(np.abs(dWdx_fd - dWdx_true))
        err_y = np.mean(np.abs(dWdy_fd - dWdy_true))

        print(f"Test {i}: mean |∂w/∂x| error = {err_x:.3e}, |∂w/∂y| error = {err_y:.3e}")
        if err_x > 5e-3 or err_y > 5e-3:
            print("derivative test failed\n")
        else:
            print("derivative test passed\n")



def test_second_derivative_and_viscous():
    mesh = Mesh(-5, 5, 100, -5, 5, 100)
    ops = SpatialOperators(Re=2.0)
    X, Y = np.meshgrid(mesh.x, mesh.y, indexing="ij")

    X_t = torch.tensor(X, dtype=torch.float64, requires_grad=True)
    Y_t = torch.tensor(Y, dtype=torch.float64, requires_grad=True)
    W_t = torch.sin(X_t) * torch.cos(Y_t)

    grad_outputs = torch.ones_like(W_t)
    dWdx, dWdy = torch.autograd.grad(W_t, (X_t, Y_t), grad_outputs=grad_outputs, create_graph=True)

    d2Wdx2, = torch.autograd.grad(dWdx, X_t, grad_outputs=torch.ones_like(dWdx))
    d2Wdy2, = torch.autograd.grad(dWdy, Y_t, grad_outputs=torch.ones_like(dWdy))
    laplace_true = d2Wdx2 + d2Wdy2

    W = W_t.detach().numpy()
    laplace_fd = ops.evaluate_viscous_term(mesh, W) * ops.Re
    laplace_true = laplace_true[1:-1, 1:-1]
    laplace_fd = laplace_fd

    ny, nx = laplace_fd.shape
    laplace_true = laplace_true.detach().numpy()[:ny, :nx]
    err = np.mean(np.abs(laplace_fd - laplace_true))
    print(f"Laplacian test: mean error = {err:.3e}")
    if err > 5e-2:
        print("viscous-term test failed\n")
    else:
        print("viscous-term test passed\n")


if __name__ == "__main__":
    print("Running derivative operator tests...\n")
    test_first_derivative()
    test_second_derivative_and_viscous()
