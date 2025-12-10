import torch
import numpy as np


def compute_torch_derivative(x, f):
    return torch.autograd.grad(
        f, x, grad_outputs=torch.ones_like(f), create_graph=True
    )[0]


def evaluate_fun_and_derivatives(X, Y, fun, xp):
    f = fun(X, Y)
    df_dX = compute_torch_derivative(X, f)
    df_dY = compute_torch_derivative(Y, f)
    d2f_dX2 = compute_torch_derivative(X, df_dX)
    d2f_dY2 = compute_torch_derivative(Y, df_dY)
    torch_tensors = [f, df_dX, df_dY, d2f_dX2, d2f_dY2]
    xp_tensors = [xp.asarray(t.detach().cpu().numpy()) for t in torch_tensors]
    return xp_tensors


def analytical_functions():
    r"""Create analytical functions for velocities and pressure."""
    alpha, beta = 10, 10
    pfun = lambda X, Y: 1 - alpha * torch.exp(
        -beta * ((Y - 0.0) ** 2 + (X - 0.0) ** 2)
    )
    ufun = lambda X, Y: 1 - alpha * torch.exp(
        -beta * ((Y - 0.25) ** 2 + (X - 0.25) ** 2)
    )
    vfun = lambda X, Y: 1 - alpha * torch.exp(
        -beta * ((Y + 0.25) ** 2 + (X + 0.25) ** 2)
    )
    return ufun, vfun, pfun
