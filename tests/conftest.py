def try_import(name):
    try:
        return __import__(name)
    except ImportError:
        return None


import pytest
import numpy as np
import axisymflow as axf


@pytest.fixture(scope="session")
def library(device_and_order):
    r"""Return cuda if device is gpu else numpy"""
    device, _ = device_and_order
    return cp if device == "gpu" else np


@pytest.fixture(
    scope="session",
    params=[
        (xvec, yvec, spacings)
        for xvec, yvec, spacings in zip(
            [[-3, -1, -0.5, 0.5, 1, 2], [-1, 1]],
            [[-3, -1, -0.5, 0], [-1, 0]],
            [
                [
                    np.asarray([0.4, 0.2, 0.1, 0.2, 0.3]),
                    np.asarray([0.4, 0.2, 0.1]),
                ],
                [np.asarray([0.1]), np.asarray([0.1])],
            ],
        )
    ],
    ids=["Stretched grid", "Uniform grid"],
)
def domain_and_Reynolds(request):
    Re = 100
    xvec, yvec, spacings = request.param
    return (xvec, yvec, spacings, Re)


@pytest.fixture(scope="session")
def grid_sizes():
    dxs_min = np.asarray([0.1, 0.05, 0.025, 0.01, 5e-3])
    return dxs_min
