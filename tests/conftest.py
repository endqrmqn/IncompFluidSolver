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


@pytest.fixture(scope="session")
def domain_and_Reynolds():
    x0, x1 = -1, 1
    y0, y1 = -0.5, 0.5
    Re = 100
    return (x0, x1, y0, y1, Re)


@pytest.fixture(scope="session")
def grid_sizes():
    nys = np.asarray([100, 200, 400, 800, 1000, 1600, 3200])
    nxs = 2 * nys
    return (nxs, nys)
