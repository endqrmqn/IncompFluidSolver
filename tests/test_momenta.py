import torch                                #type: ignore
import numpy as np                          #type: ignore
from ibfs import Mesh, SpatialOperators     #type: ignore

#print(torch.__version__)
#print(np.__version__)

"""
    Momenta tests for the SpatialOperators class.
    
    Two tests are performed to validate the implementation of the derivative
    operators in the SpatialOperators class. The tests compare numerical methods
    and analytical derivatives obtained via PyTorch's automatic differentiation 
    using exact equations found in the literature.
    
"""

x0 = -5
x1 = 10
nx = 300
y0 = -5
y1 = 5
ny = 200