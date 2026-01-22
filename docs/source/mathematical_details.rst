Code Description
================

We solve the Navier-Stokes equation in the form

.. math::
    :label: eq:ns_equation

    \begin{aligned}
        \partial_t \mathbf{u} + \mathbf{u}\cdot \nabla \mathbf{u} &= 
        -\nabla p + Re^{-1}\nabla^2 \mathbf{u} + \int_{\mathcal{S}}\mathbf{f}(\pmb{\xi}(s,t))\delta(\pmb{\xi} - \mathbf{x})\,ds
        + \mathbf{v}\\
        \nabla\cdot\mathbf{u} &= 0 \\
        \mathbf{u}(\pmb{\xi}(s,t)) &= \int_{\mathcal{X}} \mathbf{u}(\mathbf{x})\delta(\mathbf{x}-\pmb{\xi})\,d\mathbf{x} = \mathbf{u}_{B}(\pmb{\xi}(s,t))
    \end{aligned}

where :math:`\mathbf{u}=(u,v)` is the velocity field, :math:`\mathbf{v} = (v_x, v_y)`
is an external volumetric forcing vector, :math:`\pmb{\xi}=(\xi, \eta)` is the location of an immersed body
moving with velocity :math:`\mathbf{u}_B`, and :math:`\mathbf{f} = (f_x, f_y)` is a body force associated 
with the presence of the immersed body. 
The governing equations may be written compactly as

.. math::
    :label: eq:compact_ns

    \begin{aligned}
        \begin{bmatrix}
        I & G & H \\
        D & 0 & 0 \\
        E & 0 & 0
        \end{bmatrix}\frac{\partial}{\partial t}
        \begin{bmatrix}
        \mathbf{u} \\ p \\ \mathbf{f}
        \end{bmatrix} = \begin{bmatrix}
            N(\mathbf{u}) \\ 0 \\ \mathbf{u}_{B}
        \end{bmatrix}.
    \end{aligned}


Functionalities
---------------

- Simulation of two-dimensional fluid flows with (or without) immersed boundaries,
- Extraction of linear operators in sparse COO format for linear analyses,
- Compatibility with the ``resolvent4py`` package (see `here <https://github.com/albertopadovan/resolvent4py>`_),
- Support for the assembly of Petrov-Galerkin reduced-order models giving bases :math:`\Phi` and :math:`\Psi`.




.. - The mesh and arrays to store the velocity, pressure, and forcing fields are held in the :class:`~ibfs.mesh.Mesh` class
.. - The gradient, divergence, and momentum are evaluated in the :class:`~ibfs.spatial_operators.SpatialOperators` class
.. - The operations associated with the immersed body (i.e., the last row and column of equation :eq:`eq:compact_ns`) are performed in the :class:`~ibfs.immersed_body.ImmersedBody` class
