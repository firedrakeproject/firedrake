Assimilating point data
=======================

.. rst-class:: emphasis

    This example is based on work done by Reuben Nixon-Hill et al. in :cite:`Nixon-Hill:2024`, and was written up by Leo Collins.


General theory
--------------

We have a model

.. math::

    F(u,m)=0

where :math:`u` is our solution, :math:`m` are the model parameters, and :math:`F` is our model, for example a PDE. 
We have a set of :math:`N` observations :math:`u_{\text{obs}}^i` at locations :math:`X_i`, for :math:`i=1,\ldots,N`.
We want our solution field :math:`u` to match the observations at the locations :math:`X_i`, so we define a "misfit" (or "objective") function

.. math::

    J=J_{\text{model-data misfit}} + J_{\text{regularisation}}.

Here :math:`J_{\text{regularisation}}` is a regularisation term, which is there to ensure that the problem is well-posed. Often this uses some known properties of the model, such as smoothness requirements.
The :math:`J_{\text{model-data misfit}}` term is taken to be the :math:`L^2` norm of the difference between the observations :math:`u_{\text{obs}}^i` and the model solution :math:`u` point evaluated 
at the observation locations: :math:`u(X_i)`, i.e. :math:`\lVert u_{\text{obs}}^i-u(X_{i}) \rVert_{L^2}`.

The aim is to minimise the misfit functional :math:`J`. This is possible because point evaluation as implemented using the vertex-only mesh formalism is differentiable 


Vertex-only mesh formalism
----------------------------

Our mesh consists of the our observation locations :math:`X_i`,

.. math::

    \Omega_{v}=\{X_i\}_{i=1}^{N}

which is embedded in the parent mesh :math:`\Omega`, and our observations :math:`u_{\text{obs}}^i` live in the function space of zero-order discontinuous Lagrange polynomials,

.. math::
  
    u_{\text{obs}}^i \in \operatorname{P0DG}(\Omega_{v}).


Integrating a function over a vertex-only mesh is equivalent to summing the function values at the vertices of the mesh, i.e for a function :math:`f\in\operatorname{P0DG}(\Omega_{v})` we have

.. math::

    \int_{\Omega_{v}} f \, dx = \sum_{i=1}^{N} f(X_{i}).

We can interpolate a function defined on some function space on the parent mesh :math:`u\in\operatorname{FS}(\Omega)` into the function space :math:`\operatorname{P0DG}(\Omega_{v})` by evaluating the function :math:`u` at the vertices :math:`X_i` of the mesh.
This is done by the interpolation operator

.. math::

    \mathcal{I}_{\operatorname{P0DG}(\Omega_{v})}&\rightarrow\operatorname{FS}(\Omega)
    \mathcal{I}_{\operatorname{P0DG}(\Omega_{v})}(u)&\mapsto u_{v}.


Unknown conductivity
--------------------

We consider the PDE 

.. math::

    -\nabla\cdot k\nabla u=1

defined on the domain :math:`\Omega`. Our solution field is :math:`u:\Omega\rightarrow\mathbb{R}` and :math:`k` is the conductivity. We take the Dirichlet boundary condition

.. math::

    u=0 \text{ on } \partial\Omega.

We assume that the conductivity is of the form

.. math::

    k=k_{0}e^{q}

with :math:`k_{0}=\frac{1}{2}`, where :math:`q` is the log-conductivity field. We want to estimate the log-conductivity field :math:`q` from the (noisy) point observations :math:`u_{\text{obs}}^i`, which are taken at the locations :math:`X_i`.
We take the true conductivity :math:`q_{\text{true}}` to be in :math:`\operatorname{P2CG}(\Omega)`, and solve the PDE on the same function space for :math:`u_{\text{true}}\in\operatorname{P2CG}(\Omega)`.
The PDE can be written in weak form as

.. math::

    k_{0}e^{q}\int_{\Omega}\nabla u\cdot\nabla v \, dx = \int_{\Omega} fv\,dx

where :math:`v` is a test function in :math:`\operatorname{P2CG}(\Omega)`. 

Our :math:`J_{\text{model-data misfit}}` term is then 

.. math::

    J_{\text{model-data misfit}} &= \sum_{i=1}^{N} \lVert u_{\text{obs}}^i-u(X_{i}) \rVert_{L^2}^2
    &= \int_{\Omega_{v}} (u_{\text{obs}}^i-\mathcal{I}_{\operatorname{P0DG}(\Omega_{v})}(u))^2 \, dx
    &= \sum_{i=1}^{N} (u_{\text{obs}}^i-u(X_{i))^2.

For the regularisation term :math:`J_{\text{regularisation}}` we take 

.. math::

    J_{\text{regularisation}} = \alpha^2\int_{\Omega} \lVert \nabla q \rVert_{L^2}^2 \, dx.

This ensures the smoothness of the solution :math:`u`, and :math:`\alpha` gives a weighting to this assertion.

Firedrake implementation
------------------------

We begin by importing Firedrake, Firedrake-Adjoint, and Numpy and starting the tape::

    import firedrake as fd
    import numpy as np
    from firedrake.__future__ import interpolate
    from firedrake.adjoint import continue_annotation
    continue_annotation()

We'll then create our mesh and define the solution and control function spaces ::

    mesh = fd.UnitSquareMesh(10, 10)
    V = fd.FunctionSpace(mesh, "CG", 2)  # solution space
    Q = fd.FunctionSpace(mesh, "CG", 2)  # control space

Now we'll create our :math:`q_{\text{true}}` and :math:`u_{\text{true}}` fields. 
To get our :math:`u_{\text{true}}` field we solve the PDE with :math:`q_{\text{true}}`. 
We don't want to write this to the tape, so we use a `stop_annotating` context manager::

    with fd.adjoint.stop_annotating():
        rng = np.random.default_rng(seed=42)
        degree = 5
        x = fd.SpatialCoordinate(mesh)
        q_true = fd.Function(Q)
        for k in range(degree):
            for l in range(int(np.sqrt(degree**2 - k**2))):
                Z = np.sqrt(1 + k**2 + l**2)
                phi = 2 * fd.pi * (k * x[0] + l * x[1])

                A_kl = rng.standard_normal() / Z
                B_kl = rng.standard_normal() / Z

                expr = fd.Constant(A_kl) * fd.cos(phi) + fd.Constant(B_kl) * fd.sin(phi)
                mode = fd.assemble(interpolate(expr, Q))

                q_true += mode

        u_true = fd.Function(V)
        v = fd.TestFunction(V)
        f = fd.Constant(1.0)
        k0 = fd.Constant(0.5)
        bc = fd.DirichletBC(V, 0, 'on_boundary')
        F = (k0 * fd.exp(q_true) * fd.inner(fd.grad(u_true), fd.grad(v)) - f * v) * fd.dx
        fd.solve(F == 0, u_true, bc)

Now we'll randomly generate our point data observations and add some Gaussian noise ::

    num_obs = 10
    X_i = rng.random((num_obs, 2))
    signal_to_noise = 20
    U = u_true.dat.data_ro[:]
    u_range = U.max() - U.min()
    sigma = fd.Constant(u_range / signal_to_noise)
    zeta = rng.standard_normal(len(X_i))
    u_obs_vals = np.array(u_true.at(X_i)) + float(sigma) * zeta

We can now solve the model PDE with :math:`q=0` as an initial guess ::

    u = fd.Function(V)
    v = fd.TestFunction(V)
    q = fd.Function(Q)
    bc = fd.DirichletBC(V, 0, 'on_boundary')
    F = (k0 * fd.exp(q) * fd.inner(fd.grad(u), fd.grad(v)) - f * v) * fd.dx
    fd.solve(F == 0, u, bc)

Now we write down our misfit functional ::

    alpha = fd.Constant(0.02)
    point_cloud = fd.VertexOnlyMesh(mesh, X_i)
    P0DG = fd.FunctionSpace(point_cloud, 'DG', 0)
    u_obs = fd.Function(P0DG)
    u_obs.dat.data[:] = u_obs_vals
    
    misfit_expr = (u_obs - fd.assemble(interpolate(u, P0DG)))**2
    regularisation_expr = alpha**2 * fd.inner(fd.grad(q), fd.grad(q))

    J = fd.assemble(misfit_expr * fd.dx) + fd.assemble(regularisation_expr * fd.dx)
  
We now minimise our functional :math:`J` ::

    q_hat = fd.adjoint.Control(q)
    J_hat = fd.adjoint.ReducedFunctional(J, q_hat)

    q_min = fd.adjoint.minimize(
        J_hat, method='Newton-CG', options={'disp': True}
    )