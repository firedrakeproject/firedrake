Assimilating point data
=======================

.. rst-class:: emphasis

    This example is taken from work done by Reuben Nixon-Hill et al. in :cite:`Nixon-Hill:2024`, and was written up by Leo Collins. The paper contains an additional example of assimilating point data in a model of the Larsen C ice shelf using the `Icepack <https://icepack.github.io/>`_ library :cite:`Shapero:2021`, which is built on Firedrake. 


This demo will show how to use Firedrake-adjoint to assimilate point data into a PDE model. 
In Firedrake we represent point data as functions in the function space of zero-order discontinuous Lagrange polynomials :math:`\operatorname{P0DG}(\Omega_{v})`, where :math:`\Omega_{v}` is a vertex-only mesh consisting of the observation locations.


Adjoint background
------------------

We have a model (typically a PDE)

.. math::

    F(u,q)=0

where :math:`u` is our solution, :math:`q` are the model parameters, and :math:`F` is our model, for example a PDE. 
We have a set of :math:`N` observations :math:`u_{\text{obs}}^i` at locations :math:`X_i`, for :math:`i=1,\ldots,N`.
We want our solution field :math:`u` to match the observations at the locations :math:`X_i`, so we define a "misfit" (or "objective") functional

.. math::

    J(u,q)=J_{\text{model-data misfit}}(u) + J_{\text{regularisation}}(q).

Here :math:`J_{\text{regularisation}}` is a regularisation term, which is there to ensure that the problem is well-posed. Often this uses some known properties of the model, such as smoothness requirements.
The :math:`J_{\text{model-data misfit}}` term is taken to be the :math:`L^2` norm of the difference between the observations :math:`u_{\text{obs}}^i` and the model solution :math:`u` point evaluated 
at the observation locations: :math:`u(X_i)`, i.e. :math:`\lVert u_{\text{obs}}^i-u(X_{i}) \rVert_{L^2}`.

We now note that if we change the model parameters :math:`q`, then the solution :math:`u` to the PDE will change, so we can write :math:`u` as a function of :math:`q`. We then write our functional :math:`J` as

.. math::

    \hat{J}(q) = J(u(q),q)

We call :math:`\hat{J}` the reduced functional. The aim is to find the value of :math:`q` which minimises the :math:`\hat{J}`. Firedrake-adjoint will compute :math:`\frac{d \hat{J}}{d q}` automatically, and we can use this to find the minimum of :math:`\hat{J}` using a gradient-based method such as Newton-CG.


Point data as finite element functions
--------------------------------------

We have our point data :math:`u_{\text{obs}}^i` observed at locations :math:`X_i`. In Firedrake, we represent our observation locations as a vertex-only mesh ,

.. math::

    \Omega_{v}=\{X_i\}_{i=1}^{N}

which is embedded in the parent mesh :math:`\Omega`, and our observations :math:`u_{\text{obs}}^i` as functions in the function space of zero-order discontinuous Lagrange polynomials on :math:`\Omega_{v}`,

.. math::
  
    u_{\text{obs}}^i \in \operatorname{P0DG}(\Omega_{v}).


Integrating a function over a vertex-only mesh is equivalent to summing the function values at the vertices of the mesh, i.e for a function :math:`f\in\operatorname{P0DG}(\Omega_{v})` we have

.. math::

    \int_{\Omega_{v}} f \, dx = \sum_{i=1}^{N} f(X_{i}).

We can interpolate a function defined on some function space on the parent mesh :math:`u\in\operatorname{FS}(\Omega)` into the function space :math:`\operatorname{P0DG}(\Omega_{v})` by evaluating the function :math:`u` at the vertices :math:`X_i` of the mesh.
This is done by the interpolation operator

.. math::

    \begin{align}
    \mathcal{I}_{\operatorname{P0DG}(\Omega_{v})}:\operatorname{FS}(\Omega)&\rightarrow\operatorname{P0DG}(\Omega_v)\\
    u&\mapsto u_{v}.
    \end{align}

Unknown thermal conductivity
--------------------

As a concrete example, we consider the steady-state heat equation 

.. math::

    -k\nabla^{2} u=f

defined on the domain :math:`\Omega`. Our solution field is :math:`u:\Omega\rightarrow\mathbb{R}`, :math:`f=1` is a forcing function, and :math:`k` is the thermal conductivity. We use the Dirichlet boundary condition

.. math::

    u=0 \text{ on } \partial\Omega.

We assume that the conductivity is of the form

.. math::

    k=k_{0}e^{q}

with :math:`k_{0}=\frac{1}{2}`, where :math:`q` is the log-conductivity field. We want to estimate the log-conductivity field :math:`q` from the (noisy) point observations :math:`u_{\text{obs}}^i`, which are taken at the locations :math:`X_i`.
We assume that the true conductivity :math:`q_{\text{true}}` is a finite element function in :math:`\operatorname{P2CG}(\Omega)`, and solve the PDE on the same function space for :math:`u_{\text{true}}\in\operatorname{P2CG}(\Omega)`.
The PDE can be written in weak form as

.. math::

    k_{0}e^{q}\int_{\Omega}\nabla u\cdot\nabla v \, dx = \int_{\Omega} fv\,dx

where :math:`v` is a test function in :math:`\operatorname{P2CG}(\Omega)`. 

Our :math:`J_{\text{model-data misfit}}` term is then 

.. math::

    \begin{align}
    J_{\text{model-data misfit}} &= \sum_{i=1}^{N} \lVert u_{\text{obs}}^i-u(X_{i}) \rVert_{L^2}^2\\
    &= \sum_{i=1}^{N}\int_{\Omega_{v}} (u_{\text{obs}}^i-\mathcal{I}_{\operatorname{P0DG}(\Omega_{v})}(u))^2 \, dx\\
    &= \sum_{i=1}^{N} (u_{\text{obs}}^i-u(X_{i}))^2.
    \end{align}

For the regularisation term :math:`J_{\text{regularisation}}` we take 

.. math::

    J_{\text{regularisation}} = \alpha^2\int_{\Omega} \lVert \nabla q \rVert_{L^2}^2 \, dx.

This ensures the smoothness of the solution :math:`u`, and :math:`\alpha` gives a weighting to this assertion.

Firedrake implementation
------------------------

We begin by importing Firedrake, Firedrake-Adjoint, and Numpy and starting the tape::

    from firedrake import *
    import numpy as np
    from firedrake.__future__ import interpolate
    from firedrake.adjoint import *
    continue_annotation()

We'll then create our mesh and define the solution and control function spaces ::

    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "CG", 2)  # solution space
    Q = FunctionSpace(mesh, "CG", 2)  # control space

Now we'll create our :math:`q_{\text{true}}` and :math:`u_{\text{true}}` fields. 
To get our :math:`u_{\text{true}}` field we solve the PDE with :math:`q_{\text{true}}`. 
We don't want to write this to the tape, so we use a :class:`~pyadjoint.stop_annotating` context manager::

    with stop_annotating():
        rng = np.random.default_rng(seed=42)
        degree = 5
        x = SpatialCoordinate(mesh)
        q_true = Function(Q)
        for k in range(degree):
            for l in range(int(np.sqrt(degree**2 - k**2))):
                Z = np.sqrt(1 + k**2 + l**2)
                phi = 2 * pi * (k * x[0] + l * x[1])

                A_kl = rng.standard_normal() / Z
                B_kl = rng.standard_normal() / Z

                expr = Constant(A_kl) * cos(phi) + Constant(B_kl) * sin(phi)
                mode = assemble(interpolate(expr, Q))

                q_true += mode

        # Now we solve the PDE with q_true to get u_true
        u_true = Function(V)
        v = TestFunction(V)
        f = Constant(1.0)
        k0 = Constant(0.5)
        bc = DirichletBC(V, 0, 'on_boundary')
        F = (k0 * exp(q_true) * inner(grad(u_true), grad(v)) - f * v) * dx
        solve(F == 0, u_true, bc)

Now we solve the PDE with :math:`q=0` as an initial guess ::

    u = Function(V)
    q = Function(Q)
    bc = DirichletBC(V, 0, 'on_boundary')
    F = (k0 * exp(q) * inner(grad(u), grad(v)) - f * v) * dx
    solve(F == 0, u, bc)

We randomly generate our observation locations and create the vertex-only mesh :math:`\Omega_{v}=\{X_{i}\}_{i=1}^{N}` and its associated function space :math:`\operatorname{P0DG}(\Omega_{v})`. ::

    N = 1000
    X_i = rng.random((N, 2))
    Omega_v = VertexOnlyMesh(mesh, X_i)
    P0DG = FunctionSpace(Omega_v, 'DG', 0)

To evaluate :obj:`!u_true` at the points :math:`X_{i}`, we interpolate it into :math:`\operatorname{P0DG}`. The resulting :class:`~.Function` will have the values of :obj:`!u_true` at the points :math:`X_i`. ::

    u_obs_vals = assemble(interpolate(u_true, P0DG)).dat.data

We add some Gaussian noise to our observations ::

    signal_to_noise = 20
    U = u_true.dat.data_ro[:]
    u_range = U.max() - U.min()
    sigma = Constant(u_range / signal_to_noise)
    zeta = rng.standard_normal(len(X_i))
    u_obs_vals += float(sigma) * zeta

Finally, we store our point observations in a :class:`~.Function` in :math:`\operatorname{P0DG}`. ::

    u_obs = Function(P0DG)
    u_obs.dat.data[:] = u_obs_vals

Next, we write down our misfit functional :math:`J` and assemble. ::

    alpha = Constant(0.02)
    
    misfit_expr = (u_obs - assemble(interpolate(u, P0DG)))**2
    regularisation_expr = alpha**2 * inner(grad(q), grad(q))

    J = assemble(misfit_expr * dx) + assemble(regularisation_expr * dx)
  
We construct our control variable :math:`\hat{q}` and our reduced functional :math:`\hat{J}`  ::

    q_hat = Control(q)
    J_hat = ReducedFunctional(J, q_hat)

Finally, we can minimise our reduced functional :math:`\hat{J}` and obtain our optimal control :math:`q_{\text{min}}`. ::

    q_min = minimize(J_hat, method='Newton-CG', options={'disp': True})

We can compare our result to :obj:`!q_true` by calculating the error between :obj:`!q_min` and :obj:`!q_true` ::

    q_err = Function(Q).assign(q_min - q_true)
    L2_err = norm(q_err, "L2")
    print(f"L2 error: {L2_err:.3e}")

A python script version of this demo can be found :demo:`here <assimilating_point_data.py>`.

.. rubric:: References

.. bibliography:: demo_references.bib
   :filter: docname in docnames

