Weak constraint 4DVar data assimilation
=======================================


.. rst-class:: emphasis

    This tutorial was contributed by `Josh Hope-Collins <mailto:joshua.hope-collins13@imperial.ac.uk>`__

Data assimilation is the process of using real world observations to improve the accuracy of a simulation.
This demo shows how to run the 4D Variational data assimilation (4DVar) method in Firedrake.
The 4DVar method is commonly used in weather and climate modelling, and there is a particular variant called "weak constraint" 4DVar (WC4DVar) that allows the use of parallel-in-time solvers.

The 4DVar method tackles the situation where we want to find an approximation :math:`x_{j}` of the true values :math:`x^{t}_{j}` of a timeseries, :math:`0<j<N`, and we have:

  1. Observation operators :math:`\mathcal{H}_{j}`, and incomplete and imperfect (noisy) observations of :math:`x^{t}_{j}` at each time point :math:`y_j=\mathcal{H}(x^{t}_{j}) + r_{j}`, where the noise is :math:`r_{j}\sim\mathcal{N}(0,R_{j})` with correlation matrix :math:`R_{j}`.
  2. An imperfect PDE model :math:`\mathcal{M}_{j}` that propagates from one value to the next :math:`x^{t}_{j}=\mathcal{M}_{j}(x^{t}_{j-1})+q_{j}`, where the noise is :math:`q_{j}\sim\mathcal{N}(0,Q_{j})` with correlation matrix :math:`Q_{j}`.
  3. A prior estimate of the initial condition :math:`x_{b}=x^{t}_{0}+b`, where the noise is :math:`b\sim\mathcal{N}(0,B)` with correlation matrix :math:`B`.

We want to find a timeseries that minimises the misfits in the observations, propagator, and background data, which we formulate as finding the minimiser :math:`x_{j}` of the following objective functional:

.. math::

   \min_{x} J(x) = \|x_{0} - x_{b}\|_{B^{-1}}^{2}
                   + \sum^{N_{w}}_{j=0}\|\mathcal{H}_{j}(x_{j}) - y_{j}\|_{R_{j}^{-1}}^{2}
                   + \sum^{N_{w}}_{j=1}\|x_{j} - \mathcal{M}_{j}(x_{j-1})\|_{Q_{j}^{-1}}^{2}

The "weak constraint" is that we have allowed our PDE model :math:`\mathcal{M}` to be imperfect, rather than requiring the entire timeseries to be a perfect trajectory of :math:`\mathcal{M}` as is the standard in "strong constraint" 4DVar.
Not only does this account for the fact that our PDE model inherently has numerical and modelling errors, it also enables time-parallelism because each model misfit term :math:`x_{j}-\mathcal{M}_{j}(x_{j-1})` only requires the two neighbouring values of :math:`x_{j}`, so can be evaluated independently of model misfit terms earlier or later in the timeseries.

The minimisation is solved using a Gauss-Newton method, where at each iteration :math:`k` the increment :math:`\delta x^{k} = x^{k+1} - x^{k}` is calculated by minimising the derivative :math:`\nabla_{x}J` (often called the "incremental formulation" in 4DVar literature).

.. math::

   (\nabla_{x}^{2}J(x^{k}))\delta x^{k} = - \nabla_{x}J(x^{k})

For WC4DVar the Hessian :math:`\mathbf{S}=\nabla_{x}^{2}J(x)` and derivative :math:`\nabla_{x}J(x)` of :math:`J` are:

.. math::

   \mathbf{S}\delta\mathbf{x} =
   (\mathbf{L}^{T}\mathbf{D}^{-1}\mathbf{L} + \mathbf{H}^{T}\mathbf{R}^{-1}\mathbf{H})\mathbf{\delta} x
   = \mathbf{L}^{T}\mathbf{R}^{-1}\mathbf{b} + \mathbf{H}^{T}\mathbf{R}^{-1}\mathbf{d}

Where the matrices are constructed using linearisations :math:`H` and :math:`M` of :math:`\mathcal{H}` and :math:`\mathcal{M}`:

.. math::
   \mathbf{L} =
   \begin{pmatrix}
      I      &        &        &   \\
      -M_{1} & I      &        &   \\
             & -M_{2} & I      &   \\
             &        & -M_{3} & I \\
   \end{pmatrix},
   \quad
   \mathbf{D} =
   \begin{pmatrix}
      B &       &       &       \\
        & Q_{1} &       &       \\
        &       & Q_{2} &       \\
        &       &       & Q_{3} \\
   \end{pmatrix},
   \quad
   \mathbf{H} =
   \begin{pmatrix}
      H_{0} &       &       &       \\
            & H_{1} &       &       \\
            &       & H_{2} &       \\
            &       &       & H_{3} \\
   \end{pmatrix},
   \quad
   \mathbf{R} =
   \begin{pmatrix}
      R_{0} &       &       &       \\
            & R_{1} &       &       \\
            &       & R_{2} &       \\
            &       &       & R_{3} \\
   \end{pmatrix}

The observation operator matrix :math:`\textbf{H}` and the correlation operator matrices :math:`\textbf{D}` and :math:`\textbf{R}` are all block diagonal, with one block per timestep, so clearly both the action and inverse of each can be applied parallel-in-time.
The model integration matrix :math:`\textbf{L}` is block lower bidiagonal so its action can be applied parallel-in-time, with a communication overhead that is independent of the number of timesteps.

Assuming that :math:`\mathbf{S}` is dominated by the first term, which describes the model integration, we can construct a preconditioner  using an approximate model integration operator :math:`\mathbf{\tilde{L}}\approx\mathbf{L}`

.. math::

   \mathbf{S} \approx \mathbf{\tilde{S}} = \mathbf{\tilde{L}}^{T}\mathbf{D}^{-1}\mathbf{\tilde{L}}

Unfortunately, using the exact integration operator :math:`\mathbf{\tilde{L}}=\mathbf{L}` is impractical because the inverse is a block dense lower triangular matrix:

.. math::

   \textbf{L}^{-1} =
   \begin{pmatrix}
      I        &          &          &   \\
      -M_{1,1} & I        &          &   \\
      -M_{1,2} & -M_{2,2} & I        &   \\
      -M_{1,3} & -M_{2,3} & -M_{3,3} & I \\
   \end{pmatrix}

where :math:`M_{i,j}=\prod^{j}_{k=i}M_{k}` is the integration from timestep :math:`i-1` to timestep :math:`j` (so :math:`M_{j,j}=M_{j}`).
Clearly, the action of the inverse :math:`\mathbf{L}^{-1}` cannot be applied parallel-in-time because each timestep depends on all previous steps, which motivates the need for cheaper approximations :math:`\mathbf{\tilde{L}}`.
Broadly, there are three approaches to building :math:`\mathbf{\tilde{L}}`.

1. Keeping the structure of :math:`\mathbf{L}` but using an approximation of the time integrator :math:`\tilde{M}\approx M`.
Examples include :math:`\tilde{M}=0` which reduces :math:`\mathbf{\tilde{L}}^{-1}` to the identity, or :math:`\tilde{M}=I`, in which case :math:`\mathbf{\tilde{L}}` has the same block sparsity as :math:`\mathbf{L}` but with every non-zero block replaced with identity.
These are both clearly cheap but discard information from the propagator.

2. Approximating :math:`\mathbf{\tilde{L}}` using a parallel-in-time method, for example a fixed number of multigrid-in-time iterations.
As for spatial multigrid, the exact structure of :math:`\mathbf{\tilde{L}}^{-1}` then becomes complicated to write out, but may retain much more information from the propagator.

3. Low rank or randomised preconditioners, applying ideas from these fields to construct preconditioners with low operation counts.

A different approach for introducing time-parallelism into WC4DVar is called the saddle point formulation, which the linear system at each Gauss-Newton iteration in terms of the KKT conditions and expands the unknowns to include two Lagrange multipliers :math:`\delta\mathbf{\eta}` and :math:`\delta\mathbf{\lambda}`:

.. math::

   \mathbf{A}\delta\mathbf{w}
   =
   \begin{pmatrix}
      \mathbf{D}     & \mathbf{0}     & \mathbf{L} \\
      \mathbf{0}     & \mathbf{R}     & \mathbf{H} \\
      \mathbf{L}^{T} & \mathbf{H}^{T} & \mathbf{0} \\
   \end{pmatrix}
   \begin{pmatrix}
      \delta\mathbf{\eta}    \\
      \delta\mathbf{\lambda} \\
      \delta\mathbf{x}       \\
   \end{pmatrix}
   =
   \begin{pmatrix}
      \mathbf{b} \\
      \mathbf{d} \\
      \mathbf{0} \\
   \end{pmatrix}
   =
   \mathbf{g}

The saddle point formulation maintains the ability to apply the matrix action parallel-in-time, and also opens the door to the zoo of preconditioning strategies developed for saddle point systems.
We will explore preconditioners which are based on a partial, and approximate, application of the Schur LDU factorisation.
Notice that the Hessian of the original Gauss-Newton method is the Schur complement of the saddle point system after elimination of :math:`\delta\mathbf{\eta}` and :math:`\delta\mathbf{\lambda}`.
Using the same approximation :math:`\mathbf{\tilde{S}}` as above, we can construct a block diagonal preconditioner, as well as upper and lower block triangular preconditioners:

.. math::

   \mathbf{P}_{D} =
   \begin{pmatrix}
      \mathbf{D} & \mathbf{0} & \mathbf{0}         \\
      \mathbf{0} & \mathbf{R} & \mathbf{0}         \\
      \mathbf{0} & \mathbf{0} & \mathbf{\tilde{S}} \\
   \end{pmatrix},
   \quad
   \mathbf{P}_{U} =
   \begin{pmatrix}
      \mathbf{D} & \mathbf{0} & \mathbf{L}         \\
      \mathbf{0} & \mathbf{R} & \mathbf{H}         \\
      \mathbf{0} & \mathbf{0} & \mathbf{\tilde{S}} \\
   \end{pmatrix},
   \quad
   \mathbf{P}_{L} =
   \begin{pmatrix}
      \mathbf{D}     & \mathbf{0}     & \mathbf{0}         \\
      \mathbf{0}     & \mathbf{R}     & \mathbf{0}         \\
      \mathbf{L}^{T} & \mathbf{H}^{T} & \mathbf{\tilde{S}} \\
   \end{pmatrix}


In this example we will use the advection-diffusion equation in one spatial dimension :math:`z`, with a spatially varying advection velocity :math:`c(z)`, a time-dependent forcing term :math:`g(t)`, and periodic boundary conditions.
The "ground-truth" initial condition is just a simple sinusoid.

.. math::

   \partial_{t}u + \vec{c}(z)\cdot\nabla u + \nu\nabla^{2}u = g(t) &

   t \in [0, T], \quad z \in \Omega = [0, 1) &

   u_{r} = 0.3, \quad u(z, 0) = u_{r}\sin(2\pi z) &

   u(0, t) = u(1, t) &

   c(z) = 1 + u_{r}\cos(2\pi z) &

For the time integration we use the trapezium rule.
Choosing a function space :math:`V` for the solution, the weak form to find the next timestep :math:`u^{n+1}\in V` from the current timestep :math:`u^{n}\in V` is:

.. math::

   \int_{\Omega}\left(\frac{u^{n+1} - u^{n}}{\Delta t}\right)v\mathrm{d}x
   + \int_{\Omega}\left(\vec{c}\cdot\nabla\left(\frac{u^{n+1} + u^{n}}{2}\right) \right)v\mathrm{d}x
   + \int_{\Omega}\nu\nabla\left(\frac{u^{n+1} + u^{n}}{2}\right)\cdot\nabla v\mathrm{d}x
   - \int_{\Omega}gv\mathrm{d}x
   = 0,
   \quad \forall v \in V.

Now we will go through how to set up and solve the WC4DVar system in Firedrake, with the following steps:

  1. Define the finite element model for the advection-diffusion equation :math:`\mathcal{M}`.
  2. Define the observation operator :math:`\mathcal{H}` and generate some synthetic "ground-truth" observation data for :math:`y_{j}`.
  3. Initialise a ``ReducedFunctional`` for :math:`\hat{J}`.
  4. Specify a solver configuration and calculate an optimised :math:`x`.


First we import Firedrake, including the components from the adjoint module.
As we will be generating some random noise, we set the random number generator seed to a fixed value..

::

  from firedrake import *
  from firedrake.adjoint import *
  from firedrake.__future__ import interpolate, Interpolator
  import numpy as np
  np.random.seed(13)

A Firedrake ``Ensemble`` splits ``COMM_WORLD`` into several ensemble members, with spatial parallelism within each ensemble member and time-parallelism between members.
Here we specify just one MPI rank per ensemble member, and the number of ensemble members automatically adjusts to use all available ranks.
The communicator ``ensemble.comm`` is used for the spatial parallelism, so is the one we use to construct the mesh.

::

  ensemble = Ensemble(COMM_WORLD, 1)

  mesh = PeriodicUnitIntervalMesh(100, comm=ensemble.comm)

Next, we start constructing the advection-diffusion scheme, using the CG1 function space for the solution, and the space of real numbers to hold the time value.

::

  V = FunctionSpace(mesh, "CG", 1)
  Vr = FunctionSpace(mesh, "R", 0)

The control :math:`x_{j}` is a timeseries distributed in time over the ``Ensemble``, with each timestep being a Firedrake ``Function``.
For this we use an ``EnsembleFunctionSpace`` which represents a mixed function space with each component living on a particular ensemble member.
To initialise the ``EnsembleFunctionSpace`` we just need the ``Ensemble`` and a list of ``FunctionSpace`` for the local components.
We give each ensemble member an equal number of observation stages, and include an extra component on the first member for the initial condition :math:`x_{0}`.

::

  rank = ensemble.ensemble_rank
  nlocal_stages = ns//ensemble.ensemble_size
  nlocal_spaces = nlocal_stages
  if rank == 0:
      nlocal_spaces += 1
  W = EnsembleFunctionSpace([V for _ in range(nlocal_spaces)], ensemble)
  control = EnsembleFunction(W)

Next, we construct the advection-diffusion scheme using UFL.
The forcing term :math:`g(t)` is rather involved, but ensures that there is some non-trivial variation in the solution, and prevents it decaying to zero over long time periods due to the diffusion.

::

  t = Function(Vr).zero()
  dt = Function(Vr).assign(8e-3)

  umax = Constant(0.3)
  c = Function(V).project(1 + umax*cos(2*pi*x))
  nu = Constant(1/80)

  x1 = 1 - x
  t1 = t + 1
  x, = SpatialCoordinate(mesh)
  g = pi*sqrt(sin(pi*x))*(
      pi*umax*(
          (x + umax*(1 + sin(pi*t1))*sin(2*pi*x1*(2+cos(2*pi*t1))))
          *cos(2*pi*x*sin(pi*t1))*sin(2*pi*x1*cos(pi*t1))

          + (x1 - umax*(1 + cos(pi*t1))*sin(pi*x*cos(pi*t1)))
          *sin(pi*x*cos(pi*t1))*cos(pi*x1*(3+sin(3*pi*t1)))
      )
      + (2*nu*(umax*pi*(1-sin(pi*t1)))**2)
        *(sin(pi*x*cos(pi*t1))*sin(pi*x1*(4+cos(4*t1)))
          + cos(pi*x*sin(pi*t1))*cos(pi*x1*sin(pi*t1)))
  )

The finite element form is written out in the code almost identically to the maths above thanks to the Unified Form Language (UFL).
If we were interested in testing a different equation, we would just need to change these lines, and possibly define a different function space above.

::

  un, un1 = Function(V), Function(V)
  v = TestFunction(V)
  uh = 0.5*(un1 + un)
  F = (inner((un1 - un)/dt, v)*dx
       + inner(vel, uh.dx(0))*v*dx
       + inner(nu*grad(uh), grad(v))*dx
       - inner(g, v)*dx
  )

Firedrake uses PETSc to solve the finite element problem for each timestep, so here we need to specify the solution strategy with a PETSc options dictionary.
For such a small problem, we just use a direct solver by specifying LU as the preconditioner (PC) and "preconditioner only" as the Krylov method (KSP).
For a more complicated equation or timestepper we could specify a wide range of solvers just by changing these options.

The ``solve_step`` convenience function just wraps up moving ``un`` to the next step and incrementing the time for us.

::

  params = {
      "snes_type": "ksponly",
      "ksp_type": "preonly",
      "pc_type": "lu",
  }

  def solve_step():
      un1.assign(un)
      solve(F==0, un1, solver_parameters=params)
      un.assign(un1)
      t.assign(t + dt)

Our observations will be point evaluations at a set of 20 random locations in the domain.
This point cloud is created as a ``VertexOnlyMesh`` from a set of coordinates and a parent mesh.
The observation operator ``H`` is then simply interpolating onto this mesh.
The observation operator can be any finite element operation expressable in Firedrake, for example we could also have interpolated the energy ``0.5*x*x``.

::

  # observations are point evaluations at random locations
  stations = np.random.rand(20, 1)
  vom = VertexOnlyMesh(mesh, stations)
  Y = FunctionSpace(vom, "DG", 0)

  def H(x):
      return assemble(interpolate(x, Y))

Now we need the correlation operators :math:`B`, :math:`Q`, and :math:`R`.
We need to do three things with correlation operators: solve the system :math:`Bv=w` for :math:`v`, apply the action :math:`Bw=v` for :math:`v`, and generate physically relevant noise.
If :math:`w_{j}\sim\mathcal{N}(0,I)` then :math:`B^{1/2}w_{j}=v_{j}\sim\mathcal{N}(0,B)`, i.e. :math:`B^{1/2}` transforms uncorrelated noise to correlated noise with covariance :math:`B`.
Firedrake provides a ``CorrelationOperatorBase`` base class with ``apply``, ``solve``, and ``correlated_noise`` methods, as well as a few implementations of specific correlation operators.

Background and model errors have spatial correlations with a particular lengthscale :math:`L` and variance :math:`\sigma^{2}`.
The action :math:`v=Bw` of these correlation operators can be approximated by integrating the diffusion equation over :math:`m` pseudo-timesteps using :math:`x` as the initial condition, with careful choice of the diffusion coefficient :math:`\kappa` and normalisation factor :math:`\omega`.
If the number of pseudo-timesteps :math:`m` is even, then :math:`B^{1/2}` can be approximated by only taking :math:`m/2` steps.

.. math::

   G = (I - \kappa\nabla^{2})^{-1}

   B = \omega G^{m} \omega, \quad B^{1/2} = \omega G^{m/2}

   L = \sqrt(2m\kappa), \quad \omega^{2} = \sigma^{2}L\sqrt(2\pi)

This type of correlation operator is particularly well suited for Firedrake as we can easily express the diffusion equation as a finite element problem.

Firedrake provides the ``ImplicitDiffusionCorrelation`` and ``ExplicitDiffusionCorrelation`` classes to implement these correlation operators using backward and forward Euler pseudo-timestepping respectively.
The variance of the model error is set to be proportional to the length of the observation stage :math:`n_{t}\Delta t`.

::

  # Background correlation operator
  sigma_b = sqrt(1e-2)
  B = ImplicitDiffusionCorrelation(V, sigma_b, L=0.2, m=4, seed=2)

  # Model correlation operator
  sigma_q = sqrt(1e-4*nt*dt)
  Q = ImplicitDiffusionCorrelation(V, sigma_q, L=0.05, m=2, seed=17)

The observations are treated as uncorrelated with a diagonal correlation operator (which becomes a mass matrix in the finite element context).

::

  # Observation correlation operator
  sigma_b = sqrt(1e-3)
  R = ExplicitMassCorrelation(Y, sigma_r, seed=18)

For our test case we will pick a known initial condition :math:`x^{t}_{0}` and use this to generate a set of "ground-truth" observations :math:`y_{i}`.
We do this by integrating forward over each observation stage, and adding some model error noise by setting :math:`x^{t}_{j}=\mathcal{M}(x^{t}_{j-1})+q_{j}` where :math:`q_{j}\sim\mathcal{N}(0,Q)`.
Then, we take a noisy observation by setting :math:`y_{j}=\mathcal{H}(x^{t}_{j})+r_{j}` where :math:`r_{j}\sim\mathcal{N}(0,R)`.
The background state is generated similarly, by adding noise to the ground truth initial condition: :math:`x_{b}=x^{t}_{0}+b_{j}` where :math:`b_{j}\sim\mathcal{N}(0,B)`.

Because our test case is distributed over the ``Ensemble``, each observation :math:`y_{j}` needs to live on the right ensemble member.
This is achieved using the ``Ensemble.sequential`` context manager, which runs the code block within the context on each ensemble member in turn.
Any kwarg passed to ``Ensemble.sequential`` is made available in the ``ctx`` context object, and is sent forward to the next ensemble member once the local code block is complete.
After running the local part of the timeseries on each ensemble member, this allows us to pass forward the state ``un`` and the time ``t`` to the next member.
After initialising :math:`y_{j}` we save the final condition :math:`x^{t}_{N_t}` to compare th optimised solution to later.

::

  truth_ics = Function(V).project(umax*sin(2*pi*x))

  background = truth_ics.copy(deepcopy=True) + B.correlated_noise()

  # take an observation at the initial condition
  if rank == 0:
      y = [H(truth_ics) + R.correlated_noise()]

  un.assign(truth_ics)
  t.assign(0.0)

  with ensemble.sequential(u=un, t=t) as ctx:
      un.assign(ctx.u)
      t.assign(ctx.t)

      for j in range(nlocal_stages):
          for i in range(nt):
              solve_step()
          un.assign(un + Q.correlated_noise())
          y.append(H(un) + R.correlated_noise())

  truth_end = ensemble.bcast(
      un.copy(deepcopy=True), root=ensemble.ensemble_size-1)

Now that we have the "ground-truth" observations, we can create a function to evaluate the error vs the observation at each timestep.

::

  def observation_error(i):
      return lambda x: Function(Y).assign(H(x) - y[i])

Now we have all the pieces ready to start assembling the 4DVar system.
``continue_annotation`` tells Pyadjoint to start recording any code that is executed from now on.
The ``FourDVarReducedFunctional`` class will manage recording, constructing, and solving the 4DVar system.
To initialise it, it needs the ``EnsembleFunction`` as a ``pyadjoint.Control``, and the components to evaluate the functional at the initial condition, i.e. the background state and covariance for :math:`\|x_{0}-x_{b}\|_{B^{-1}}^{2}`, and the observation error and covariance for :math:`\|\mathcal{H}_{0}(x_{0})-y_{0}\|_{R_{0}^{-1}}^{2}`.

The ``weak_constraint=True`` argument is needed because the ``FourDVarReducedFunctional`` can also be used for strong constraint 4DVar problems, although we won't cover that here.


::

  continue_annotation()

  # This object will construct and solve the 4DVar system
  Jhat = FourDVarReducedFunctional(
      Control(control),
      background=background,
      background_covariance=B,
      observation_covariance=R,
      observation_error=observation_error(0),
      weak_constraint=True)

All Firedrake operations are "taped" by pyadjoint, so all we need to do to initialise the stages is to run :math:`\mathcal{M}` and :math:`\mathcal{H}` within the ``FourDVarReducedFunctional.recording_stages`` context manager below.
This wraps the ``ensemble.sequential`` context manager, and additionally provides the ``stages`` iterator that we loop through to record the stages on the local ensemble member.
For each ``stage``, we integrate forward from ``stage.control`` (i.e. :math:`x_{j-1}`), and then set the observation by providing the state (i.e. :math:`\mathcal{M}_{j}(x_{j-1})`) error operator, and the covariances.
The ``control`` for the first stage is set to :math:`x_{b}`, and the ``control`` for subsequent stages is set to the value of the ``state`` passed to ``set_observation`` by the previous stage.
This ensures that the initial guess for :math:`x_{j}` is a continuous trajectory over the entire ``Ensemble``.

::

  t.assign(0.0)
  with Jhat.recording_stages(t=t) as stages:
      for stage, ctx in stages:
          un.assign(stage.control)
          un1.assign(un)
          t.assign(ctx.t)

          for i in range(nt):
              solve_step()

          stage.set_observation(
              state=un,
              observation_error=observation_error(stage.local_index),
              observation_covariance=R,
              forward_model_covariance=Q)

  pause_annotation()

``Jhat`` now has a record of all operations in the model, and can use this to a) re-evaluate :math:`\hat{J}(x)` with different control values, b) calculate the derivative with respect to the controls, and c) apply the action of the Hessian.

We will demonstrate a few different solver configurations, so we save a copy of the initialised controls to reuse as the initial guess for each solve.

::

  prior = Jhat.control.control.copy()

First, we will solve the primal (state) formulation using the approximate Schur complement :math:`\mathbf{\tilde{S}}` with :math:`\mathbf{\tilde{L}}=\mathbf{L}`.

TAO is PETSc's optimisation library, and provides a range of optimisation methods.
Just like the timestepper, the TAO solver is configured using a set of options strings.
There are several levels to this solver, which we will explain in turn:

* At the top level of the dictionary, ``'tao_gttol': 1e-3`` says that convergence tolerance is a drop in the gradient of :math:`10^{-3}`.
  Nest, we specify a Newton Linesearch method using ``'tao_type': 'nls'``. Conveniently, the options dictionaries will nest, so we can set options for the Newton iterations in the ``'tao_nls'`` dictionary.

* At each Newton iteration, the Krylov solver will converge either after a :math:`10^{-2}` drop in the residual (``'ksp_rtol'``) or 20 iterations (``'ksp_max_it'``).
  We use the Conjugate Gradient method (``'ksp_type': 'cg'``), preconditioned with the ``WC4DVarSchurPC`` preconditioner, which implements :math:`\mathbf{\tilde{S}}^{-1}`.

* The :math:`\mathbf{\tilde{L}}` and :math:`\mathbf{\tilde{L}}^{T}` solves in the Schur preconditioner are configured identically in the ``'wcschur_l'`` dictionary.
  We use a fixed number of Richardson iterations to approximate the inverses.
  One Richardson iteration is equivalent to approximating :math:`\mathbf{L}^{-1}` with :math:`\mathbf{L}` itself, but each subsequent iteration fills in another subdiagonal of :math:`\mathbf{L}^{-1}`, so after :math:`n` iterations :math:`\mathbf{\tilde{L}}^{-1}` is a low (block)-bandwidth approximation to :math:`\mathbf{L}^{-1}`.
  To start with we do a full solve using :math:`N_{w}` iterations to test the "best" possible performance of :math:`\mathbf{\tilde{S}}^{-1}`.

* The :math:`\mathbf{D}^{-1}` solve is configured in the ``'wcschur_d'`` dictionary.
  :math:`\mathbf{D}^{-1}` is block diagonal, with each block corresponding to one component of the ``EnsembleFunctionSpace``, so we use the block diagonal preconditioner ``EnsembleBJacobiPC``.
  On each block, the ``CorrelationOperatorPC`` will extract the corresponding ``CorrelationOperator`` from :math:`\mathbf{D}` and apply the action of ``B`` or ``Q``.

::

  schur_parameters = {
      'tao_converged_reason': None,  #  .  .  .  .  .  .  .  .  .  .  .  . # Print out diagnostics
      'tao_monitor': None,
      'tao_max_it': 20,  #  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  . # maximum iterations
      'tao_gttol': 1e-3,  # .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  . # stopping tolerance
      'tao_type': 'nls',  # .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  . # Newton linesearch
      'tao_nls': {
          'ksp_converged_rate': None,
          'ksp_monitor': None,
          'ksp_converged_maxits': None,
          'ksp_max_it': 20,
          'ksp_rtol': 1e-2,
          'ksp_type': 'cg',  # .  .  .  .  .  .  .  .  .  .  .  .  .  .  . # Conjugate Gradient
          'pc_type': 'python',
          'pc_python_type': 'firedrake.WC4DVarSchurPC',  #.  .  .  .  .  . # S^{-1}=L^{-1}DL^{-T}
          'wcschur_l': {
              'ksp_convergence_test': 'skip',
              'ksp_converged_maxits': None,
              'ksp_type': 'richardson',  # .  .  .  .  .  .  .  .  .  .  . # Fixed bandwidth
              'ksp_max_it': ns+1,  # .  .  .  .  .  .  .  .  .  .  .  .  . # approximation
          },
          'wcschur_d': {
              'ksp_type': 'preonly',
              'pc_type': 'python',
              'pc_python_type': 'firedrake.EnsembleBJacobiPC',  #  .  .  . # block-diagonal PC
              'sub_pc_type': 'python',
              'sub_pc_python_type': 'firedrake.CorrelationOperatorPC',  #. # Action of B,Q
          },
      },
  }

  tao = TAOSolver(MinimizationProblem(Jhat),
                  parameters=saddle_parameters)
  xopts = tao.solve()

  prior_ics = background
  prior_end = ensemble.bcast(
      prior.subfunctions[-1].copy(deepcopy=True),
      root=ensemble.ensemble_size-1)

  xopts_ics = ensemble.bcast(
      xopts.subfunctions[0].copy(deepcopy=True),
      root=0
  xopts_end = ensemble.bcast(
      xopts.subfunctions[-1].copy(deepcopy=True),
      root=ensemble.ensemble_size-1)

The output of the ``TAOSolver.solve`` call is below.
The total value of the objective function dropped from 246.5 to 22.23 in two Gauss-Newton iterations, and the gradient (residual) dropped from 4490 to 0.36.
The convergence rate during both solves was quite fast, at 0.167 and 0.265

::

  0 TAO,  Function value: 246.549,  Residual: 4490.03
    Residual norms for tao_nls_ solve.
    0 KSP preconditioned resid norm 6.552341157577e+02
    1 KSP preconditioned resid norm 1.304231164182e+02
    2 KSP preconditioned resid norm 5.649750560861e+00
    3 KSP preconditioned resid norm 8.682705626648e+00
    4 KSP preconditioned resid norm 3.348733324735e-01
  Linear tao_nls_ solve converged due to CONVERGED_RTOL iterations 4 res rate 0.167505 R^2 0.923338
  1 TAO,  Function value: 22.2748,  Residual: 659.087
    Residual norms for tao_nls_ solve.
    0 KSP preconditioned resid norm 3.348733324735e-01
    1 KSP preconditioned resid norm 1.116631530433e+00
    2 KSP preconditioned resid norm 1.172918174105e-01
    3 KSP preconditioned resid norm 3.196064727185e-01
    4 KSP preconditioned resid norm 3.061657364491e-01
    5 KSP preconditioned resid norm 9.196930369193e-02
    6 KSP preconditioned resid norm 1.390118156550e-03
    7 KSP preconditioned resid norm 5.257952356658e-06
  Linear tao_nls_ solve converged due to CONVERGED_RTOL iterations 7 res rate 0.264703 R^2 0.634282
  2 TAO,  Function value: 22.2342,  Residual: 0.356489
  TAO solve converged due to CONVERGED_GTTOL iterations 2

Comparing the breakdowns of the objective functionals for the truth values, the prior trajectory and the optimised values, we can see that while the model misfit increased slightly for the optimised values vs the prior, the observation misfit dropped significantly from 247 to 14.6, almost as low as the truth observation misfit of 13.6.

The result is that the optimised initial and final conditions match the truth values around 10 times and 20 times more accurately than the prior conditions respectively.

::

  Jhat.Jmodel(truth) = 4.0181e+02
  Jhat.Jobservations(truth) = 1.3641e+01
  Jhat(truth) = 4.1545e+02

  Jhat.Jmodel(prior) = 0.0000e+00
  Jhat.Jobservations(prior) = 2.4655e+02
  Jhat(prior) = 2.4655e+02

  Jhat.Jmodel(xopts) = 7.6481e+00
  Jhat.Jobservations(xopts) = 1.4586e+01
  Jhat(xopts) = 2.2234e+01

  errornorm(truth_ics, prior_ics)/norm(truth_ics) = 6.350e-01
  errornorm(truth_ics, xopts_ics)/norm(truth_ics) = 6.866e-02
  errornorm(truth_end, prior_end)/norm(truth_end) = 2.594e-01
  errornorm(truth_end, xopts_end)/norm(truth_end) = 1.335e-02

Next the saddle point formulation is solved using the block diagonal preconditioner :math:`P_{D}` with :math:`\mathbf{\tilde{L}}=\mathbf{L}`.
The preconditioner type is changed to the ``WC4DVarSaddlePointPC``, which sets up as KSP solver for the saddle point system, which is configured using the ``"wcsaddle"`` options dictionary.

  1. We use GMRES, and the ``"fieldsplit"`` preconditioner, which is PETSc's way of implementing block preconditioning.
  We select the diagonal approximation of the Schur factorisation :math:`P_{U}` with ``"pc_fieldsplit_schur_fact_type": "upper"``, but :math:`P_{D}`, :math:`P_{L}`, :math:`P_{F}` can be selected by changing ``"upper"`` to one of ``<"diag","lower","full">``.
  The Schur factorisation requires a 2x2 block matrix, so we need to tell PETSc to group the first two block-rows together with the ``"pc_fieldsplit_n_fields"`` options.

  2. The top left block is configured with ``"fieldsplit_0"``, and is block diagonal with :math:`\mathbf{D}` and :math:`mathbf{R}`, so we use another fieldsplit preconditioner but this time of ``"additive"`` type to split into the two blocks, which are solved like the :math:`\mathbf{D}`` matrix in the ``WC4DVarSchurPC`` above.

  3. The Schur complement is configured with ``fieldsplit_1"``, and we use a single application of the ``WC4DVarSchurPC`` with exactly the same configuration as before.

::

  saddle_parameters = {
      'tao_monitor': None,
      'tao_max_it': 20,
      'tao_converged_reason': None,
      'tao_gttol': 1e-3,
      'tao_type': 'nls',
      'tao_nls': {
          'ksp_monitor': None,
          'ksp_type': 'preonly',
          'pc_type': 'python',
          'pc_python_type': 'firedrake.WC4DVarSaddlePointPC',
          'wcsaddle': {
              'ksp_converged_rate': None,
              'ksp_monitor': None,
              'ksp_converged_maxits': None,
              'ksp_max_it': 20,
              'ksp_min_it': 6,
              'ksp_rtol': 1e-4,
              'ksp_type': 'gmres',
              'pc_type': 'fieldsplit',
              'pc_fieldsplit_type': 'schur',
              'pc_fieldsplit_schur_fact_type': 'upper',
              'pc_fieldsplit_0_fields': '0,1',
              'pc_fieldsplit_1_fields': '2',
              'fieldsplit_0': {
                  'ksp_type': 'preonly',
                  'pc_type': 'fieldsplit',
                  'pc_fieldsplit_type': 'additive',
                  'fieldsplit': {
                      'ksp_type': 'preonly',
                      'pc_type': 'python',
                      'pc_python_type': 'firedrake.EnsembleBJacobiPC',
                      'sub_pc_type': 'python',
                      'sub_pc_python_type': 'firedrake.CorrelationOperatorPC',
                  },
              },
              'fieldsplit_1': {
                  'ksp_type': 'preonly',
                  'pc_type': 'python',
                  'pc_python_type': 'firedrake.WC4DVarSchurPC',
                  'wcschur_l': {
                      'ksp_convergence_test': 'skip',
                      'ksp_converged_maxits': None,
                      'ksp_type': 'richardson',
                      'ksp_max_it': ns+1,
                  },
                  'wcschur_d': {
                      'ksp_type': 'preonly',
                      'pc_type': 'python',
                      'pc_python_type': 'firedrake.EnsembleBJacobiPC',
                      'sub_ksp_type': 'preonly',
                      'sub_pc_type': 'python',
                      'sub_pc_python_type': 'firedrake.CorrelationOperatorPC',
                  },
              },
          },
      },
  }

  Jhat(prior)
  tao = TAOSolver(MinimizationProblem(Jhat),
                  parameters=saddle_parameters)
  xopt = tao.solve()

The output of the ``TAOSolver.solve`` call is below.
A very similar drop in the objective function and gradient has been achieved, but in a single Gauss-Newton iteration and 8 KSP iterations rather than 11 KSP iterations over 2 Gauss-Newton iterations as with the primal formulation.
Because the saddle point and primal formulations solve different linear systems, a sweep of different stopping tolerances would be needed for a proper comparison, but this at least shows that the saddle point formulation requires a competitive amount of work to the primal formulation.

::

  0 TAO,  Function value: 246.549,  Residual: 4490.03
    Residual norms for tao_nls_ solve.
    0 KSP none resid norm 3.491466898106e+02
      Residual norms for tao_nls_wcsaddle_ solve.
      0 KSP preconditioned resid norm 1.867638226923e+02
      1 KSP preconditioned resid norm 9.170279277945e+01
      2 KSP preconditioned resid norm 6.175042293864e+01
      3 KSP preconditioned resid norm 5.899837006966e+01
      4 KSP preconditioned resid norm 5.438227952724e+01
      5 KSP preconditioned resid norm 2.380950200421e+01
      6 KSP preconditioned resid norm 8.813614188604e+00
      7 KSP preconditioned resid norm 3.500210795382e-01
      8 KSP preconditioned resid norm 1.254258527790e-02
    Linear tao_nls_wcsaddle_ solve converged due to CONVERGED_RTOL iterations 8 res rate 0.368248 R^2 0.754983
    1 KSP none resid norm 1.920757889527e-02
  1 TAO,  Function value: 22.2342,  Residual: 0.244281
  TAO solve converged due to CONVERGED_GTTOL iterations 1

Comparing the breakdowns of the objective functional and initial and final errors, we can see that the saddle point formulation reached essentially the same optimum as the primal formulation:

::

  Jhat.Jmodel(truth) = 4.0181e+02
  Jhat.Jobservations(truth) = 1.3641e+01
  Jhat(truth) = 4.1545e+02

  Jhat.Jmodel(prior) = 0.0000e+00
  Jhat.Jobservations(prior) = 2.4655e+02
  Jhat(prior) = 2.4655e+02

  Jhat.Jmodel(xopt) = 7.6481e+00
  Jhat.Jobservations(xopt) = 1.4586e+01
  Jhat(xopt) = 2.2234e+01

  errornorm(truth_ic, prior_ic)/norm(truth_ic) = 6.350e-01
  errornorm(truth_ic, xopts_ic)/norm(truth_ic) = 6.866e-02
  errornorm(truth_end, prior_end)/norm(truth_end) = 2.594e-01
  errornorm(truth_end, xopts_end)/norm(truth_end) = 1.335e-02

Solving the saddle point formulation using the block diagonal preconditioner :math:`P_{D}` only requires changing the type of Schur factorisation that PETSc applies, from ``'upper'`` to ``'diag'`` or ``'lower'``. The initial guess is also reset to the ``prior`` before setting up the new solver.

::

  saddle_parameters = {
      ...
      'tao_type': 'nls',
      'tao_nls': {
          ...
          'pc_python_type': 'firedrake.WC4DVarSaddlePointPC',
          'wcsaddle': {
              ...
              'ksp_type': 'gmres',
              'pc_type': 'fieldsplit',
              'pc_fieldsplit_type': 'schur',
              'pc_fieldsplit_schur_fact_type': 'diag',
              'pc_fieldsplit_0_fields': '0,1',
              'pc_fieldsplit_1_fields': '2',
              ...
              },
          },
      },
  }

  Jhat(prior)
  tao = TAOSolver(MinimizationProblem(Jhat),
                  parameters=saddle_parameters)
  xopt = tao.solve()

The output of the ``TAOSolver.solve`` call is below.
After 8 Gauss-Newton iterations and 52 linear iterations, the solver reaches as similar value of the objective functional, but the gradient remains much higher than the previous solvers.

After the first Gauss-Newton iteration, the saddle point system solve tends to have a very steep residual drop in the first iteration, followed by a much slower drop at later iterations, often only dropping appreciably at every other iteration.
This alternating pattern is not uncommon for saddle point systems with :math:`P_{D}` because both the matrix and preconditioner are symmetric, so many of the eigenvalues of the preconditioned matrix appear as complex conjugate pairs.
The hand-wavy explanation is that if each of the two eigenvalues in a pair contribute very similarly to the error, the overall residual doesn't drop until the Krylov space is large enough to account for both of them - although proper explanations for GMRES convergence is much more complicated and far beyond the scope of this demo.

::

  0 TAO,  Function value: 246.549,  Residual: 4490.03
    Residual norms for tao_nls_ solve.
    0 KSP none resid norm 3.491466898106e+02
      Residual norms for tao_nls_wcsaddle_ solve.
      0 KSP preconditioned resid norm 1.867638226923e+02
      1 KSP preconditioned resid norm 1.622514551474e+02
      2 KSP preconditioned resid norm 8.689559908553e+01
      3 KSP preconditioned resid norm 4.589487692165e+01
      4 KSP preconditioned resid norm 3.921194610886e+01
      5 KSP preconditioned resid norm 2.523504918296e+00
      6 KSP preconditioned resid norm 2.523254639822e+00
      7 KSP preconditioned resid norm 1.772143699375e+00
      8 KSP preconditioned resid norm 1.772139483993e+00
      9 KSP preconditioned resid norm 1.375401695966e-01
    Linear tao_nls_wcsaddle_ solve converged due to CONVERGED_RTOL iterations 9 res rate 0.461891 R^2 0.922734
    1 KSP none resid norm 5.120422007750e+01
  1 TAO,  Function value: 22.2751,  Residual: 661.727
    Residual norms for tao_nls_ solve.
    0 KSP none resid norm 3.491466898106e+02
      Residual norms for tao_nls_wcsaddle_ solve.
      0 KSP preconditioned resid norm 5.666346548957e+01
      1 KSP preconditioned resid norm 1.375402020536e-01
      2 KSP preconditioned resid norm 1.375401582305e-01
      3 KSP preconditioned resid norm 1.280210104103e-01
      4 KSP preconditioned resid norm 1.280210075402e-01
      5 KSP preconditioned resid norm 1.056872813905e-01
      6 KSP preconditioned resid norm 1.056872793722e-01
      7 KSP preconditioned resid norm 1.579870963188e-02
    Linear tao_nls_wcsaddle_ solve converged due to CONVERGED_RTOL iterations 7 res rate 0.493038 R^2 0.527536
    1 KSP none resid norm 3.179831263705e+01
  2 TAO,  Function value: 22.2416,  Residual: 410.896
    Residual norms for tao_nls_ solve.
    0 KSP none resid norm 3.491466898106e+02
      Residual norms for tao_nls_wcsaddle_ solve.
      0 KSP preconditioned resid norm 6.090045650298e+01
      1 KSP preconditioned resid norm 1.579870998764e-02
      2 KSP preconditioned resid norm 1.579870983202e-02
      3 KSP preconditioned resid norm 1.331711232674e-02
      4 KSP preconditioned resid norm 1.331711232671e-02
      5 KSP preconditioned resid norm 1.177020202294e-02
      6 KSP preconditioned resid norm 1.177020202179e-02
    Linear tao_nls_wcsaddle_ solve converged due to CONVERGED_RTOL iterations 6 res rate 0.38932 R^2 0.410217
    1 KSP none resid norm 2.962544749103e+01
  3 TAO,  Function value: 22.2372,  Residual: 270.5
    Residual norms for tao_nls_ solve.
    0 KSP none resid norm 3.491466898106e+02
      Residual norms for tao_nls_wcsaddle_ solve.
      0 KSP preconditioned resid norm 6.393291237136e+01
      1 KSP preconditioned resid norm 4.376723164178e-02
      2 KSP preconditioned resid norm 4.376722734766e-02
      3 KSP preconditioned resid norm 5.440883341734e-03
      4 KSP preconditioned resid norm 5.440883341710e-03
      5 KSP preconditioned resid norm 5.218586839293e-03
      6 KSP preconditioned resid norm 5.218586838995e-03
    Linear tao_nls_wcsaddle_ solve converged due to CONVERGED_RTOL iterations 6 res rate 0.290852 R^2 0.604412
    1 KSP none resid norm 1.935613884378e+01
  4 TAO,  Function value: 22.2367,  Residual: 250.105
    Residual norms for tao_nls_ solve.
    0 KSP none resid norm 3.491466898106e+02
      Residual norms for tao_nls_wcsaddle_ solve.
      0 KSP preconditioned resid norm 6.441191310403e+01
      1 KSP preconditioned resid norm 5.218594264479e-03
      2 KSP preconditioned resid norm 5.218594257190e-03
      3 KSP preconditioned resid norm 5.159352110478e-03
      4 KSP preconditioned resid norm 5.159352110453e-03
      5 KSP preconditioned resid norm 3.034323324138e-03
      6 KSP preconditioned resid norm 3.034323324065e-03
    Linear tao_nls_wcsaddle_ solve converged due to CONVERGED_RTOL iterations 6 res rate 0.330678 R^2 0.431481
    1 KSP none resid norm 7.092026444460e+00
  5 TAO,  Function value: 22.2345,  Residual: 91.6169
    Residual norms for tao_nls_ solve.
    0 KSP none resid norm 3.491466898106e+02
      Residual norms for tao_nls_wcsaddle_ solve.
      0 KSP preconditioned resid norm 6.836103402260e+01
      1 KSP preconditioned resid norm 3.034332809473e-03
      2 KSP preconditioned resid norm 3.034332807607e-03
      3 KSP preconditioned resid norm 1.965080888225e-03
      4 KSP preconditioned resid norm 1.965080888223e-03
      5 KSP preconditioned resid norm 1.466387171929e-03
      6 KSP preconditioned resid norm 1.466387171927e-03
    Linear tao_nls_wcsaddle_ solve converged due to CONVERGED_RTOL iterations 6 res rate 0.295461 R^2 0.445541
    1 KSP none resid norm 6.650120853902e+00
  6 TAO,  Function value: 22.2343,  Residual: 63.0724
    Residual norms for tao_nls_ solve.
    0 KSP none resid norm 3.491466898106e+02
      Residual norms for tao_nls_wcsaddle_ solve.
      0 KSP preconditioned resid norm 6.911909967112e+01
      1 KSP preconditioned resid norm 1.072662173026e-02
      2 KSP preconditioned resid norm 1.072662164851e-02
      3 KSP preconditioned resid norm 3.682151620914e-03
      4 KSP preconditioned resid norm 3.682151620771e-03
      5 KSP preconditioned resid norm 8.845860757639e-04
      6 KSP preconditioned resid norm 8.845860757639e-04
    Linear tao_nls_wcsaddle_ solve converged due to CONVERGED_RTOL iterations 6 res rate 0.240864 R^2 0.621579
    1 KSP none resid norm 4.613966875512e+00
  7 TAO,  Function value: 22.2343,  Residual: 59.6018
    Residual norms for tao_nls_ solve.
    0 KSP none resid norm 3.491466898106e+02
      Residual norms for tao_nls_wcsaddle_ solve.
      0 KSP preconditioned resid norm 6.921206161094e+01
      1 KSP preconditioned resid norm 8.845903066387e-04
      2 KSP preconditioned resid norm 8.845903066304e-04
      3 KSP preconditioned resid norm 8.579657722316e-04
      4 KSP preconditioned resid norm 8.579657721271e-04
      5 KSP preconditioned resid norm 8.148465444716e-04
      6 KSP preconditioned resid norm 8.148465441703e-04
    Linear tao_nls_wcsaddle_ solve converged due to CONVERGED_RTOL iterations 6 res rate 0.294347 R^2 0.382276
    1 KSP none resid norm 4.170861012370e+00
  8 TAO,  Function value: 22.2342,  Residual: 30.98
  TAO solve converged due to CONVERGED_GTTOL iterations 8

Despite the larger gradient for the :math:`x_{j}` from this solver configuration, the misfits errors in the initial and final conditions are again almost exactly the same as the previous solvers.

::

  Jhat.Jmodel(truth) = 4.0181e+02
  Jhat.Jobservations(truth) = 1.3641e+01
  Jhat(truth) = 4.1545e+02

  Jhat.Jmodel(prior) = 0.0000e+00
  Jhat.Jobservations(prior) = 2.4655e+02
  Jhat(prior) = 2.4655e+02

  Jhat.Jmodel(xopt) = 7.6459e+00
  Jhat.Jobservations(xopt) = 1.4588e+01
  Jhat(xopt) = 2.2234e+01

  errornorm(truth_ic, prior_ic)/norm(truth_ic) = 6.350e-01
  errornorm(truth_ic, xopts_ic)/norm(truth_ic) = 6.869e-02
  errornorm(truth_end, prior_end)/norm(truth_end) = 2.594e-01
  errornorm(truth_end, xopts_end)/norm(truth_end) = 1.336e-02
