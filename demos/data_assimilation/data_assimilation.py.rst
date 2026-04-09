Weak constraint 4DVar data assimilation
=======================================


.. rst-class:: emphasis

    This tutorial was contributed by `Josh Hope-Collins <mailto:joshua.hope-collins13@imperial.ac.uk>`__

Data assimilation is the process of using real world observations to improve the accuracy of a simulation, and is commonly used in weather and climate modelling.
A particular variant called "weak constraint" 4DVar (WC4DVar) allows the use of parallel-in-time solvers.

In data assimilation problems we want to find an approximation :math:`x_{j}` of the true values :math:`x^{t}_{j}` of a timeseries, :math:`0<j<N`. and we have the following available to us:

  #. Observation operators :math:`\mathcal{H}_{j}`, and incomplete and imperfect (noisy) observations of :math:`x^{t}_{j}` at each time point :math:`y_j=\mathcal{H}(x^{t}_{j}) + r_{j}`, where the noise is :math:`r_{j}\sim\mathcal{N}(0,R_{j})` with correlation matrix :math:`R_{j}`.

  #. An imperfect PDE model :math:`\mathcal{M}_{j}` that propagates from one value to the next :math:`x^{t}_{j}=\mathcal{M}_{j}(x^{t}_{j-1})+q_{j}`, where the noise is :math:`q_{j}\sim\mathcal{N}(0,Q_{j})` with correlation matrix :math:`Q_{j}`.

  #. A prior estimate of the initial condition, called the "background", :math:`x_{b}=x^{t}_{0}+b`, where the noise is :math:`b\sim\mathcal{N}(0,B)` with correlation matrix :math:`B`.

We want to find a timeseries that minimises the misfits with the background, the observations, and the propagator, which we formulate as finding the minimiser :math:`\mathbf{x}=(x_{0}, x_{1}, \dots, x_{N})` of the following objective functional:

.. math::

   \min_{\mathbf{x}} \mathcal{J}(\mathbf{x})
      = \|x_{0} - x_{b}\|_{B^{-1}}^{2}
      + \sum^{N_{w}}_{j=0}\|\mathcal{H}_{j}(x_{j}) - y_{j}\|_{R_{j}^{-1}}^{2}
      + \sum^{N_{w}}_{j=1}\|x_{j} - \mathcal{M}_{j}(x_{j-1})\|_{Q_{j}^{-1}}^{2}

The "weak constraint" is that we have allowed our PDE model :math:`\mathcal{M}` to be imperfect, rather than requiring the entire timeseries to be a perfect trajectory of :math:`\mathcal{M}` as is the standard in "strong constraint" 4DVar.
This accounts for numerical and modelling errors in the PDE model, and also enables time-parallelism.
Each model misfit term :math:`x_{j}-\mathcal{M}_{j}(x_{j-1})` only requires the two neighbouring values of :math:`x_{j}`, so can be evaluated independently of model misfit terms earlier or later in the timeseries.

The minimisation is solved using a Gauss-Newton method, where at each iteration :math:`k` the increment :math:`\delta x^{k} = x^{k+1} - x^{k}` is calculated by minimising the linearised objective functional :math:`J(\mathbf{\delta x})` (often called the "incremental formulation" in 4DVar literature).

.. math::

   \min_{\mathbf{\delta x}} J(\mathbf{\delta x})
      = \|\delta x_{0} - b_{0}\|_{B^{-1}}^{2}
      + \sum^{N_{w}}_{j=0}\|d_{i} - H_{i}\delta x_{i}\|_{R_{j}^{-1}}^{2}
      + \sum^{N_{w}}_{j=1}\|\delta x_{j} - M_{j}\delta x_{j-1} - c_{i}\|_{Q_{j}^{-1}}^{2}

where :math:`H` and :math:`M` are linearisations of :math:`\mathcal{H}` and :math:`\mathcal{M}` respectively, and the "misfits" are defined as

.. math::

   b_{0} = x_{b} - x^{k}_{0},
   \quad
   d_{i} = y_{i} - \mathcal{H}_{i}(x^{k}_{i}),
   \quad
   c_{i} = \mathcal{M}_{i}(x^{k}_{i-1}) - x^{k}_{i}.

This is a linear least squares problem which can be written in terms of the Hessian matrix :math:`\mathbf{S}` of :math:`J`.

.. math::

   \mathbf{S}\delta\mathbf{x} =
   (\mathbf{L}^{T}\mathbf{D}^{-1}\mathbf{L} + \mathbf{H}^{T}\mathbf{R}^{-1}\mathbf{H})\mathbf{\delta x}
   = \mathbf{L}^{T}\mathbf{R}^{-1}\mathbf{b} + \mathbf{H}^{T}\mathbf{R}^{-1}\mathbf{d}

with :math:`\mathbf{b}=(b_{0}, c_{1}, c_{2}, \dots, c_{N})^{T}` and :math:`\mathbf{d}=(d_{0}, d_{1}, d_{2}, \dots, d_{N})^{T}`.
The matrices in the Hessian are constructed from the linearised propagator and observation operators and the covariance operators:

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
The model integration matrix :math:`\textbf{L}` is block lower bidiagonal so its action can be applied parallel-in-time, with a communication overhead for time-halos that is independent of the number of timesteps.

Assuming that :math:`\mathbf{S}` is dominated by the first term, which describes the model integration, we can construct a preconditioner :math:`\mathbf{\tilde{S}}`  using an approximate model integration operator :math:`\mathbf{\tilde{L}}\approx\mathbf{L}`

.. math::

   \mathbf{\tilde{S}} = \mathbf{\tilde{L}}^{T}\mathbf{\tilde{D}}^{-1}\mathbf{\tilde{L}} \approx \mathbf{S}

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
Clearly, the inverse :math:`\mathbf{L}^{-1}` cannot be applied parallel-in-time because each timestep depends on all previous steps, which motivates the need for cheaper approximations :math:`\mathbf{\tilde{L}}`.
For example, we could build :math:`\mathbf{\tilde{L}}` using an approximation :math:`\tilde{M}\approx M`.
A very simple approximation sometimes used in WC4DVar is :math:`\tilde{M}=I`, in which case :math:`\mathbf{\tilde{L}}^{-1}` must still be applied sequentially, but can be done so very cheaply because :math:`M_{i,j}=I\;\forall i,j`.

Finding :math:`\mathbf{\delta x}` by solving the Hessian :math:`\mathbf{S}` is referred to as the "primal" formulation.
An alternative is the "saddle point" formulation, which finds :math:`\mathbf{\delta x}` at each Gauss-Newton iteration by solving the saddle point matrix :math:`\mathbf{A}` for the KKT conditions:

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

where :math:`\delta\mathbf{\eta}` and :math:`\delta\mathbf{\lambda}` are Lagrange multipliers for the model and observation misfits respectively.
The action of :math:`\mathbf{A}` is also parallel-in-time.

There is a wide range of research on preconditioning saddle point systems, often based on partial and/or approximate Schur LDU factorisations.
Notice that the primal Hessian :math:`\mathbf{S}` is the Schur complement of the saddle point matrix :math:`\mathbf{A}` after eliminating :math:`\delta\mathbf{\eta}` and :math:`\delta\mathbf{\lambda}`.
Using the same approximation :math:`\mathbf{\tilde{S}}` as above we can construct several block preconditioners:

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


In this demo we will solve the WC4DVar system for the advection-diffusion equation using the saddle point formulation preconditioned with :math:`\mathbf{P_{U}}`.
We will go through how to set up and solve the WC4DVar system in Firedrake, with the following steps:

  #. Define the finite element model for the advection-diffusion equation :math:`\mathcal{M}`.
  #. Define the observation operator :math:`\mathcal{H}`.
  #. Define the error covariance operators :math:`B`, :math:`R`, and :math:`Q`.
  #. Generate synthetic "ground-truth" observation data for :math:`y_{j}`.
  #. Create a ``ReducedFunctional`` for :math:`\mathcal{J}`.
  #. Specify a solver configuration and calculate an optimised :math:`\mathbf{x}`.

First we import Firedrake, including the components from the adjoint module.
As we will be generating some random noise, we set the random number generator seed to a fixed value..

::

  import numpy as np
  from firedrake import *
  from firedrake.adjoint import *
  np.random.seed(13)

We use the advection-diffusion equation in one spatial dimension :math:`z`, with a spatially varying advection velocity :math:`c(z)`, a time-dependent forcing term :math:`g(t)`, and periodic boundary conditions.

.. math::

   \partial_{t}u + \vec{c}(z)\cdot\nabla u + \nu\nabla^{2}u = g(t) &

   t \in [0, T], \quad z \in \Omega = [0, 1) &

   u(0, t) = u(1, t) &

   c(z) = 1 + \overline{c}\cos(2\pi z)

The reference state :math:`\hat{u}` that we will use to generate the "ground-truth" trajectory :math:`x^{t}` is just a simple sinusoid.

.. math::

   \hat{u} = \overline{u}\sin(2\pi z)

For the time integration we use the implicit midpoint rule with the semi-discrete weak form:

.. math::

   \int_{\Omega}\left(\partial_{t}u\right)v\mathrm{d}x
   + \int_{\Omega}\left(\vec{c}\cdot\nabla u_{h} \right)v\mathrm{d}x
   + \int_{\Omega}\nu\nabla u_{h}\cdot\nabla v\mathrm{d}x
   - \int_{\Omega}gv\mathrm{d}x
   = 0,
   \quad \forall v \in V

where :math:`V` is the function space for the solution.

First we create the mesh and function spaces.
To enable time-parallelism we use Firedrake's :class:`~firedrake.ensemble.ensemble.Ensemble`, which splits ``COMM_WORLD`` into several ensemble members, with spatial parallelism within each ensemble member and time-parallelism between members.
Here we specify just one MPI rank per ensemble member, and the number of ensemble members automatically adjusts to use all available ranks.
The communicator ``ensemble.comm`` is used for the spatial parallelism, so is the one we use to construct the mesh.
We create the CG1 function space for :math:`V`, and the space of real numbers to hold the time :math:`t`.

::

  ensemble = Ensemble(COMM_WORLD, 1)
  ensemble_rank = ensemble.ensemble_rank
  ensemble_size = ensemble.ensemble_size

  mesh = PeriodicUnitSquareMesh(20, 20, direction="x", comm=ensemble.comm)

  V = FunctionSpace(mesh, "CG", 1)
  Vv = VectorFunctionSpace(mesh, "CG", 1)
  Vr = FunctionSpace(mesh, "R", 0)

The control :math:`\mathbf{x}` is a timeseries distributed in time over the ``Ensemble``, with each timestep :math:`x_{j}` being a Firedrake ``Function``.
For this we use an :class:`~firedrake.ensemble.ensemble_functionspace.EnsembleFunctionSpace` which represents a mixed function space with each component living on a particular ensemble member.
To initialise the ``EnsembleFunctionSpace`` we just need the ``Ensemble`` and a list of ``FunctionSpace`` for the local components.
We split the number of observation stages ``N`` equally across the ensemble members, and include an extra component on the first member for the initial condition :math:`x_{0}`.

::

  N = 8

  nlocal_stages = N//ensemble_size
  nlocal_spaces = nlocal_stages + int(ensemble_rank == 0)

  W = EnsembleFunctionSpace([V for _ in range(nlocal_spaces)], ensemble)

**Define the propagator.**

We construct the propagator :math:`\mathcal{M}` for the advection-diffusion scheme, using `Irksome <https://www.firedrakeproject.org/Irksome>`_ to provide the time integrator.
The forcing term :math:`g(t)` is rather involved, but just ensures that there is some non-trivial variation in the solution and prevents it decaying to zero over long time periods due to the diffusion.
The observations are taken at intervals of :math:`T_{\textrm{stage}}=n_{t}\Delta t`, where :math:`n_{t}` is the number of timesteps between each observation.

::

  from irksome import Dt, TimeStepper, GaussLegendre

  Tstage = 1e-1
  nt = 3

  dt = Function(Vr).assign(Tstage/nt)
  t = Function(Vr).zero()
  x_, y_ = SpatialCoordinate(mesh)

  cbar = Constant(0.2)
  c = Function(Vv).project(as_vector([1 + cbar*cos(2*pi*x_), 0.0]))

  reynolds = 100
  nu = Constant(1/reynolds)

  u = Function(V)
  v = TestFunction(V)

  ubar = Constant(0.3)
  reference_ic = Function(V).project(ubar*sin(2*pi*x_))

  g = (
      ubar*cos(2*pi*x_)*(
          - sin(2*pi*(y_ + 0.1*sin(2*pi*t)))
          + ubar*cos(2*pi*t + 1)*sin(2*pi*(3*x_ - 2*t))
      )
  )

  F = (
      inner(Dt(u), v)*dx
      + inner(dot(c, grad(u)), v)*dx
      + inner(nu*grad(u), grad(v))*dx
      - inner(g, v)*dx(degree=4)
  )

  solver_parameters = {
      "snes_type": "ksponly",
      "ksp_type": "preonly",
      "pc_type": "lu",
      "ksp_reuse_preconditioner": None,
  }

  tableau = GaussLegendre(1)

  stepper = TimeStepper(
      F, tableau, t, dt, u,
      solver_parameters=solver_parameters,
      options_prefix="irk")

For convenience we make a Python function for the propagator :math:`\mathcal{M}(x)`.

::

  def M(x):
      stepper.u0.assign(x)
      for _ in range(nt):
          stepper.stages.zero()
          stepper.advance()
          t.assign(t + dt)
      return stepper.u0.copy(deepcopy=True)

  # **Define the observation operator.**
  #
  # Our observations will be point evaluations at a set of random locations in the domain, which are defined using a :class:`~firedrake.mesh.VertexOnlyMesh`.
  # The observation operator :math:`\mathcal{H}` is then simply interpolating onto this mesh.
  #
  # ::
  line = UnitIntervalMesh(10, comm=ensemble.comm)
  x, = SpatialCoordinate(line)
  lfs = VectorFunctionSpace(line, "CG", 1, dim=2)
  new_coords = Function(lfs).interpolate(as_vector([x, x]))
  new_line = Mesh(new_coords)
  U = FunctionSpace(new_line, "CG", 2)

  R_line = FunctionSpace(new_line, "R", 0)

  def H(x) -> "Function":
      a = inner(TestFunction(R_line), TrialFunction(R_line)) * dx(domain=new_line)
      b = inner(TestFunction(R_line), interpolate(x, U)) * dx(domain=new_line)
      u = Function(R_line)
      solve(a == b, u)
      return u

**Define the error covariance operators.**

We need to do three things with correlation operators: apply the action :math:`B`, the inverse :math:`B^{-1}`, and and generate physically relevant noise.
If :math:`w\sim\mathcal{N}(0,I)` is a vector of white noise then :math:`B^{1/2}w=v\sim\mathcal{N}(0,B)`, i.e. :math:`B^{1/2}` transforms uncorrelated noise to correlated noise with covariance :math:`B`.

Firedrake provides an implementation of diffusion-based autoregressive covariance operators with the :class:`~firedrake.adjoint.covariance_operator.AutoregressiveCovariance` class.
The action :math:`Bx=y` of an `m`-th order autoregressive covariance operator is equivalent to :math:`m` Backward Euler steps of a diffusion equation with initial condition :math:`x`, where the diffusion coefficient depends on the correlation lengthscale.
This makes this type of covariance operator well suited to finite element models.
If :math:`m` is even then an efficient square root :math:`B^{1/2}` can be calculated by taking just :math:`m/2` Backward Euler steps.

We create the background and model error covariance operators with specified lengthscales :math:`L` and standard deviations :math:`\sigma`.
The variance of the model error is made proportional to the length of the observation stage :math:`T_{\textrm{stage}}`.

::

  sigma_b = sqrt(1e-2)
  B = AutoregressiveCovariance(V, L=0.2, sigma=sigma_b, m=2, seed=2)

  sigma_q = sqrt(1e-3*Tstage)
  Q = AutoregressiveCovariance(V, L=0.05, sigma=sigma_q, m=2, seed=17)

The observations are treated as uncorrelated, i.e. a diagonal covariance operator, which is created by setting :math:`m=0`.

::

  sigma_r = sqrt(1e-3)
  R = AutoregressiveCovariance(R_line, L=0, sigma=sigma_r, m=0, seed=18)

Firedrake provides an abstract base class :class:`~firedrake.adjoint.covariance_operator.CovarianceOperatorBase` for implementing new covariance operators. 

**Generate observational data.**

We can use a known reference initial condition :math:`\hat{x}` to generate synthetic "ground-truth" observations :math:`y_{i}`.
We do this by adding noise consistently with the original definition of the problem, i.e. we add noise :math:`b_{j}\sim\mathcal{N}(0,B)` at the initial condition, then at each observation time we add noise :math:`q_{j}\sim\mathcal{N}(0,Q)` to the solution and add noise :math:`r_{j}\sim\mathcal{N}(0,R)` to the observations.
This process is detailed below:

  1. :math:`x_{b}     \leftarrow \hat{x} + b_{b}`
  2. :math:`x^{t}_{0} \leftarrow \hat{x} + b_{0}`
  3. :math:`y_{0}     \leftarrow \mathcal{H}(x^{t}_{0}) + r_{0}`
  4. **for** :math:`j=1` **to** :math:`j=N` **do**

    #. :math:`x^{t}_{j} \leftarrow \mathcal{M}(x^{t}_{j-1}) + q_{j}`
    #. :math:`y_{j}     \leftarrow \mathcal{H}(x^{t}_{j}) + r_{j}`

  5. **end for**

See that we generate both the background :math:`x_{b}` and the "truth" initial condition :math:`x^{t}_{0}` by perturbing :math:`\hat{x}`, which means that both states will contain noise (rather than one or the other being completely deterministic).

The code below uses this process to generate synthetic observation data.
Because our timeseries is distributed over the ``Ensemble``, each observation :math:`y_{j}` needs to live on the right ensemble member.
To do this we use the :meth:`Ensemble.sequential <firedrake.ensemble.ensemble.Ensemble.sequential>` context manager, which runs the code within the context on each ensemble member in turn.
Any kwarg passed to ``Ensemble.sequential`` is made available in the ``ctx`` object, and is sent forward to the next ensemble member once the local code block is complete.
After running the local part of the timeseries on each ensemble member, this allows us to pass forward the state ``xt`` and the time ``t`` to the next member.

::

  xb = Function(V).assign(reference_ic + B.sample())
  xt = Function(V).assign(reference_ic + B.sample())

  # send ground-truth initial condition to all ranks.
  truth_ic = ensemble.bcast(xt, root=0).copy(deepcopy=True)

  if ensemble_rank == 0:
      y = [Function(U).assign(H(xt) + R.sample())]
  else:
      y = []

  t.assign(0)
  with ensemble.sequential(state=xt, t=t) as ctx:
      t.assign(ctx.t)
      xt.assign(ctx.state)

      for _ in range(nlocal_stages):
          xt.assign(M(xt) + Q.sample())
          y.append(Function(U).assign(H(xt) + R.sample()))

      ctx.state.assign(xt)

  # send ground-truth end condition to all ranks.
  truth_end = ensemble.bcast(xt.copy(deepcopy=True), root=ensemble_size-1)

print("finishing now")
sys.exit(0)

Now that we have the "ground-truth" observations, we can create a function to generate callbacks for the error vs the observation at each timestep ``i``.

::

  def observation_error(i):
      return lambda x: Function(U).assign(H(x) - y[i])

**Define the reduced functional.**

Now we have all the pieces ready to start assembling the 4DVar system.
:func:`pyadjoint.continue_annotation` tells Pyadjoint to start recording any code that is executed from now on.
The :class:`~firedrake.adjoint.fourdvar_reduced_functional.WC4DVarReducedFunctional` class will manage recording, constructing, and solving the 4DVar system.
To initialise it, it needs an :class:`~firedrake.ensemble.ensemble_function.EnsembleFunction` as a :class:`pyadjoint.Control`, and the components to evaluate the functional at the initial condition, i.e. the background state and covariance for :math:`\|x_{0}-x_{b}\|_{B^{-1}}^{2}`, and the observation error and covariance for :math:`\|\mathcal{H}_{0}(x_{0})-y_{0}\|_{R_{0}^{-1}}^{2}`.


::

  continue_annotation()

  control = EnsembleFunction(W)

  Jhat = WC4DVarReducedFunctional(
      Control(control),
      background=xb,
      background_covariance=B,
      observation_covariance=R,
      observation_error=observation_error(0),
      gauss_newton=True)

All Firedrake operations are "taped" by pyadjoint, so all we need to do to initialise the stages is to run :math:`\mathcal{M}` and :math:`\mathcal{H}` within the ``WC4DVarReducedFunctional.recording_stages`` context manager below.
For each ``stage``, we integrate forward from ``stage.control`` (i.e. :math:`x_{j-1}`), and then set the observation by providing the state (i.e. :math:`x_{j}=\mathcal{M}_{j}(x_{j-1})`) error operator, and the covariances.

::

  t.assign(0.0)
  with Jhat.recording_stages(t=t) as stages:
      for stage, ctx in stages:
          t.assign(ctx.t)
          xn1 = M(stage.control)

          obs_error = observation_error(stage.observation_index)

          stage.set_observation(
              state=xn1,
              observation_error=obs_error,
              observation_covariance=R,
              forward_model_covariance=Q)

  pause_annotation()

To ensure that the initial guess for :math:`x_{j}` is a continuous trajectory over the entire ``Ensemble``, the ``recording_stages`` context manager wraps ``ensemble.sequential``.
The ``control`` for the first stage is set to :math:`x_{b}`, and the ``control`` for subsequent stages is set to the value of the ``state`` passed to ``set_observation`` by the previous stage.

``Jhat`` now has a record of all operations in the model, and can use this to a) re-evaluate :math:`\hat{J}(x)` with different control values, b) calculate the derivative with respect to the controls, and c) apply the action of the Hessian.

We save a copy of the initial control to compare the optimised state to.

::

  prior = control.copy()

**Configure the WC4DVar solver.**

TAO is PETSc's optimisation library and provides a range of optimisation methods.
Just like the timestepper, the TAO solver is configured using a set of options strings.
We will configure the solver to use the saddle point formulation :math:`\mathbf{A}` preconditioned by the upper triangular Schur factorisation :math:`\mathbf{P}_{U}` with the approximate Schur complement :math:`\mathbf{\tilde{S}}=\mathbf{\tilde{L}}^{T}\mathbf{D}^{-1}\mathbf{\tilde{L}}` where :math:`\mathbf{\tilde{L}}` is constructed using :math:`\tilde{M}=I`.

To make this a bit simpler, we will define a couple of options sets for components of the full solver.
The ``covariance_parameters`` below can be used to solve the matrices :math:`\mathbf{D}` and :math:`\mathbf{R}` in :math:`\mathbf{P}_{U}`, and :math:`\mathbf{D}^{-1}` in :math:`\mathbf{\tilde{S}}^{-1}`.
These matrices are block diagonal with one block per component of an ``EnsembleFunctionSpace``, so we can use the :class:`~firedrake.ensemble.ensemble_pc.EnsembleBJacobiPC`.
Just like PETSc's ``PCBJacobi`` this creates a ``sub`` KSP for each block (i.e. for each covariance operator :math:`B`, :math:`Q`, or :math:`R`).
On each block we use the :class:`~firedrake.adjoint.covariance_operator.CovariancePC` which will automatically apply the inverse or action depending on if it acts on e.g. :math:`\mathbf{D}` or :math:`\mathbf{D}^{-1}`.

::

  covariance_parameters = {
      'pc_type': 'python',
      'pc_python_type': 'firedrake.EnsembleBJacobiPC',
      'sub_pc_type': 'python',
      'sub_pc_python_type': 'firedrake.adjoint.CovariancePC',
  }

The ``schur_parameters`` specify the approximate Schur complement :math:`\mathbf{\tilde{S}}`, which is implemented with the :class:`~firedrake.preconditioners.adjoint.wc4dvar.WC4DVarSchurPC`.
This preconditioner requires options to solve :math:`\mathbf{D}^{-1}`, given in ``wcschur_d``, and to solve :math:`\mathbf{\tilde{L}}`, given in ``wcschur_l``.
For :math:`\mathbf{D}^{-1}` we can use the ``covariance_parameters``.
For :math:`\mathbf{\tilde{L}}` we use the :class:`~firedrake.preconditioners.adjoint.allatonce.AllAtOnceRFGaussSeidelPC`, which uses forward substitution so solve :math:`\mathbf{\tilde{L}}`.
This preconditioner has one option, ``pc_aaogs_type``, which can be a) ``'model'`` i.e. :math:`\tilde{M}=M` and :math:`\mathbf{\tilde{L}}=\mathbf{L}` or b) ``'identity'`` i.e. :math:`\tilde{M}=I`.

::

  schur_parameters = {
      'ksp_type': 'preonly',
      'pc_type': 'python',
      'pc_python_type': 'firedrake.WC4DVarSchurPC',
      'wcschur_l': {
          'pc_type': 'python',
          'pc_python_type': 'firedrake.AllAtOnceRFGaussSeidelPC',
          'pc_aaogs_type': 'identity',
      },
      'wcschur_d': covariance_parameters,
  }

Now we set up the full solver options in the ``'tao_parameters'`` below.

* At the top level of the dictionary, ``'tao_gttol': 1e-2`` sets the convergence tolerance for the reduction drop in the gradient norm.
  Next, we specify a Newton method using ``'tao_type': 'nls'``, which needs options for the linear solver in the ``'tao_nls'`` dictionary.

* At each Newton iteration, we use ``'ksp_type': 'preonly'`` to replace the linear solve with the :class:`~firedrake.preconditioners.adjoint.wc4dvar.WC4DVarSaddlePC` preconditioner, which solves the saddle point system :math:`\mathbf{A}` and returns the :math:`\mathbf{\delta x}` part of the solution.

* To use a Schur complement factorisation so we have to tell PETSc's `PCFieldsplit <https://petsc.org/release/manualpages/PC/PCFIELDSPLIT/>`_ how to reinterpret the :math:`3\times3` matrix :math:`\mathbf{A}` as a :math:`2\times2` matrix :math:`\mathbf{\hat{A}}`.
  This is done using the ``'pc_fieldsplit_{0,1}_fields'`` options, where the Schur complement will be formed on the ``'1_fields'`` after eliminating the ``'0_fields'``.

.. math::

   \mathbf{\hat{A}} =
   \begin{pmatrix}
      \mathbf{\hat{A}}_{00} & \mathbf{\hat{A}}_{01} \\
      \mathbf{\hat{A}}_{10} & \mathbf{0}             \\
   \end{pmatrix},
   \quad
   \mathbf{\hat{A}}_{00} =
   \begin{pmatrix}
      \mathbf{D} & \mathbf{0} \\
      \mathbf{0} & \mathbf{R} \\
   \end{pmatrix},
   \quad
   \mathbf{\hat{A}}_{01} =
   \begin{pmatrix}
      \mathbf{L} \\
      \mathbf{H} \\
   \end{pmatrix},
   \quad
   \mathbf{\hat{A}}_{10} =
   \begin{pmatrix}
      \mathbf{L}^{T} & \mathbf{H}^{T} \\
   \end{pmatrix}.

* The upper triangular :math:`\mathbf{P}_{U}` preconditioner is specified using the ``'pc_fieldsplit_schur_fact_type'`` option.

.. math::

   \mathbf{P}_{U} =
   \begin{pmatrix}
      \mathbf{\hat{A}}_{00} & \mathbf{\hat{A}}_{01} \\
      \mathbf{0}            & \mathbf{\tilde{S}}    \\
   \end{pmatrix}

* The solver options for :math:`\mathbf{\hat{A}}_{00}` are in the ``'fieldsplit_0'`` dictionary.
  This block-diagonal matrix is solved by splitting it apart using an ``'additive'`` fieldsplit, then solving :math:`\mathbf{D}` and :math:`\mathbf{R}` separately using the ``'covariance_parameters'``.

* The Schur complement is specified in the ``'fieldsplit_1'`` dictionary using the ``'schur_parameters'`` above.

::

  tao_parameters = {
      'tao_monitor': None,  #  .  .  .  .  # Print out diagnostics.
      'tao_converged_reason': None,
      'tao_gttol': 5e-2,  # .  .  .  .  .  # Gradient reduction.
      'tao_max_it': 30,
      'tao_type': 'nls',  # .  .  .  .  .  # Newton iterations
      'tao_ls_type': 'unit',  #.  .  .  .  # without linesearch.
      'tao_nls': {
          'ksp_type': 'preonly',  #  .  .  # Replace the hessian solve with the PC.
          'ksp_monitor_short': None,
          'pc_type': 'python',
          'pc_python_type': 'firedrake.WC4DVarSaddlePC',
          'wcsaddle': {
              'ksp_monitor_short': None,
              'ksp_converged_rate': None,  #  .  .  .  # Print contraction rate.
              'ksp_converged_maxits': None,
              'ksp_rtol': 1e-2,
              'ksp_min_it': 10,
              'ksp_max_it': 100,
              'ksp_gmres_restart': 100,
              'ksp_type': 'gmres',
              'pc_type': 'fieldsplit',
              'pc_fieldsplit_type': 'schur',  #  .  .  .  .  # Use a schur (LDU) factorisation,
              'pc_fieldsplit_0_fields': '0,1',  #.  .  .  .  # eliminating the first two fields,
              'pc_fieldsplit_1_fields': '2',  #  .  .  .  .  # forming the schur complement on the third,
              'pc_fieldsplit_schur_fact_type': 'upper',  #.  # and using just the DU part of the LDU.
              'fieldsplit_0': {
                  'ksp_type': 'preonly',
                  'pc_type': 'fieldsplit',  # .  .  .  .  .  # Solve the covariance
                  'pc_fieldsplit_type': 'additive',  # .  .  # matrices separately.
                  'fieldsplit_ksp_type': 'preonly',
                  'fieldsplit': covariance_parameters,
              },
              'fieldsplit_1': schur_parameters,
          },
      }
  }

Now we have a reduced functional and a set of TAO parameters we can solve the optimisation problem using Pyadjoint's :class:`~pyadjoint.TAOSolver`.

::

  tao = TAOSolver(MinimizationProblem(Jhat),
                  parameters=tao_parameters,
                  options_prefix="")
  xopts = tao.solve()

Lastly, we compare the error between the optimised solution and ground truth data with the error between the initial guess and the ground truth data, at both the initial and final times.

::

  prior_ic = ensemble.bcast(prior.subfunctions[0], root=0)
  xopts_ic = ensemble.bcast(xopts.subfunctions[0], root=0)

  prior_end = ensemble.bcast(prior.subfunctions[-1], root=ensemble_size-1)
  xopts_end = ensemble.bcast(xopts.subfunctions[-1], root=ensemble_size-1)

  PETSc.Sys.Print()

  PETSc.Sys.Print("Errors at initial timestep:")
  prior_error = errornorm(truth_ic, prior_ic)/norm(truth_ic)
  xopts_error = errornorm(truth_ic, xopts_ic)/norm(truth_ic)
  PETSc.Sys.Print(f"{prior_error = :.3e}")
  PETSc.Sys.Print(f"{xopts_error = :.3e}")
  PETSc.Sys.Print(f"Error reduction factor = {xopts_error/prior_error:.3e}")
  PETSc.Sys.Print()

  PETSc.Sys.Print("Errors at final timestep:")
  prior_error = errornorm(truth_end, prior_end)/norm(truth_end)
  xopts_error = errornorm(truth_end, xopts_end)/norm(truth_end)
  PETSc.Sys.Print(f"{prior_error = :.3e}")
  PETSc.Sys.Print(f"{xopts_error = :.3e}")
  PETSc.Sys.Print(f"Error reduction factor = {xopts_error/prior_error:.3e}")
  PETSc.Sys.Print()

The output of these print statements is shown below.
At the initial and final conditions the optimised solution matches the ground truth around 13 times and 20 times more accurately than the prior solution respectively.

::

Errors at initial timestep:
prior_error = 6.723e-01
xopts_error = 4.925e-02
Error reduction factor = 7.326e-02

Errors at final timestep:
prior_error = 8.843e-01
xopts_error = 4.333e-02
Error reduction factor = 4.900e-02

A runnable python version of this demo can be found :demo:`here<data_assimilation.py>`.
This demo can be run in parallel as long as the number of observations stages :math:`N` is divisible by the number of MPI ranks.
