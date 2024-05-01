Full-waveform inversion: automated gradient, ensemble parallelism and checkpointing
===================================================================================

*This short tutorial was prepared by `Daiane I. Dolci <mailto:d.dolci@imperial.ac.uk>`__ and Jack Betteridge*


Full-waveform inversion (FWI) consists of a local optimisation, where the goal is to minimise
the misfit between observed and predicted seismogram data. The misfit is quantified by a functional,
which in general is a summation of the cost functions for multiple sources:

.. math::

       J = \sum_{s=1}^{N_s} J_s(u, u^{obs}),

where :math:`N_s` is the number of sources, and :math:`J_s(u, u^{obs})` is the cost function
for a single source. Following :cite:`Tarantola:1984`, the cost function for a single
source can be measured by the :math:`L^2` norm:

.. math::
    
    J_s(u, u^{obs}) = \sum_{r=0}^{N-1} \int_\Omega \left(
        u(c,\mathbf{x},t) - u^{obs}(c, \mathbf{x},t)\right)^2 \delta(\mathbf{x} - \mathbf{x}_r
        ) \, dx

where :math:`u = u(c, \mathbf{x},t)` and :math:`u_{obs} = u_{obs}(c,\mathbf{x},t)`,
are respectively the computed and observed data, both recorded at a finite number
of receivers (:math:`N_r`), located at the point positions :math:`\mathbf{x}_r \in \Omega`,
in a time interval :math:`\tau\equiv[t_0, t_f]\subset \mathbb{R}`, where :math:`t_0` is the
initial time and :math:`t_f` is the final time. The spatial domain of interest is defined
as :math:`\Omega`.

The predicted data is here modeled here by an acoustic wave equation,

.. math::

    \frac{\partial^2 u}{\partial t^2}- c^2\frac{\partial^2 u}{\partial \mathbf{x}^2} = f(\mathbf{x}_s,t),

where :math:`c(\mathbf{x}):\Omega\rightarrow \mathbb{R}` is the pressure wave velocity,
which is assumed here a piecewise-constant and positive. The force term
:math:`f(\mathbf{x},t):\Omega\rightarrow \mathbb{R}` models the source
of waves and is usually described by a `Ricker Wavelet
<https://wiki.seg.org/wiki/Dictionary:Ricker_wavelet>`__. The acoustic wave equation
should satisfy the initial conditions :math:`u(\mathbf{x}, 0) = 0 = u_t(\mathbf{x}, 0) = 0`.
We are employing no-reflective absorbing boundary condition :cite:`Clayton:1977`:

.. math::  \frac{\partial u}{\partial t}- c\frac{\partial u}{\partial \mathbf{x}} = 0, \, \, 
           \forall \mathbf{x} \, \in \partial \Omega 

To solve the wave equation, we consider the following weak form over the domain :math:`\Omega`:

.. math:: \int_{\Omega} \left(
    \frac{\partial^2 u}{\partial t^2}v + c^2\nabla u \cdot \nabla v\right
    ) \, dx = \int_{\Omega} f v \, dx,

for an arbitrary test function :math:`v\in V`, where :math:`V` is a function space. The weak form
implementation in Firedrake is written as follows::

    import finat
    from firedrake import *
    from firedrake.__future__ import Interpolator, interpolate
    
    def wave_equation_solver(c, source_function, dt, V):
        u = TrialFunction(V)
        v = TestFunction(V)
        u_np1 = Function(V) # timestep n+1
        u_n = Function(V) # timestep n
        u_nm1 = Function(V) # timestep n-1
        # Quadrature rule for lumped mass matrix.
        quad_rule = finat.quadrature.make_quadrature(V.finat_element.cell, V.ufl_element().degree(), "KMV")
        time_term = (1 / (c * c)) * (u - 2.0 * u_n + u_nm1) / Constant(dt**2) * v * dx(scheme=quad_rule)
        nf = (1 / c) * ((u_n - u_nm1) / dt) * v * ds
        a = dot(grad(u_n), grad(v)) * dx(scheme=quad_rule)
        F = time_term + a + nf
        lin_var = LinearVariationalProblem(lhs(F), rhs(F) + source_function, u_np1)
        solver_parameters = {"mat_type": "matfree", "ksp_type": "preonly", "pc_type": "jacobi"}
        solver = LinearVariationalSolver(lin_var,solver_parameters=solver_parameters)
        return solver, u_np1, u_n, u_nm1

You can find more details about the wave equation with mass lumping on this
`Firedrake demos <https://www.firedrakeproject.org/demos/higher_order_mass_lumping.py.html>`_.

The wave equation forcing :math:`f = f(\mathbf{x}_s, t)` represents a time-dependent wave source
locate at the position :math:`\mathbf{x}_s`, and it is given by:

.. math::

    f(\mathbf{x}_s,t) = r(t) \delta(\mathbf{x} - \mathbf{x}_s)

where :math:`r(t)` is the `Ricker wavelet <https://wiki.seg.org/wiki/Dictionary:Ricker_wavelet>`__, and
:math:`\delta(\mathbf{x} - \mathbf{x}_s)` is the Dirac delta function. The implementation of `Ricker
wavelet <https://wiki.seg.org/wiki/Dictionary:Ricker_wavelet>`__ is given by the following code::

    def ricker_wavelet(t, fs, amp=1.0):
        ts = 1.5
        t0 = t - ts * np.sqrt(6.0) / (np.pi * fs)
        return (amp * (1.0 - (1.0 / 2.0) * (2.0 * np.pi * fs) * (2.0 * np.pi * fs) * t0 * t0)
                * np.exp((-1.0 / 4.0) * (2.0 * np.pi * fs) * (2.0 * np.pi * fs) * t0 * t0))


In Firedrake, we can compute in parralell the functional values and their gradients for multiple sources.
That is achieved by using the :class:`~.ensemble.Ensemble`, which allows for problem the spatial and
source parallelism. This example demonstrates how to make  use of the :class:`~.ensemble.Ensemble` in this
optimisation problem using autotmated gradient. We have first to define an ensemble object::

    from firedrake import Ensemble, COMM_WORLD
    M = 2
    my_ensemble = Ensemble(COMM_WORLD, M)

``my_ensemble`` requires a communicator (which by default is ``COMM_WORLD``) and a value ``M``, the "team" size,
used to configure the ensemble parallelism. Based on the value of ``M`` and the number of MPI processes,
:class:`~.ensemble.Ensemble` will split the total number of MPI processes in ``COMM_WORLD`` into two
sub-communicators: ``Ensemble.comm`` the spatial communicator having a unique source that each mesh is
distributed over and ``Ensemble.ensemble_comm``. ``Ensemble.ensemble_comm`` is used to communicate information
about the functionals and their gradients computation between different wave sources.

In this example, we want to distribute each mesh over 2 ranks and compute the functional and its gradient
for 3 wave sources. So we set ``M=2`` and execute this code with 6 MPI ranks. That is: 3 (number of sources) x 2 (M).
To have a better understanding of the ensemble parallelism, please refer to the
`Firedrake manual <hhttps://www.firedrakeproject.org/parallelism.html#id8>`__.

The number of sources are set according the source ``my_ensemble.ensemble_comm.size`` (3 in this case)::

    num_sources = my_ensemble.ensemble_comm.size

The source number is defined according to the rank of the ``Ensemble.ensemble_comm``::

    source_number = my_ensemble.ensemble_comm.rank

We consider a two dimensional square domain with side length 1.0 km. The mesh is created over the
``my_ensemble.comm`` communicator::
    
    Lx, Lz = 1.0, 1.0
    mesh = UnitSquareMesh(80, 80, comm=my_ensemble.comm)

We define the basic input for the FWI problem::

    import numpy as np
    source_locations = np.linspace((0.3, 0.05), (0.7, 0.05), num_sources)
    receiver_locations = np.linspace((0.2, 0.85), (0.8, 0.85), 10)
    dt = 0.002  # time step
    final_time = 0.8  # final time
    frequency_peak = 7.0  # The dominant frequency of the Ricker wavelet.

The firedrake functions will be displayed using the following function::

    import matplotlib.pyplot as plt
    from firedrake.pyplot import tricontourf
    def plot_function(function, file_name="function.png"):
        fig, axes = plt.subplots()
        contours = tricontourf(function, 10, axes=axes)
        fig.colorbar(contours, ax=axes, fraction=0.1, cmap="seismic", format="%.3f")
        plt.gca().invert_yaxis()
        axes.set_aspect("equal")
        plt.savefig(file_name + ".png")

        
FWI seeks to estimate the pressure wave velocity based on the observed data stored at the receivers.
The observed data at the receivers is subject to influences of the subsurface medium while waves
propagate from the sources. In this example, we emulate observed data by executing the acoustic wave
equation with a synthetic pressure wave velocity model. The synthetic pressure wave velocity model is
referred to here as the true velocity model (``c_true``). For the sake of simplicity, we consider ``c_true``
consisting of a circle in the centre of the domain, as shown in the coming code cell::

    x, z = SpatialCoordinate(mesh)
    c_true = Function(V).interpolate(2.5 + 1 * tanh(200 * (0.125 - sqrt((x - 0.5) ** 2 + (z - 0.5) ** 2))))
    plot_function(c_true, "c_true")

.. image:: c_true.png

We define the function space to solve the wave equation, :math:`V`. In addition, we define the receivers mesh and its
function space :math:`V_r`::

    V = FunctionSpace(mesh, "KMV", 1)
    receiver_mesh = VertexOnlyMesh(mesh, receiver_locations)
    V_r = FunctionSpace(receiver_mesh, "DG", 0)

We need to define the receiver mesh in order to interpolate the wave equation solution at the receivers.

To model the wave source term in the wave equation, we first create a mesh based on the the source locations
and define the function space (:math:`V_s`) for the source term::

    source_mesh = VertexOnlyMesh(mesh, [source_locations[source_number]])
    V_s = FunctionSpace(source_mesh, "DG", 0)

As recommended in the `Firedrake manual <https://www.firedrakeproject.org/point-evaluation.html#id13>`__,
we define the external Dirac delta value (equal to 1.0) via the
:py:attr:`~.VertexOnlyMeshTopology.input_ordering` property::

    P1DG = FunctionSpace(source_mesh.input_ordering, "DG", 0)
    f_p1DG = Function(P1DG)
    f_p1DG.assign(1.0)

We then interpolate the Dirac delta onto the source function space :math:`V_s`::

    f_s = assemble(interpolate(f_p1DG, V_s)),

which result in a function :math:`f_s \in V_s` such that :math:`f_s(\mathbf{x}_s) = 1.0`. We finally interpolate
the point source onto :math:`V` (function space to solve wave equation solver)::

    cofunction_s = assemble(forcing_point * TestFunction(source_space) * dx)
    source_cofunction = Cofunction(V.dual()).interpolate(cofunction_source_space)

We get the synthetic data recorded on the receivers by executing the acoustic wave equation with the
true velocity model ``c_true``.

.. code-block:: python

    true_data_receivers = []
    total_steps = int(final_time / dt) + 1
    f = Cofunction(V.dual()) # Wave equation forcing term.
    solver, u_np1, u_n, u_nm1 = wave_equation_solver(c_true, f, dt, V)
    interpolate_receivers = Interpolator(u_np1, P0DG).interpolate()

    for t in range(total_steps):
        f.assign(ricker_wavelet(step * dt, frequency_peak) * source_cofunction)
        solver.solve()
        u_nm1.assign(u_n)
        u_n.assign(u_np1)
        true_data_receivers.append(assemble(interpolate_receivers))

Next, we execute an FWI problem, which involves the following steps:

1. Set the initial guess for the parameter ``c_guess``;

2. Solve the wave equation with the initial guess for the parameter ``c_guess``;

3. Compute the functional :math:`J`;

4. Compute the adjoint-based gradient of the functional :math:`J` witt respect to the parameter ``c_guess``;

5. Update the parameter ``c_guess`` using a gradient-based optimisation method, on this case the L-BFGS-B method;

6. Repeat steps 2-5 until the stopping criterion is satisfied.

The initial guess is set (step 1) as a constant field with a value of 1.5 km/s::

    c_guess = Function(V).assign(1.5)
    plot_function(c_guess, "c_initial")


.. image:: c_initial.png


To have the step 4, we need first to tape the forward problem. That is done by calling::

    from firedrake.adjoint import *
    continue_annotation()


We also enable checkpointing in order to reduce the memory usage inherent to the adjoint-based gradient::
    
    from checkpoint_schedules import Revolve
    tape = get_working_tape()
    tape.enable_checkpointing(Revolve(total_steps, 100))

The schedules for checkpointing are generated from the
`checkpoint_schedules <https://www.firedrakeproject.org/checkpoint_schedules/>`__ package.

We then solve the wave equation and compute the functional (steps 2-3)::

    f = Cofunction(V.dual())  # Wave equation forcing term.
    solver, u_np1, u_n, u_nm1 = wave_equation_solver(c_guess, f, dt, V)
    interpolate_receivers = Interpolator(u_np1, P0DG).interpolate()
    J_val = 0.0
    for step in tape.timestepper(iter(range(total_steps))):
        f.assign(ricker_wavelet(step * dt, frequency_peak) * source_cofunction)
        solver.solve()
        u_nm1.assign(u_n)
        u_n.assign(u_np1)
        guess_receiver = assemble(interpolate_receivers)
        misfit = guess_receiver - true_data_receivers[step]
        J_val += 0.5 * assemble(inner(misfit, misfit) * dx)

:class:`~.EnsembleReducedFunctional` is employed to recompute in parallel the functional and
its gradient associated with the multiple sources (3 in this case)::

    J_hat = EnsembleReducedFunctional(J_val, Control(c_guess), my_ensemble)

The ``J_hat`` instance of :class:`~.EnsembleReducedFunctional` is passed as an argument
to the ``minimize`` function, which executes steps 4-6. In the backend, what happens is that
the :class:`~.EnsembleReducedFunctional` computes the functional and gradient for each source
in parallel and returns their sum that is used by the optimisation method.

.. code-block:: python

    c_optimised = minimize(J_hat, method="L-BFGS-B", options={"disp": True, "maxiter": 5}, bounds=(1.5, 3.5))

The optimised parameter ``c_optimised`` for 5 iterations is shown below:

.. code-block:: python

    plot_function(c_optimised, "c_opt_parallel")


.. image:: c_opt_parallel.png

.. note::

    In this demo, we use an acoustic wave equation and a simple FWI problem with only 5 iterations.
    Probably you will get a better result by increasing the number of iterations. Feel free to explore this
    example, which is just a starting point for more complex FWI problems.

.. rubric:: References

.. bibliography:: demo_references.bib
   :filter: docname in docnames

