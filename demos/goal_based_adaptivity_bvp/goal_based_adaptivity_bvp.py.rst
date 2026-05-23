Goal-based adaptivity for stationary boundary value problems
============================================================

.. rst-class:: emphasis

    The dual-weighted residual (DWR) method is a technique for designing global and local
    error estimators for the error in a goal functional :math:`J(u)`, where :math:`u`
    is the solution of a partial differential equation. Deriving the DWR method for
    a specific problem usually involves substantial expertise, in deriving the appropriate
    adjoint equation, residual formulation, etc. In this demo we show how the DWR method
    can be automatically implemented in Firedrake for stationary boundary value problems.

    The demo was contributed by `Patrick Farrell <mailto:patrick.farrell@maths.ox.ac.uk>`__, based on the MSc project of
    `Joseph Flood <mailto:josephdflood01@gmail.com>`__.

The dual-weighted residual (DWR) method :cite:`Becker2001` is a technique for designing global and local error estimators for the error in a goal functional :math:`J(u)`. While implementing DWR by hand involves substantial expertise, the high-level symbolic UFL representation of the problem to solve permits the *automation* of DWR :cite:`Rognes2010`.

In this demo we demonstrate how to automatically apply DWR to a nonlinear stationary boundary-value problem, the :math:`p`-Laplacian:

.. math::
    -\nabla \cdot \left( |\nabla u|^{p-2} \nabla u \right) = f \text{ in } \Omega, \quad u = 0 \text{ on } \partial \Omega.

We solve the problem on a unit square with known analytical solution, so that we can compute effectivity indices of our error estimates.
Since we will be adapting the mesh, :doc:`we must build the domain with Netgen <netgen_mesh.py>`.
Notice that we can set very coarse tolerances on the nonlinear solvers, as DWR also provides an estimate of the algebraic solver error: ::

    from firedrake import *
    from netgen.occ import *
    square = WorkPlane().Rectangle(1, 1).Face().bc("all")
    square.edges.Max(Y).name = "top"
    geo = OCCGeometry(square, dim=2)
    ngmesh = geo.GenerateMesh(maxh=0.1)
    mesh = Mesh(ngmesh)

    degree = 3
    V = FunctionSpace(mesh, "CG", degree)
    (x, y) = SpatialCoordinate(mesh)

    p = Constant(5)
    u_exact = x*(1-x)*y*(1-y)*exp(2*pi*x)*cos(pi*y)
    f = -div(inner(grad(u_exact), grad(u_exact))**((p-2)/2) * grad(u_exact))

    # Since the problem is highly nonlinear, for the purposes of this demo we will
    # cheat and pick our initial guess really close to the exact solution.
    u = Function(V, name="Solution")
    u.interpolate(0.99*u_exact)

    v = TestFunction(V)
    F = (inner(inner(grad(u), grad(u))**((p-2)/2) * grad(u), grad(v)) * dx(degree=degree+10)
         - inner(f, v) * dx(degree=degree+10)
    )
    bcs = DirichletBC(V, u_exact, "on_boundary")
    solver_parameters = {
                 "snes_monitor": None,
                 "snes_atol": 1e-6,
                 "snes_rtol": 1e-1, # very coarse!
                 "snes_linesearch_monitor": None,
                 "snes_linesearch_type": "l2",
                 "snes_linesearch_maxlambda": 1}

To apply goal-based adaptivity, we need a goal functional. For this we will employ the integral of the normal derivative of the solution on the top boundary: ::

    top = tuple(i + 1 for (i, name) in enumerate(ngmesh.GetRegionNames(codim=1)) if name == "top")
    n = FacetNormal(mesh)
    J = inner(grad(u), n)*ds(top)

We now specify options for how the goal-based adaptivity should proceed.
We set the absolute tolerance on the error estimate, and the maximum number of iterations.
We choose to use an expensive/robust approach,
where the adjoint solution is approximated in a higher-degree function space, and where both the adjoint and primal residuals
are employed for the error estimate. This requires four solves on every grid (primal and adjoint solutions with degree :math:`p`
and :math:`p+1`), and gives a provably efficient and reliable error estimator under a saturation assumption up to a term that is cubic in the error :cite:`Endtmayer2024`.
It is possible to employ cheaper and more practical approximations by setting the options for the :code:`GoalAdaptiveNonlinearVariationalSolver`
appropriately, as discussed below. ::

    solver_parameters["goal_adaptive"] = {
        "tolerance": 1e-4,
        "max_it": 100,
        "dorfler_alpha": 0.5,
        "use_adjoint_residual": True,
        "dual_low_method": "solve",
        "primal_low_method": "solve",
        "dual_extra_degree": 1,
    }

We then solve the problem, passing the goal functional :math:`J`. We also pass the exact solution, so that
the DWR automation can compute effectivity indices, but this is not generally required: ::

    problem = NonlinearVariationalProblem(F, u, bcs)

    adaptive_solver = GoalAdaptiveNonlinearVariationalSolver(problem, J,
                                                             solver_parameters=solver_parameters,
                                                             exact_solution=u_exact)
    adaptive_solution = adaptive_solver.solve()
    error_estimate = adaptive_solver.get_error_estimate()

The initial error in the goal functional is :math:`-3.5 \times 10^{-2}`. The solver terminates with the goal functional computed to :math:`10^{-4}` after 4 refinements. Each nonlinear solve only required one Newton iteration. The error estimates :math:`\eta` are very accurate: their effectivity indices

.. math::

    I = \frac{\eta}{J(u) - J(u_h)}

are very close to one throughout, except in the final step:

+-----------------------+-------------------------------+
| Number of refinements | Effectivity index :math:`I`   |
+=======================+===============================+
| 0                     | 1.0034                        |
+-----------------------+-------------------------------+
| 1                     | 1.0354                        |
+-----------------------+-------------------------------+
| 2                     | 1.0749                        |
+-----------------------+-------------------------------+
| 3                     | 1.0406                        |
+-----------------------+-------------------------------+
| 4                     | 4.1940                        |
+-----------------------+-------------------------------+

The effectivity index in the final step is larger than one because the solver error is larger than the discretisation error. (The code prints a warning to refine the solver tolerances if this happens, which in this case we can safely ignore.)

Changing the tolerance to :math:`10^{-8}` takes 40 refinements. The resulting mesh is plotted below. The mesh resolution is adaptively concentrated at the top boundary, since the goal functional is localised there.

.. image:: mesh_40.png
    :align: center
    :width: 60%

We now discuss more practical variants. The configuration above solves four PDEs per adaptive step (primal and adjoint, degree :math:`p` and :math:`p+1`). Changing `goal_adaptive_options` to `{"use_adjoint_residual": False, "dual_low_method": "interpolate"}` instead only solves two PDEs per adaptive step (primal at degree :math:`p`, and adjoint at degree :math:`p+1`), and is thus much faster. For this problem with tolerance :math:`10^{-4}` this barely makes a difference to the effectivity indices: most are around 1, with only one step where :math:`I \approx 1.25`. We therefore recommend this as the default settings for production use.

:demo:`A Python script version of this demo can be found here
<goal_based_adaptivity_bvp.py>`.

.. rubric:: References

.. bibliography:: demo_references.bib
   :filter: docname in docnames
