Goal-based adaptivity for computing functionals of the solution
===============================================================

.. rst-class:: emphasis

    The dual-weighted residual (DWR) method is a technique for designing global and local
    error estimators for the error in a goal functional :math:`J(u)`, where :math:`u`
    is the solution of a partial differential equation. Deriving the DWR method for
    a specific problem usually involves substantial expertise, in deriving the appropriate
    adjoint equation, residual formulation, etc. In this demo we show how the DWR method
    can be automatically implemented using the :code:`GoalAdaptiveNonlinearVariationalSolver`
    class in Firedrake.

    The demo was contributed by `Patrick Farrell
    <mailto:patrick.farrell@maths.ox.ac.uk>`__, based on the MSc project of Joseph Flood.

The dual-weighted residual (DWR) method :cite:`Becker2001` is a technique for designing global and local error estimators for the error in a goal functional :math:`J(u)`. While implementing DWR by hand involves substantial expertise, the high-level symbolic UFL representation of the problem to solve permits the *automation* of DWR :cite:`Rognes2010`.

In this demo we demonstrate how to automatically apply DWR to a nonlinear stationary boundary-value problem, the 𝑝-Laplacian:

.. math::
    :name: eq:plaplace

    -\nabla \cdot \left( |\nabla u|^{p-2} \nabla u \right) = f \text{ in } \Omega, \quad u = 0 \text{ on } \partial \Omega.

We solve the problem on a unit square with known analytical solution, so that we can compute effectivity indices of our error estimates.
Since we will be adapting the mesh, :doc:`we must build the domain with Netgen <netgen_mesh.py>`: ::

    from firedrake import *
    from netgen.occ import *
    square = WorkPlane().Rectangle(1, 1).Face().bc("all")
    square.edges.Max(Y).name = "top"
    geo = OCCGeometry(square, dim=2)
    ngmesh = geo.GenerateMesh(maxh=0.1)
    mesh = Mesh(ngmesh)

    V = FunctionSpace(mesh, "CG", 3)
    (x, y) = SpatialCoordinate(mesh)

    p = Constant(5)
    u_exact = x*(1-x)*y*(1-y)*exp(2*pi*x)*exp(cos(pi*y))
    f = -div(inner(grad(u_exact), grad(u_exact))**((p-2)/2) * grad(u_exact))

    u = Function(V, name="Solution")
    u.interpolate(0.99*u_exact)
    v = TestFunction(V)

    dx = dx(degree=20)
    ds = ds(degree=20)
    F = (
          inner(inner(grad(u), grad(u))**((p-2)/2) * grad(u), grad(v))*dx
        - inner(f, v)*dx
        )
    bcs = DirichletBC(V, 0, "on_boundary")
    solver_parameters = {"snes_monitor": None,
                 "snes_atol": 1e-6,
                 "snes_rtol": 1e-12,
                 "snes_linesearch_monitor": None,
                 "snes_linesearch_type": "l2",
                 "snes_linesearch_maxlambda": 1}

To apply goal-based adaptivity, we need a goal functional. For this we will employ the integral of the normal derivative of the solution on the top boundary: ::

    top = tuple(i + 1 for (i, name) in enumerate(ngmesh.GetRegionNames(codim=1)) if name == "top")
    n = FacetNormal(mesh)
    J = inner(grad(u), n)*ds(top)

We now specify options for how the goal-based adaptivity should proceed. We choose to use an expensive/robust approach,
where the adjoint solution is approximated in a higher-degree function space. This tends to give better error estimates. 
It is possible to employ cheaper approximations by setting the parameters for the :code:`GoalAdaptiveNonlinearVariationalSolver` appropriately. ::

    dwr_parameters = {
        "max_iterations": 30,
        "output_dir": "output/p-laplace",
        "run_name": "p-laplace",
        "use_adjoint_residual": True,
        "dual_low_method": "solve",
        "primal_low_method": "solve",
        "dorfler_alpha": 0.5,
        "dual_extra_degree": 1,
    }

We then solve the problem, passing the goal functional :math:`J` and our specified tolerance. We also pass the exact solution, so that
the DWR automation can compute the effectivity indices (ratio of the estimated error in the goal functional to the true error): ::


    tolerance = 1e-4
    problem = NonlinearVariationalProblem(F, u, bcs)
    GoalAdaptiveNonlinearVariationalSolver(problem, J, tolerance, dwr_parameters,
                                           exact_solution=u_exact, primal_solver_parameters=solver_parameters).solve()

:demo:`A Python script version of this demo can be found here
<goal_based_adaptivity.py>`.

.. rubric:: References

.. bibliography:: demo_references.bib
   :filter: docname in docnames
