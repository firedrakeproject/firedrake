"""Tests for GoalAdaptiveNonlinearVariationalSolver."""

import pytest
from firedrake import *


@pytest.mark.parallel([1, 3])
@pytest.mark.skipnetgen
def test_goal_adaptive_poisson():
    """DWR goal-adaptive solver on Poisson with a known exact solution.

    Problem: -Δu = f on [0,1]², u = 0 on ∂Ω,
    with exact solution u = sin(πx)sin(πy), f = 2π²sin(πx)sin(πy).

    Goal functional: J(u) = ∫_top (∇u·n) ds (normal flux on the top edge).

    The exact value is J(u) = -2.  We verify that the adaptive solver drives
    its error estimate below the requested tolerance.
    """
    from netgen.occ import WorkPlane, OCCGeometry, Y

    square = WorkPlane().Rectangle(1, 1).Face().bc("all")
    square.edges.Max(Y).name = "top"
    geo = OCCGeometry(square, dim=2)
    ngmesh = geo.GenerateMesh(maxh=0.25)
    mesh = Mesh(ngmesh)

    degree = 2
    V = FunctionSpace(mesh, "CG", degree)
    x, y = SpatialCoordinate(mesh)

    u_exact = sin(pi*x) * sin(pi*y)
    f = 2 * pi**2 * sin(pi*x) * sin(pi*y)

    u = Function(V, name="Solution")
    v = TestFunction(V)
    F = inner(grad(u), grad(v))*dx - inner(f, v)*dx
    bcs = DirichletBC(V, 0, "on_boundary")

    top = tuple(i + 1 for (i, name) in enumerate(ngmesh.GetRegionNames(codim=1)) if name == "top")
    n = FacetNormal(mesh)
    J = inner(grad(u), n)*ds(top)

    solver_parameters = {
        "mat_type": "aij",
        "snes_type": "ksponly",
        "ksp_type": "preonly",
        "pc_type": "lu",
    }
    goal_adaptive_options = {
        "max_iterations": 8,
        "use_adjoint_residual": False,
        "dual_low_method": "interpolate",
        "write_solution": False,
        "verbose": False,
    }

    J_exact = assemble(replace(J, {u: u_exact}))

    tolerance = 1e-3
    problem = NonlinearVariationalProblem(F, u, bcs)
    adaptive_solver = GoalAdaptiveNonlinearVariationalSolver(
        problem, J, tolerance,
        goal_adaptive_options=goal_adaptive_options,
        primal_solver_parameters=solver_parameters,
    )
    adaptive_solver.solve()
    assert abs(adaptive_solver.Juh - J_exact) < tolerance
