import pytest
from firedrake import *
from firedrake import adapt, dmhooks
from firedrake.mg.utils import get_level
from ufl.domain import extract_unique_domain


def test_marking_callback_configures_refine_adaptor():
    def mark_cells(ctx, current_solution):
        M = FunctionSpace(current_solution.mesh(), "DG", 0)
        return Function(M).assign(1)

    mesh = UnitSquareMesh(1, 1)
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V)
    v = TestFunction(V)
    F = inner(u - 1.0, v) * dx
    problem = NonlinearVariationalProblem(F, u)
    solver = NonlinearVariationalSolver(problem, marking_callback=mark_cells)

    assert solver.parameters["adaptor_criterion"] == "refine"
    assert solver._ctx._marking_callback is mark_cells
    assert solver._ctx.snes == solver.snes
    with pytest.raises(RuntimeError):
        solver._ctx.set_snes(solver.snes)

    ctx = solver._ctx.reconstruct()
    ctx.set_snes(solver._ctx.snes)
    assert ctx.snes == solver.snes


def test_solve_accepts_marking_callback():
    def mark_cells(ctx, current_solution):
        M = FunctionSpace(current_solution.mesh(), "DG", 0)
        return Function(M).assign(1)

    mesh = UnitSquareMesh(1, 1)
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V)
    v = TestFunction(V)
    F = inner(u - 1.0, v) * dx

    result = solve(F == 0, u, marking_callback=mark_cells)
    assert result is u


def poisson_problem(n):
    mesh = UnitSquareMesh(n, n)
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V)
    v = TestFunction(V)
    F = inner(grad(u), grad(v))*dx - inner(1, v)*dx
    bcs = DirichletBC(V, 0, "on_boundary")
    return NonlinearVariationalProblem(F, u, bcs=bcs)


def goal_parameters(refinements):
    return {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "goal_cell_ksp_type": "preonly",
        "goal_cell_pc_type": "jacobi",
        "goal_facet_ksp_type": "preonly",
        "goal_facet_pc_type": "jacobi",
        "snes_adapt_sequence": refinements,
        "adaptor_criterion": "refine",
    }


@pytest.mark.parallel([1, 2])
def test_goal_oriented_marker_refines_mesh():
    problem = poisson_problem(2)
    u = problem.u
    mesh = u.function_space().mesh()
    marker = GoalOrientedMarker(u*dx)
    solver = NonlinearVariationalSolver(
        problem, solver_parameters=goal_parameters(3), marking_callback=marker,
    )
    result = solver.solve()

    hierarchy, level = get_level(result.function_space().mesh())
    assert level == 3
    assert hierarchy[0] is mesh
    assert solver._ctx.snes == solver.snes
    # The marker runs on every mesh except the finest.
    num_dofs = [FunctionSpace(m, "CG", 1).dim() for m in hierarchy[:-1]]
    assert [e.num_dofs for e in marker.estimates] == num_dofs
    assert marker.goal_functional.ufl_domain() is hierarchy[2]


@pytest.mark.parallel([1, 2])
def test_goal_oriented_marker_solver_reuse():
    problem = poisson_problem(2)
    u = problem.u
    marker = GoalOrientedMarker(u*dx)
    solver = NonlinearVariationalSolver(
        problem, solver_parameters=goal_parameters(1), marking_callback=marker,
    )
    dim0 = u.function_space().dim()

    u1 = solver.solve()
    _, level = get_level(u1.function_space().mesh())
    assert level == 1
    dim1 = u1.function_space().dim()
    assert dim1 > dim0

    u2 = solver.solve()
    hierarchy, level = get_level(u2.function_space().mesh())
    assert level == 2
    assert hierarchy[1] is u1.function_space().mesh()
    assert u2.function_space().dim() > dim1
    # The second solve starts from the mesh of the first solve.
    assert [e.num_dofs for e in marker.estimates] == [dim0, dim1]


@pytest.mark.parallel([1, 2])
def test_goal_oriented_marker_stops_at_tolerance():
    problem = poisson_problem(4)
    marker = GoalOrientedMarker(problem.u*dx)
    markers = []

    def mark_cells(ctx, current_solution):
        markers.append(marker(ctx, current_solution))
        return markers[-1]

    atol = 2.0e-3
    refinements = 8
    parameters = goal_parameters(refinements)
    parameters["goal_atol"] = atol
    solver = NonlinearVariationalSolver(
        problem, solver_parameters=parameters, marking_callback=mark_cells,
    )
    result = solver.solve()

    _, level = get_level(result.function_space().mesh())
    assert 0 < level < refinements
    assert len(markers) == level + 1
    assert markers[-1] is None
    assert len(marker.estimates) == level + 1
    assert abs(marker.estimates[-1].error_estimate) < atol


@pytest.mark.parallel([1, 2])
def test_goal_oriented_marker_solver_options(monkeypatch):
    solves = []
    for name in ("LinearVariationalSolver", "NonlinearVariationalSolver"):
        class RecordingSolver(getattr(adapt, name)):
            def solve(self, *args, **kwargs):
                super().solve(*args, **kwargs)
                ksp = self.snes.ksp
                solves.append((self.options_prefix, ksp.getType(), ksp.pc.getType()))

        monkeypatch.setattr(adapt, name, RecordingSolver)

    refinements = 3
    problem = poisson_problem(4)
    marker = GoalOrientedMarker(problem.u*dx)
    parameters = goal_parameters(refinements)
    parameters["ksp_type"] = "cg"
    parameters["pc_type"] = "jacobi"
    solver = NonlinearVariationalSolver(
        problem, solver_parameters=parameters, marking_callback=marker,
    )
    solver.solve()

    # The options of each solve come from the prefix of the solver on every
    # mesh, although the refined contexts have their own prefixes.
    prefix = solver.options_prefix
    expected = refinements * [
        (prefix + "goal_enriched_", "cg", "jacobi"),
        (prefix + "goal_cell_", "preonly", "jacobi"),
        (prefix + "goal_facet_", "preonly", "jacobi"),
    ]
    assert sorted(solves) == sorted(expected)


@pytest.mark.parallel([1, 2])
@pytest.mark.parametrize("snes_type", ["newtonls", "ksponly"])
def test_goal_oriented_marker_effectivity_index(snes_type):
    problem = poisson_problem(4)
    mesh = problem.u.function_space().mesh()
    x, y = SpatialCoordinate(mesh)
    marker = GoalOrientedMarker(problem.u*dx, exact_solution=x*(1 - x)*y*(1 - y))
    parameters = goal_parameters(2)
    parameters["snes_type"] = snes_type
    parameters["goal_monitor"] = None
    solver = NonlinearVariationalSolver(
        problem, solver_parameters=parameters, marking_callback=marker,
    )
    result = solver.solve()

    hierarchy, level = get_level(result.function_space().mesh())
    assert level == 2
    assert extract_unique_domain(marker.exact_solution) is hierarchy[1]
    assert len(marker.estimates) == 2
    for estimate in marker.estimates:
        assert estimate.effectivity_index == estimate.error_estimate / estimate.true_error
        # The marker sees the converged solution, so the solver error is small.
        assert abs(estimate.solver_error) < 1e-10 * abs(estimate.discretisation_error)


@pytest.mark.parallel([1, 2])
def test_adaptive_refine_without_marking_callback_is_uniform():
    problem = poisson_problem(2)
    parameters = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "snes_adapt_sequence": 2,
        "adaptor_criterion": "refine",
    }
    solver = NonlinearVariationalSolver(problem, solver_parameters=parameters)
    result = solver.solve()

    _, level = get_level(result.function_space().mesh())
    assert level == 2
    # Two uniform refinements of a 2x2 mesh give a 8x8 mesh.
    assert result.function_space().dim() == 81


@pytest.mark.skipnetgen
def test_marking_callback_refine_hook_reconstructs_problem():
    from netgen.geom2d import SplineGeometry
    seen = []

    def mark_cells(ctx, current_solution):
        current_mesh = current_solution.function_space().mesh()
        seen.append(current_mesh)
        M = FunctionSpace(current_mesh, "DG", 0)
        markers = Function(M)
        markers.assign(1)
        return markers

    geo = SplineGeometry()
    geo.AddRectangle((0, 0), (1, 1), bc="boundary")
    mesh = Mesh(geo.GenerateMesh(maxh=0.5))
    V = FunctionSpace(mesh, "CG", 1)
    old_dim = V.dim()
    u = Function(V)
    v = TestFunction(V)
    F = inner(u - 1.0, v) * dx
    problem = NonlinearVariationalProblem(F, u)
    solver = NonlinearVariationalSolver(problem, marking_callback=mark_cells)

    dm = solver.snes.getDM()
    with dmhooks.add_hooks(dm, solver, appctx=solver._ctx):
        newdm = dm.refine()
        solver._ctx = dmhooks.get_appctx(newdm)

    adapted = solver.get_solution()
    adapted_mesh = adapted.function_space().mesh()
    hierarchy, level = get_level(adapted_mesh)

    assert seen[0] is mesh
    assert newdm == solver._ctx._problem.dm
    assert adapted_mesh is not mesh
    assert level == 1
    assert hierarchy[1] is adapted_mesh
    assert adapted.function_space().dim() > old_dim


@pytest.mark.skipnetgen
@pytest.mark.parallel([1, 2])
def test_snes_adapt_sequence_with_adaptive_multigrid():
    from netgen.occ import WorkPlane, Axes, OCCGeometry, X, Z

    rect1 = WorkPlane(Axes((0, 0, 0), n=Z, h=X)).Rectangle(1, 2).Face()
    rect2 = WorkPlane(Axes((0, 1, 0), n=Z, h=X)).Rectangle(2, 1).Face()
    mesh = Mesh(OCCGeometry(rect1 + rect2, dim=2).GenerateMesh(maxh=0.8))
    mh = MeshHierarchy(mesh)

    V = FunctionSpace(mesh, "CG", 1)
    old_dim = V.dim()
    u = TrialFunction(V)
    v = TestFunction(V)
    uh = Function(V, name="solution")
    a = inner(grad(u), grad(v))*dx
    L = inner(Constant(1), v)*dx
    bcs = DirichletBC(V, 0, "on_boundary")
    problem = LinearVariationalProblem(a, L, uh, bcs=bcs)

    def estimate_error(current_solution):
        current_mesh = current_solution.function_space().mesh()
        Q = FunctionSpace(current_mesh, "DG", 0)
        eta_sq = Function(Q)
        p = TrialFunction(Q)
        q = TestFunction(Q)
        residual = Constant(1) + div(grad(current_solution))
        h = CellDiameter(current_mesh)
        n = FacetNormal(current_mesh)
        vol = CellVolume(current_mesh)

        a = inner(p, q / vol) * dx
        L = (inner(residual**2, q * h**2) * dx
             + inner(jump(grad(current_solution), n)**2, avg(q * h)) * dS)
        sp = {"mat_type": "matfree", "ksp_type": "preonly", "pc_type": "jacobi"}
        solve(a == L, eta_sq, solver_parameters=sp)
        return Function(Q).interpolate(sqrt(eta_sq))

    seen = []

    def mark_cells(ctx, current_solution):
        current_mesh = current_solution.function_space().mesh()
        seen.append(current_mesh)
        eta = estimate_error(current_solution)
        with eta.dat.vec_ro as eta_vec:
            _, eta_max = eta_vec.max()
        markers = Function(eta.function_space())
        markers.interpolate(conditional(gt(abs(eta), 0.5 * eta_max), 1, 0))
        return markers

    refinements = 5
    params = {
        "mat_type": "aij",
        "snes_adapt_sequence": refinements,
        "ksp_type": "cg",
        "ksp_max_it": 10,
        "ksp_monitor": None,
        "pc_type": "mg",
        "mg_levels": {
            "ksp_type": "chebyshev",
            "ksp_max_it": 1,
            "pc_type": "jacobi",
        },
        "mg_levels_0": {
            "mat_type": "aij",
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
    }
    solver = LinearVariationalSolver(problem,
                                     solver_parameters=params,
                                     marking_callback=mark_cells)
    u_adapted = solver.solve()

    adapted_mesh = u_adapted.function_space().mesh()
    hierarchy, level = get_level(adapted_mesh)

    assert seen[0] == mesh
    assert hierarchy is mh
    assert level == refinements
    assert len(mh) == refinements + 1
    assert adapted_mesh is not mesh
    assert u_adapted is not uh
    assert u_adapted.function_space().dim() > old_dim
