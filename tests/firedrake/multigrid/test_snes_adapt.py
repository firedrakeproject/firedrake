import subprocess
import sys

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
    F = (u - 1.0)*v*dx

    result = solve(F == 0, u, marking_callback=mark_cells)

    assert result is u


# Make sure that we don't segfault when collecting the adapted-away mesh. That
# kills the interpreter rather than raising, so it has to run in a subprocess.
_COLLECT_AFTER_ADAPT = """
import gc
from firedrake import *

def mark_cells(ctx, current_solution):
    M = FunctionSpace(current_solution.function_space().mesh(), "DG", 0)
    return Function(M).assign(1)

mesh = UnitSquareMesh(2, 2)
V = FunctionSpace(mesh, "CG", 1)
u = Function(V)
v = TestFunction(V)
F = inner(grad(u), grad(v))*dx - inner(1, v)*dx
bc = DirichletBC(V, 0, "on_boundary")
problem = NonlinearVariationalProblem(F, u, bcs=bc)
solver = NonlinearVariationalSolver(
    problem, marking_callback=mark_cells,
    solver_parameters={"ksp_type": "preonly", "pc_type": "lu",
                       "snes_adapt_sequence": 1, "adaptor_criterion": "refine"})
solver.solve()
del solver, problem, u, v, V, mesh
gc.collect()
"""


def test_collect_mesh_after_adaptive_solve():
    subprocess.run([sys.executable, "-c", _COLLECT_AFTER_ADAPT], check=True)


def _goal_poisson_solver_parameters(adapt_option, criterion, num_refinements):
    direct = {"ksp_type": "preonly", "pc_type": "lu"}
    goal_local = {
        "goal_cell_ksp_type": "preonly",
        "goal_cell_pc_type": "jacobi",
        "goal_facet_ksp_type": "preonly",
        "goal_facet_pc_type": "jacobi",
    }
    return {
        **direct,
        **goal_local,
        adapt_option: num_refinements,
        "adaptor_criterion": criterion,
    }


@pytest.mark.parallel([1, 2])
@pytest.mark.parametrize(
    ("adapt_option", "criterion"),
    # ("snes_adapt_multigrid", "none") requires
    # https://gitlab.com/petsc/petsc/-/merge_requests/9447
    (("snes_adapt_sequence", "refine"),),
)
def test_goal_oriented_marker_builds_poisson_markers(adapt_option, criterion):
    mesh = UnitSquareMesh(2, 2)
    V = FunctionSpace(mesh, "CG", 1)
    old_dim = V.dim()
    u = Function(V)
    v = TestFunction(V)
    F = inner(grad(u), grad(v))*dx - v*dx
    goal = u*dx
    callback = GoalOrientedMarker(goal)

    bc = DirichletBC(V, 0, "on_boundary")
    parameters = _goal_poisson_solver_parameters(adapt_option, criterion, 1)

    problem = NonlinearVariationalProblem(F, u, bcs=bc)
    solver = NonlinearVariationalSolver(
        problem, solver_parameters=parameters, marking_callback=callback,
    )
    result = solver.solve()

    assert result.function_space().mesh() is not mesh
    assert result.function_space().dim() > old_dim
    assert solver._ctx.snes == solver.snes
    hierarchy, level = get_level(result.function_space().mesh())
    assert level == 1
    assert hierarchy[0] is mesh


@pytest.mark.parallel([1, 2])
def test_goal_oriented_marker_multiple_levels():
    mesh = UnitSquareMesh(2, 2)
    V = FunctionSpace(mesh, "CG", 1)
    old_dim = V.dim()
    u = Function(V)
    v = TestFunction(V)
    F = inner(grad(u), grad(v))*dx - v*dx
    goal = u*dx
    callback = GoalOrientedMarker(goal)

    bc = DirichletBC(V, 0, "on_boundary")
    parameters = _goal_poisson_solver_parameters("snes_adapt_sequence", "refine", 3)

    problem = NonlinearVariationalProblem(F, u, bcs=bc)
    solver = NonlinearVariationalSolver(
        problem, solver_parameters=parameters, marking_callback=callback,
    )
    result = solver.solve()

    assert result.function_space().dim() > old_dim
    hierarchy, level = get_level(result.function_space().mesh())
    assert level == 3
    assert len(hierarchy) == 4
    assert hierarchy[0] is mesh
    # The marker estimates the error on every mesh except the finest one.
    assert [e.num_dofs for e in callback.estimates] == [
        FunctionSpace(m, "CG", 1).dim() for m in hierarchy[:-1]
    ]
    assert callback.goal_functional.ufl_domain() is hierarchy[2]


@pytest.mark.parallel([1, 2])
def test_goal_oriented_marker_solver_reuse():
    mesh = UnitSquareMesh(2, 2)
    V = FunctionSpace(mesh, "CG", 1)
    old_dim = V.dim()
    u = Function(V)
    v = TestFunction(V)
    F = inner(grad(u), grad(v))*dx - v*dx
    goal = u*dx
    callback = GoalOrientedMarker(goal)

    bc = DirichletBC(V, 0, "on_boundary")
    parameters = _goal_poisson_solver_parameters("snes_adapt_sequence", "refine", 1)

    problem = NonlinearVariationalProblem(F, u, bcs=bc)
    solver = NonlinearVariationalSolver(
        problem, solver_parameters=parameters, marking_callback=callback,
    )

    first_result = solver.solve()
    hierarchy, first_level = get_level(first_result.function_space().mesh())
    assert first_level == 1
    first_dim = first_result.function_space().dim()

    second_result = solver.solve()
    hierarchy, second_level = get_level(second_result.function_space().mesh())
    assert second_level == 2
    assert hierarchy[1] is first_result.function_space().mesh()
    assert second_result.function_space().dim() > first_dim

    assert first_dim > old_dim
    # The second solve marks the mesh that the first solve adapted to.
    assert [e.num_dofs for e in callback.estimates] == [old_dim, first_dim]


def _goal_poisson_problem(n=4):
    mesh = UnitSquareMesh(n, n)
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V)
    v = TestFunction(V)
    F = inner(grad(u), grad(v))*dx - v*dx
    bc = DirichletBC(V, 0, "on_boundary")
    problem = NonlinearVariationalProblem(F, u, bcs=bc)
    goal = u*dx
    return mesh, V, problem, goal


@pytest.mark.parallel([1, 2])
def test_goal_oriented_marker_stops_at_tolerance(monkeypatch):
    markings = []
    mark = GoalOrientedMarker.__call__

    def record_marking(self, ctx, current_solution):
        markers = mark(self, ctx, current_solution)
        markings.append(markers)
        return markers

    monkeypatch.setattr(GoalOrientedMarker, "__call__", record_marking)
    atol = 2.0e-3
    requested = 8
    mesh, V, problem, goal = _goal_poisson_problem()
    parameters = _goal_poisson_solver_parameters("snes_adapt_sequence", "refine", requested)
    parameters["goal_atol"] = atol
    callback = GoalOrientedMarker(goal)
    solver = NonlinearVariationalSolver(
        problem, solver_parameters=parameters, marking_callback=callback,
    )
    result = solver.solve()

    hierarchy, level = get_level(result.function_space().mesh())
    assert 0 < level < requested
    # The callback marks once per refinement, and the sequence stops after the
    # callback marks nothing.
    assert len(markings) == level + 1
    assert markings[-1] is None
    assert len(callback.estimates) == level + 1
    assert abs(callback.estimates[-1].error_estimate) < atol


@pytest.mark.parallel([1, 2])
def test_goal_oriented_marker_reads_options_after_refinement(monkeypatch):
    # Every goal_ option must keep coming from the prefix of the solver the
    # callback was attached to. The reconstructed context is renamed after
    # the multigrid level it becomes.
    used = []
    monkeypatch.setattr(adapt, "LinearVariationalSolver",
                        _recording_solver(LinearVariationalSolver, used))
    monkeypatch.setattr(adapt, "NonlinearVariationalSolver",
                        _recording_solver(NonlinearVariationalSolver, used))

    mesh, V, problem, goal = _goal_poisson_problem()
    callback = GoalOrientedMarker(goal)
    parameters = _goal_poisson_solver_parameters("snes_adapt_sequence", "refine", 3)
    solver = NonlinearVariationalSolver(
        problem, solver_parameters=parameters, marking_callback=callback,
    )
    solver.solve()

    assert solver._ctx._marking_callback is callback
    prefixes = {prefix for prefix, _ in used}
    expected = {solver.options_prefix + suffix
                for suffix in ("goal_enriched_", "goal_cell_", "goal_facet_")}
    assert prefixes == expected


def _recording_solver(base, record):
    """Return a solver class that records the solver that each solve used."""
    class RecordingSolver(base):
        def solve(self, *args, **kwargs):
            super().solve(*args, **kwargs)
            ksp = self.snes.ksp
            record.append((self.options_prefix, (ksp.getType(), ksp.pc.getType())))

    return RecordingSolver


@pytest.mark.parallel([1, 2])
def test_goal_oriented_marker_auxiliary_solver_options_survive_refinement(monkeypatch):
    # The auxiliary solvers are rebuilt on every adapted mesh, and each one
    # deletes from the options database the options that it reads. The callback
    # must therefore hold its own copy of them. Otherwise every mesh after the
    # first one is solved with the default preonly and lu, which is what these
    # parameters are chosen to differ from.
    used = []
    monkeypatch.setattr(adapt, "LinearVariationalSolver",
                        _recording_solver(LinearVariationalSolver, used))
    monkeypatch.setattr(adapt, "NonlinearVariationalSolver",
                        _recording_solver(NonlinearVariationalSolver, used))

    refinements = 3
    mesh, V, problem, goal = _goal_poisson_problem()
    parameters = {
        "ksp_type": "cg",
        "pc_type": "jacobi",
        "goal_cell_ksp_type": "preonly",
        "goal_cell_pc_type": "jacobi",
        "goal_facet_ksp_type": "preonly",
        "goal_facet_pc_type": "jacobi",
        "snes_adapt_sequence": refinements,
        "adaptor_criterion": "refine",
    }
    callback = GoalOrientedMarker(goal)
    solver = NonlinearVariationalSolver(
        problem, solver_parameters=parameters, marking_callback=callback,
    )
    solver.solve()

    localization = [solve for prefix, solve in used
                    if prefix.endswith(("goal_cell_", "goal_facet_"))]
    enriched = [solve for prefix, solve in used if prefix.endswith("goal_enriched_")]
    # One cell solve and one facet solve localize the estimate on every mesh.
    assert len(localization) == 2*refinements
    assert set(localization) == {("preonly", "jacobi")}
    # The enriched solve inherits the iterative solver of the parent.
    assert len(enriched) == refinements
    assert set(enriched) == {("cg", "jacobi")}


@pytest.mark.parallel([1, 2])
@pytest.mark.parametrize("snes_type", ["newtonls", "ksponly"])
def test_goal_oriented_marker_effectivity_index(snes_type):
    mesh, V, problem, goal = _goal_poisson_problem()
    x, y = SpatialCoordinate(mesh)
    callback = GoalOrientedMarker(goal, exact_solution=x*(1 - x)*y*(1 - y))
    parameters = _goal_poisson_solver_parameters("snes_adapt_sequence", "refine", 2)
    parameters["snes_type"] = snes_type
    parameters["goal_monitor"] = None
    solver = NonlinearVariationalSolver(
        problem, solver_parameters=parameters, marking_callback=callback,
    )
    result = solver.solve()

    hierarchy, level = get_level(result.function_space().mesh())
    assert level == 2
    # The exact solution follows the goal to the last marked mesh.
    assert extract_unique_domain(callback.exact_solution) is hierarchy[1]
    assert len(callback.estimates) == 2
    for estimate in callback.estimates:
        assert estimate.true_error is not None
        assert estimate.effectivity_index == estimate.error_estimate / estimate.true_error
        # The marker must see the solution, not the initial guess.
        assert abs(estimate.solver_error) < 1e-10 * abs(estimate.discretisation_error)


@pytest.mark.parallel([1, 2])
def test_goal_oriented_marker_biharmonic():
    mesh = UnitSquareMesh(4, 4)
    x, y = SpatialCoordinate(mesh)
    u_exact = x**2*(1 - x)**2*y**2*(1 - y)**2
    f = div(grad(div(grad(u_exact))))
    V = FunctionSpace(mesh, "HCT-red", 3)
    u = Function(V)
    v = TestFunction(V)
    F = inner(grad(grad(u)), grad(grad(v)))*dx - inner(f, v)*dx
    bc = DirichletBC(V, 0, "on_boundary")
    problem = NonlinearVariationalProblem(F, u, bcs=bc)

    callback = GoalOrientedMarker(u*dx, exact_solution=u_exact)
    parameters = {
        "snes_type": "ksponly",
        "ksp_type": "preonly",
        "pc_type": "lu",
        "goal_cell_ksp_type": "preonly",
        "goal_cell_pc_type": "lu",
        "goal_facet_ksp_type": "preonly",
        "goal_facet_pc_type": "lu",
        "snes_adapt_sequence": 6,
        "adaptor_criterion": "refine",
        "goal_enriched_family": "HCT",
        "goal_enrichment_degree": 0,
        "goal_test_derivative_order": 2,
    }
    solver = NonlinearVariationalSolver(
        problem, solver_parameters=parameters, marking_callback=callback,
    )
    solver.solve()

    assert len(callback.estimates) == 6
    estimate = callback.estimates[-1]
    assert abs(estimate.true_error) < 1e-5
    assert abs(estimate.effectivity_index - 1) < 0.05


@pytest.mark.parallel([1, 2])
def test_adaptive_refine_without_marking_callback_is_uniform():
    mesh, V, problem, _ = _goal_poisson_problem(n=2)
    solver = NonlinearVariationalSolver(
        problem,
        solver_parameters={"ksp_type": "preonly", "pc_type": "lu",
                           "snes_adapt_sequence": 2, "adaptor_criterion": "refine"},
    )
    result = solver.solve()

    hierarchy, level = get_level(result.function_space().mesh())
    assert level == 2
    # Two rounds of red refinement of the 8 cells of a 2x2 unit square.
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
