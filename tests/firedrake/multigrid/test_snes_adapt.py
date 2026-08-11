import subprocess
import sys

import numpy as np
import pytest
from firedrake import *
from firedrake import dmhooks
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
    problem = NonlinearVariationalProblem((u - 1.0)*v*dx, u)
    solver = NonlinearVariationalSolver(problem, marking_callback=mark_cells)

    assert solver.parameters["adaptor_criterion"] == "refine"
    assert solver._ctx._marking_callback is mark_cells
    assert solver._ctx.get_snes() is solver.snes


def test_solve_accepts_marking_callback():
    def mark_cells(ctx, current_solution):
        M = FunctionSpace(current_solution.mesh(), "DG", 0)
        return Function(M).assign(1)

    mesh = UnitSquareMesh(1, 1)
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V)
    v = TestFunction(V)

    result = solve((u - 1.0)*v*dx == 0, u, marking_callback=mark_cells)

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
problem = NonlinearVariationalProblem(inner(grad(u), grad(v))*dx - inner(1, v)*dx,
                                      u, bcs=DirichletBC(V, 0, "on_boundary"))
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


def _jacobian_solver(V, u):
    v = TestFunction(V)
    F = inner(grad(u), grad(v))*dx - inner(Constant(1), v)*dx
    problem = NonlinearVariationalProblem(F, u, bcs=DirichletBC(V, 0, "on_boundary"))
    return NonlinearVariationalSolver(
        problem, solver_parameters={"ksp_type": "preonly", "pc_type": "lu"}
    )


def test_solve_jacobian_uses_its_own_solvers_operator():
    # Every solver on a function space used to share one DM-composed KSP, so
    # building a second solver on V hijacked the first one's Jacobian solve.
    mesh = UnitSquareMesh(4, 4)
    V = FunctionSpace(mesh, "CG", 1)

    first = _jacobian_solver(V, Function(V))
    first.solve()

    b = assemble(inner(Constant(1), TestFunction(V))*dx)
    expected = Function(V)
    first._ctx.solve_jacobian(b, expected)

    # A second solver on the same V, with a deliberately different Jacobian.
    u2 = Function(V)
    v2 = TestFunction(V)
    second_problem = NonlinearVariationalProblem(
        Constant(7)*inner(grad(u2), grad(v2))*dx - inner(Constant(1), v2)*dx,
        u2, bcs=DirichletBC(V, 0, "on_boundary"))
    second = NonlinearVariationalSolver(
        second_problem, solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})
    second.solve()

    actual = Function(V)
    first._ctx.solve_jacobian(b, actual)
    assert np.allclose(actual.dat.data_ro, expected.dat.data_ro)

    # ... and the second solver really does have a different operator, so the
    # check above is not vacuous.
    other = Function(V)
    second._ctx.solve_jacobian(b, other)
    assert not np.allclose(other.dat.data_ro, expected.dat.data_ro)


def test_solve_jacobian_matches_assembled_jacobian():
    mesh = UnitSquareMesh(4, 4)
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V)
    solver = _jacobian_solver(V, u)
    solver.solve()

    bc = DirichletBC(V, 0, "on_boundary")
    b = assemble(inner(Constant(1), TestFunction(V))*dx, bcs=bc)
    J = assemble(derivative(solver._problem.F, u), bcs=bc)

    for transpose in (False, True):
        actual = Function(V)
        solver._ctx.solve_jacobian(b, actual, transpose=transpose)
        expected = Function(V)
        solve(J, expected, b, solver_parameters={"ksp_type": "preonly",
                                                 "pc_type": "lu"})
        assert np.allclose(actual.dat.data_ro, expected.dat.data_ro)


def test_solve_jacobian_without_snes_raises():
    mesh = UnitSquareMesh(2, 2)
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V)
    solver = _jacobian_solver(V, u)
    solver.solve()

    # Contexts rebuilt for coarse levels or field splits describe a different
    # problem than the outer SNES, so they deliberately do not inherit it.
    reconstructed = solver._ctx.reconstruct()
    assert reconstructed.get_snes() is None

    b = assemble(inner(Constant(1), TestFunction(V))*dx)
    with pytest.raises(RuntimeError, match="not attached to a SNES"):
        reconstructed.solve_jacobian(b, Function(V))


def _dwr_poisson_solver_parameters(adapt_option, criterion, num_refinements):
    direct = {"ksp_type": "preonly", "pc_type": "lu"}
    dwr_local = {
        "dwr_cell_ksp_type": "preonly",
        "dwr_cell_pc_type": "jacobi",
        "dwr_facet_ksp_type": "preonly",
        "dwr_facet_pc_type": "jacobi",
    }
    return {
        **direct,
        **dwr_local,
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
def test_dwr_marking_callback_builds_poisson_markers(adapt_option, criterion):
    mesh = UnitSquareMesh(2, 2)
    V = FunctionSpace(mesh, "CG", 1)
    old_dim = V.dim()
    u = Function(V)
    v = TestFunction(V)
    F = inner(grad(u), grad(v))*dx - v*dx
    goal = u*dx
    callback = DWRMarkingCallback(goal)

    problem = NonlinearVariationalProblem(
        F, u, bcs=DirichletBC(V, 0, "on_boundary")
    )
    solver = NonlinearVariationalSolver(
        problem,
        solver_parameters=_dwr_poisson_solver_parameters(adapt_option, criterion, 1),
        marking_callback=callback,
    )
    result = solver.solve()

    assert result.function_space().mesh() is not mesh
    assert result.function_space().dim() > old_dim
    hierarchy, level = get_level(result.function_space().mesh())
    assert level == 1
    assert hierarchy[0] is mesh


@pytest.mark.parallel([1, 2])
def test_dwr_marking_callback_multiple_levels():
    mesh = UnitSquareMesh(2, 2)
    V = FunctionSpace(mesh, "CG", 1)
    old_dim = V.dim()
    u = Function(V)
    v = TestFunction(V)
    F = inner(grad(u), grad(v))*dx - v*dx
    goal = u*dx
    callback = DWRMarkingCallback(goal)

    problem = NonlinearVariationalProblem(
        F, u, bcs=DirichletBC(V, 0, "on_boundary")
    )
    solver = NonlinearVariationalSolver(
        problem,
        solver_parameters=_dwr_poisson_solver_parameters("snes_adapt_sequence", "refine", 3),
        marking_callback=callback,
    )
    result = solver.solve()

    assert result.function_space().dim() > old_dim
    hierarchy, level = get_level(result.function_space().mesh())
    assert level == 3
    assert len(hierarchy) == 4
    assert hierarchy[0] is mesh
    adapted_goal = solver.get_goal_functional()
    assert adapted_goal.arguments() == ()


@pytest.mark.parallel([1, 2])
def test_dwr_marking_callback_solver_reuse():
    mesh = UnitSquareMesh(2, 2)
    V = FunctionSpace(mesh, "CG", 1)
    old_dim = V.dim()
    u = Function(V)
    v = TestFunction(V)
    F = inner(grad(u), grad(v))*dx - v*dx
    goal = u*dx
    callback = DWRMarkingCallback(goal)

    problem = NonlinearVariationalProblem(
        F, u, bcs=DirichletBC(V, 0, "on_boundary")
    )
    solver = NonlinearVariationalSolver(
        problem,
        solver_parameters=_dwr_poisson_solver_parameters("snes_adapt_sequence", "refine", 1),
        marking_callback=callback,
    )

    first_result = solver.solve()
    hierarchy, first_level = get_level(first_result.function_space().mesh())
    assert first_level == 1
    first_dim = first_result.function_space().dim()
    first_goal = assemble(solver.get_goal_functional())

    second_result = solver.solve()
    hierarchy, second_level = get_level(second_result.function_space().mesh())
    assert second_level == 2
    assert hierarchy[1] is first_result.function_space().mesh()
    assert second_result.function_space().dim() > first_dim
    second_goal = assemble(solver.get_goal_functional())

    assert first_dim > old_dim
    assert isinstance(first_goal, float)
    assert isinstance(second_goal, float)


def _dwr_poisson_problem(n=4):
    mesh = UnitSquareMesh(n, n)
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V)
    v = TestFunction(V)
    F = inner(grad(u), grad(v))*dx - v*dx
    problem = NonlinearVariationalProblem(F, u, bcs=DirichletBC(V, 0, "on_boundary"))
    return mesh, V, problem, u*dx


@pytest.mark.parallel([1, 2])
def test_dwr_marking_callback_stops_at_tolerance():
    atol = 2.0e-3
    requested = 8
    mesh, V, problem, goal = _dwr_poisson_problem()
    parameters = _dwr_poisson_solver_parameters("snes_adapt_sequence", "refine", requested)
    parameters["dwr_atol"] = atol
    solver = NonlinearVariationalSolver(
        problem, solver_parameters=parameters, marking_callback=DWRMarkingCallback(goal),
    )
    result = solver.solve()

    hierarchy, level = get_level(result.function_space().mesh())
    assert 0 < level < requested
    assert solver._ctx._adapt_converged
    assert abs(solver.get_error_estimate()) < atol


@pytest.mark.parallel([1, 2])
def test_dwr_marking_callback_reads_options_after_refinement():
    # Every dwr_ option must keep coming from the prefix of the solver the
    # callback was attached to. The reconstructed context is renamed after
    # the multigrid level it becomes.
    mesh, V, problem, goal = _dwr_poisson_problem()
    callback = DWRMarkingCallback(goal)
    parameters = _dwr_poisson_solver_parameters("snes_adapt_sequence", "refine", 3)
    solver = NonlinearVariationalSolver(
        problem, solver_parameters=parameters, marking_callback=callback,
    )
    solver.solve()

    assert solver._ctx._marking_callback._options_prefix == solver.options_prefix


@pytest.mark.parallel([1, 2])
def test_dwr_marking_callback_reconstructs_exact_solution():
    mesh, V, problem, goal = _dwr_poisson_problem()
    x, y = SpatialCoordinate(mesh)
    callback = DWRMarkingCallback(goal, exact_solution=x*(1 - x)*y*(1 - y))
    parameters = _dwr_poisson_solver_parameters("snes_adapt_sequence", "refine", 2)
    parameters["dwr_monitor"] = None
    solver = NonlinearVariationalSolver(
        problem, solver_parameters=parameters, marking_callback=callback,
    )
    result = solver.solve()

    adapted = solver._ctx._marking_callback
    adapted_mesh = result.function_space().mesh().unique()
    assert adapted_mesh is not mesh
    assert extract_unique_domain(adapted.exact_solution) is adapted_mesh
    assert solver.get_error_estimate() != 0.0


@pytest.mark.parallel([1, 2])
def test_adaptive_refine_without_marking_callback_is_uniform():
    mesh, V, problem, _ = _dwr_poisson_problem(n=2)
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
    problem = NonlinearVariationalProblem((u - 1.0)*v*dx, u)
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
        markers.interpolate(conditional(gt(eta, 0.5 * eta_max), 1, 0))
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
