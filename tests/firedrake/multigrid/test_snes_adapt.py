import gc

import pytest
from firedrake import *
from firedrake import dmhooks
from firedrake.mg.utils import get_level


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


def poisson_problem(n):
    mesh = UnitSquareMesh(n, n)
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V)
    v = TestFunction(V)
    F = inner(grad(u), grad(v))*dx - inner(1, v)*dx
    bcs = DirichletBC(V, 0, "on_boundary")
    return NonlinearVariationalProblem(F, u, bcs=bcs)


@pytest.mark.parallel([1, 2])
@pytest.mark.parametrize("refine", [False, True], ids=["noop", "refine"])
def test_adaptive_refine_references(refine: bool) -> None:
    problem = poisson_problem(2)
    mesh = problem.u.function_space().mesh()
    markers = Function(FunctionSpace(mesh, "DG", 0)).assign(1) if refine else None
    params = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "snes_grid_sequence": 0,
    }
    solver = NonlinearVariationalSolver(problem, solver_parameters=params)
    solver.solve()
    dm = solver.snes.getDM()
    work = solver.snes.getSolution()
    gc.collect()
    dm_references = dm.getRefCount()
    vector_references = work.getRefCount()
    if refine:
        solver.set_marking_callback(lambda ctx, u: markers)
        solver.parameters["snes_grid_sequence"] = 1
        with solver.inserted_options():
            solver.snes.setFromOptions()

    result = solver.solve()
    gc.collect()

    # The SNES, KSP and PC release their DM references, and the refined
    # context retains a reference to the parent DM.
    assert dm.getRefCount() == dm_references - 2 * int(refine)
    # Refinement releases the SNES reference to the original vector.
    assert work.getRefCount() == vector_references - int(refine)
    assert (result.function_space().mesh() is mesh) == (not refine)


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


@pytest.mark.parallel([1, 2])
def test_snes_grid_sequence_solver_reuse():
    def mark_cells(ctx, current_solution):
        M = FunctionSpace(current_solution.function_space().mesh(), "DG", 0)
        return Function(M).assign(1)

    mesh = UnitSquareMesh(2, 2)
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V)
    v = TestFunction(V)
    F = inner(grad(u), grad(v))*dx - inner(1, v)*dx
    problem = NonlinearVariationalProblem(F, u, bcs=DirichletBC(V, 0, "on_boundary"))
    params = {
        "snes_type": "ksponly",
        "ksp_type": "preonly",
        "pc_type": "lu",
        "snes_grid_sequence": 1,
    }
    solver = NonlinearVariationalSolver(problem, solver_parameters=params, marking_callback=mark_cells)
    u1 = solver.solve()
    # The second solve starts from the mesh of the first solve.
    u2 = solver.solve()

    hierarchy, level = get_level(u2.function_space().mesh())
    assert level == 2
    assert hierarchy[1] is u1.function_space().mesh()


@pytest.mark.skipnetgen
@pytest.mark.parallel([1, 2])
def test_snes_grid_sequence_with_adaptive_multigrid():
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
        "snes_grid_sequence": refinements,
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


def mark_no_cells(ctx, current_solution):
    mesh = current_solution.function_space().mesh()
    return Function(FunctionSpace(mesh, "DG", 0))


@pytest.mark.parallel([1, 3])
def test_snes_grid_sequence_noop_refinement_with_multigrid():
    base = UnitSquareMesh(2, 2)
    mh = MeshHierarchy(base, 1)
    mesh = mh[-1]
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V)
    v = TestFunction(V)
    F = inner(grad(u), grad(v)) * dx - inner(Constant(1), v) * dx
    bcs = DirichletBC(V, 0, "on_boundary")
    params = {
        "snes_type": "ksponly",
        "snes_grid_sequence": 1,
        "mat_type": "aij",
        "ksp_type": "cg",
        "ksp_rtol": 1e-12,
        "pc_type": "mg",
        "mg_levels": {"ksp_type": "chebyshev", "pc_type": "jacobi"},
        "mg_coarse": {"ksp_type": "preonly", "pc_type": "lu"},
    }
    solver = NonlinearVariationalSolver(
        NonlinearVariationalProblem(F, u, bcs=bcs),
        solver_parameters=params,
        marking_callback=mark_no_cells,
    )
    adapted = solver.solve()
    _, level = get_level(adapted.function_space().mesh())
    assert level == 2
    assert adapted.function_space().dim() == V.dim()

    u_ref = Function(V)
    solve(replace(F, {u: u_ref}) == 0, u_ref, bcs=bcs)
    assert abs(norm(adapted) - norm(u_ref)) < 1e-10 * norm(u_ref)
