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


lu_parameters = {
    "mat_type": "aij",
    "ksp_type": "preonly",
    "pc_type": "lu",
}

mg_parameters = {
    "mat_type": "aij",
    "ksp_type": "cg",
    "ksp_rtol": 1e-12,
    "pc_type": "mg",
    "mg_levels": {"ksp_type": "chebyshev", "pc_type": "jacobi"},
    "mg_coarse": {"ksp_type": "preonly", "pc_type": "lu"},
}


@pytest.fixture(params=["lu", "mg"])
def linear_parameters(request):
    return {"lu": lu_parameters, "mg": mg_parameters}[request.param]


def poisson(V):
    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(grad(u), grad(v)) * dx
    L = inner(Constant(1), v) * dx
    bcs = DirichletBC(V, 0, "on_boundary")
    return a, L, bcs


def test_get_coefficient():
    def mark_cells(ctx, current_solution):
        M = FunctionSpace(current_solution.function_space().mesh(), "DG", 0)
        return Function(M).assign(1)

    mesh = UnitSquareMesh(2, 2)
    V = FunctionSpace(mesh, "CG", 1)
    f = Function(V).assign(1)
    c = Constant(2)
    g = Function(V)
    u = Function(V)
    v = TestFunction(V)
    F = inner(u - c * f, v) * dx
    bcs = DirichletBC(V, g, 1)
    problem = NonlinearVariationalProblem(F, u, bcs=bcs)
    solver = NonlinearVariationalSolver(problem,
                                        solver_parameters={"snes_adapt_sequence": 1},
                                        marking_callback=mark_cells)
    assert solver.get_coefficient(f) is f
    assert solver.get_coefficient(c) is c
    with pytest.raises(ValueError):
        solver.get_coefficient(Function(V))

    uh = solver.solve()
    adapted_mesh = uh.function_space().mesh()
    for w in (f, g, u):
        assert solver.get_coefficient(w).function_space().mesh() is adapted_mesh
    assert solver.get_coefficient(u) is uh
    assert solver.get_coefficient(c) is c
    with pytest.raises(ValueError):
        solver.get_coefficient(Function(V))


class TransientMarkingCallback:
    """Refines the cells that a moving window covers, and coarsens the others.

    The window has radius ``radius`` and its centre is at ``(centre, 1/2)``.
    A cell that is still as large as a base cell is marked with +1 when its
    centroid is inside the window. A cell outside the window is marked with -1.
    """

    def __init__(self, centre, base_volume, radius):
        self.centre = centre
        self.base_volume = base_volume
        self.radius = radius
        self.meshes = []

    def distance(self, mesh):
        x = SpatialCoordinate(mesh)
        return sqrt((x[0] - self.centre)**2 + (x[1] - 0.5)**2)

    def __call__(self, ctx, current_solution):
        mesh = current_solution.function_space().mesh()
        self.meshes.append(mesh)
        inside = lt(self.distance(mesh), self.radius)
        unrefined = gt(CellVolume(mesh), 0.75 * self.base_volume)
        marker = conditional(inside, conditional(unrefined, 1, 0), -1)
        M = FunctionSpace(mesh, "DG", 0)
        return Function(M).interpolate(marker)


@pytest.mark.parallel([1, 3])
def test_snes_adapt_transient(linear_parameters):
    base = UnitSquareMesh(16, 16)
    base_volume = 0.5 / 16**2
    t = Constant(0)
    dt = Constant(0.1)
    centre = Constant(0)
    radius = 0.1
    speed = 1.2
    nsteps = 5
    nsteps_still = 2

    def heat_equation(u_old):
        V = u_old.function_space()
        x = SpatialCoordinate(V.mesh())
        source = exp(-((x[0] - centre)**2 + (x[1] - 0.5)**2) / 0.01)
        u = TrialFunction(V)
        v = TestFunction(V)
        a = inner(u, v) * dx + dt * inner(grad(u), grad(v)) * dx
        L = inner(u_old + dt * source, v) * dx
        return a, L, DirichletBC(V, 0, "on_boundary")

    def global_max(expr, mesh):
        M = FunctionSpace(mesh, "DG", 0)
        f = Function(M).interpolate(expr)
        with f.dat.vec_ro as v:
            return v.max()[1]

    params = {"snes_adapt_sequence": 1, **linear_parameters}
    marking_callback = TransientMarkingCallback(centre, base_volume, radius)
    V = FunctionSpace(base, "CG", 1)
    u_old = Function(V)
    a, L, bc = heat_equation(u_old)
    uh = Function(V)
    problem = LinearVariationalProblem(a, L, uh, bcs=bc)
    solver = LinearVariationalSolver(problem, solver_parameters=params,
                                     marking_callback=marking_callback)

    u_ref = Function(V)
    for step in range(nsteps + nsteps_still):
        t.assign(t + dt)
        moving = step < nsteps
        if moving:
            centre.assign(speed * t)
        previous_u = solver.get_solution()
        previous_ctx = solver._ctx
        previous_mesh = previous_u.function_space().mesh()
        u = solver.solve()
        mesh = u.function_space().mesh()

        # The marker sees the solution of this step on the previous mesh.
        assert marking_callback.meshes[-1] is previous_mesh
        if moving:
            # The step is solved again on the adapted mesh.
            assert mesh is not previous_mesh
        else:
            # The window does not move, so the mesh, the solution and the context are kept.
            assert u is previous_u
            assert solver._ctx is previous_ctx

        # Each adapted mesh is refined once from the base mesh, so it replaces
        # the previous one on the finest level.
        hierarchy, level = get_level(mesh)
        assert level == 1 and len(hierarchy) == 2
        assert hierarchy[0] is base

        # The window is refined, and the cells far from it are coarsened back to the base mesh.
        distance = marking_callback.distance(mesh)
        volume = CellVolume(mesh)
        assert global_max(conditional(lt(distance, radius), volume, 0), mesh) < 0.75 * base_volume
        far = gt(distance, radius + 3 / 16)
        assert global_max(conditional(far, abs(volume - base_volume), 0), mesh) < 1e-12 * base_volume

        # The same step, solved without adaptation on the adapted mesh.
        Vh = u.function_space()
        u_ref_old = assemble(interpolate(u_ref, Vh)) if moving else u_ref
        a, L, bc = heat_equation(u_ref_old)
        u_ref = Function(Vh)
        solve(a == L, u_ref, bcs=bc, solver_parameters=lu_parameters)
        assert errornorm(u_ref, u) < 1e-10 * norm(u_ref)

        solver.get_coefficient(u_old).assign(u)


@pytest.mark.parallel([1, 3])
def test_snes_adapt_repeated_mixed_solve():
    mesh = UnitSquareMesh(16, 16)
    base_volume = 0.5 / 16**2
    centre = Constant(0.1)
    radius = 0.12
    V = FunctionSpace(mesh, "CG", 1)
    W = V * V
    u = Function(W)
    u0, u1 = split(u)
    v0, v1 = TestFunctions(W)
    F = inner(u0 - 1.0, v0) * dx + inner(u1 - 2.0, v1) * dx
    marked_meshes = []
    mat_types = []

    def mark_cells(ctx, current_solution):
        mesh = current_solution.function_space().mesh().unique()
        marked_meshes.append(mesh)
        mat_types.append(ctx.mat_type)
        x = SpatialCoordinate(mesh)
        inside = lt(sqrt((x[0] - centre)**2 + (x[1] - 0.5)**2), radius)
        unrefined = gt(CellVolume(mesh), 0.75 * base_volume)
        marker = conditional(inside, conditional(unrefined, 1, 0), -1)
        M = FunctionSpace(mesh, "DG", 0)
        return Function(M).interpolate(marker)

    problem = NonlinearVariationalProblem(F, u)
    solver = NonlinearVariationalSolver(
        problem,
        solver_parameters={
            "snes_adapt_sequence": 1,
            "mat_type": "aij",
            "ksp_type": "gmres",
            "pc_type": "none",
            "mg_levels_1": {
                "mat_type": "aij",
                "ksp_type": "gmres",
                "pc_type": "none",
            },
            "mg_levels_2": {
                "mat_type": "matfree",
                "pmat_type": "matfree",
                "ksp_type": "gmres",
                "pc_type": "none",
            },
        },
        marking_callback=mark_cells,
    )
    for step in range(3):
        centre.assign(0.1 + 0.1 * step)
        previous_mesh = solver.get_solution().function_space().mesh().unique()
        solution = solver.solve()
        adapted_mesh = solution.function_space().mesh().unique()
        assert adapted_mesh is not previous_mesh
        hierarchy, level = get_level(adapted_mesh)
        sequence_hierarchy, sequence_level = get_level(solution.function_space().mesh())
        assert (len(sequence_hierarchy), sequence_level) == (len(hierarchy), level)
        assert marked_meshes[-1] is previous_mesh
        assert adapted_mesh.cell_set.size < 3 * mesh.cell_set.size
    assert mat_types == ["aij", "aij", "matfree"]


refine_left_half = lambda x: conditional(lt(x[0], 0.5), 1, 0)  # noqa: E731
refine_left_quarter = lambda x: conditional(lt(x[0], 0.25), 1, 0)  # noqa: E731
coarsen_left_quarter = lambda x: conditional(lt(x[0], 0.25), -1, 0)  # noqa: E731
coarsen_left_half = lambda x: conditional(lt(x[0], 0.5), -1, 0)  # noqa: E731
coarsen_all_but_right = lambda x: conditional(gt(x[0], 0.75), 1, -1)  # noqa: E731


@pytest.mark.parallel([1, 3])
@pytest.mark.parametrize("markers, levels", [
    # Two refinements, and then a coarsening that undoes the first one.
    ([refine_left_half, refine_left_quarter, coarsen_all_but_right], [0, -1]),
    # Two refinements, and then a coarsening that returns the first refined mesh.
    ([refine_left_half, refine_left_quarter, coarsen_left_quarter], [0, 1]),
    # The same, and then a refinement of the returned mesh.
    ([refine_left_half, refine_left_quarter, coarsen_left_quarter, refine_left_quarter], [0, 1, -1]),
    # A refinement, and then a coarsening that returns the base mesh.
    ([refine_left_half, coarsen_left_half], [0]),
], ids=["coarsen_to_new_mesh", "coarsen_to_ancestor", "refine_ancestor", "coarsen_to_base"])
def test_snes_adapt_hierarchy_follows_adaptive_parents(markers, levels):
    """The levels of the hierarchy are the adaptive ancestors of the adapted mesh.

    ``levels`` indexes the meshes that the markers see, followed by the
    adapted mesh.
    """
    meshes = []

    def mark_cells(ctx, current_solution):
        mesh = current_solution.function_space().mesh()
        meshes.append(mesh)
        marker = markers[len(meshes) - 1](SpatialCoordinate(mesh))
        M = FunctionSpace(mesh, "DG", 0)
        return Function(M).interpolate(marker)

    base = UnitSquareMesh(8, 8)
    V = FunctionSpace(base, "CG", 1)
    a, L, bcs = poisson(V)
    uh = Function(V)
    problem = LinearVariationalProblem(a, L, uh, bcs=bcs)
    params = {"snes_adapt_sequence": len(markers), **mg_parameters}
    solver = LinearVariationalSolver(problem, solver_parameters=params, marking_callback=mark_cells)
    u = solver.solve()
    mesh = u.function_space().mesh()

    assert len(meshes) == len(markers)
    candidates = [*meshes, mesh]
    expected = [candidates[i] for i in levels]
    hierarchy, level = get_level(mesh)
    assert list(hierarchy) == expected
    assert level == len(expected) - 1
    assert mesh._adaptive_parent is (expected[-2] if level else None)
    for m in candidates:
        if not any(m is e for e in expected):
            assert get_level(m) == (None, None)
    assert solver.snes.getDM().getRefineLevel() == level

    # The solution on the adapted mesh is the one of the same problem solved directly.
    Vh = u.function_space()
    a, L, bcs = poisson(Vh)
    u_ref = Function(Vh)
    solve(a == L, u_ref, bcs=bcs, solver_parameters=lu_parameters)
    assert errornorm(u_ref, u) < 1e-10 * norm(u_ref)


@pytest.mark.parallel([1, 3])
def test_snes_adapt_noop_refinement(linear_parameters):
    # Marking no cells keeps the mesh and the solution.
    def mark_no_cells(ctx, current_solution):
        M = FunctionSpace(current_solution.function_space().mesh(), "DG", 0)
        return Function(M)

    mesh = UnitSquareMesh(4, 4)
    V = FunctionSpace(mesh, "CG", 1)
    a, L, bcs = poisson(V)
    uh = Function(V)
    problem = LinearVariationalProblem(a, L, uh, bcs=bcs)
    params = {"snes_adapt_sequence": 1, **linear_parameters}
    solver = LinearVariationalSolver(problem, solver_parameters=params, marking_callback=mark_no_cells)
    u = solver.solve()
    assert u is uh
    assert get_level(mesh) == (None, None)

    u_ref = Function(V)
    solve(a == L, u_ref, bcs=bcs, solver_parameters=lu_parameters)
    assert errornorm(u_ref, u) < 1e-10 * norm(u_ref)


@pytest.mark.parallel([1, 2])
@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("shape", ["scalar", "vector", "mixed"])
def test_snes_adapt_project_preserves_coefficient_mass(periodic, shape):
    mesh_type = PeriodicUnitSquareMesh if periodic else UnitSquareMesh
    mesh = mesh_type(6, 6)
    V = FunctionSpace(mesh, "CG", 1)
    if shape == "vector":
        V = VectorFunctionSpace(mesh, "CG", 1)
    elif shape == "mixed":
        V = V * V
    source = Function(V)
    u = Function(V)
    v = TestFunction(V)
    marker = Constant(1)

    def mark_cells(ctx, solution):
        mesh = solution.function_space().mesh().unique()
        x, y = SpatialCoordinate(mesh)
        return Function(FunctionSpace(mesh, "DG", 0)).interpolate(
            conditional(gt(marker, 0), conditional(lt(x, 0.4), 1, 0), -1)
        )

    scale = Function(FunctionSpace(mesh, "R", 0)).assign(1)
    problem = NonlinearVariationalProblem(inner(u - scale*source, v)*dx, u)
    solver = NonlinearVariationalSolver(
        problem,
        solver_parameters={
            "snes_adapt_sequence": 1,
            "snes_adapt_transfer": "project",
            "mat_type": "aij",
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
        marking_callback=mark_cells,
    )
    for _ in range(2):
        solver.solve()
    fine_source = solver.get_coefficient(source)
    fine_mesh = fine_source.function_space().mesh().unique()
    x, y = SpatialCoordinate(fine_mesh)
    pulse = exp(-10*(sin(pi*(x - 0.27))**2 + sin(pi*(y - 0.43))**2))
    if shape == "vector":
        fine_source.interpolate(as_vector([pulse, 2*pulse]))
    elif shape == "mixed":
        fine_source.sub(0).interpolate(pulse)
        fine_source.sub(1).interpolate(2*pulse)
    else:
        fine_source.interpolate(pulse)

    def masses(f):
        return [assemble(c*dx) for c in (split(f) if f.ufl_shape else (f,))]

    expected = masses(fine_source)
    marker.assign(-1)
    for _ in range(2):
        solution = solver.solve()
        assert masses(solution) == pytest.approx(expected, rel=0, abs=1e-10)
        assert masses(solver.get_coefficient(source)) == pytest.approx(expected, rel=0, abs=1e-10)
    assert solution.function_space().mesh().unique() is mesh


def test_snes_adapt_rejects_unknown_transfer():
    def mark_cells(ctx, current_solution):
        M = FunctionSpace(current_solution.function_space().mesh(), "DG", 0)
        return Function(M).assign(1)

    mesh = UnitSquareMesh(1, 1)
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V)
    problem = NonlinearVariationalProblem(inner(u, TestFunction(V))*dx, u)
    params = {"snes_adapt_sequence": 1, "snes_adapt_transfer": "invalid"}
    solver = NonlinearVariationalSolver(problem, solver_parameters=params, marking_callback=mark_cells)
    with pytest.raises(PETSc.Error) as error:
        solver.solve()
    assert isinstance(error.value.__cause__, ValueError)
