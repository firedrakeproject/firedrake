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


def test_get_coefficient():
    def mark_cells(ctx, current_solution):
        return Function(FunctionSpace(current_solution.function_space().mesh(), "DG", 0)).assign(1)

    mesh = UnitSquareMesh(2, 2)
    V = FunctionSpace(mesh, "CG", 1)
    f = Function(V).assign(1)
    c = Constant(2)
    g = Function(V)
    u = Function(V)
    v = TestFunction(V)
    F = inner(u - c * f, v) * dx
    solver = NonlinearVariationalSolver(NonlinearVariationalProblem(F, u, bcs=DirichletBC(V, g, 1)),
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

    The window has radius ``radius`` and its centre moves along ``y = 1/2``
    as ``speed * t``. A cell that is still as large as a base cell is marked
    with +1 when its centroid is inside the window. A cell outside the window
    is marked with -1.
    """

    def __init__(self, t, base_volume, radius, speed):
        self.t = t
        self.base_volume = base_volume
        self.radius = radius
        self.speed = speed
        self.meshes = []

    def distance(self, mesh):
        x = SpatialCoordinate(mesh)
        return sqrt((x[0] - self.speed * self.t)**2 + (x[1] - 0.5)**2)

    def __call__(self, ctx, current_solution):
        mesh = current_solution.function_space().mesh()
        self.meshes.append(mesh)
        inside = lt(self.distance(mesh), self.radius)
        unrefined = gt(CellVolume(mesh), 0.75 * self.base_volume)
        marker = conditional(inside, conditional(unrefined, 1, 0), -1)
        return Function(FunctionSpace(mesh, "DG", 0)).interpolate(marker)


@pytest.mark.parallel([1, 3])
@pytest.mark.parametrize("pc_type", ["lu", "mg"])
def test_snes_adapt_transient(pc_type):
    base = UnitSquareMesh(16, 16)
    base_volume = 0.5 / 16**2
    t = Constant(0)
    dt = Constant(0.1)
    radius = 0.1
    speed = 1.2
    nsteps = 5

    def heat_equation(u_old):
        V = u_old.function_space()
        x = SpatialCoordinate(V.mesh())
        source = exp(-((x[0] - speed * t)**2 + (x[1] - 0.5)**2) / 0.01)
        u = TrialFunction(V)
        v = TestFunction(V)
        a = inner(u, v) * dx + dt * inner(grad(u), grad(v)) * dx
        L = inner(u_old + dt * source, v) * dx
        return a, L, DirichletBC(V, 0, "on_boundary")

    def global_max(expr, mesh):
        f = Function(FunctionSpace(mesh, "DG", 0)).interpolate(expr)
        with f.dat.vec_ro as v:
            return v.max()[1]

    params = {
        "snes_adapt_sequence": 1,
        "mat_type": "aij",
        "ksp_type": "preonly",
        "pc_type": "lu",
    }
    if pc_type == "mg":
        params.update({
            "ksp_type": "cg",
            "ksp_rtol": 1e-12,
            "pc_type": "mg",
            "mg_levels": {"ksp_type": "chebyshev", "pc_type": "jacobi"},
            "mg_coarse": {"ksp_type": "preonly", "pc_type": "lu"},
        })
    marking_callback = TransientMarkingCallback(t, base_volume, radius, speed)
    u_old = Function(FunctionSpace(base, "CG", 1))
    a, L, bc = heat_equation(u_old)
    uh = Function(u_old.function_space())
    problem = LinearVariationalProblem(a, L, uh, bcs=bc)
    solver = LinearVariationalSolver(problem, solver_parameters=params,
                                     marking_callback=marking_callback)

    u_ref = Function(u_old.function_space())
    for step in range(nsteps):
        t.assign(t + dt)
        previous_mesh = solver.get_solution().function_space().mesh()
        u = solver.solve()
        mesh = u.function_space().mesh()

        # The marker sees the solution of this step on the previous mesh,
        # and the step is solved again on the adapted mesh.
        assert marking_callback.meshes[-1] is previous_mesh
        assert mesh is not previous_mesh

        # The window is refined, and the cells far from it are coarsened back to the base mesh.
        distance = marking_callback.distance(mesh)
        volume = CellVolume(mesh)
        assert global_max(conditional(lt(distance, radius), volume, 0), mesh) < 0.75 * base_volume
        far = gt(distance, radius + 3 / 16)
        assert global_max(conditional(far, abs(volume - base_volume), 0), mesh) < 1e-12 * base_volume

        # The same step, solved without adaptation on the adapted mesh.
        V = u.function_space()
        u_ref_old = prolong(u_ref, Function(V))
        a, L, bc = heat_equation(u_ref_old)
        u_ref = Function(V)
        solve(a == L, u_ref, bcs=bc, solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})
        assert errornorm(u_ref, u) < 1e-10 * norm(u_ref)

        solver.get_coefficient(u_old).assign(u)


@pytest.mark.parallel([1, 3])
@pytest.mark.parametrize("pc_type", ["lu", "mg"])
def test_snes_adapt_transient_noop(pc_type):
    base = UnitSquareMesh(16, 16)
    t = Constant(0)
    marking_callback = TransientMarkingCallback(t, 0.5 / 16**2, radius=0.1, speed=0)
    V = FunctionSpace(base, "CG", 1)
    u_old = Function(V)
    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(u, v) * dx + inner(grad(u), grad(v)) * dx
    L = inner(u_old + t, v) * dx
    problem = LinearVariationalProblem(a, L, Function(V), bcs=DirichletBC(V, 0, "on_boundary"))
    params = {"snes_adapt_sequence": 1, "mat_type": "aij", "ksp_type": "preonly", "pc_type": "lu"}
    if pc_type == "mg":
        params.update({
            "ksp_type": "cg",
            "pc_type": "mg",
            "mg_levels": {"ksp_type": "chebyshev", "pc_type": "jacobi"},
            "mg_coarse": {"ksp_type": "preonly", "pc_type": "lu"},
        })
    solver = LinearVariationalSolver(problem, solver_parameters=params,
                                     marking_callback=marking_callback)

    # The first step refines the window, and the window does not move afterwards.
    t.assign(1)
    uh = solver.solve()
    ctx = solver._ctx
    mesh = uh.function_space().mesh()
    hierarchy, level = get_level(mesh)
    for step in range(2, 4):
        t.assign(step)
        solver.get_coefficient(u_old).assign(uh)
        assert solver.solve() is uh
        assert marking_callback.meshes[-1] is mesh
        assert solver._ctx is ctx
        assert get_level(mesh) == (hierarchy, level)
        assert len(hierarchy) == level + 1


@pytest.mark.parallel([1, 3])
def test_snes_adapt_noop_refinement_with_multigrid():
    # Marking no cells keeps the mesh and the solution.
    def mark_no_cells(ctx, current_solution):
        return Function(FunctionSpace(current_solution.function_space().mesh(), "DG", 0))

    mesh = UnitSquareMesh(4, 4)
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V)
    v = TestFunction(V)
    F = inner(grad(u), grad(v)) * dx - inner(Constant(1), v) * dx
    bcs = DirichletBC(V, 0, "on_boundary")
    params = {
        "snes_type": "ksponly",
        "snes_adapt_sequence": 1,
        "mat_type": "aij",
        "ksp_type": "cg",
        "ksp_rtol": 1e-12,
        "pc_type": "mg",
        "mg_levels": {"ksp_type": "chebyshev", "pc_type": "jacobi"},
        "mg_coarse": {"ksp_type": "preonly", "pc_type": "lu"},
    }
    solver = NonlinearVariationalSolver(NonlinearVariationalProblem(F, u, bcs=bcs),
                                        solver_parameters=params,
                                        marking_callback=mark_no_cells)
    uh = solver.solve()
    assert uh is u
    assert get_level(mesh) == (None, None)

    u_ref = Function(V)
    solve(replace(F, {u: u_ref}) == 0, u_ref, bcs=bcs)
    assert abs(norm(uh) - norm(u_ref)) < 1e-10 * norm(u_ref)
