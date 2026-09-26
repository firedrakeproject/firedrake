import gc

from firedrake import *
import numpy
import pytest
import warnings

from firedrake.mg import ufl_utils


def solver_parameters(solver_type):
    max_its = 4
    if solver_type == "mg":
        parameters = {"snes_type": "ksponly",
                      "ksp_type": "preonly",
                      "mat_type": "aij",
                      "pc_type": "mg",
                      "pc_mg_type": "full",
                      "mg_levels_ksp_type": "chebyshev",
                      "mg_levels_ksp_max_it": max_its,
                      "mg_levels_pc_type": "jacobi"}
    elif solver_type == "mgmatfree":
        parameters = {"snes_type": "ksponly",
                      "ksp_type": "preonly",
                      "mat_type": "matfree",
                      "pc_type": "mg",
                      "pc_mg_type": "full",
                      "mg_coarse_mat_type": "aij",
                      "mg_coarse_ksp_type": "preonly",
                      "mg_coarse_pc_type": "lu",
                      "mg_levels_ksp_type": "chebyshev",
                      "mg_levels_ksp_max_it": max_its,
                      "mg_levels_pc_type": "jacobi"}
    elif solver_type == "fas":
        parameters = {"snes_type": "fas",
                      "snes_fas_type": "full",
                      "fas_coarse_snes_type": "ksponly",
                      "fas_coarse_ksp_type": "preonly",
                      "fas_coarse_pc_type": "redundant",
                      "fas_coarse_redundant_pc_type": "lu",
                      "fas_levels_snes_type": "ksponly",
                      "fas_levels_ksp_type": "chebyshev",
                      "fas_levels_ksp_max_it": max_its,
                      "fas_levels_pc_type": "jacobi",
                      "fas_levels_ksp_convergence_test": "skip",
                      "snes_max_it": 1,
                      "snes_convergence_test": "skip"}
    elif solver_type == "newtonfas":
        parameters = {"snes_type": "newtonls",
                      "ksp_type": "preonly",
                      "pc_type": "none",
                      "snes_linesearch_type": "secant",
                      "snes_max_it": 1,
                      "snes_convergence_test": "skip",
                      "npc_snes_type": "fas",
                      "npc_snes_fas_type": "full",
                      "npc_fas_coarse_snes_type": "ksponly",
                      "npc_fas_coarse_ksp_type": "preonly",
                      "npc_fas_coarse_pc_type": "redundant",
                      "npc_fas_coarse_redundant_pc_type": "lu",
                      "npc_fas_coarse_snes_linesearch_type": "basic",
                      "npc_fas_levels_snes_type": "ksponly",
                      "npc_fas_levels_ksp_type": "chebyshev",
                      "npc_fas_levels_ksp_max_it": max_its,
                      "npc_fas_levels_pc_type": "jacobi",
                      "npc_fas_levels_ksp_convergence_test": "skip",
                      "npc_snes_max_it": 1,
                      "npc_snes_convergence_test": "skip"}
    else:
        raise RuntimeError("Unknown parameter set '%s' request", solver_type)
    return parameters


def manufacture_solution(V):
    # Choose a forcing function such that the exact solution is not an
    # eigenmode.  This stresses the preconditioner much more.  e.g. 10
    # iterations of ilu fails to converge this problem sufficiently.
    x = SpatialCoordinate(V.mesh())
    f = function.Function(V)
    f.interpolate(-0.5*pi*pi*(4*cos(pi*x[0]) - 5*cos(pi*x[0]*0.5) + 2)*sin(pi*x[1]))

    exact = Function(V[-1])
    exact.interpolate(sin(pi*x[0])*tan(pi*x[0]*0.25)*sin(pi*x[1]))
    return exact, f


def run_poisson(solver_type, rhs_type="form"):
    parameters = solver_parameters(solver_type)
    mesh = UnitSquareMesh(10, 10)

    nlevel = 2

    mh = MeshHierarchy(mesh, nlevel)

    V = FunctionSpace(mh[-1], 'CG', 2)
    exact, f = manufacture_solution(V)
    u = function.Function(V)
    v = TestFunction(V)

    L = inner(f, v)*dx
    if rhs_type == "cofunction":
        L = assemble(L)
    elif rhs_type != "form":
        raise ValueError("Unexpected RHS type")
    F = inner(grad(u), grad(v))*dx - L
    bcs = DirichletBC(V, 0.0, (1, 2, 3, 4))

    solve(F == 0, u, bcs=bcs, solver_parameters=parameters)

    return norm(assemble(exact - u))


def _baseform_solver_parameters(solver_type: str) -> dict:
    """Return solver parameters used by the BaseForm diagnostics.

    Parameters
    ----------
    solver_type : str
        Multigrid solver configuration to use.

    Returns
    -------
    dict
        PETSc solver parameters for the diagnostic.
    """
    parameters = dict(solver_parameters(solver_type))
    parameters.update({
        "snes_rtol": 1.0E-10,
        "snes_atol": 0.0,
        "ksp_type": "gmres",
        "ksp_rtol": 1.0E-12,
        "ksp_atol": 0.0,
    })
    return parameters


def _baseform_problem(mixed: bool, hierarchy=None) -> tuple:
    """Construct the problem and equivalent right-hand sides.

    Parameters
    ----------
    mixed : bool
        Whether to use two copies of the scalar space.
    hierarchy : MeshHierarchy, optional
        Mesh hierarchy to reuse, or ``None`` to construct one.

    Returns
    -------
    tuple
        The hierarchy, function space, bilinear form, boundary conditions,
        and equivalent right-hand sides.
    """
    if hierarchy is None:
        base = UnitSquareMesh(2, 2)
        hierarchy = MeshHierarchy(base, 2, refinements_per_level=2)
    mesh = hierarchy[-1]
    V = FunctionSpace(mesh, "CG", 1)
    _, f = manufacture_solution(V)
    if mixed:
        V = V * V

    bcs = []
    forms = []
    a_terms = []
    for Vsub, v, u in zip(V, TestFunctions(V), TrialFunctions(V)):
        bcs.append(DirichletBC(Vsub, 1.0, (2, 3, 4)))
        forms.extend([inner(f, v) * dx, inner(Constant(1), v) * ds(1)])
        a_terms.append(inner(grad(u), grad(v)) * dx)
    a = sum(a_terms)

    # These are equivalent right-hand sides.
    sources = [sum(forms),
               assemble(sum(forms), bcs=bcs),
               sum(assemble(form, bcs=bcs) for form in forms),
               forms[0] + assemble(sum(forms[1:]), bcs=bcs),
               ]
    return hierarchy, V, a, bcs, sources


def _dummy_solve(mesh: object) -> None:
    """Run a solver whose objects can be collected before the next probe.

    Parameters
    ----------
    mesh : firedrake.MeshGeometry
        Mesh on which to run the dummy solve.

    Returns
    -------
    None
        The solve is used only to exercise solver construction and teardown.
    """
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V)
    trial = TrialFunction(V)
    v = TestFunction(V)
    solve(inner(trial, v) * dx == inner(Constant(1), v) * dx, u,
          solver_parameters={"snes_type": "ksponly",
                             "ksp_type": "preonly",
                             "pc_type": "lu"})


class _TrackingTransferManager(TransferManager):
    """Count transfers made while a solver owns the transfer manager."""

    def __init__(self) -> None:
        super().__init__()
        self.prolong_calls = 0
        self.restrict_calls = 0

    def prolong(self, coarse: object, fine: object) -> None:
        self.prolong_calls += 1
        super().prolong(coarse, fine)

    def restrict(self, fine_dual: object, coarse_dual: object) -> None:
        self.restrict_calls += 1
        super().restrict(fine_dual, coarse_dual)


def _collect(mesh: object) -> None:
    """Collect Python and PETSc objects associated with a mesh.

    Parameters
    ----------
    mesh : firedrake.MeshGeometry
        Mesh supplying the communicator for PETSc cleanup.

    Returns
    -------
    None
        Garbage collection is performed for its side effects.
    """
    gc.collect()
    PETSc.garbage_cleanup(mesh.comm)


def _solve_baseform_source(hierarchy: object, source_index: int,
                           solver_type: str = "mg",
                           transfer_manager: object | None = None) -> float:
    """Solve one BaseForm diagnostic right-hand side.

    Parameters
    ----------
    hierarchy : MeshHierarchy
        Hierarchy on which to construct the problem.
    source_index : int
        Index of the equivalent right-hand side to solve.
    solver_type : str, optional
        Multigrid solver configuration.
    transfer_manager : TransferManager, optional
        Transfer manager to attach to the solver.

    Returns
    -------
    float
        Norm of the computed solution.
    """
    _, V, a, bcs, sources = _baseform_problem(False, hierarchy)
    uh = Function(V)
    problem = LinearVariationalProblem(a, sources[source_index], uh, bcs=bcs)
    solver = LinearVariationalSolver(
        problem, solver_parameters=_baseform_solver_parameters(solver_type))
    if transfer_manager is not None:
        solver.set_transfer_manager(transfer_manager)
    solver.solve()
    return norm(uh)


def _run_transfer_probe(hierarchy: object, transfer_op: str) -> int:
    """Run one solver and count an appctx-owned transfer direction.

    Parameters
    ----------
    hierarchy : MeshHierarchy
        Hierarchy on which to solve the symbolic first right-hand side.
    transfer_op : {"prolong", "restrict"}
        Transfer direction whose calls should be counted.

    Returns
    -------
    int
        Number of calls to the selected transfer operation.
    """
    transfer = _TrackingTransferManager()
    _solve_baseform_source(hierarchy, 0, transfer_manager=transfer)
    if transfer_op == "prolong":
        return transfer.prolong_calls
    if transfer_op == "restrict":
        return transfer.restrict_calls
    raise ValueError(f"Unknown transfer operation: {transfer_op}")


@pytest.mark.parametrize("solver_type",
                         ["mg", "mgmatfree", "fas", "newtonfas"])
def test_poisson_gmg(solver_type):
    assert run_poisson(solver_type) < 4e-6


def test_poisson_gmg_cofunction():
    assert run_poisson("mg", rhs_type="cofunction") < 4e-6


@pytest.mark.parallel
def test_poisson_gmg_parallel_mg():
    errmat = run_poisson("mg")
    errmatfree = run_poisson("mgmatfree")
    assert numpy.allclose(errmat, errmatfree)
    assert errmat < 4e-6
    assert errmatfree < 4e-6


@pytest.mark.parallel
def test_poisson_gmg_parallel_fas():
    assert run_poisson("fas") < 4e-6


@pytest.mark.parallel
def test_poisson_gmg_parallel_newtonfas():
    assert run_poisson("newtonfas") < 4e-6


@pytest.mark.parametrize("solver_type", ["mg", "mgmatfree"])
def test_preconditioner_coarsening(solver_type):
    nlevel = 2
    base = UnitSquareMesh(10, 10)
    mh = MeshHierarchy(base, nlevel)
    mesh = mh[-1]
    V = FunctionSpace(mesh, 'CG', 2)
    R = FunctionSpace(mesh, 'R', 0)
    alpha = Function(R)
    alpha.assign(0.01)
    beta = Function(R)
    beta.assign(100)

    exact, f = manufacture_solution(V)
    v = TestFunction(V)
    u = TrialFunction(V)
    a = inner(alpha * grad(u), grad(v))*dx
    # Rescaled a as the preconditioner
    Jp = inner(beta * alpha * grad(u), grad(v))*dx
    bcs = DirichletBC(V, 0.0, (1, 2, 3, 4))
    L = inner(alpha * f, v)*dx

    uh = function.Function(V)
    parameters = solver_parameters(solver_type)
    parameters.update({
        "snes_type": "ksponly",
        "ksp_convergence_test": "skip",
        "ksp_type": "richardson",
        "ksp_max_it": 1,
        "ksp_richardson_scale": float(beta),  # undo the rescaling
        "pc_use_amat": False,
    })
    solve(a == L, uh, bcs=bcs, J=a, Jp=Jp, solver_parameters=parameters)

    assert norm(assemble(exact - uh)) < 4e-6


@pytest.mark.parametrize("solver_type",
                         ["mg", "mgmatfree", "fas", "newtonfas"])
@pytest.mark.parametrize("mixed", [False, True], ids=["scalar", "mixed"])
@pytest.mark.skip(reason="Test stochastically fails. See https://github.com/firedrakeproject/firedrake/issues/5421")
def test_baseform_coarsening(solver_type, mixed):
    parameters = _baseform_solver_parameters(solver_type)
    _, V, a, bcs, sources = _baseform_problem(mixed)
    solutions = []
    for L in sources:
        uh = Function(V)
        solve(a == L, uh, bcs=bcs, solver_parameters=parameters)
        solutions.append(uh)

    for s in solutions[1:]:
        assert errornorm(s, solutions[0]) < 1E-14


def test_baseform_coarsening_after_dummy_solve_gc():
    """Check BaseForm coarsening after an isolated solver is collected."""
    base = UnitSquareMesh(2, 2)
    hierarchy = MeshHierarchy(base, 2, refinements_per_level=2)
    mesh = hierarchy[-1]

    _solve_baseform_source(hierarchy, 0)
    _collect(mesh)
    _dummy_solve(mesh)
    _collect(mesh)

    assert numpy.isfinite(_solve_baseform_source(hierarchy, 1))
    _collect(mesh)


@pytest.mark.parametrize("transfer_op", ["prolong", "restrict"])
def test_baseform_coarsening_after_isolated_transfer_gc(transfer_op):
    """Check BaseForm coarsening after one collected transfer direction."""
    base = UnitSquareMesh(2, 2)
    hierarchy = MeshHierarchy(base, 2, refinements_per_level=2)
    mesh = hierarchy[-1]

    # The transfer manager is local to the probe solver, so both it and the
    # solver appctx are out of scope before the intervening solver is built.
    assert _run_transfer_probe(hierarchy, transfer_op) > 0
    _collect(mesh)
    # Exercise solver construction after either transfer direction has been
    # collected, before entering the BaseForm coarsening path.
    _dummy_solve(mesh)
    _collect(mesh)

    source_index = {"prolong": 0, "restrict": 1}[transfer_op]
    assert numpy.isfinite(_solve_baseform_source(hierarchy, source_index))
    _collect(mesh)


def test_baseform_coarsening_with_tracked_coefficient_mapping(monkeypatch):
    """Check coarsening with a tracked coefficient mapping and GC boundary."""
    mappings = []
    original_coarsen = ufl_utils.coarsen

    class TrackingMapping(dict):
        def __init__(self, *args):
            super().__init__(*args)
            self.lookups = 0
            self.assignments = 0

        def get(self, key, default=None):
            self.lookups += 1
            return super().get(key, default)

        def __setitem__(self, key, value):
            self.assignments += 1
            super().__setitem__(key, value)

    def tracked_coarsen(expr, dispatch, coefficient_mapping=None):
        if not isinstance(coefficient_mapping, TrackingMapping):
            coefficient_mapping = TrackingMapping(coefficient_mapping or {})
            mappings.append(coefficient_mapping)
        return original_coarsen(expr, tracked_coarsen,
                                coefficient_mapping=coefficient_mapping)

    monkeypatch.setattr(ufl_utils, "coarsen", tracked_coarsen)

    base = UnitSquareMesh(2, 2)
    hierarchy = MeshHierarchy(base, 2, refinements_per_level=2)
    mesh = hierarchy[-1]
    assert _run_transfer_probe(hierarchy, "restrict") > 0
    _collect(mesh)
    _dummy_solve(mesh)
    _collect(mesh)

    assert numpy.isfinite(_solve_baseform_source(hierarchy, 1))
    assert mappings
    assert any(mapping.lookups and mapping.assignments for mapping in mappings)
    _collect(mesh)


@pytest.mark.parametrize("solver_type",
                         ["mg", "mgmatfree"])
def test_reinjection_mass_then_poisson(solver_type):
    parameters = solver_parameters(solver_type)
    parameters = dict(parameters)
    parameters["ksp_type"] = "gmres"
    parameters["ksp_rtol"] = 1.0E-12
    parameters["ksp_atol"] = 0.0

    base = UnitSquareMesh(10, 10)
    nlevel = 4

    mh = MeshHierarchy(base, nlevel)
    mesh = mh[-1]
    R = FunctionSpace(mesh, 'R', 0)
    V = FunctionSpace(mesh, 'CG', 1)
    v = TestFunction(V)
    uh = Function(V)
    alpha = Function(R)
    one = Function(R)
    one.assign(1.0)

    uexact, _ = manufacture_solution(V)

    # The problem is parametrized such that
    # alpha = 0 gives the mass matrix, and alpha = 1 gives Poisson
    a = lambda v, u: inner((one - alpha) * u, v)*dx + inner(alpha * grad(u), grad(v))*dx
    F = a(v, uh - uexact)
    bcs = DirichletBC(V, 0.0, (1, 2, 3, 4))

    transfer = TransferManager()
    problem = NonlinearVariationalProblem(F, uh, bcs=bcs)
    solver = NonlinearVariationalSolver(problem, solver_parameters=parameters)
    solver.set_transfer_manager(transfer)

    # We first solve a problem with the mass matrix, then change the
    # coefficients to obtain Poisson, and test that the second solve propagates
    # the updated coefficients across the multigrid hierarchy
    for val in (0.0, 1.0):
        alpha.assign(val)
        uh.assign(0)
        with warnings.catch_warnings():
            warnings.filterwarnings("error", "Creating new TransferManager", RuntimeWarning)
            solver.solve()

    ksp_its_reused = solver.snes.ksp.getIterationNumber()
    snes_its_reused = solver.snes.getIterationNumber()
    res_reused = solver.snes.getFunctionNorm()

    # Test that the reused solver behaves like a new solver
    new_solver = NonlinearVariationalSolver(problem, solver_parameters=parameters)
    new_solver.set_transfer_manager(transfer)
    uh.assign(0)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", "Creating new TransferManager", RuntimeWarning)
        new_solver.solve()

    ksp_its_new = new_solver.snes.ksp.getIterationNumber()
    snes_its_new = new_solver.snes.getIterationNumber()
    res_new = new_solver.snes.getFunctionNorm()
    assert ksp_its_reused == ksp_its_new
    assert snes_its_reused == snes_its_new
    assert numpy.isclose(res_reused, res_new)
