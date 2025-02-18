from firedrake import *
import numpy
import pytest
import warnings


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
                      "mg_coarse_ksp_type": "preonly",
                      "mg_coarse_pc_type": "python",
                      "mg_coarse_pc_python_type": "firedrake.AssembledPC",
                      "mg_coarse_assembled_pc_type": "lu",
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
                      "snes_linesearch_type": "l2",
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


def run_poisson(solver_type):
    parameters = solver_parameters(solver_type)
    mesh = UnitSquareMesh(10, 10)

    nlevel = 2

    mh = MeshHierarchy(mesh, nlevel)

    V = FunctionSpace(mh[-1], 'CG', 2)
    exact, f = manufacture_solution(V)
    u = function.Function(V)
    v = TestFunction(V)
    F = inner(grad(u), grad(v))*dx - inner(f, v)*dx
    bcs = DirichletBC(V, 0.0, (1, 2, 3, 4))

    solve(F == 0, u, bcs=bcs, solver_parameters=parameters)

    return norm(assemble(exact - u))


@pytest.mark.parametrize("solver_type",
                         ["mg", "mgmatfree", "fas", "newtonfas"])
def test_poisson_gmg(solver_type):
    assert run_poisson(solver_type) < 4e-6


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
    # If we are providing Jp we need to also specify a python preconditioner
    # This is to force a separate _SNESContext for the preconditioner
    parameters = {
        "mat_type": "matfree",
        "snes_type": "ksponly",
        "ksp_convergence_test": "skip",
        "ksp_type": "richardson",
        "ksp_max_it": 1,
        "ksp_richardson_scale": float(beta),  # undo the rescaling
        "pc_type": "python",
        "pc_python_type": "firedrake.AssembledPC",
        "assembled": solver_parameters(solver_type),
        "assembled_pc_use_amat": False
    }
    solve(a == L, uh, bcs=bcs, J=a, Jp=Jp, solver_parameters=parameters)

    assert norm(assemble(exact - uh)) < 4e-6


@pytest.mark.parametrize("solver_type",
                         ["mg", "mgmatfree", "fas", "newtonfas"])
@pytest.mark.parametrize("mixed", [False, True], ids=["scalar", "mixed"])
def test_baseform_coarsening(solver_type, mixed):
    parameters = solver_parameters(solver_type)
    parameters = dict(parameters)
    parameters["snes_rtol"] = 1.0E-10
    parameters["snes_atol"] = 0.0
    parameters["ksp_type"] = "gmres"
    parameters["ksp_rtol"] = 1.0E-12
    parameters["ksp_atol"] = 0.0
    base = UnitSquareMesh(2, 2)
    mh = MeshHierarchy(base, 2, refinements_per_level=2)
    mesh = mh[-1]
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

    # These are equivalent right-hand sides
    sources = [sum(forms),  # purely symbolic linear form
               assemble(sum(forms), bcs=bcs),  # purely numerical cofunction
               sum(assemble(form, bcs=bcs) for form in forms),  # symbolic combination of numerical cofunctions
               forms[0] + assemble(sum(forms[1:]), bcs=bcs),  # symbolic plus numerical
               ]
    solutions = []
    for L in sources:
        uh = Function(V)
        solve(a == L, uh, bcs=bcs, solver_parameters=parameters)
        solutions.append(uh)

    for s in solutions[1:]:
        assert errornorm(s, solutions[0]) < 1E-14


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
