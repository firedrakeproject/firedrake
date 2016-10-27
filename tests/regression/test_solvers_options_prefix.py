from firedrake import *
from firedrake.petsc import PETSc
import pytest


@pytest.fixture(params=[None, "", "foo_"])
def prefix(request):
    return request.param


@pytest.fixture
def global_parameters():
    return {"ksp_type": "fgmres",
            "pc_type": "none"}


@pytest.fixture
def opts(request, prefix, global_parameters):
    opts = PETSc.Options()
    if prefix is None:
        prefix = ""

    for k, v in global_parameters.iteritems():
        opts["%s%s" % (prefix, k)] = v

    def finalize():
        for k in global_parameters.keys():
            del opts["%s%s" % (prefix, k)]

    request.addfinalizer(finalize)


@pytest.fixture
def V():
    m = UnitSquareMesh(1, 1)
    return FunctionSpace(m, "CG", 1)


@pytest.fixture
def a(V):
    u = TrialFunction(V)
    v = TestFunction(V)

    return u*v*dx


@pytest.fixture
def L(V):
    v = TestFunction(V)
    return v*dx


@pytest.fixture
def u(V):
    return Function(V)


@pytest.fixture
def parameters():
    return {"ksp_type": "cg",
            "pc_type": "jacobi"}


@pytest.fixture
def ls(a, prefix, parameters):
    A = assemble(a)
    solver = LinearSolver(A, options_prefix=prefix, solver_parameters=parameters)

    return solver, prefix


@pytest.fixture
def lvs(a, L, u, prefix, parameters):
    problem = LinearVariationalProblem(a, L, u)
    solver = LinearVariationalSolver(problem, options_prefix=prefix,
                                     solver_parameters=parameters)

    return solver, prefix


@pytest.fixture
def nlvs(a, L, u, prefix, parameters):
    problem = NonlinearVariationalProblem(action(a, u) - L, u)
    solver = NonlinearVariationalSolver(problem, options_prefix=prefix,
                                        solver_parameters=parameters)

    return solver, prefix


def test_linear_solver_options_prefix(opts, ls, u, L, parameters, global_parameters):
    solver, prefix = ls

    b = assemble(L)

    solver.solve(u, b)

    ksp_type = solver.ksp.getType()
    pc_type = solver.ksp.pc.getType()

    parms = global_parameters
    if prefix is None:
        parms = parameters

    expect_ksp_type = parms["ksp_type"]
    expect_pc_type = parms["pc_type"]

    assert ksp_type == expect_ksp_type
    assert pc_type == expect_pc_type


def test_lvs_options_prefix(opts, lvs, parameters, global_parameters):
    solver, prefix = lvs

    solver.solve()

    ksp_type = solver.snes.ksp.getType()
    pc_type = solver.snes.ksp.pc.getType()

    parms = global_parameters
    if prefix is None:
        parms = parameters

    expect_ksp_type = parms["ksp_type"]
    expect_pc_type = parms["pc_type"]

    assert ksp_type == expect_ksp_type
    assert pc_type == expect_pc_type


def test_nlvs_options_prefix(opts, nlvs, parameters, global_parameters):
    solver, prefix = nlvs

    solver.solve()

    ksp_type = solver.snes.ksp.getType()
    pc_type = solver.snes.ksp.pc.getType()

    parms = global_parameters
    if prefix is None:
        parms = parameters

    expect_ksp_type = parms["ksp_type"]
    expect_pc_type = parms["pc_type"]

    assert ksp_type == expect_ksp_type
    assert pc_type == expect_pc_type


def test_options_database_cleared():
    opts = PETSc.Options()
    expect = len(opts.getAll())

    mesh = UnitIntervalMesh(1)
    V = FunctionSpace(mesh, "DG", 0)
    u = TrialFunction(V)
    v = TestFunction(V)
    A = assemble(u*v*dx)
    b = assemble(v*dx)
    u = Function(V)
    solvers = []
    for i in range(100):
        solver = LinearSolver(A, solver_parameters={"ksp_type": "preonly",
                                                    "pc_type": "lu"})
        solver.solve(u, b)
        solvers.append(solver)
    assert expect == len(opts.getAll())
