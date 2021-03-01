from firedrake import *
from firedrake.matrix import ImplicitMatrix
from firedrake.petsc import PETSc, OptionsManager
import pytest


@pytest.fixture(params=[None, "", "foo_"])
def prefix(request):
    return request.param


@pytest.fixture(params=["aij", "matfree"])
def mat_type(request, prefix):
    return request.param


@pytest.fixture
def global_parameters(mat_type):
    return {"ksp_type": "fgmres",
            "pc_type": "none",
            "mat_type": mat_type}


@pytest.fixture
def opts(request, prefix, global_parameters):
    opts = PETSc.Options()
    if prefix is None:
        prefix = ""

    for k, v in global_parameters.items():
        opts[prefix + k] = v

    # Pretend these came from the commandline
    OptionsManager.commandline_options = frozenset(opts.getAll())

    def finalize():
        for k in global_parameters.keys():
            del opts[prefix + k]
        # And remove again
        OptionsManager.commandline_options = frozenset(opts.getAll())

    request.addfinalizer(finalize)


@pytest.fixture
def V():
    m = UnitSquareMesh(1, 1)
    return FunctionSpace(m, "CG", 1)


@pytest.fixture
def a(V):
    u = TrialFunction(V)
    v = TestFunction(V)

    return inner(u, v) * dx


@pytest.fixture
def L(V):
    v = TestFunction(V)
    return conj(v) * dx


@pytest.fixture
def u(V):
    return Function(V)


@pytest.fixture
def parameters(opts):
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


def test_linear_solver_options_prefix(ls, u, L, parameters, global_parameters):
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


def test_lvs_options_prefix(lvs, parameters, global_parameters):
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

    J = solver._ctx._jac
    if prefix is not None and global_parameters["mat_type"] == "matfree":
        assert isinstance(J, ImplicitMatrix)
        assert J.petscmat.getType() == "python"


def test_nlvs_options_prefix(nlvs, parameters, global_parameters):
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
    J = solver._ctx._jac
    if prefix is not None and global_parameters["mat_type"] == "matfree":
        assert isinstance(J, ImplicitMatrix)
        assert J.petscmat.getType() == "python"


def test_options_database_cleared():
    opts = PETSc.Options()
    expect = len(opts.getAll())

    mesh = UnitIntervalMesh(1)
    V = FunctionSpace(mesh, "DG", 0)
    u = TrialFunction(V)
    v = TestFunction(V)
    A = assemble(inner(u, v) * dx)
    b = assemble(conj(v) * dx)
    u = Function(V)
    solvers = []
    for i in range(100):
        solver = LinearSolver(A, solver_parameters={"ksp_type": "preonly",
                                                    "pc_type": "lu"})
        solver.solve(u, b)
        solvers.append(solver)
    assert expect == len(opts.getAll())


def test_same_options_prefix_different_solve():
    mesh = UnitSquareMesh(2, 2)
    V = FunctionSpace(mesh, "CG", 1)

    u = Function(V)
    v = TestFunction(V)

    F = inner(u, v) * dx - conj(v) * dx

    problem = NonlinearVariationalProblem(F, u)
    solver1 = NonlinearVariationalSolver(problem, solver_parameters={"ksp_type": "cg"},
                                         options_prefix="foo_")
    solver2 = NonlinearVariationalSolver(problem, solver_parameters={"ksp_type": "gcr"},
                                         options_prefix="foo_")

    assert solver1.snes.ksp.getType() == "cg"
    assert solver2.snes.ksp.getType() == "gcr"

    with pytest.raises(PETSc.Error) as excinfo:
        solver2 = NonlinearVariationalSolver(problem, solver_parameters={"ksp_type": "bork"},
                                             options_prefix="foo_")
    # Unknown KSP type
    assert excinfo.value.ierr == 86
