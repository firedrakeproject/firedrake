from firedrake import *
import pytest


benchmark = pytest.mark.benchmark(warmup=True, disable_gc=True, warmup_iterations=1)


@benchmark
@pytest.mark.parametrize("bcs", [False, True],
                         ids=["no bcs", "bcs"])
def test_linearsolver(bcs, benchmark):
    m = UnitSquareMesh(1, 1)
    V = FunctionSpace(m, 'CG', 1)

    u = TrialFunction(V)
    v = TestFunction(V)

    a = u*v*dx

    L = v*dx

    if bcs:
        bcs = DirichletBC(V, 0, 1)
    else:
        bcs = None
    A = assemble(a, bcs=bcs)
    b = assemble(L)

    solver = LinearSolver(A)

    f = Function(V)

    benchmark(lambda: solver.solve(f, b))


@benchmark
@pytest.mark.parametrize("bcs", [False, True],
                         ids=["no bcs", "bcs"])
def test_assembled_solve(bcs, benchmark):
    m = UnitSquareMesh(1, 1)
    V = FunctionSpace(m, 'CG', 1)

    u = TrialFunction(V)
    v = TestFunction(V)

    a = u*v*dx

    L = v*dx

    if bcs:
        bcs = DirichletBC(V, 0, 1)
    else:
        bcs = None
    A = assemble(a, bcs=bcs)
    b = assemble(L)

    f = Function(V)

    benchmark(lambda: solve(A, f, b, bcs=bcs))


@benchmark
@pytest.mark.parametrize("bcs", [False, True],
                         ids=["no bcs", "bcs"])
def test_linearvariationalsolver(bcs, benchmark):
    m = UnitSquareMesh(1, 1)
    V = FunctionSpace(m, 'CG', 1)

    u = TrialFunction(V)
    v = TestFunction(V)

    a = u*v*dx

    L = v*dx

    if bcs:
        bcs = DirichletBC(V, 0, 1)
    else:
        bcs = None

    f = Function(V)
    problem = LinearVariationalProblem(a, L, f, bcs=bcs)
    solver = LinearVariationalSolver(problem)

    benchmark(lambda: solver.solve())


@benchmark
@pytest.mark.parametrize("bcs", [False, True],
                         ids=["no bcs", "bcs"])
def test_nonlinearvariationalsolver(bcs, benchmark):
    m = UnitSquareMesh(1, 1)
    V = FunctionSpace(m, 'CG', 1)

    u = Function(V)
    v = TestFunction(V)

    F = u*v*dx - v*dx
    if bcs:
        bcs = DirichletBC(V, 0, 1)
    else:
        bcs = None

    problem = NonlinearVariationalProblem(F, u, bcs=bcs)
    solver = NonlinearVariationalSolver(problem)

    benchmark(lambda: solver.solve())


@benchmark
@pytest.mark.parametrize("bcs", [False, True],
                         ids=["no bcs", "bcs"])
def test_linear_solve(bcs, benchmark):
    m = UnitSquareMesh(1, 1)
    V = FunctionSpace(m, 'CG', 1)

    u = TrialFunction(V)
    v = TestFunction(V)

    a = u*v*dx

    L = v*dx

    if bcs:
        bcs = DirichletBC(V, 0, 1)
    else:
        bcs = None

    f = Function(V)

    benchmark(lambda: solve(a == L, f, bcs=bcs))


@benchmark
@pytest.mark.parametrize("bcs", [False, True],
                         ids=["no bcs", "bcs"])
def test_nonlinear_solve(bcs, benchmark):
    m = UnitSquareMesh(1, 1)
    V = FunctionSpace(m, 'CG', 1)

    u = Function(V)
    v = TestFunction(V)

    F = u*v*dx - v*dx
    if bcs:
        bcs = DirichletBC(V, 0, 1)
    else:
        bcs = None
    benchmark(lambda: solve(F == 0, u, bcs=bcs))
