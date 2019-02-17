from firedrake import *
import pytest


benchmark = pytest.mark.benchmark(warmup=True, disable_gc=True, warmup_iterations=1)


@benchmark
@pytest.mark.parametrize("fresh_form",
                         [False, True],
                         ids=["reuse_form", "fresh_form"])
@pytest.mark.parametrize("fresh_tensor",
                         [False, True],
                         ids=["reuse_tensor", "fresh_tensor"])
def test_assemble_residual(fresh_tensor, fresh_form, benchmark):
    m = UnitTriangleMesh()
    V = FunctionSpace(m, 'DG', 0)

    v = TestFunction(V)
    f = Function(V)
    g = Function(V)
    if fresh_form:
        L = lambda: inner(f, v)*dx
    else:
        L_ = inner(f, v)*dx
        L = lambda: L_

    if fresh_tensor:
        call = lambda: assemble(L())
    else:
        call = lambda: assemble(L(), tensor=g)
    benchmark(lambda: call())


@benchmark
@pytest.mark.parametrize("fresh_form",
                         [False, True],
                         ids=["reuse_form", "fresh_form"])
@pytest.mark.parametrize("fresh_tensor",
                         [False, True],
                         ids=["reuse_tensor", "fresh_tensor"])
def test_assemble_mass(fresh_tensor, fresh_form, benchmark):
    m = UnitTriangleMesh()
    V = FunctionSpace(m, 'DG', 0)

    u = TrialFunction(V)
    v = TestFunction(V)
    if fresh_form:
        L = lambda: inner(u, v)*dx
    else:
        L_ = inner(u, v)*dx
        L = lambda: L_

    if fresh_tensor:
        call = lambda: assemble(L()).M
    else:
        g = assemble(L())
        call = lambda: assemble(L(), tensor=g).M
    benchmark(lambda: call())


@benchmark
@pytest.mark.parametrize("fresh_tensor",
                         [False, True],
                         ids=["reuse_tensor", "fresh_tensor"])
def test_assemble_mass_with_bcs(fresh_tensor, benchmark):
    m = UnitSquareMesh(1, 1)
    V = FunctionSpace(m, 'CG', 1)

    u = TrialFunction(V)
    v = TestFunction(V)
    bcs = DirichletBC(V, 0, (1, 2, 3, 4))
    L = inner(u, v)*dx

    if fresh_tensor:
        call = lambda: assemble(L, bcs=bcs).M
    else:
        g = assemble(L)
        call = lambda: assemble(L, tensor=g, bcs=bcs).M
    benchmark(lambda: call())


@benchmark
@pytest.mark.parametrize("nspaces", range(2, 6))
@pytest.mark.parametrize("fresh_tensor",
                         [False, True],
                         ids=["reuse_tensor", "fresh_tensor"])
def test_assemble_mixed_mass(fresh_tensor, nspaces, benchmark):
    m = UnitTriangleMesh()
    V = FunctionSpace(m, 'DG', 0)
    W = MixedFunctionSpace([V]*nspaces)
    u = TrialFunction(W)
    v = TestFunction(W)
    L = inner(u, v)*dx

    if fresh_tensor:
        call = lambda: assemble(L, mat_type="aij").M
    else:
        g = assemble(L)
        call = lambda: assemble(L, mat_type="aij", tensor=g).M
    benchmark(lambda: call())


@benchmark
def test_dat_zero(benchmark):
    m = UnitTriangleMesh()
    V = FunctionSpace(m, 'DG', 0)
    f = Function(V)
    benchmark(lambda: f.dat.zero())


@benchmark
def test_assign_zero(benchmark):
    m = UnitTriangleMesh()
    V = FunctionSpace(m, 'DG', 0)
    f = Function(V)
    benchmark(lambda: f.assign(0))


@benchmark
def test_assign_function(benchmark):
    m = UnitTriangleMesh()
    V = FunctionSpace(m, 'DG', 0)
    f = Function(V)
    g = Function(V)
    benchmark(lambda: f.assign(g))


@benchmark
@pytest.mark.parametrize("fresh_expr",
                         [False, True],
                         ids=["reuse_expr", "fresh_expr"])
def test_assign_complicated(fresh_expr, benchmark):
    m = UnitTriangleMesh()
    V = FunctionSpace(m, 'DG', 0)
    f = Function(V)
    g = Function(V)
    if fresh_expr:
        expr = lambda: g*sqrt(Constant(10) + Constant(11)) + Constant(2)**Constant(4)
    else:
        expr_ = g*sqrt(Constant(10) + Constant(11)) + Constant(2)**Constant(4)
        expr = lambda: expr_
    benchmark(lambda: f.assign(expr()))


@benchmark
@pytest.mark.parametrize("val",
                         [lambda V: Constant(0),
                          lambda V: Function(V)],
                         ids=["Constant", "Function"])
def test_apply_bc(val, benchmark):
    m = UnitSquareMesh(1, 1)
    V = FunctionSpace(m, 'CG', 1)
    bc = DirichletBC(V, val(V), (1, 2, 3, 4))
    f = Function(V)
    benchmark(lambda: bc.apply(f))
