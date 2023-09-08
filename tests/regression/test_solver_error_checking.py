import pytest
from firedrake import *


@pytest.fixture
def V():
    return FunctionSpace(UnitSquareMesh(1, 1), "CG", 1)


@pytest.fixture
def a(V):
    return inner(TrialFunction(V), TestFunction(V)) * dx


@pytest.fixture
def f(V):
    return Function(V)


@pytest.fixture
def L(V, f):
    return inner(f, TestFunction(V)) * dx


@pytest.fixture
def c(V):
    return Constant(1)


def test_too_few_arguments(a, L):
    with pytest.raises(TypeError):
        solve(a == L)


def test_invalid_solution_type(a, L, c):
    with pytest.raises(TypeError):
        solve(a == L, c)


def test_invalid_lhs_type(L, f):
    with pytest.raises(TypeError):
        solve(f == L, f)


def test_invalid_rhs_type(a, L, f):
    with pytest.raises(TypeError):
        solve(a == f, f)


def test_invalid_lhs_arity(L, f):
    with pytest.raises(ValueError):
        solve(L == L, f)


def test_invalid_rhs_arity(a, L, f):
    with pytest.raises(ValueError):
        solve(a == a, f)


def test_invalid_bc_type(a, L, f, c):
    with pytest.raises(TypeError):
        solve(a == L, f, bcs=(c, ))


def test_la_invalid_matrix_type(a, L, f):
    with pytest.raises(TypeError):
        solve(a, f, assemble(L))


def test_la_invalid_function_type(a, L, f):
    with pytest.raises(TypeError):
        solve(assemble(a), f, L)


def test_la_invalid_solution_type(a, L, c):
    with pytest.raises(TypeError):
        solve(assemble(a), c, assemble(L))


def test_la_too_few_arguments(a, f):
    with pytest.raises(TypeError):
        solve(a, f)
