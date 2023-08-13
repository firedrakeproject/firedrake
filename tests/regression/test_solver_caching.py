import pytest
from pyop2.caching import cache_manager

from firedrake import *


@pytest.fixture(scope="module")
def comm():
    return COMM_WORLD


@pytest.fixture(scope="module")
def mesh(comm):
    return UnitSquareMesh(5, 5, comm=comm)


@pytest.fixture(scope="module")
def function_space(mesh):
    return FunctionSpace(mesh, "CG", 1)


@pytest.fixture
def trial(function_space):
    return TrialFunction(function_space)


@pytest.fixture
def test(function_space):
    return TestFunction(function_space)


@pytest.fixture
def coeff(function_space):
    x, y = SpatialCoordinate(function_space.ufl_domain())
    return Function(function_space).interpolate(x + y)


@pytest.fixture
def sol(function_space):
    return Function(function_space)


@pytest.fixture
def a(trial, test):
    return trial * test * dx


@pytest.fixture
def L(coeff, test):
    return coeff * test * dx


@pytest.fixture
def F():
    ...


@pytest.fixture
def solver_cache():
    return cache_manager["firedrake.solver"]


def test_repeated_solves_with_new_identical_forms_reuses_lvs(
    trial, test, coeff, sol, solver_cache, comm,
):
    solver_cache.clear(comm)

    solve(inner(trial, test) * dx == inner(coeff, test) * dx, sol)
    assert solver_cache.currsize(comm) == 1

    solve(inner(trial, test) * dx == inner(coeff, test) * dx, sol)
    assert solver_cache.currsize(comm) == 1


def test_repeated_solves_with_new_identical_forms_reuses_nlvs(
    trial, test, coeff, sol, solver_cache, comm,
):
    solver_cache.clear(comm)

    a = inner(trial, test) * dx

    F = action(a, sol) - inner(coeff, test) * dx
    solve(F == 0, sol)
    assert solver_cache.currsize(comm) == 1

    F = action(a, sol) - inner(coeff, test) * dx
    solve(F == 0, sol)
    assert solver_cache.currsize(comm) == 1


def test_lvs_cache_key_includes_coefficients(
    trial, test, coeff, sol, solver_cache, comm,
):
    solver_cache.clear(comm)

    a = inner(trial, test) * dx
    f, g = coeff, coeff.copy()

    L = inner(f, test) * dx
    solve(a == L, sol)
    assert solver_cache.currsize(comm) == 1

    L = inner(g, test) * dx
    solve(a == L, sol)
    assert solver_cache.currsize(comm) == 2


def test_nlvs_cache_key_includes_coefficients(
    trial, test, coeff, sol, solver_cache, comm,
):
    solver_cache.clear(comm)

    a = inner(trial, test) * dx
    f, g = coeff, coeff.copy()

    F = action(a, sol) - inner(f, test) * dx
    solve(F == 0, sol)
    assert solver_cache.currsize(comm) == 1

    F = action(a, sol) - inner(g, test) * dx
    solve(F == 0, sol)
    assert solver_cache.currsize(comm) == 2
