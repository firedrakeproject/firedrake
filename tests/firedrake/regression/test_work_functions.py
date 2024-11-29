import pytest
from firedrake import *


@pytest.fixture
def mesh():
    return UnitSquareMesh(1, 1)


@pytest.fixture
def V(mesh):
    return FunctionSpace(mesh, "CG", 1)


@pytest.fixture
def Q(mesh):
    return FunctionSpace(mesh, "CG", 2)


def test_max_work_functions(V):
    # Default is 25
    assert V.max_work_functions == 25

    # Can set number
    V.max_work_functions = 2
    assert V.max_work_functions == 2


def test_get_work_function(V):
    f = V.get_work_function()
    assert V.num_work_functions == 1
    assert f.function_space() is V

    V.restore_work_function(f)
    assert V.num_work_functions == 0


def test_get_work_function_valueerror(V):
    V.max_work_functions = 2

    f = V.get_work_function()
    g = V.get_work_function()
    assert V.num_work_functions == 2

    with pytest.raises(ValueError):
        V.get_work_function()

    V.restore_work_function(f)
    V.restore_work_function(g)
    assert V.num_work_functions == 0


def test_restore_work_function_valueerror(V):
    V.max_work_functions = 2

    f = V.get_work_function()
    g = Function(V)
    assert V.num_work_functions == 1

    with pytest.raises(ValueError):
        V.restore_work_function(g)

    V.restore_work_function(f)
    assert V.num_work_functions == 0

    with pytest.raises(ValueError):
        V.restore_work_function(f)


def test_set_max_work_functions_valueerror(V):
    fns = []
    for _ in range(4):
        fns.append(V.get_work_function())

    assert V.num_work_functions == 4
    # Can't resize when have more checked out than new size.
    with pytest.raises(ValueError):
        V.max_work_functions = 3

    V.restore_work_function(fns.pop())
    assert V.num_work_functions == 3

    # Can resize
    V.max_work_functions = 3

    assert V.max_work_functions == 3


def test_get_restore_get(V):
    f = V.get_work_function()
    V.restore_work_function(f)
    g = V.get_work_function()

    assert f is g

    assert V.num_work_functions == 1


def test_get_get(V):
    f = V.get_work_function()
    g = V.get_work_function()

    assert f is not g
    assert V.num_work_functions == 2


def test_different_spaces(V, Q):
    f = V.get_work_function()
    g = Q.get_work_function()

    assert f is not g

    assert f.function_space() is not g.function_space()

    assert V.num_work_functions == 1
    assert Q.num_work_functions == 1


def test_max_work_functions_shared_across_instances(V, Q):
    V2 = FunctionSpace(V.mesh(), V.ufl_element())

    assert V.max_work_functions == V2.max_work_functions

    V.max_work_functions = 1

    assert V.max_work_functions == 1
    assert V2.max_work_functions == 1

    f = V.get_work_function()
    g = Q.get_work_function()
    assert f is not g
    Q.restore_work_function(g)
    with pytest.raises(ValueError):
        V2.get_work_function()

    with pytest.raises(ValueError):
        V.get_work_function()

    V.restore_work_function(f)

    g = V2.get_work_function()

    assert f is g
