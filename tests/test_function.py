import pytest

from firedrake import *


@pytest.fixture
def V():
    mesh = UnitIntervalMesh(2)
    V = FunctionSpace(mesh, "CG", 1)
    return V


def test_firedrake_function(V):
    f = Function(V)
    f.interpolate(Expression("1"))
    assert (f.dat.data_ro == 1.0).all()

    g = Function(f)
    assert (g.dat.data_ro == 1.0).all()

    # Check that g is indeed a deep copy
    f.interpolate(Expression("2"))

    assert (f.dat.data_ro == 2.0).all()
    assert (g.dat.data_ro == 1.0).all()


def test_function_name(V):
    f = Function(V, name="foo")
    assert f.name() == "foo"

    f.rename(label="bar")
    assert f.label() == "bar" and f.name() == "foo"

    f.rename(name="baz")
    assert f.name() == "baz" and f.label() == "bar"

    f.rename("foo", "quux")
    assert f.name() == "foo" and f.label() == "quux"

    f.rename(name="bar", label="baz")
    assert f.name() == "bar" and f.label() == "baz"

if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
