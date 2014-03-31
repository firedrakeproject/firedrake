import pytest
import numpy as np
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


def test_mismatching_rank_interpolation(V):
    f = Function(V)
    with pytest.raises(RuntimeError):
        f.interpolate(Expression(('1', '2')))
    VV = VectorFunctionSpace(V.mesh(), 'CG', 1)
    f = Function(VV)
    with pytest.raises(RuntimeError):
        f.interpolate(Expression(('1', '2')))


def test_mismatching_shape_interpolation(V):
    VV = VectorFunctionSpace(V.mesh(), 'CG', 1)
    f = Function(VV)
    with pytest.raises(RuntimeError):
        f.interpolate(Expression(['1'] * (VV.ufl_element().value_shape()[0] + 1)))


def test_function_val(V):
    """Initialise a Function with a NumPy array."""
    f = Function(V, np.ones((V.node_count, V.dim)))
    assert (f.dat.data_ro == 1.0).all()


def test_function_dat(V):
    """Initialise a Function with an op2.Dat."""
    f = Function(V, op2.Dat(V.node_set**V.dim))
    f.interpolate(Expression("1"))
    assert (f.dat.data_ro == 1.0).all()


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
