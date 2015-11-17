import pytest
import numpy as np
from firedrake import *


@pytest.fixture
def V():
    mesh = UnitIntervalMesh(2)
    V = FunctionSpace(mesh, "CG", 1)
    return V


@pytest.fixture
def W():
    mesh = UnitSquareMesh(5, 5)
    W = TensorFunctionSpace(mesh, "CG", 1)
    return W


@pytest.fixture
def W_nonstandard_shape():
    mesh = UnitSquareMesh(5, 5)
    W_nonstandard_shape = TensorFunctionSpace(mesh, "CG", 1, shape=(2, 5, 3))
    return W_nonstandard_shape


def test_firedrake_scalar_function(V):
    f = Function(V)
    f.interpolate(Expression("1"))
    assert (f.dat.data_ro == 1.0).all()

    g = Function(f)
    assert (g.dat.data_ro == 1.0).all()

    # Check that g is indeed a deep copy
    f.interpolate(Expression("2"))

    assert (f.dat.data_ro == 2.0).all()
    assert (g.dat.data_ro == 1.0).all()


def test_firedrake_tensor_function(W):
    f = Function(W)
    vals = np.array([1.0, 2.0, 10.0, 20.0]).reshape(2, 2)
    f.interpolate(Expression(vals.astype("string")))
    assert np.allclose(f.dat.data_ro, vals)

    g = Function(f)
    assert np.allclose(g.dat.data_ro, vals)

    # Check that g is indeed a deep copy
    fvals = np.array([5.0, 6.0, 7.0, 8.0]).reshape(2, 2)
    f.interpolate(Expression(fvals.astype("string")))

    assert np.allclose(f.dat.data_ro, fvals)
    assert np.allclose(g.dat.data_ro, vals)


def test_firedrake_tensor_function_nonstandard_shape(W_nonstandard_shape):
    f = Function(W_nonstandard_shape)
    vals = np.arange(1, W_nonstandard_shape.dim+1).reshape(f.ufl_shape)
    f.interpolate(Expression(vals.astype("string")))
    assert np.allclose(f.dat.data_ro, vals)

    g = Function(f)
    assert np.allclose(g.dat.data_ro, vals)

    # Check that g is indeed a deep copy
    fvals = vals + 10
    f.interpolate(Expression(fvals.astype("string")))

    assert np.allclose(f.dat.data_ro, fvals)
    assert np.allclose(g.dat.data_ro, vals)


def test_mismatching_rank_interpolation(V):
    f = Function(V)
    with pytest.raises(RuntimeError):
        f.interpolate(Expression(('1', '2')))
    VV = VectorFunctionSpace(V.mesh(), 'CG', 1)
    f = Function(VV)
    with pytest.raises(RuntimeError):
        f.interpolate(Expression(('1', '2')))
    VVV = TensorFunctionSpace(V.mesh(), 'CG', 1)
    f = Function(VVV)
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
