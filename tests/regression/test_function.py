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


@pytest.fixture
def Rvector():
    mesh = UnitSquareMesh(5, 5)
    return VectorFunctionSpace(mesh, "R", 0, dim=4)


def test_firedrake_scalar_function(V):
    f = Function(V)
    f.interpolate(Constant(1))
    assert (f.dat.data_ro == 1.0).all()

    g = Function(f)
    assert (g.dat.data_ro == 1.0).all()

    # Check that g is indeed a deep copy
    f.interpolate(Constant(2))

    assert (f.dat.data_ro == 2.0).all()
    assert (g.dat.data_ro == 1.0).all()


def test_firedrake_tensor_function(W):
    f = Function(W)
    vals = np.array([1.0, 2.0, 10.0, 20.0]).reshape(2, 2)
    f.interpolate(as_tensor(vals))
    assert np.allclose(f.dat.data_ro, vals)

    g = Function(f)
    assert np.allclose(g.dat.data_ro, vals)

    # Check that g is indeed a deep copy
    fvals = np.array([5.0, 6.0, 7.0, 8.0]).reshape(2, 2)
    f.interpolate(as_tensor(fvals))

    assert np.allclose(f.dat.data_ro, fvals)
    assert np.allclose(g.dat.data_ro, vals)


def test_firedrake_tensor_function_nonstandard_shape(W_nonstandard_shape):
    f = Function(W_nonstandard_shape)
    vals = np.arange(1.0, W_nonstandard_shape.value_size+1).reshape(f.ufl_shape)
    f.interpolate(as_tensor(vals))
    assert np.allclose(f.dat.data_ro, vals)

    g = Function(f)
    assert np.allclose(g.dat.data_ro, vals)

    # Check that g is indeed a deep copy
    fvals = vals + 10
    f.interpolate(as_tensor(fvals))

    assert np.allclose(f.dat.data_ro, fvals)
    assert np.allclose(g.dat.data_ro, vals)


def test_mismatching_rank_interpolation(V):
    f = Function(V)
    with pytest.raises(RuntimeError):
        f.interpolate(Constant((1, 2)))
    VV = VectorFunctionSpace(V.mesh(), 'CG', 1)
    f = Function(VV)
    with pytest.raises(RuntimeError):
        f.interpolate(Constant((1, 2)))
    VVV = TensorFunctionSpace(V.mesh(), 'CG', 1)
    f = Function(VVV)
    with pytest.raises(RuntimeError):
        f.interpolate(Constant((1, 2)))


def test_mismatching_shape_interpolation(V):
    VV = VectorFunctionSpace(V.mesh(), 'CG', 1)
    f = Function(VV)
    with pytest.raises(RuntimeError):
        f.interpolate(Constant([1] * (VV.value_shape[0] + 1)))


def test_function_val(V):
    """Initialise a Function with a NumPy array."""
    f = Function(V, np.ones((V.node_count, V.value_size)))
    assert (f.dat.data_ro == 1.0).all()


def test_function_dat(V):
    """Initialise a Function with an op2.Dat."""
    f = Function(V, op2.Dat(V.node_set**V.value_size))
    f.interpolate(Constant(1))
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


def test_copy(V):
    f = Function(V, name="foo")
    f.assign(1)
    g = f.copy(deepcopy=True)

    assert f.name() == "foo"
    assert g.name() == "foo"

    assert np.allclose(f.dat.data_ro, 1.0)
    assert np.allclose(g.dat.data_ro, 1.0)

    g.assign(2)
    assert np.allclose(g.dat.data_ro, 2.0)
    assert np.allclose(f.dat.data_ro, 1.0)

    h = f.copy()

    assert np.allclose(h.dat.data_ro, 1.0)

    h.assign(3.0)
    assert np.allclose(h.dat.data_ro, 3.0)
    assert np.allclose(f.dat.data_ro, 3.0)

    assert h.name() == "foo"


def test_scalar_function_zero(V):
    f = Function(V)

    f.assign(1)
    assert np.allclose(f.dat.data_ro, 1.0)

    g = f.zero()
    assert f is g
    assert np.allclose(f.dat.data_ro, 0.0)


def test_scalar_function_zero_with_subset(V):
    f = Function(V)
    # create an arbitrary subset consisting of the first two nodes
    assert V.node_set.size > 2
    subset = op2.Subset(V.node_set, [0, 1])

    f.assign(1)
    assert np.allclose(f.dat.data_ro, 1.0)

    f.zero(subset=subset)
    assert np.allclose(f.dat.data_ro[:2], 0.0)
    assert np.allclose(f.dat.data_ro[2:], 1.0)


def test_tensor_function_zero(W):
    f = Function(W)

    f.assign(1)
    assert np.allclose(f.dat.data_ro, 1.0)

    g = f.zero()
    assert f is g
    assert np.allclose(f.dat.data_ro, 0.0)


def test_tensor_function_zero_with_subset(W):
    f = Function(W)
    # create an arbitrary subset consisting of the first three nodes
    assert W.node_set.size > 3
    subset = op2.Subset(W.node_set, [0, 1, 2])

    f.assign(1)
    assert np.allclose(f.dat.data_ro, 1.0)

    f.zero(subset=subset)
    assert np.allclose(f.dat.data_ro[:3], 0.0)
    assert np.allclose(f.dat.data_ro[3:], 1.0)


def test_component_function_zero(W):
    f = Function(W)

    f.assign(1)
    assert np.allclose(f.dat.data_ro, 1.0)

    g = f.sub(0).zero()
    assert f.sub(0) is g
    for i, j in np.ndindex(f.dat.data_ro.shape[1:]):
        expected = 0.0 if i == 0 and j == 0 else 1.0
        assert np.allclose(f.dat.data_ro[..., i, j], expected)


def test_component_function_zero_with_subset(W):
    f = Function(W)
    # create an arbitrary subset consisting of the first three nodes
    assert W.node_set.size > 3
    subset = op2.Subset(W.node_set, [0, 1, 2])

    f.assign(1)
    assert np.allclose(f.dat.data_ro, 1.0)

    f.sub(0).zero(subset=subset)
    for i, j in np.ndindex(f.dat.data_ro.shape[1:]):
        expected = 0.0 if i == 0 and j == 0 else 1.0
        assert np.allclose(f.dat.data_ro[:3, i, j], expected)
        assert np.allclose(f.dat.data_ro[3:, i, j], 1.0)


@pytest.mark.parametrize("value", [
    1,
    1.0,
    (1, 2, 3, 4),
    [1, 2, 3, 4],
    np.array([5, 6, 7, 8]),
    range(4)], ids=type)
def test_vector_real_space_assign(Rvector, value):
    f = Function(Rvector)
    f.assign(value)
    assert np.allclose(f.dat.data_ro, value)


def test_vector_real_space_assign_function(Rvector):
    value = [9, 10, 11, 12]
    fvalue = Function(Rvector, val=value)
    f = Function(Rvector)
    f.assign(fvalue)
    assert np.allclose(f.dat.data_ro, value)


def test_vector_real_space_assign_constant(Rvector):
    value = [9, 10, 11, 12]
    fvalue = Constant(value)
    f = Function(Rvector)
    f.assign(fvalue)
    assert np.allclose(f.dat.data_ro, value)


def test_vector_real_space_assign_zero(Rvector):
    f = Function(Rvector, val=[9, 10, 11, 12])
    f.assign(zero())
    assert np.allclose(f.dat.data_ro, 0)
