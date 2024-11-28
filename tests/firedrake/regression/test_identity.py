import numpy as np
import pytest

from firedrake import *


def identity(family, degree):
    mesh = UnitCubeMesh(3, 3, 3)
    fs = FunctionSpace(mesh, family, degree)
    x = SpatialCoordinate(mesh)

    f = Function(fs)
    out = Function(fs)

    u = TrialFunction(fs)
    v = TestFunction(fs)

    a = inner(u, v) * dx

    f.interpolate(x[0])

    L = inner(f, v) * dx

    solve(a == L, out)

    return norm(assemble(f - out))


def vector_identity(family, degree):
    mesh = UnitSquareMesh(4, 4)
    fs = VectorFunctionSpace(mesh, family, degree)
    x = SpatialCoordinate(mesh)
    f = Function(fs)
    out = Function(fs)
    u = TrialFunction(fs)
    v = TestFunction(fs)

    f.interpolate(as_vector((x[0], x[1])))
    solve(inner(u, v)*dx == inner(f, v)*dx, out)

    return norm(assemble(f - out))


def tensor_identity(family, degree):
    mesh = UnitSquareMesh(4, 4)
    fs = TensorFunctionSpace(mesh, family, degree)
    x = SpatialCoordinate(mesh)
    f = Function(fs)
    out = Function(fs)
    u = TrialFunction(fs)
    v = TestFunction(fs)

    f.interpolate(as_tensor([(x[0], x[1]), (x[0], x[1])]))
    solve(inner(u, v)*dx == inner(f, v)*dx, out)

    return norm(assemble(f - out))


def tensor_identity_nonstandard_shape(family, degree):
    mesh = UnitSquareMesh(4, 4)
    fs = TensorFunctionSpace(mesh, family, degree, shape=(2, 3))
    x = SpatialCoordinate(mesh)
    f = Function(fs)
    out = Function(fs)
    u = TrialFunction(fs)
    v = TestFunction(fs)

    f.interpolate(as_tensor([(x[0], x[1], x[0]*x[1]), (x[0], x[1], x[0]*x[1])]))
    solve(inner(u, v)*dx == inner(f, v)*dx, out)

    return norm(assemble(f - out))


def run_test():
    family = "Lagrange"
    degree = range(1, 5)
    return np.array([identity(family, d) for d in degree])


def run_vector_test():
    family = "Lagrange"
    degree = range(1, 5)
    return np.array([vector_identity(family, d) for d in degree])


def run_tensor_test():
    family = "Lagrange"
    degree = range(1, 5)
    return np.array([tensor_identity(family, d) for d in degree])


def run_tensor_test_nonstandard_shape():
    family = "Lagrange"
    degree = range(1, 5)
    return np.array([tensor_identity_nonstandard_shape(family, d) for d in degree])


def test_identity():
    assert (run_test() < 1e-6).all()


def test_vector_identity():
    assert (run_vector_test() < 1e-6).all()


def test_tensor_identity():
    assert (run_tensor_test() < 1e-6).all()


def test_tensor_identity_nonstandard_shape():
    assert (run_tensor_test_nonstandard_shape() < 1e-6).all()


@pytest.mark.parallel
def test_identity_parallel():
    assert (run_test() < 1e-6).all()


@pytest.mark.parallel(nprocs=2)
def test_vector_identity_parallel():
    assert (run_vector_test() < 1e-6).all()


@pytest.mark.parallel(nprocs=2)
def test_tensor_identity_parallel():
    assert (run_tensor_test() < 1e-6).all()


@pytest.mark.parallel(nprocs=2)
def test_tensor_identity_nonstandard_shape_parallel():
    assert (run_tensor_test_nonstandard_shape() < 1e-6).all()
