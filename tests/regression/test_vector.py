from firedrake import *
import pytest
import numpy as np


@pytest.fixture
def f():
    mesh = UnitIntervalMesh(2)
    V = FunctionSpace(mesh, "CG", 1)
    f = Function(V)
    f.interpolate(Constant(1))
    return f


def test_vector_array(f):
    v = f.vector()
    assert (v.array() == 1.0).all()


def test_vector_setitem(f):
    v = f.vector()
    v[:] = 2.0

    assert (v.array() == 2.0).all()


def test_vector_getitem(f):
    v = f.vector()
    assert v[0] == 1.0


def test_vector_len(f):
    v = f.vector()
    assert len(v) == f.dof_dset.size


def test_vector_returns_copy(f):
    v = f.vector()
    a = v.array()
    a[:] = 5.0
    assert v.array() is not a
    assert (v.array() == 1.0).all()
    assert (a == 5.0).all()


def test_mixed_vector_copy():
    mesh = UnitIntervalMesh(2)
    V = FunctionSpace(mesh, "CG", 1)
    W = V*V
    f = Function(W)
    f.assign(1)

    v = f.vector()
    assert np.allclose(v.array(), 1.0)

    copy = v.copy()

    assert isinstance(copy, Vector)
    assert np.allclose(copy.array(), 1.0)

    local = copy.get_local()
    local[:] = 10.0
    copy.set_local(local)

    assert np.allclose(copy.array(), 10.0)

    assert np.allclose(v.array(), 1.0)


def test_vector_gather_works(f):
    f.interpolate(Constant(2))
    v = f.vector()
    gathered = v.gather([0])
    assert len(gathered) == 1 and gathered[0] == 2.0


def test_axpy(f):
    f.interpolate(Constant(2))
    v = f.vector()
    y = Vector(v)
    y[:] = 4

    v.axpy(3, y)

    assert (v.array() == 14.0).all()


def test_addition(f):
    f.interpolate(Constant(2))
    v = f.vector()
    y = Vector(v)
    w = v + y
    assert (w.array() == 4.).all()

    w = v + 3.
    assert (w.array() == 5.).all()

    w = 3. + v
    assert (w.array() == 5.).all()


def test_iadd(f):
    f.interpolate(Constant(2))
    v = f.vector()
    y = Vector(v)
    v += y
    assert (v.array() == 4.).all()


def test_subtraction(f):
    f.interpolate(Constant(2))
    v = f.vector()
    y = Vector(v)
    w = v - y
    assert (w.array() == 0.).all()

    w = v - 3.0
    assert (w.array() == -1.).all()

    w = 3.0 - v
    assert (w.array() == 1.).all()


def test_isub(f):
    f.interpolate(Constant(2))
    v = f.vector()
    y = Vector(v)
    v -= y
    assert (v.array() == 0.).all()


def test_scale(f):
    f.interpolate(Constant(3))
    v = f.vector()
    v._scale(7)

    assert (v.array() == 21.0).all()


@pytest.mark.parallel(nprocs=2)
def test_parallel_gather():
    from mpi4py import MPI
    mesh = UnitSquareMesh(2, 2)
    V = FunctionSpace(mesh, "CG", 1)
    f = Function(V)
    v = f.vector()
    rank = MPI.COMM_WORLD.rank
    v[:] = rank

    assert (f.dat.data_ro[:f.dof_dset.size] == rank).all()

    lsum = sum(v.array())
    lsum = MPI.COMM_WORLD.allreduce(lsum, op=MPI.SUM)
    gathered = v.gather()
    gsum = sum(gathered)
    assert lsum == gsum
    assert len(gathered) == v.size()

    gathered = v.gather([0])
    assert len(gathered) == 1 and gathered[0] == 0
