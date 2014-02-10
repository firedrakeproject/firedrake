import numpy as np
import pytest

from firedrake import *


def identity(family, degree):
    mesh = UnitCubeMesh(1, 1, 1)
    fs = FunctionSpace(mesh, family, degree)

    f = Function(fs)
    out = Function(fs)

    u = TrialFunction(fs)
    v = TestFunction(fs)

    a = u * v * dx

    f.interpolate(Expression("x[0]"))

    L = f * v * dx

    solve(a == L, out)

    return np.max(np.abs(out.dat.data - f.dat.data))


def vector_identity(family, degree):
    mesh = firedrake.UnitSquareMesh(2, 2)
    fs = firedrake.VectorFunctionSpace(mesh, family, degree)
    f = firedrake.Function(fs)
    out = firedrake.Function(fs)
    u = firedrake.TrialFunction(fs)
    v = firedrake.TestFunction(fs)

    f.interpolate(firedrake.Expression(("x[0]", "x[1]")))
    firedrake.solve(firedrake.inner(u, v)*firedrake.dx == firedrake.inner(f, v)*firedrake.dx, out)

    return np.max(np.abs(out.dat.data - f.dat.data))


def run_test():
    family = "Lagrange"
    degree = range(1, 5)
    return np.array([identity(family, d) for d in degree])


def run_vector_test():
    family = "Lagrange"
    degree = range(1, 5)
    return np.array([vector_identity(family, d) for d in degree])


def test_firedrake_identity():
    assert (run_test() < 1e-6).all()


def test_vector_identity():
    assert (run_vector_test() < 1e-6).all()


@pytest.mark.parallel
def test_firedrake_identity_parallel():
    from mpi4py import MPI
    error = run_test()
    MPI.COMM_WORLD.allreduce(MPI.IN_PLACE, error, MPI.MAX)
    print '[%d]' % MPI.COMM_WORLD.rank, 'error:', error
    assert (error < np.array([1.0e-11, 1.0e-6, 1.0e-6, 1.0e-5])).all()


@pytest.mark.parallel(nprocs=2)
def test_vector_identity_parallel():
    from mpi4py import MPI
    error = run_vector_test()
    MPI.COMM_WORLD.allreduce(MPI.IN_PLACE, error, MPI.MAX)
    print '[%d]' % MPI.COMM_WORLD.rank, 'error:', error
    assert (error < 1e-6).all()


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
