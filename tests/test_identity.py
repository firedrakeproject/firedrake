import firedrake
import numpy as np
import pytest


def identity(family, degree):
    mesh = firedrake.UnitCubeMesh(1, 1, 1)
    fs = firedrake.FunctionSpace(mesh, family, degree)

    f = firedrake.Function(fs)
    out = firedrake.Function(fs)

    u = firedrake.TrialFunction(fs)
    v = firedrake.TestFunction(fs)

    firedrake.assemble(u * v * firedrake.dx)

    f.interpolate(firedrake.Expression("x[0]"))

    firedrake.assemble(f * v * firedrake.dx)

    firedrake.solve(u * v * firedrake.dx == f * v * firedrake.dx, out)

    return np.max(np.abs(out.dat.data - f.dat.data))


def run_test():
    family = "Lagrange"
    degree = range(1, 5)
    return np.array([identity(family, d) for d in degree])


def test_firedrake_identity():
    assert (run_test() < np.array([1.0e-13, 1.0e-6, 1.0e-6, 1.0e-5])).all()


@pytest.mark.parallel
def test_firedrake_identity_parallel():
    from mpi4py import MPI
    error = run_test()
    MPI.COMM_WORLD.allreduce(MPI.IN_PLACE, error, MPI.MAX)
    print '[%d]' % MPI.COMM_WORLD.rank, 'error:', error
    assert (error < np.array([1.0e-13, 1.0e-6, 1.0e-6, 1.0e-5])).all()
