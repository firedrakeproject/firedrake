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

    assemble(u * v * dx)

    f.interpolate(Expression("x[0]"))

    assemble(f * v * dx)

    solve(u * v * dx == f * v * dx, out)

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

if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
