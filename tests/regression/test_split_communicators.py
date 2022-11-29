import pytest
import numpy as np
from firedrake import *
from mpi4py import MPI


@pytest.mark.parallel(nprocs=4)
def test_split_communicators():

    wcomm = COMM_WORLD

    if wcomm.rank == 0:
        # On rank zero, we build a unit triangle,
        wcomm.Split(MPI.UNDEFINED)

        m = UnitTriangleMesh(comm=COMM_SELF)
        V = FunctionSpace(m, 'DG', 0)

        u = TrialFunction(V)
        v = TestFunction(V)

        volume = assemble(inner(u, v) * dx).M.values

        assert np.allclose(volume, 0.5)
    else:
        # On the other ranks, we'll build a collective mesh
        comm = wcomm.Split(0)

        m = UnitSquareMesh(4, 4, quadrilateral=True, comm=comm)

        V = VectorFunctionSpace(m, 'DG', 0)

        f = Function(V)

        u = TrialFunction(V)
        v = TestFunction(V)

        const = Constant((1, 0), domain=m)
        solve(inner(u, v) * dx == inner(const, v) * dx, f)

        expect = Function(V).interpolate(const)
        assert np.allclose(expect.dat.data, f.dat.data)
