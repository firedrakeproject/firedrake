import pytest
from firedrake import *
import numpy as np


@pytest.fixture(scope="module",
                params=[False, True],
                ids=["simplex", "quad"])
def mesh(request):
    return UnitSquareMesh(2, 2, quadrilateral=request.param)


@pytest.fixture(params=range(1, 4))
def degree(request):
    return request.param


@pytest.fixture(params=["CG"])
def fs(request):
    return request.param


@pytest.fixture
def dumpfile(tmpdir):
    return str(tmpdir.join("dump.h5"))


def run_store_load(mesh, fs, degree, dumpfile):

    V = FunctionSpace(mesh, fs, degree)

    f = Function(V, name="f")

    expr = Expression("x[0]*x[1]")

    f.interpolate(expr)

    f2 = Function(V, name="f")

    dumpfile = op2.MPI.comm.bcast(dumpfile, root=0)
    chk = DumbCheckpoint(dumpfile, mode=FILE_CREATE)

    chk.store(f)

    chk.load(f2)

    assert np.allclose(f.dat.data_ro, f2.dat.data_ro)


def test_store_load(mesh, fs, degree, dumpfile):
    run_store_load(mesh, fs, degree, dumpfile)


@pytest.mark.parallel(nprocs=2)
def test_store_load_parallel(mesh, fs, degree, dumpfile):
    run_store_load(mesh, fs, degree, dumpfile)


@pytest.fixture
def serial_checkpoint(dumpfile):
    # Make the checkpoint file on rank zero, writing the COMM_SELF (size 1)
    if op2.MPI.comm.rank == 0:
        from mpi4py import MPI
        chk = DumbCheckpoint(dumpfile, mode=FILE_CREATE, comm=MPI.COMM_SELF)
        chk.close()
    # Make sure it's written
    return op2.MPI.comm.bcast(dumpfile, root=0)


@pytest.mark.parallel(nprocs=2)
def test_serial_checkpoint_parallel_load_fails(serial_checkpoint):
    # serial_checkpoint fixture makes the checkpoint file (on one
    # process), which we now try and read on two, and therefore expect
    # an error.
    with pytest.raises(ValueError):
        with DumbCheckpoint(serial_checkpoint, mode=FILE_READ):
            pass


@pytest.mark.parametrize("obj",
                         [lambda: Constant(1),
                          lambda: np.arange(10)])
def test_checkpoint_fails_for_non_function(obj, dumpfile):
    with DumbCheckpoint(dumpfile, mode=FILE_CREATE) as chk:
        with pytest.raises(ValueError):
            chk.store(obj)


def test_checkpoint_read_not_exist_ioerror(dumpfile):
    with pytest.raises(IOError):
        with DumbCheckpoint(dumpfile, mode=FILE_READ):
            pass


if __name__ == "__main__":
    import os
    pytest.main(os.path.abspath(__file__))
