import os
import tempfile

import pytest
from mpi4py import MPI


@pytest.fixture
def dumpdir():
    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        tmp = tempfile.TemporaryDirectory()
        yield comm.bcast(tmp.name, root=0)
        comm.barrier()
        tmp.cleanup()
    else:
        yield comm.bcast(None, root=0)
        comm.barrier()


@pytest.fixture
def dumpfile(dumpdir):
    yield os.path.join(dumpdir, "dump")
