from firedrake import COMM_WORLD
import tempfile
import pytest


@pytest.fixture
def dumpdir():
    comm = COMM_WORLD
    if comm.rank == 0:
        tmp = tempfile.TemporaryDirectory()
        yield comm.bcast(tmp.name, root=0)
        comm.barrier()
        tmp.cleanup()
    else:
        yield comm.bcast(None, root=0)
        comm.barrier()
