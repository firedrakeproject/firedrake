from mpi4py import MPI
import pytest
import time


def test_timeout_works():
    time.sleep(1e8)


@pytest.mark.parallel
def test_timeout_works_parallel():
    if MPI.COMM_WORLD.rank == 0:
        MPI.COMM_WORLD.allreduce(1, op=MPI.SUM)
