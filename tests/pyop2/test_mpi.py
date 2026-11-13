import pytest
import pytest_mpi
from mpi4py import MPI

import pyop2.mpi


def passing_test():
    return "pass"


def failing_test():
    raise RuntimeError("This test has failed")


@pytest.mark.parallel(2)
@pytest.mark.parametrize("root", [0, 1])
def test_branches_on_rank_do_not_deadlock(root):
    result = pyop2.mpi.safe_noncollective(MPI.COMM_WORLD, passing_test, root=root)
    pytest_mpi.parallel_assert(result == "pass")

    try:
        result = pyop2.mpi.safe_noncollective(MPI.COMM_WORLD, failing_test, root=root)
    except BaseException as e:
        result = e
    pytest_mpi.parallel_assert(isinstance(result, RuntimeError) and str(result) == "This test has failed")
