import gc
import re

import pytest
from mpi4py import MPI
from pyop3.mpi import temp_internal_comm
from pytest_mpi.parallel_assert import parallel_assert

from firedrake import *
from firedrake.petsc import garbage_cleanup


@pytest.mark.parallel(2)
def test_making_many_meshes_does_not_exhaust_comms():
    # Clean up garbage first, in case the test suite is already using lots of comms
    garbage_cleanup(COMM_WORLD)

    for ii in range(3000):
        mesh = UnitIntervalMesh(2)
        if ii % 800 == 0:
            garbage_cleanup(mesh)

    # Clean up garbage after too
    garbage_cleanup(COMM_WORLD)


# perform this test last so it will catch anything leaking from earlier tests
@pytest.mark.order("last")
@pytest.mark.parallel(3)
def test_no_petsc_objects_on_private_comm(request, capfd):
    """Check that PETSc objects are being created with the correct comm.

    If objects are being created using Firedrake's private communicator then
    they will not be destroyed using `PETSc.garbage_cleanup`.

    """
    # Put ref cycle objects into the garbage
    gc.collect()

    with temp_internal_comm(MPI.COMM_WORLD) as private_comm:
        PETSc.garbage_view(private_comm)
    captured = MPI.COMM_WORLD.bcast(capfd.readouterr().out)

    pattern = r"Rank \d+:: Total entries: (\d+)"
    all_zero = True
    nhits = 0
    for line in captured.splitlines():
        if match := re.fullmatch(pattern, line):
            nhits += 1
            if match.groups()[0] != "0":
                all_zero = False
    parallel_assert(nhits == MPI.COMM_WORLD.size)
    parallel_assert(
        all_zero,
        msg=f"Objects found on private communicator, got:\n{captured}"
    )
