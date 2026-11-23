import pytest
import numpy as np

from firedrake import *
from firedrake.utils import _new_uid
from functools import partial


def make_mesh(comm):
    return UnitSquareMesh(5, 5, comm=comm)


def make_function(comm):
    mesh = UnitSquareMesh(5, 5, comm=comm)
    V = FunctionSpace(mesh, 'Lagrange', 1)
    return Function(V)


def make_cofunction(comm):
    mesh = UnitSquareMesh(5, 5, comm=comm)
    V = FunctionSpace(mesh, 'Lagrange', 1)
    return Cofunction(V.dual())


def make_constant(comm):
    return Constant(677)


@pytest.fixture(
    params=[Function, Cofunction, Constant, UnitSquareMesh],
    ids=["Function", "Cofunction", "Constant", "Mesh"]
)
def obj(request):
    ''' Make a callable for creating a Firedrake object
    '''
    case = {
        Function: make_function,
        Cofunction: make_cofunction,
        Constant: make_constant,
        UnitSquareMesh: make_mesh
    }
    return partial(case[request.param])


# Attempting to solve this has been a tremendous headache. Combined with
# tests/firedrake/adjoint/test_disk_checkpointing.py::test_disk_checkpointing_parallel
# the UIDs will sometimes go out of sync. I have tracked it down to
# pyadjoint.BlockVariable.save_output() where BlockVariable.checkpoint is either
# None or a Function. Reproducing it is really difficult.
@pytest.mark.skip(reason="Something is afoot")
@pytest.mark.parallel(nprocs=[1, 2, 3, 4])
def test_monotonic_uid(obj):
    object_parallel = obj(comm=COMM_WORLD)  # noqa: F841

    if COMM_WORLD.rank == 0:
        object_serial = obj(comm=COMM_SELF)  # noqa: F841

    for comm in [COMM_WORLD, COMM_SELF]:
        new = np.array([_new_uid(comm)])
        all_new = np.array([-1]*comm.size)
        comm.Allgather(new, all_new)
        assert all([a == all_new[comm.rank] for a in all_new])
