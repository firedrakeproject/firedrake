from operator import attrgetter

import numpy as np
import pytest
from mpi4py import MPI
from petsc4py import PETSc

import pyop3 as op3


@pytest.fixture
def comm():
    return MPI.COMM_WORLD


@pytest.fixture
def buffer(comm):
    """Return a distributed buffer.

    The point SF for the distributed axis is given by

                  g   g *
    [rank 0]      0---1-*-2---3---4---5
                  |   | * |   |
    [rank 1]  0---1---2-*-3---4
                        * g   g

    where 'g' means a ghost point (leaves of the SF).

    """
    # abort in serial
    if comm.size == 1:
        return

    # build the point SF
    if comm.rank == 0:
        npoints = 6
        ilocal = np.asarray([0, 1], dtype=op3.IntType)
        iremote = np.asarray([(1, i) for i in (1, 2)], dtype=op3.IntType)
    else:
        assert comm.rank == 1
        npoints = 5
        ilocal = np.asarray([3, 4], dtype=op3.IntType)
        iremote = np.asarray([(0, i) for i in (2, 3)], dtype=op3.IntType)
    sf = op3.StarForest.from_graph(npoints, ilocal, iremote, comm=comm)

    # build the DoF SF
    axis_component = op3.AxisComponent(npoints, sf=sf)
    axis = op3.Axis(axis_component)
    axes = op3.AxisTree.from_iterable([axis, op3.Axis(3)])
    return op3.ArrayBuffer(np.arange(axes.local_size), axes.sf)


@pytest.mark.parallel(2)
def test_new_array_has_valid_roots_and_leaves(buffer):
    assert buffer._roots_valid and buffer._leaves_valid


@pytest.mark.parallel(2)
@pytest.mark.parametrize("accessor", ["data_rw", "data_ro", "data_wo"])
def test_accessors_update_roots_and_leaves(comm, buffer, accessor):
    if comm.rank == 0:
        self_num = 1
        other_num = 2
    else:
        assert comm.rank == 1
        self_num = 2
        other_num = 1

    # invalidate root and leaf data
    buffer._current_device_array[...] = self_num
    buffer._leaves_valid = False
    buffer._pending_reduction = op3.INC

    attrgetter(accessor)(buffer)

    # core points (not in SF) should be unchanged
    assert (buffer._current_device_array[buffer.sf.icore] == self_num).all()

    assert buffer._pending_reduction is None

    if accessor == "data_ro":
        assert buffer._leaves_valid
        assert (buffer._current_device_array[buffer.sf.iroot] == self_num + other_num).all()
        assert (buffer._current_device_array[buffer.sf.ileaf] == self_num + other_num).all()

    elif accessor == "data_rw":
        assert not buffer._leaves_valid
        assert (buffer._current_device_array[buffer.sf.iroot] == self_num + other_num).all()
        assert (buffer._current_device_array[buffer.sf.ileaf] == self_num + other_num).all()

    else:
        assert accessor == "data_wo"
        # roots should be considered up-to-date but the pending reduction
        # will have been ignored
        assert (buffer._current_device_array[buffer.sf.iroot] == self_num).all()
        assert (buffer._current_device_array[buffer.sf.ileaf] == self_num).all()
