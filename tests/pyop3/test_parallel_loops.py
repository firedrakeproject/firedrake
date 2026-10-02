import loopy as lp
import numpy as np
import pytest
from petsc4py import PETSc
from pyrsistent import freeze

import pyop3 as op3
from pyop3.ir import LOOPY_LANG_VERSION, LOOPY_TARGET
from pyop3.utils import just_one


def set_kernel(size, intent):
    return op3.Function(
        lp.make_kernel(
            f"{{ [i]: 0 <= i < {size} }}",
            "y[i] = x[0]",
            [
                lp.GlobalArg("x", int, (1,), is_input=True, is_output=False),
                lp.GlobalArg("y", int, (size,), is_input=False, is_output=True),
            ],
            target=LOOPY_TARGET,
            lang_version=LOOPY_LANG_VERSION,
        ),
        [op3.READ, intent],
    )


@pytest.fixture
def mesh_axis(comm):
    """Return an axis corresponding to an interval mesh distributed between two ranks.

    The mesh looks like the following:

                         r      g  g
             6  2  5  1  4  *   0  3
    [rank 0] x-----x-----x  * -----x
                            *
    [rank 1]             x  * -----x-----x-----x-----x
                         4      0  5  1  6  2  7  3  8
                         g      r  r

    Ghost points (leaves) are marked with "g" and roots with "r".

    The axes are also given an arbitrary numbering.

    """
    # abort in serial
    if comm.size == 1:
        return

    # the sf is created independently of the renumbering
    if comm.rank == 0:
        nroots = 1
        ilocal = [0, 3]
        iremote = [(1, 0), (1, 5)]
    else:
        assert comm.rank == 1
        nroots = 2
        ilocal = [4]
        iremote = [(0, 4)]
    sf = PETSc.SF().create(comm)
    sf.setGraph(nroots, ilocal, iremote)

    # numberings chosen to stress ghost partitioning algorithms
    if comm.rank == 0:
        ncells = 3
        nverts = 4
        numbering = [1, 5, 4, 0, 6, 3, 2]
    else:
        ncells = 4
        nverts = 5
        numbering = [3, 4, 7, 0, 2, 1, 6, 8, 5]
    serial = op3.Axis(
        [op3.AxisComponent(ncells, "cells"), op3.AxisComponent(nverts, "verts")],
        "mesh",
        numbering=numbering,
    )
    return op3.Axis.from_serial(serial, sf)


@pytest.fixture
def cone_map(comm, mesh_axis):
    """Return a map from cells to incident vertices."""
    # abort in serial
    if comm.size == 1:
        return

    ncells = mesh_axis.components[0].count
    nverts = mesh_axis.components[1].count
    arity = 2
    maxes = op3.AxisTree.from_nest(
        {op3.Axis({"cells": ncells}, "mesh"): op3.Axis(arity)},
    )

    if comm.rank == 0:
        mdata = np.asarray([[4, 3], [5, 4], [6, 5]])
    else:
        assert comm.rank == 1
        mdata = np.asarray([[4, 5], [5, 6], [6, 7], [7, 8]])

    # renumber the map
    mdata_renum = np.empty_like(mdata)
    for old_cell in range(ncells):
        # new_cell = cell_renumbering[old_cell]
        new_cell = mesh_axis.default_to_applied_component_number("cells", old_cell)
        for i, old_pt in enumerate(mdata[old_cell]):
            component, old_vert = mesh_axis.axis_to_component_number(old_pt)
            assert component.label == "verts"
            new_vert = mesh_axis.default_to_applied_component_number("verts", old_vert)
            mdata_renum[new_cell, i] = new_vert

    mdat = op3.Dat(maxes, name="cone", data=mdata_renum.flatten())
    return op3.Map(
        {
            freeze({"mesh": "cells"}): [
                op3.TabulatedMapComponent("mesh", "verts", mdat),
            ]
        },
        "cone",
    )


@pytest.mark.parallel(nprocs=2)
# @pytest.mark.parametrize("intent", [op3.INC, op3.MIN, op3.MAX])
@pytest.mark.parametrize(["intent", "fill_value"], [(op3.WRITE, 0), (op3.INC, 0)])
# @pytest.mark.timeout(5)  for now
def test_parallel_loop(comm, paxis, intent, fill_value):
    assert comm.size == 2

    rank_dat = op3.Dat(
        op3.Axis(1), name="rank", data=np.asarray([comm.rank + 1]), dtype=int
    )
    dat = op3.Dat(paxis, data=np.full(paxis.size, fill_value, dtype=int))
    knl = set_kernel(1, intent)

    op3.do_loop(
        p := paxis.index(),
        knl(rank_dat, dat[p]),
    )

    assert np.equal(dat.array._data[: paxis.owned_count], comm.rank + 1).all()
    assert np.equal(dat.array._data[paxis.owned_count :], fill_value).all()

    # since we do not modify ghost points no reduction is needed
    assert dat.array._pending_reduction is None


# can try with P1 and P2
@pytest.mark.parallel(nprocs=2)
@pytest.mark.timeout(5)
def test_parallel_loop_with_map(comm, mesh_axis, cone_map, scalar_copy_kernel):
    assert comm.size == 2
    rank = comm.rank
    other_rank = (comm.rank + 1) % 2

    # could parametrise these
    intent = op3.INC
    fill_value = 0
    write_value = rank + 1
    other_write_value = other_rank + 1

    rank_dat = op3.Dat(
        op3.Axis(1), name="rank", data=np.asarray([write_value]), dtype=int
    )
    dat = op3.Dat(
        mesh_axis, data=np.full(mesh_axis.size, fill_value), dtype=int
    )

    knl = set_kernel(2, intent)

    op3.do_loop(
        c := mesh_axis.as_tree().owned["cells"].index(),
        knl(rank_dat, dat[cone_map(c)]),
    )

    # we now expect the (renumbered) values to look like
    #             1  0  2  0  1  *   0  0
    #    [rank 0] x-----x-----x  * -----x
    #                            *
    #    [rank 1]             x  * -----x-----x-----x-----x
    #                         2  *   0  4  0  4  0  4  0  2
    if comm.rank == 0:
        assert np.count_nonzero(dat.buffer._data == 0) == 4
        assert np.count_nonzero(dat.buffer._data == 1) == 2
        assert np.count_nonzero(dat.buffer._data == 2) == 1
    else:
        assert np.count_nonzero(dat.buffer._data == 0) == 4
        assert np.count_nonzero(dat.buffer._data == 2) == 2
        assert np.count_nonzero(dat.buffer._data == 4) == 3

    # there should be a pending reduction
    assert dat.buffer._pending_reduction == intent
    assert not dat.buffer._roots_valid
    assert not dat.buffer._leaves_valid

    # now do the reduction
    dat.buffer._reduce_leaves_to_roots()
    assert dat.buffer._pending_reduction is None
    assert dat.buffer._roots_valid
    # leaves are still not up-to-date, requires a broadcast
    assert not dat.buffer._leaves_valid

    # we now expect the (renumbered) values to look like
    #             1  0  2  0  3  *   0  0
    #    [rank 0] x-----x-----x  * -----x
    #                            *
    #    [rank 1]             x  * -----x-----x-----x-----x
    #                         2  *   0  4  0  4  0  4  0  2
    if comm.rank == 0:
        assert np.count_nonzero(dat.array._data == 0) == 4
        assert np.count_nonzero(dat.array._data == 1) == 1
        assert np.count_nonzero(dat.array._data == 2) == 1
        assert np.count_nonzero(dat.array._data == 3) == 1
    else:
        assert np.count_nonzero(dat.array._data == 0) == 4
        assert np.count_nonzero(dat.array._data == 2) == 2
        assert np.count_nonzero(dat.array._data == 4) == 3

    # now broadcast to leaves
    dat.array._broadcast_roots_to_leaves()
    assert dat.array._leaves_valid

    # we now expect the (renumbered) values to look like
    #             1  0  2  0  3  *   0  4
    #    [rank 0] x-----x-----x  * -----x
    #                            *
    #    [rank 1]             x  * -----x-----x-----x-----x
    #                         3  *   0  4  0  4  0  4  0  2
    if comm.rank == 0:
        assert np.count_nonzero(dat.array._data == 0) == 3
        assert np.count_nonzero(dat.array._data == 1) == 1
        assert np.count_nonzero(dat.array._data == 2) == 1
        assert np.count_nonzero(dat.array._data == 3) == 1
        assert np.count_nonzero(dat.array._data == 4) == 1
    else:
        assert np.count_nonzero(dat.array._data == 0) == 4
        assert np.count_nonzero(dat.array._data == 2) == 1
        assert np.count_nonzero(dat.array._data == 3) == 1
        assert np.count_nonzero(dat.array._data == 4) == 3


@pytest.mark.parallel(nprocs=2)
@pytest.mark.timeout(5)
def test_same_reductions_commute():
    ...


@pytest.mark.parallel(nprocs=2)
@pytest.mark.timeout(5)
def test_different_reductions_do_not_commute():
    ...
