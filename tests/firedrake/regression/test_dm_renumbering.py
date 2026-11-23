"""Tests for the point renumbering computed by ``dmcommon.compute_dm_renumbering``.

The renumbering visits the cells in the order given by a cell permutation and
numbers the points of each cell closure the first time that they are seen.
"""
import numpy as np
import pytest
from petsc4py import PETSc

from firedrake import COMM_SELF, UnitCubeMesh, UnitSquareMesh
from firedrake.cython import dmcommon


class CountingIS(PETSc.IS):
    """An index set that counts the Python accesses to its indices.

    ``IS.indices``, ``IS.getIndices`` and ``IS.array`` each return a new
    NumPy array. A copy of the whole cell permutation for each cell makes the
    renumbering cost grow with the square of the number of cells, so the
    renumbering must read the permutation through the C pointer instead.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.naccess = 0

    @property
    def indices(self):
        self.naccess += 1
        # Call the base method directly: the base property calls
        # self.getIndices, which would count this access twice.
        return PETSc.IS.getIndices(self)

    def getIndices(self):
        self.naccess += 1
        return super().getIndices()

    @property
    def array(self):
        self.naccess += 1
        return super().array


def reference_renumbering(dm, new_to_old_cell):
    """Return the renumbering of ``compute_dm_renumbering``, computed in Python.

    Parameters
    ----------
    dm : PETSc.DMPlex
        The mesh topology.
    new_to_old_cell : numpy.ndarray
        The cell permutation: entry ``c`` is the plex cell that is visited
        in position ``c``.

    Returns
    -------
    numpy.ndarray
        The plex points in the order of the first visit (new to old).
    """
    seen = set()
    ordering = []
    for cell in new_to_old_cell:
        closure, _ = dm.getTransitiveClosure(cell)
        for p in closure:
            if p not in seen:
                seen.add(p)
                ordering.append(p)
    return np.asarray(ordering)


@pytest.mark.parametrize("make_mesh", [
    lambda: UnitSquareMesh(6, 5, comm=COMM_SELF),
    lambda: UnitCubeMesh(3, 2, 2, comm=COMM_SELF),
], ids=["triangle", "tetrahedron"])
def test_compute_dm_renumbering_reads_permutation_once(make_mesh):
    """The renumbering follows the cell permutation and does not copy it for each cell."""
    topology = make_mesh().topology
    dm = topology.topology_dm
    cStart, cEnd = dm.getHeightStratum(0)

    # A fixed, non-trivial cell permutation.
    rng = np.random.default_rng(seed=5)
    new_to_old_cell = rng.permutation(np.arange(cStart, cEnd, dtype=PETSc.IntType))
    cell_is = CountingIS().createGeneral(new_to_old_cell, comm=COMM_SELF)

    renumbering = dmcommon.compute_dm_renumbering(topology, cell_is)

    assert np.array_equal(renumbering.indices, reference_renumbering(dm, new_to_old_cell))
    # At most one copy of the permutation, independent of the number of cells.
    assert cell_is.naccess <= 1


def test_compute_dm_renumbering_without_permutation():
    """Without a permutation the cells are visited in plex order."""
    topology = UnitSquareMesh(4, 3, comm=COMM_SELF).topology
    dm = topology.topology_dm
    cStart, cEnd = dm.getHeightStratum(0)

    renumbering = dmcommon.compute_dm_renumbering(topology, None)

    assert np.array_equal(renumbering.indices, reference_renumbering(dm, np.arange(cStart, cEnd)))


def test_compute_dm_renumbering_wrong_permutation_size():
    """A cell permutation of the wrong size gives an error."""
    topology = UnitSquareMesh(2, 2, comm=COMM_SELF).topology
    dm = topology.topology_dm
    cStart, cEnd = dm.getHeightStratum(0)
    cell_is = PETSc.IS().createGeneral(np.arange(cStart, cEnd - 1, dtype=PETSc.IntType), comm=COMM_SELF)

    with pytest.raises(ValueError):
        dmcommon.compute_dm_renumbering(topology, cell_is)
