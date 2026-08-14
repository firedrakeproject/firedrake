"""Dof layouts that a ``DMLabel``'s strata induce on a function space.

A :class:`SubspaceLayout` reads the strata of a ``DMLabel``. It reports how
many copies of its dofs each mesh point holds. A point takes one copy from
every stratum that marks it. It takes no dofs at all if no stratum marks it,
which restricts the space to the marked region. The space is therefore
continuous within a stratum, and discontinuous between strata.

The caller must mark every point that carries dofs, not the cells alone.
``DMPlexLabelComplete`` extends a cell marking to its closure. An interface
point then lies in every stratum that touches it. This module reads the
strata as it finds them, and never completes a label itself.
"""

import numpy

from firedrake.petsc import PETSc
from firedrake.utils import IntType


__all__ = ("SubspaceLayout",)


class SubspaceLayout:
    """The dof multiplicity that a label's strata induce on the mesh points.

    Parameters
    ----------
    dm : PETSc.DM
        The topological DM. The layout covers its chart.
    label : PETSc.DMLabel
        The label that marks the subspaces. Its strata may overlap.

    Raises
    ------
    ValueError
        If ``label`` marks no points at all.

    Notes
    -----
    Each point stores its strata as one sorted row of a compressed sparse row
    array. A short scan of that row then gives the copy a stratum owns.
    """

    def __init__(self, dm: PETSc.DM, label: PETSc.DMLabel):
        pStart, pEnd = dm.getChart()
        values = numpy.sort(label.getValueIS().indices).astype(IntType)

        points = []
        strata = []
        for value in values:
            if label.getStratumSize(value) == 0:
                continue
            marked = label.getStratumIS(value).indices
            points.append(marked)
            strata.append(numpy.full(marked.size, value, dtype=IntType))
        if not points:
            raise ValueError("The label marks no points, so the subspace is empty")

        points = numpy.concatenate(points).astype(IntType)
        strata = numpy.concatenate(strata)
        # Order the memberships by point, and by stratum within each point, so
        # that a point's strata form one sorted row
        order = numpy.lexsort((strata, points))
        points = points[order]
        strata = strata[order]

        npoints = pEnd - pStart
        multiplicity = numpy.bincount(points - pStart, minlength=npoints)
        offsets = numpy.zeros(npoints + 1, dtype=IntType)
        numpy.cumsum(multiplicity, out=offsets[1:])

        self.chart = (pStart, pEnd)
        self.values = values
        self.multiplicity = multiplicity.astype(IntType)
        """The number of dof copies each point of the chart carries."""
        self.offsets = offsets
        """The start of each point's row of strata, with ``npoints + 1`` entries."""
        self.strata = strata
        """The strata of each point, one sorted row per point."""

    @property
    def nstrata(self) -> int:
        """The number of strata in the label."""
        return self.values.size

    def node_classes(self, dm: PETSc.DM, nodes_per_entity) -> tuple:
        """Count the nodes of the core, owned and ghost classes.

        Parameters
        ----------
        dm : PETSc.DM
            The topological DM. The layout covers its chart.
        nodes_per_entity : sequence of int
            The number of nodes an element puts on an entity of each
            dimension.

        Returns
        -------
        tuple of int
            The running totals after the core, the owned and the ghost class,
            in the form `~firedrake.mesh.MeshTopology.node_classes` returns
            for a whole space.

        Notes
        -----
        A stratum gives each of its points one copy of the dofs. The count
        therefore weights a point by its multiplicity.
        `~firedrake.functionspacedata.get_node_set` asserts that this total
        matches the storage size of the section.
        """
        pStart, pEnd = self.chart
        # A point's class follows the pyop2 label that holds it
        classes = numpy.full(pEnd - pStart, 2, dtype=IntType)
        for index, name in enumerate(("pyop2_core", "pyop2_owned")):
            if dm.getStratumSize(name, 1) > 0:
                classes[dm.getStratumIS(name, 1).indices - pStart] = index

        counts = numpy.zeros(3, dtype=IntType)
        for dim, nodes in enumerate(nodes_per_entity):
            if nodes == 0:
                continue
            dStart, dEnd = dm.getDepthStratum(dim)
            here = slice(dStart - pStart, dEnd - pStart)
            weights = numpy.bincount(classes[here], weights=self.multiplicity[here],
                                     minlength=3)
            counts += nodes * weights.astype(IntType)
        return tuple(numpy.cumsum(counts))

    def cell_strata(self, cStart: int, cEnd: int) -> numpy.ndarray:
        """Return the stratum of each cell.

        Parameters
        ----------
        cStart, cEnd : int
            The first cell of the chart, and one past the last.

        Returns
        -------
        numpy.ndarray
            The stratum value of each cell. A row of ``-1`` marks a cell
            outside every stratum. Such a cell carries no dofs of the
            subspace, so assembly skips it.

        Raises
        ------
        NotImplementedError
            If a cell lies in more than one stratum. Assembly then visits that
            cell once per stratum, which needs an iteration set of cell and
            stratum pairs. This class iterates over cells.
        """
        pStart, _ = self.chart
        counts = self.multiplicity[cStart - pStart:cEnd - pStart]
        if numpy.any(counts > 1):
            raise NotImplementedError(
                "A cell belongs to more than one stratum. Overlapping subdomains "
                "must overlap on their interfaces, not on their cells.")
        strata = numpy.full(cEnd - cStart, -1, dtype=IntType)
        starts = self.offsets[cStart - pStart:cEnd - pStart]
        held = counts > 0
        strata[held] = self.strata[starts[held]]
        return strata
