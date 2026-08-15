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
from mpi4py import MPI

from firedrake.petsc import PETSc
from firedrake.utils import IntType


__all__ = ("SubspaceLayout", "subspace", "complete_strata",
           "parent_local_to_global_map")


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
        from firedrake.cython import dmcommon

        pStart, pEnd = dm.getChart()
        values, multiplicity, offsets, strata = dmcommon.label_point_strata(label, pStart, pEnd)
        if strata.size == 0:
            raise ValueError("The label marks no points, so the subspace is empty")

        self.chart = (pStart, pEnd)
        self.values = values
        self.multiplicity = multiplicity
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


def subspace(V, cellwise: bool = False, label=None):
    """Return the subspace that gives each subdomain its own copy of the dofs.

    Parameters
    ----------
    V : WithGeometry
        The space to break apart.
    cellwise : bool
        Whether to take each cell as a subdomain of its own.
    label : PETSc.DMLabel
        A label whose strata mark the subdomains, or None to take the cells of
        a process as one subdomain.

    Returns
    -------
    WithGeometry
        A subspace that duplicates the dofs on a subdomain interface, one copy
        per subdomain that touches it. This is ``V`` itself when a process
        holds a single subdomain.
    """
    if label is not None:
        from firedrake.functionspace import SubspaceDecomposition
        return SubspaceDecomposition(V, label)
    elif cellwise:
        return V.broken_space()
    else:
        return V


def complete_strata(mesh, label, name=None) -> PETSc.DMLabel:
    """Complete the strata of a shared point with those of the other processes.

    Parameters
    ----------
    mesh : MeshTopology
        The mesh whose points the label marks.
    label : PETSc.DMLabel
        A label in which each process marks the closure of its own cells only,
        so that no stratum spans a process boundary.
    name : str
        The name of the new label, by default that of ``label``.

    Returns
    -------
    PETSc.DMLabel
        The same label, with a shared point carrying the same strata on every
        process that holds it.

    Notes
    -----
    ``DMPlexLabelComplete`` extends a marking across the closure of a point.
    This extends one across the processes that hold the point.

    A subdomain refines the partition that distributed the mesh, so no two
    processes contribute the same stratum to a shared point. Their counts at
    that point add up instead of overlapping, and the point star forest can
    therefore sum them onto the owner and hand the total back.

    The strata themselves travel as one row per point, of a fixed width that
    the largest of those totals sets. The owner merges the rows of every
    process holding the point, then broadcasts the merged row back.

    Without this, two processes disagree on how many dofs a shared point
    carries, and the global section built from their local ones is wrong.
    """
    from firedrake.cython import dmcommon

    plex = mesh.topology_dm
    sf = plex.getPointSF()
    pStart, pEnd = plex.getChart()
    npoints = pEnd - pStart
    unit = MPI._typedict[numpy.dtype(IntType).char]

    _, multiplicity, offsets, strata = dmcommon.label_point_strata(label, pStart, pEnd)

    # The contributions do not overlap, so the counts of a shared point add up
    total = multiplicity.copy()
    sf.reduceBegin(unit, multiplicity, total, MPI.SUM)
    sf.reduceEnd(unit, multiplicity, total, MPI.SUM)
    sf.bcastBegin(unit, total, total, MPI.REPLACE)
    sf.bcastEnd(unit, total, total, MPI.REPLACE)
    width = max(mesh.comm.allreduce(int(total.max(initial=0)), op=MPI.MAX), 1)

    rows = _pack_rows(npoints, width, multiplicity, offsets, strata)

    # A single process shares no point, so its own strata are already complete
    if mesh.comm.size > 1:
        rows = _merge_shared_rows(sf, rows, unit)

    return _label_from_rows(name or label.getName(), pStart, rows)


def _merge_shared_rows(sf, rows, unit) -> numpy.ndarray:
    """Merge the rows of strata that the processes sharing a point hold.

    Parameters
    ----------
    sf : PETSc.SF
        The point star forest of the mesh, whose graph names the sharers of
        each point.
    rows : numpy.ndarray
        The strata of each point, one fixed width row per point.
    unit : MPI.Datatype
        The datatype of one entry of ``rows``.

    Returns
    -------
    numpy.ndarray
        The same array, with a shared point carrying the strata of every
        process that holds it.

    Notes
    -----
    The owner of a point collects the rows of its sharers, merges them, and
    broadcasts the merged row back.
    """
    npoints, width = rows.shape
    block = unit.Create_contiguous(width)
    block.Commit()
    try:
        degree = sf.computeDegree()
        gathered = numpy.full((int(degree.sum()), width), -1, dtype=IntType)
        sf.gatherBegin(block, rows, gathered)
        sf.gatherEnd(block, rows, gathered)

        points = numpy.concatenate([numpy.repeat(numpy.arange(npoints, dtype=IntType), width),
                                    numpy.repeat(numpy.repeat(numpy.arange(npoints, dtype=IntType),
                                                              degree), width)])
        values = numpy.concatenate([rows.ravel(), gathered.ravel()])
        held = values >= 0
        points, values = points[held], values[held]
        order = numpy.lexsort((values, points))
        points, values = points[order], values[order]
        keep = numpy.ones(points.size, dtype=bool)
        keep[1:] = (points[1:] != points[:-1]) | (values[1:] != values[:-1])
        points, values = points[keep], values[keep]

        multiplicity = numpy.bincount(points, minlength=npoints).astype(IntType)
        offsets = numpy.zeros(npoints + 1, dtype=IntType)
        numpy.cumsum(multiplicity, out=offsets[1:])
        rows = _pack_rows(npoints, width, multiplicity, offsets, values)

        sf.bcastBegin(block, rows, rows, MPI.REPLACE)
        sf.bcastEnd(block, rows, rows, MPI.REPLACE)
    finally:
        block.Free()
    return rows


def _pack_rows(npoints, width, multiplicity, offsets, strata) -> numpy.ndarray:
    """Lay a compressed sparse row array out as one fixed width row per point."""
    rows = numpy.full((npoints, width), -1, dtype=IntType)
    point = numpy.repeat(numpy.arange(npoints, dtype=IntType), multiplicity)
    column = numpy.arange(strata.size, dtype=IntType) - numpy.repeat(offsets[:-1], multiplicity)
    rows[point, column] = strata
    return rows


def _label_from_rows(name, pStart, rows) -> PETSc.DMLabel:
    """Build a label from one fixed width row of strata per point."""
    npoints, width = rows.shape
    points = numpy.repeat(numpy.arange(pStart, pStart + npoints, dtype=IntType), width)
    values = rows.ravel()
    held = values >= 0
    points, values = points[held], values[held]
    order = numpy.argsort(values, kind="stable")
    points, values = points[order], values[order]

    label = PETSc.DMLabel().create(name, comm=PETSc.COMM_SELF)
    unique, starts = numpy.unique(values, return_index=True)
    label.addStrata(unique)
    # One pass per stratum, since a label resolves a stratum in constant time
    # but scans them all to answer for a point
    for value, points_of in zip(unique, numpy.split(points, starts[1:])):
        label.setStratumIS(value, PETSc.IS().createGeneral(points_of, comm=PETSc.COMM_SELF))
    return label


def parent_local_to_global_map(space, parent) -> PETSc.LGMap:
    """Map the local nodes of a subspace to the global nodes of its parent.

    Parameters
    ----------
    space : WithGeometry
        A subspace of ``parent``, such as a decomposition or a broken space.
        Its nodes may duplicate a node of the parent, and may leave one out.
    parent : WithGeometry
        The space that ``space`` was built from.

    Returns
    -------
    PETSc.LGMap
        The global node of the parent that each local node of ``space``
        duplicates. A node that no owned cell touches at all takes ``-1``.

    Notes
    -----
    A subdomain matrix assembled on ``space`` sums to the operator on
    ``parent`` under this map, which is what ``MatIS`` asks of its own map.

    The two spaces share a mesh and an element, so entry ``(c, j)`` of either
    cell node map names the same node of the same cell. One scatter of the
    parent's global numbers through the subspace's map therefore builds the
    whole map, with no loop over cells.

    An extruded cell node map stores the base layer alone, so the two maps
    walk their columns in step, through the same
    `~pyop2.types.mat.layer_nodes` that `~pyop2.types.mat.mask_ghost_cells`
    uses. The mask comes from that same function, which also decides the size
    of the local matrix, so the two cannot disagree on which nodes the
    subdomain holds.
    """
    from pyop2.types.mat import layer_nodes, mask_ghost_cells

    cell_nodes = space.cell_node_map()
    parent_cell_nodes = parent.cell_node_map()
    indices = numpy.full(space.node_set.total_size, -1, dtype=IntType)
    for nodes, parent_nodes in zip(layer_nodes(cell_nodes, cell_nodes.values_with_halo),
                                   layer_nodes(parent_cell_nodes, parent_cell_nodes.values_with_halo)):
        held = nodes >= 0
        indices[nodes[held]] = parent.dof_dset.lgmap.block_indices[parent_nodes[held]]
    indices[mask_ghost_cells(cell_nodes)] = -1
    return PETSc.LGMap().create(indices, bsize=parent.block_size, comm=parent.comm)
