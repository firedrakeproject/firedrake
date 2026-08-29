from __future__ import annotations

import numpy
from fractions import Fraction
from mpi4py import MPI
from pyop2 import op2
from firedrake.petsc import PETSc
from firedrake.utils import IntType
from firedrake.functionspacedata import entity_dofs_key
import finat.ufl
import firedrake
from firedrake.cython import mgimpl as impl
from firedrake.halo import _get_mtype


def fine_node_to_coarse_node_map(Vf, Vc):
    if len(Vf) > 1:
        assert len(Vf) == len(Vc)
        return op2.MixedMap(map(fine_node_to_coarse_node_map, Vf, Vc))
    mesh = Vf.mesh()
    assert hasattr(mesh, "_shared_data_cache")
    hierarchyf, levelf = get_level(Vf.mesh())
    hierarchyc, levelc = get_level(Vc.mesh())

    if hierarchyc != hierarchyf:
        raise ValueError("Can't map across hierarchies")

    hierarchy = hierarchyf
    increment = Fraction(1, hierarchyf.refinements_per_level)
    if levelc + increment != levelf:
        raise ValueError("Can't map between level %s and level %s" % (levelc, levelf))

    key = _cache_key(Vc, Vf)
    cache = mesh._shared_data_cache["hierarchy_fine_node_to_coarse_node_map"]
    try:
        return cache[key]
    except KeyError:
        assert Vc.extruded == Vf.extruded
        if Vc.mesh().variable_layers or Vf.mesh().variable_layers:
            raise NotImplementedError("Not implemented for variable layers, sorry")
        if Vc.extruded and not ((Vf.mesh().layers - 1)/(Vc.mesh().layers - 1)).is_integer():
            raise ValueError("Coarse and fine meshes must have an integer ratio of layers")

        fine_to_coarse = hierarchy.fine_to_coarse_cells[levelf]
        fine_to_coarse_nodes = impl.fine_to_coarse_nodes(Vf, Vc, fine_to_coarse)
        return cache.setdefault(key, op2.Map(Vf.node_set, Vc.node_set,
                                             fine_to_coarse_nodes.shape[1],
                                             values=fine_to_coarse_nodes))


def coarse_node_to_fine_node_map(Vc, Vf):
    if len(Vf) > 1:
        assert len(Vf) == len(Vc)
        return op2.MixedMap(map(coarse_node_to_fine_node_map, Vf, Vc))
    mesh = Vc.mesh()
    assert hasattr(mesh, "_shared_data_cache")
    hierarchyf, levelf = get_level(Vf.mesh())
    hierarchyc, levelc = get_level(Vc.mesh())

    if hierarchyc != hierarchyf:
        raise ValueError("Can't map across hierarchies")

    hierarchy = hierarchyf
    increment = Fraction(1, hierarchyf.refinements_per_level)
    if levelc + increment != levelf:
        raise ValueError("Can't map between level %s and level %s" % (levelc, levelf))

    key = _cache_key(Vc, Vf)
    cache = mesh._shared_data_cache["hierarchy_coarse_node_to_fine_node_map"]
    try:
        return cache[key]
    except KeyError:
        assert Vc.extruded == Vf.extruded
        if Vc.mesh().variable_layers or Vf.mesh().variable_layers:
            raise NotImplementedError("Not implemented for variable layers, sorry")
        if Vc.extruded and not ((Vf.mesh().layers - 1)/(Vc.mesh().layers - 1)).is_integer():
            raise ValueError("Coarse and fine meshes must have an integer ratio of layers")

        coarse_to_fine = hierarchy.coarse_to_fine_cells[levelc]
        coarse_to_fine_nodes = impl.coarse_to_fine_nodes(Vc, Vf, coarse_to_fine)
        # op2.Map cannot hold the -1 that pads a short row, so fill each
        # padded slot with a real entry from its own row. The injection
        # kernel picks the candidate that matches the coarse node's physical
        # location, so a repeated entry changes nothing.
        valid = coarse_to_fine_nodes >= 0
        nonempty = valid.any(axis=1)
        if not nonempty[:Vc.node_set.size].all():
            raise RuntimeError("Adaptive coarse-to-fine map has empty node candidates")
        replacement = numpy.zeros(coarse_to_fine_nodes.shape[0],
                                  dtype=coarse_to_fine_nodes.dtype)
        rows = numpy.nonzero(nonempty)[0]
        replacement[rows] = coarse_to_fine_nodes[rows, valid[rows].argmax(axis=1)]
        coarse_to_fine_nodes = numpy.where(valid, coarse_to_fine_nodes,
                                           replacement[:, None])
        return cache.setdefault(key, op2.Map(Vc.node_set, Vf.node_set,
                                             coarse_to_fine_nodes.shape[1],
                                             values=coarse_to_fine_nodes))


def coarse_cell_to_fine_node_map(Vc, Vf):
    if len(Vf) > 1:
        assert len(Vf) == len(Vc)
        return op2.MixedMap(coarse_cell_to_fine_node_map(f, c) for f, c in zip(Vf, Vc))
    mesh = Vc.mesh()
    assert hasattr(mesh, "_shared_data_cache")
    hierarchyf, levelf = get_level(Vf.mesh())
    hierarchyc, levelc = get_level(Vc.mesh())

    if hierarchyc != hierarchyf:
        raise ValueError("Can't map across hierarchies")

    hierarchy = hierarchyf
    increment = Fraction(1, hierarchyf.refinements_per_level)
    if levelc + increment != levelf:
        raise ValueError("Can't map between level %s and level %s" % (levelc, levelf))

    key = _cache_key(Vc, Vf, needs_coarse_entity_dofs=False)
    cache = mesh._shared_data_cache["hierarchy_coarse_cell_to_fine_node_map"]
    try:
        return cache[key]
    except KeyError:
        assert Vc.extruded == Vf.extruded
        if Vc.mesh().variable_layers or Vf.mesh().variable_layers:
            raise NotImplementedError("Not implemented for variable layers, sorry")
        if Vc.extruded:
            level_ratio = (Vf.mesh().layers - 1) // (Vc.mesh().layers - 1)
        else:
            level_ratio = 1
        coarse_to_fine = hierarchy.coarse_to_fine_cells[levelc]
        _, ncell = coarse_to_fine.shape
        iterset = Vc.mesh().cell_set
        fine_per_cell = Vf.finat_element.space_dimension()
        arity = fine_per_cell * ncell
        coarse_to_fine_nodes = numpy.full((iterset.total_size, arity*level_ratio), -1, dtype=IntType)
        values = numpy.full((iterset.size, ncell, fine_per_cell), -1, dtype=IntType)
        owned_coarse_to_fine = coarse_to_fine[:iterset.size, :]
        valid = owned_coarse_to_fine >= 0
        values[valid, :] = Vf.cell_node_map().values[owned_coarse_to_fine[valid], :]
        values = values.reshape(iterset.size, arity)

        if Vc.extruded:
            off = numpy.tile(Vf.offset, ncell)
            coarse_to_fine_nodes[:Vc.mesh().cell_set.size, :] = numpy.hstack([
                numpy.where(values >= 0, values + off*i, -1) for i in range(level_ratio)
            ])
        else:
            coarse_to_fine_nodes[:Vc.mesh().cell_set.size, :] = values
        offset = Vf.offset
        if offset is not None:
            offset = numpy.tile(offset*level_ratio, ncell*level_ratio)
        return cache.setdefault(key, op2.Map(iterset, Vf.node_set,
                                             arity=arity*level_ratio, values=coarse_to_fine_nodes,
                                             offset=offset))


def coarse_cell_child_count(
    Vc: firedrake.functionspaceimpl.WithGeometry,
    Vf: firedrake.functionspaceimpl.WithGeometry,
) -> op2.Dat:
    """Count the fine cells that each coarse cell was refined into.

    A row of `HierarchyBase.coarse_to_fine_cells` is as wide as the busiest
    coarse cell's count, so its width overstates how many children most cells
    have. The DG injection kernel reads this count to stop at a coarse cell's
    own children, and so leaves the padding alone.

    Parameters
    ----------
    Vc : firedrake.functionspaceimpl.WithGeometry
        The coarse function space.
    Vf : firedrake.functionspaceimpl.WithGeometry
        The fine function space, on the next level of the same hierarchy.

    Returns
    -------
    pyop2.types.dat.Dat
        One count per cell of ``Vc``'s mesh, over that mesh's cell set. Halo
        cells are left at zero: a par_loop visits the core and owned parts
        only, so the kernel never reads them.

    """
    mesh = Vc.mesh()
    assert hasattr(mesh, "_shared_data_cache")
    hierarchyf, levelf = get_level(Vf.mesh())
    hierarchyc, levelc = get_level(Vc.mesh())

    if hierarchyc != hierarchyf:
        raise ValueError("Can't map across hierarchies")

    hierarchy = hierarchyf
    increment = Fraction(1, hierarchyf.refinements_per_level)
    if levelc + increment != levelf:
        raise ValueError("Can't map between level %s and level %s" % (levelc, levelf))

    key = (levelc, Vc.extruded and (Vf.mesh().layers, Vc.mesh().layers))
    cache = mesh._shared_data_cache["hierarchy_coarse_cell_child_count"]
    try:
        return cache[key]
    except KeyError:
        if Vc.extruded:
            level_ratio = (Vf.mesh().layers - 1) // (Vc.mesh().layers - 1)
        else:
            level_ratio = 1
        coarse_to_fine = hierarchy.coarse_to_fine_cells[levelc]
        iterset = mesh.cell_set
        counts = numpy.zeros(iterset.total_size, dtype=IntType)
        # Each child of a coarse cell becomes level_ratio cells once extruded.
        counts[:iterset.size] = (coarse_to_fine[:iterset.size] >= 0).sum(axis=1) * level_ratio
        # A count belongs to a base cell, and every layer of that cell shares
        # it. An ExtrudedSet holds no data of its own, so hang the counts off
        # the base set that it was built on.
        dset = op2.DataSet(iterset.parent if Vc.extruded else iterset, 1)
        return cache.setdefault(key, op2.Dat(dset, counts, dtype=IntType))


def _preserved_point_sf(coarse_mesh, fine_mesh, coarse_to_fine):
    """Create the SF that pairs unrefined points with their coarse originals.

    Adaptive refinement leaves some cells untouched. This SF maps each
    unrefined point in ``fine_mesh`` back to the coarse point it came from.

    Parameters
    ----------
    coarse_mesh : firedrake.mesh.AbstractMeshTopology
        The mesh before refinement.
    fine_mesh : firedrake.mesh.AbstractMeshTopology
        The mesh after refinement.
    coarse_to_fine : numpy.ndarray
        The coarse-to-fine cell map that relates the two meshes.

    Returns
    -------
    PETSc.SF
        An SF with roots on the points of ``coarse_mesh`` and leaves on the
        unrefined points of ``fine_mesh``. Returns `None` if refinement
        changed every cell, as a uniform refinement does.

    """
    coarse_plex = coarse_mesh.topology_dm
    fine_plex = fine_mesh.topology_dm
    fine_to_coarse_points = impl.preserved_points(
        coarse_plex, coarse_mesh._cell_numbering,
        fine_plex, fine_mesh._cell_numbering,
        coarse_to_fine,
    )
    leaves, = numpy.nonzero(fine_to_coarse_points >= 0)
    # A uniform refinement preserves no points. Every rank must agree on
    # whether to build the SF at all, not just the ranks with no leaves.
    if not fine_plex.comm.tompi4py().allreduce(len(leaves) > 0, op=MPI.LOR):
        return None
    leaves = leaves.astype(IntType)
    # Refinement acts on each rank's own plex. A fine point and the coarse
    # point it was copied from always live on the same rank.
    remote = numpy.empty((len(leaves), 2), dtype=IntType)
    remote[:, 0] = coarse_plex.comm.rank
    remote[:, 1] = fine_to_coarse_points[leaves]
    pStart, pEnd = coarse_plex.getChart()
    point_sf = PETSc.SF().create(comm=coarse_plex.comm)
    point_sf.setGraph(pEnd - pStart, leaves, remote)
    return point_sf


def preserved_node_sf(Vc, Vf):
    """Find the nodes that adaptive refinement leaves unchanged.

    An unrefined cell has the same nodes in both spaces. The transfer
    operators can then copy values between them instead of evaluating them.
    This is cheaper, and exact.

    Parameters
    ----------
    Vc : firedrake.functionspaceimpl.WithGeometry
        The coarse function space.
    Vf : firedrake.functionspaceimpl.WithGeometry
        The fine function space, on the next level of the same hierarchy.

    Returns
    -------
    PETSc.SF
        An SF with roots on the nodes of ``Vc`` and leaves on the matching
        nodes of ``Vf``. Returns `None` if no nodes match.

    """
    if Vc.ufl_element() != Vf.ufl_element() or Vc.boundary_set != Vf.boundary_set:
        # A space and its counterpart on the refined mesh use the same node
        # layout on an unrefined cell only when the element and the boundary
        # set both match.
        return None
    if Vc.extruded or Vf.extruded:
        # The DMPlex of an extruded mesh stores only the 2D base mesh. Each
        # point there represents a whole vertical column of nodes, and a
        # Section cannot address one node within that column. Give up here
        # and let the transfer kernel evaluate every node instead.
        return None
    hierarchy, levelc = get_level(Vc.mesh())
    _, levelf = get_level(Vf.mesh())
    if hierarchy is None or levelc + Fraction(1, hierarchy.refinements_per_level) != levelf:
        return None
    cache = Vf.mesh().topology._shared_data_cache["hierarchy_preserved_node_sf"]
    key = _cache_key(Vc, Vf)
    try:
        return cache[key]
    except KeyError:
        coarse_to_fine = hierarchy.coarse_to_fine_cells[levelc]
        point_sf = _preserved_point_sf(Vc.mesh().topology, Vf.mesh().topology,
                                       coarse_to_fine)
        if point_sf is None:
            return cache.setdefault(key, None)
        root_section = Vc.dm.getSection()
        leaf_section = Vf.dm.getSection()
        # `distributeSection` builds its own section over the range of points
        # that the SF touches. Only the broadcast root offsets are needed
        # here. Pad them back out to the full chart that `createSectionSF`
        # expects.
        remote_offsets, distributed_section = point_sf.distributeSection(root_section)
        pStart, pEnd = leaf_section.getChart()
        lpStart, lpEnd = distributed_section.getChart()
        offsets = numpy.zeros(pEnd - pStart, dtype=IntType)
        offsets[lpStart - pStart:lpEnd - pStart] = remote_offsets
        section_sf = point_sf.createSectionSF(root_section, offsets, leaf_section)
        # The transfer kernels compute only the owned fine nodes and leave
        # the halo to a later exchange. Keep only the owned leaves here too:
        # a ghost fine node reduced onto its coarse node would count twice.
        nroots, ilocal, iremote = section_sf.getGraph()
        owned = ilocal < Vf.node_set.size
        trimmed = PETSc.SF().create(comm=section_sf.comm)
        trimmed.setGraph(nroots, ilocal[owned], iremote[owned])
        return cache.setdefault(key, trimmed)


def transfer_node_subset(Vc, Vf):
    """Find the fine nodes that the transfer kernels must evaluate.

    These are the nodes of ``Vf`` that :func:`preserved_node_sf` does not
    already account for. Prolongation and restriction can copy the rest.

    Parameters
    ----------
    Vc : firedrake.functionspaceimpl.WithGeometry
        The coarse function space.
    Vf : firedrake.functionspaceimpl.WithGeometry
        The fine function space, on the next level of the same hierarchy.

    Returns
    -------
    pyop2.types.set.Set or pyop2.types.set.Subset
        A subset of the nodes of ``Vf``, or ``Vf.node_set`` itself if
        :func:`preserved_node_sf` found no preserved nodes.

    """
    section_sf = preserved_node_sf(Vc, Vf)
    if section_sf is None:
        return Vf.node_set
    cache = Vf.mesh().topology._shared_data_cache["hierarchy_transfer_node_subset"]
    key = _cache_key(Vc, Vf)
    try:
        return cache[key]
    except KeyError:
        _, preserved, _ = section_sf.getGraph()
        nodes = numpy.setdiff1d(numpy.arange(Vf.node_set.size, dtype=IntType),
                                preserved)
        return cache.setdefault(key, op2.Subset(Vf.node_set, nodes))


def prolong_preserved_nodes(coarse, fine):
    """Copy coarse values onto the fine nodes that adaptive refinement preserved.

    Parameters
    ----------
    coarse : firedrake.function.Function
        The function on the coarse mesh.
    fine : firedrake.function.Function
        The function on the refined mesh. The transfer kernel has already
        computed its other nodes.

    """

    section_sf = preserved_node_sf(coarse.function_space(), fine.function_space())
    if section_sf is None:
        return
    mtype, _ = _get_mtype(fine.dat)
    # The source coarse node can be a ghost node. Only owned fine nodes are
    # written here, the same as the transfer kernel writes.
    source = coarse.dat.data_ro_with_halos
    target = fine.dat.data_wo
    section_sf.bcastBegin(mtype, source, target, MPI.REPLACE)
    section_sf.bcastEnd(mtype, source, target, MPI.REPLACE)


def restrict_preserved_nodes(fine_dual, coarse_dual):
    """Add the contribution of preserved nodes to the coarse dual.

    Prolongation copies a preserved node's value without change. Restriction
    is its transpose, so it adds the fine value to the coarse node unchanged.

    Parameters
    ----------
    fine_dual : firedrake.cofunction.Cofunction
        The cofunction on the refined mesh.
    coarse_dual : firedrake.cofunction.Cofunction
        The cofunction on the coarse mesh. It already holds the contribution
        that the transfer kernel accumulated from the other fine nodes.

    """

    coarse_V = coarse_dual.function_space()
    section_sf = preserved_node_sf(coarse_V, fine_dual.function_space())
    if section_sf is None:
        return
    buffer = firedrake.Function(coarse_V)
    mtype, _ = _get_mtype(buffer.dat)
    source = fine_dual.dat.data_ro
    target = buffer.dat.data_wo_with_halos
    section_sf.reduceBegin(mtype, source, target, MPI.SUM)
    section_sf.reduceEnd(mtype, source, target, MPI.SUM)
    # A preserved coarse node can be a ghost on the rank that owns the
    # matching fine node. Reduce the contributions onto the owning rank.
    buffer.dat.local_to_global_begin(op2.INC)
    buffer.dat.local_to_global_end(op2.INC)
    coarse_dual.dat.data[...] += buffer.dat.data_ro


def physical_node_locations(V):
    element = V.ufl_element()
    if V.value_shape:
        assert isinstance(element, (finat.ufl.VectorElement, finat.ufl.TensorElement))
        element = element.sub_elements[0]
    mesh = V.mesh()
    # This is a defaultdict, so the first time we access the key we
    # get a fresh dict for the cache.
    cache = mesh.geometric_shared_data_cache["hierarchy_physical_node_locations"]
    key = (element, V.boundary_set)
    try:
        return cache[key]
    except KeyError:
        Vc = V.collapse().reconstruct(element=finat.ufl.VectorElement(element, dim=mesh.geometric_dimension))

        # FIXME: This is unsafe for DG coordinates and CG target spaces.
        locations = firedrake.assemble(firedrake.interpolate(firedrake.SpatialCoordinate(mesh), Vc))
        return cache.setdefault(key, locations)


def set_level(obj, hierarchy, level):
    """Attach hierarchy and level info to an object."""
    setattr(obj.topological, "__level_info__", (hierarchy, level))
    return obj


def get_level(obj):
    """Try and obtain hierarchy and level info from an object.

    If no level info is available, return ``None, None``."""
    try:
        return getattr(obj.topological, "__level_info__")
    except AttributeError:
        return None, None


def has_level(obj):
    """Does the provided object have level info?"""
    return hasattr(obj.topological, "__level_info__")


def _cache_key(Vc, Vf, needs_coarse_entity_dofs=True):
    """Construct a cache key for node maps"""
    _, levelf = get_level(Vf.mesh())
    _, levelc = get_level(Vc.mesh())

    if needs_coarse_entity_dofs:
        key = entity_dofs_key(Vc.finat_element.entity_dofs())
    else:
        key = ()
    key += entity_dofs_key(Vf.finat_element.entity_dofs())
    key += (levelc, levelf)
    key += (Vc.boundary_set, Vf.boundary_set)
    return key
