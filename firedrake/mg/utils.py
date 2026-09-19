from __future__ import annotations

import numpy
from fractions import Fraction
from mpi4py import MPI
from pyop2 import op2
from firedrake.utils import IntType
from firedrake.functionspacedata import entity_dofs_key
import finat.ufl
import firedrake
from firedrake.cython import mgimpl as impl
from firedrake.halo import _get_mtype
from firedrake.petsc import PETSc


def identity_node_map(V):
    cache = V.mesh()._shared_data_cache["hierarchy_identity_node_map"]
    key = (V.ufl_element(), V.boundary_set)
    try:
        return cache[key]
    except KeyError:
        values = numpy.arange(V.node_set.total_size, dtype=IntType).reshape(-1, 1)
        return cache.setdefault(key, op2.Map(V.node_set, V.node_set, 1, values=values))


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
        if not Vc.comm.allreduce(bool(nonempty[:Vc.node_set.size].all()), op=MPI.LAND):
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
        # The DG injection kernel skips the padded slots of a row, but PyOP2
        # still reads through them. Fill each one with the row's first child.
        children = coarse_to_fine[:iterset.size, :]
        children = numpy.where(children >= 0, children, children[:, :1])
        values = Vf.cell_node_map().values[children]
        if Vc.extruded:
            # Keep the layers of each child together, so that the children of
            # a coarse cell come before its padded slots.
            values = values[:, :, None, :] + numpy.arange(level_ratio)[:, None] * Vf.offset
        coarse_to_fine_nodes[:iterset.size, :] = values.reshape(iterset.size, arity*level_ratio)
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
        raise ValueError(f"Can't map between level {levelc} and level {levelf}")

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


def transfer_mesh(mesh):
    """Return the mesh that grid transfer operates on.

    A redistributed mesh has no cell maps relating it to the coarse mesh.
    Transfers therefore go through the mesh it was redistributed from. The
    values are then assigned across the two.

    Parameters
    ----------
    mesh : firedrake.mesh.MeshGeometry
        A mesh in a `HierarchyBase`.

    Returns
    -------
    firedrake.mesh.MeshGeometry
        ``mesh`` itself, or the mesh it was redistributed from.

    """
    return mesh.submesh_parent if mesh.submesh_point_sf is not None else mesh


def _redistribution_ancestors(topology):
    """Yield a mesh topology together with the topologies it was redistributed from.

    The transfer operators work on the mesh a redistributed mesh came from,
    so both must carry the same multigrid level.

    Parameters
    ----------
    topology : firedrake.mesh.AbstractMeshTopology
        The topology to start from.

    Yields
    ------
    firedrake.mesh.AbstractMeshTopology
        ``topology``, then each mesh topology it was redistributed from, in
        order.

    """
    yield topology
    while topology.submesh_point_sf is not None:
        topology = topology.submesh_parent
        yield topology


def set_dm_refine_level(mesh, level):
    """Set the refinement level of a mesh and of the meshes it was redistributed from.

    Parameters
    ----------
    mesh : firedrake.mesh.MeshGeometry
        The mesh to set the refinement level of.
    level : int
        The refinement level to set.

    """
    for topology in _redistribution_ancestors(mesh.topology):
        topology.topology_dm.setRefineLevel(level)


def set_level(obj, hierarchy, level):
    """Attach hierarchy and level info to an object.

    Parameters
    ----------
    obj : firedrake.mesh.MeshGeometry or firedrake.functionspaceimpl.WithGeometry
        The object to attach the hierarchy and level info to.
    hierarchy : HierarchyBase
        The hierarchy ``obj`` belongs to.
    level : Fraction
        The level of ``obj`` in ``hierarchy``.

    Returns
    -------
    firedrake.mesh.MeshGeometry or firedrake.functionspaceimpl.WithGeometry
        ``obj``, unchanged.

    """
    for topology in _redistribution_ancestors(obj.topological):
        setattr(topology, "__level_info__", (hierarchy, level))
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


def _preserved_point_sf(coarse_mesh, fine_to_coarse_points):
    """Create an SF that pairs unchanged fine points with coarse points.

    A coarse point that occurs once in the fine-to-coarse point map was copied
    unchanged by the transform. The SF maps each such fine point to that
    coarse point.

    Parameters
    ----------
    coarse_mesh : firedrake.mesh.AbstractMeshTopology
        The mesh before refinement.
    fine_to_coarse_points : numpy.ndarray
        The point map from fine mesh points to coarse mesh points.

    Returns
    -------
    PETSc.SF
        An SF whose roots are coarse mesh points and whose leaves are the
        corresponding unchanged fine mesh points.

    """
    coarse_plex = coarse_mesh.topology_dm
    source_points, counts = numpy.unique(fine_to_coarse_points[fine_to_coarse_points >= 0],
                                         return_counts=True)
    leaves = numpy.flatnonzero(
        numpy.isin(fine_to_coarse_points, source_points[counts == 1])).astype(IntType)
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
    """Find the nodes unchanged by adaptive refinement.

    An unrefined cell has the same nodes in both spaces. Values on those nodes
    can be copied instead of evaluated, which is both cheaper and exact.

    Parameters
    ----------
    Vc : firedrake.functionspaceimpl.WithGeometry
        The coarse function space.
    Vf : firedrake.functionspaceimpl.WithGeometry
        The fine function space, on the next level of the same hierarchy.

    Returns
    -------
    PETSc.SF or None
        An SF whose roots are nodes in ``Vc`` and whose leaves are matching
        nodes in ``Vf``. ``None`` is returned if no nodes match.

    """
    if Vc.ufl_element() != Vf.ufl_element() or Vc.boundary_set != Vf.boundary_set:
        # The spaces have matching node layouts on an unrefined cell only when
        # their elements and boundary sets match.
        return None
    if Vc.extruded or Vf.extruded:
        # DMPlex stores only the 2D base mesh for an extruded mesh. Each point
        # represents a vertical column of nodes, and a Section cannot address
        # individual nodes in that column. Let the transfer kernel evaluate
        # every node instead.
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
        coarse_to_fine_cells = hierarchy.coarse_to_fine_cells[levelc]
        fine_to_coarse_points = hierarchy.fine_to_coarse_points.get(levelf)
        if fine_to_coarse_points is None:
            return cache.setdefault(key, None)
        # No coarse cell is preserved by uniform refinement or by refinement
        # that splits every coarse cell. Avoid building an SF in this case.
        has_preserved_cells = numpy.any(
            (coarse_to_fine_cells >= 0).sum(axis=1) == 1)
        if not Vc.comm.allreduce(bool(has_preserved_cells), op=MPI.LOR):
            return cache.setdefault(key, None)
        point_sf = _preserved_point_sf(Vc.mesh().topology, fine_to_coarse_points)
        root_section = Vc.dm.getSection()
        leaf_section = Vf.dm.getSection()
        # `distributeSection` creates a section for the points in the SF graph.
        # `createSectionSF` expects offsets for the full point chart, so pad
        # the returned root offsets with zeros.
        remote_offsets, distributed_section = point_sf.distributeSection(root_section)
        pStart, pEnd = leaf_section.getChart()
        lpStart, lpEnd = distributed_section.getChart()
        offsets = numpy.zeros(pEnd - pStart, dtype=IntType)
        offsets[lpStart - pStart:lpEnd - pStart] = remote_offsets
        section_sf = point_sf.createSectionSF(root_section, offsets, leaf_section)
        # Only owned fine nodes are computed here; halo nodes are updated
        # later. Keep only owned leaves because reducing a ghost fine node onto
        # its coarse node would count that contribution twice.
        nroots, ilocal, iremote = section_sf.getGraph()
        owned = ilocal < Vf.node_set.size
        trimmed = PETSc.SF().create(comm=section_sf.comm)
        trimmed.setGraph(nroots, ilocal[owned], iremote[owned])
        return cache.setdefault(key, trimmed)


def transfer_node_subset(Vc, Vf):
    """Find the fine nodes that the transfer kernels must evaluate.

    These are the nodes of ``Vf`` that the preserved-node SF does not cover.
    The remaining nodes are copied during prolongation and restriction.

    Parameters
    ----------
    Vc : firedrake.functionspaceimpl.WithGeometry
        The coarse function space.
    Vf : firedrake.functionspaceimpl.WithGeometry
        The fine function space, on the next level of the same hierarchy.

    Returns
    -------
    pyop2.types.set.Set or pyop2.types.set.Subset
        A subset of the nodes of ``Vf``, or ``Vf.node_set`` itself when no
        preserved-node SF exists.

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
    """Copy coarse values to fine nodes preserved by adaptive refinement.

    Parameters
    ----------
    coarse : firedrake.function.Function
        The function on the coarse mesh.
    fine : firedrake.function.Function
        The function on the fine mesh. Its other nodes have already been
        computed by the transfer kernel.

    """

    section_sf = preserved_node_sf(coarse.function_space(), fine.function_space())
    if section_sf is None:
        return
    mtype, _ = _get_mtype(fine.dat)
    # The source coarse node can be a ghost. Only owned fine nodes are written,
    # as in the transfer kernel.
    source = coarse.dat.data_ro_with_halos
    target = fine.dat.data_wo
    section_sf.bcastBegin(mtype, source, target, MPI.REPLACE)
    section_sf.bcastEnd(mtype, source, target, MPI.REPLACE)


def restrict_preserved_nodes(fine_dual, coarse_dual):
    """Add preserved fine-node contributions to the coarse dual.

    Prolongation copies values at preserved nodes unchanged. Restriction is
    its transpose, so the same fine values are added to the corresponding
    coarse nodes.

    Parameters
    ----------
    fine_dual : firedrake.cofunction.Cofunction
        The cofunction on the fine mesh.
    coarse_dual : firedrake.cofunction.Cofunction
        The cofunction on the coarse mesh. Contributions from the other fine
        nodes have already been accumulated.

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
    # A preserved coarse node can be a ghost on the rank that owns its
    # matching fine node. Reduce its contribution to the owning rank.
    buffer.dat.local_to_global_begin(op2.INC)
    buffer.dat.local_to_global_end(op2.INC)
    coarse_dual.dat.data[...] += buffer.dat.data_ro
