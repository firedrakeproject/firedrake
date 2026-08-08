import numpy
from fractions import Fraction
from pyop2 import op2
from firedrake.utils import IntType
from firedrake.functionspacedata import entity_dofs_key
import finat.ufl
import firedrake
from firedrake.cython import mgimpl as impl


def fine_node_to_coarse_node_map(Vf, Vc):
    """Map each fine node to the nodes of its parent coarse cell.

    A fine cell has exactly one parent. This holds for uniform refinement
    and for adaptive refinement. ``hierarchy.fine_to_coarse_cells`` is
    therefore one column wide, and it holds no padding. This map needs no
    special handling for adaptive refinement.

    `coarse_node_to_fine_node_map` runs in the other direction. There the
    number of children varies, and padding does appear.

    `prolong` and `restrict` both read through this map.

    Parameters
    ----------
    Vf : firedrake.functionspaceimpl.WithGeometry
        The fine function space.
    Vc : firedrake.functionspaceimpl.WithGeometry
        The coarse function space, on the previous level of the same
        hierarchy.

    Returns
    -------
    pyop2.types.map.Map
        A map from the nodes of ``Vf`` to the nodes of ``Vc``.

    """
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
    """Map each coarse node to the fine nodes it could come from.

    A coarse node gets one candidate per fine cell that descends from it.
    Uniform refinement gives every coarse cell the same number of children.
    Every row is then full. Adaptive refinement gives them different
    numbers. ``hierarchy.coarse_to_fine_cells`` then right-pads its short
    rows with -1, out to the busiest coarse cell's count.

    This map fills that padding, because `op2.Map` cannot hold a negative
    index. `fine_node_to_coarse_node_map` runs in the other direction. One
    parent per fine cell makes padding impossible there.

    Injection into a space with a pointwise dual basis reads through this
    map. Injection into a DG space uses `coarse_cell_to_fine_node_map`
    instead. That map handles its padding a different way.

    Parameters
    ----------
    Vc : firedrake.functionspaceimpl.WithGeometry
        The coarse function space.
    Vf : firedrake.functionspaceimpl.WithGeometry
        The fine function space, on the next level of the same hierarchy.

    Returns
    -------
    pyop2.types.map.Map
        A map from the nodes of ``Vc`` to the nodes of ``Vf``.

    Raises
    ------
    RuntimeError
        If an owned coarse node has no fine node to inject from.

    """
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
        # Adaptive refinement gives coarse cells different numbers of fine
        # descendants. Each row of coarse_to_fine_nodes is therefore padded
        # with -1, out to the busiest coarse cell's count, and op2.Map cannot
        # hold a negative index. Fill each padded slot with a duplicate of a
        # real entry from its own row. The injection kernel only reads
        # through this map, and picks the candidate that matches the coarse
        # node's physical location, so a repeated entry changes nothing.
        #
        # Every rank runs this fill. The partition decides which rows hold
        # padding, so the fill must not depend on a rank-local test.
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
    """Map each coarse cell to the fine nodes of all its children.

    Each row holds one block of nodes per child cell. The blocks are stored
    back to back. Uniform refinement gives every coarse cell the same number
    of children, so every row is full. Adaptive refinement gives them
    different numbers. ``hierarchy.coarse_to_fine_cells`` then right-pads
    its short rows with -1, and that padding reaches this map.

    The padding stays as it is. The DG injection kernel reads
    `coarse_cell_child_count`. It stops at a coarse cell's own children, and
    so never reads a padded block.

    `coarse_node_to_fine_node_map` fills its padding instead. The kernel
    that reads that map visits every candidate.

    The DG branch of `inject` reads through this map.

    Parameters
    ----------
    Vc : firedrake.functionspaceimpl.WithGeometry
        The coarse function space.
    Vf : firedrake.functionspaceimpl.WithGeometry
        The fine function space, on the next level of the same hierarchy.

    Returns
    -------
    pyop2.types.map.Map
        A map from the cells of ``Vc``'s mesh to the nodes of ``Vf``.

    """
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


def coarse_cell_child_count(Vc, Vf):
    """Count the fine cells that each coarse cell was refined into.

    Uniform refinement gives every coarse cell the same number of children.
    Every count then equals the width of a `coarse_cell_to_fine_node_map`
    row. Adaptive refinement leaves some coarse cells alone, and splits
    others. A coarse cell then has from one child up to the busiest cell's
    count. The map pads its short rows out to that busiest count.

    The DG injection kernel reads this count. It stops at a coarse cell's
    own children, and so leaves that padding alone.

    Parameters
    ----------
    Vc : firedrake.functionspaceimpl.WithGeometry
        The coarse function space.
    Vf : firedrake.functionspaceimpl.WithGeometry
        The fine function space, on the next level of the same hierarchy.

    Returns
    -------
    pyop2.types.dat.Dat
        One count per cell of ``Vc``'s mesh, over that mesh's cell set.

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
