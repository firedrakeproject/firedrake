import numpy
from immutabledict import immutabledict as idict
from fractions import Fraction
import pyop3 as op3
from firedrake.utils import IntType
from firedrake.functionspacedata import entity_dofs_key
import finat.ufl
import firedrake
from firedrake.cython import mgimpl as impl


def fine_node_to_coarse_node_map(Vf, Vc):
    if len(Vf) > 1:
        raise NotImplementedError
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
        if Vc.extruded and not ((Vf.mesh().layers - 1)/(Vc.mesh().layers - 1)).is_integer():
            raise ValueError("Coarse and fine meshes must have an integer ratio of layers")

        fine_to_coarse = hierarchy.fine_to_coarse_cells[levelf]
        fine_to_coarse_nodes = impl.fine_to_coarse_nodes(Vf, Vc, fine_to_coarse)

        src_axis = Vf.nodal_axes.root
        target_axis = op3.Axis(fine_to_coarse_nodes.shape[1])
        node_map_axes = op3.AxisTree.from_iterable([src_axis, target_axis])
        node_map_dat = op3.Dat(node_map_axes, data=fine_to_coarse_nodes.flatten())
        node_map = op3.Map(
            {
                idict({"nodes": None}): [[op3.TabulatedMapComponent("nodes", None, node_map_dat)]],
            },
            # TODO: This is only here so labels resolve, ideally we would relabel to make this fine
            name=target_axis.label,
        )
        return cache.setdefault(key, node_map)


def coarse_node_to_fine_node_map(Vc, Vf):
    if len(Vf) > 1:
        raise NotImplementedError
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
        if Vc.extruded and not ((Vf.mesh().layers - 1)/(Vc.mesh().layers - 1)).is_integer():
            raise ValueError("Coarse and fine meshes must have an integer ratio of layers")

        coarse_to_fine = hierarchy.coarse_to_fine_cells[levelc]
        coarse_to_fine_nodes = impl.coarse_to_fine_nodes(Vc, Vf, coarse_to_fine)

        src_axis = Vc.nodal_axes.root
        target_axis = op3.Axis(coarse_to_fine_nodes.shape[1])
        node_map_axes = op3.AxisTree.from_iterable([src_axis, target_axis])
        node_map_dat = op3.Dat(node_map_axes, data=coarse_to_fine_nodes.flatten())
        node_map = op3.Map(
            {
                idict({"nodes": None}): [[op3.TabulatedMapComponent("nodes", None, node_map_dat)]],
            }, 
            # TODO: This is only here so labels resolve, ideally we would relabel to make this fine
            name=target_axis.label
        )
        return cache.setdefault(key, node_map)


def coarse_cell_to_fine_node_map(Vc, Vf):
    if len(Vf) > 1:
        raise NotImplementedError
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
        if Vc.extruded:
            level_ratio = (Vf.mesh().layers - 1) // (Vc.mesh().layers - 1)
        else:
            level_ratio = 1
        coarse_to_fine = hierarchy.coarse_to_fine_cells[levelc]
        _, ncell = coarse_to_fine.shape
        iterset = Vc.mesh().cells
        arity = Vf.finat_element.space_dimension() * ncell
        coarse_to_fine_nodes = numpy.full((iterset.local_size, arity*level_ratio), -1, dtype=IntType)
        values = Vf.cell_node_list[coarse_to_fine, :].reshape(iterset.local_size, arity)

        if Vc.extruded:
            off = numpy.tile(Vf.offset, ncell)
            coarse_to_fine_nodes[:iterset.local_size, :] = numpy.hstack([values + off*i for i in range(level_ratio)])
        else:
            coarse_to_fine_nodes[:iterset.local_size, :] = values
        offset = Vf.offset
        if offset is not None:
            offset = numpy.tile(offset*level_ratio, ncell*level_ratio)

        src_axis = iterset.root
        target_axis = op3.Axis(coarse_to_fine_nodes.shape[1])
        node_map_axes = op3.AxisTree.from_iterable([src_axis, target_axis])
        node_map_dat = op3.Dat(node_map_axes, data=coarse_to_fine_nodes.flatten())
        node_map = op3.Map(
            {
                idict({src_axis.label: src_axis.component.label}): [[op3.TabulatedMapComponent("nodes", None, node_map_dat)]],
            }, 
            # TODO: This is only here so labels resolve, ideally we would relabel to make this fine
            name=target_axis.label
        )
        # return cache.setdefault(key, op2.Map(iterset, Vf.node_set,
        #                                      arity=arity*level_ratio, values=coarse_to_fine_nodes,
        #                                      offset=offset))
        return cache.setdefault(key, node_map)


def physical_node_locations(V):
    element = V.ufl_element()
    if V.value_shape:
        assert isinstance(element, (finat.ufl.VectorElement, finat.ufl.TensorElement))
        element = element.sub_elements[0]
    mesh = V.mesh()
    # This is a defaultdict, so the first time we access the key we
    # get a fresh dict for the cache.
    cache = mesh._geometric_shared_data_cache["hierarchy_physical_node_locations"]
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
