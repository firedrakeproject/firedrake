import numpy
from fractions import Fraction
from pyop2 import op2
from pyop2.datatypes import IntType
from firedrake.functionspacedata import entity_dofs_key
import ufl
import firedrake
from firedrake.cython import mgimpl as impl


def get_or_set_mg_hierarchy_map_cache(cache_dict, entity_dofs_key,
        create_map_on_cpu):
    """
    :arg cache_dict: An instance of :class:`dict` that maps from tuple
        ``(entity_dofs_key, compute_backend)`` to the corresponding map.
    :arg create_host_map: A callable that takes no argument and returns the map
        on the CPU backend.
    :returns map: An instance of :class:`pyop2.base.Map`.
    """
    try:
        return cache_dict[(entity_dofs_key, op2.compute_backend)]
    except KeyError:
        from pyop2.sequential import sequential_cpu_backend
        host_map = cache_dict.setdefault((entity_dofs_key,
            sequential_cpu_backend), create_map_on_cpu())
        return cache_dict.setdefault((entity_dofs_key, op2.compute_backend),
                op2.compute_backend.Map(host_map))


def fine_node_to_coarse_node_map(Vf, Vc):
    if len(Vf) > 1:
        assert len(Vf) == len(Vc)
        return op2.compute_backend.MixedMap(fine_node_to_coarse_node_map(f, c) for f, c in zip(Vf, Vc))
    mesh = Vf.mesh()
    assert hasattr(mesh, "_shared_data_cache")
    hierarchyf, levelf = get_level(Vf.ufl_domain())
    hierarchyc, levelc = get_level(Vc.ufl_domain())

    if hierarchyc != hierarchyf:
        raise ValueError("Can't map across hierarchies")

    hierarchy = hierarchyf
    increment = Fraction(1, hierarchyf.refinements_per_level)
    if levelc + increment != levelf:
        raise ValueError("Can't map between level %s and level %s" % (levelc, levelf))

    key = (entity_dofs_key(Vc.finat_element.entity_dofs())
           + entity_dofs_key(Vf.finat_element.entity_dofs())
           + (levelc, levelf))

    cache = mesh._shared_data_cache["hierarchy_fine_node_to_coarse_node_map"]

    def create_map_on_cpu():
        from pyop2.sequential import sequential_cpu_backend
        assert Vc.extruded == Vf.extruded
        if Vc.mesh().variable_layers or Vf.mesh().variable_layers:
            raise NotImplementedError("Not implemented for variable layers, sorry")
        if Vc.extruded and not ((Vf.mesh().layers - 1)/(Vc.mesh().layers - 1)).is_integer():
            raise ValueError("Coarse and fine meshes must have an integer ratio of layers")

        fine_to_coarse = hierarchy.fine_to_coarse_cells[levelf]
        fine_to_coarse_nodes = impl.fine_to_coarse_nodes(Vf, Vc, fine_to_coarse)
        return sequential_cpu_backend.Map(Vf.node_set, Vc.node_set,
            fine_to_coarse_nodes.shape[1], values=fine_to_coarse_nodes)

    return get_or_set_mg_hierarchy_map_cache(cache, key, create_map_on_cpu)


def coarse_node_to_fine_node_map(Vc, Vf):
    if len(Vf) > 1:
        assert len(Vf) == len(Vc)
        return op2.compute_backend.MixedMap(coarse_node_to_fine_node_map(f, c) for f, c in zip(Vf, Vc))
    mesh = Vc.mesh()
    assert hasattr(mesh, "_shared_data_cache")
    hierarchyf, levelf = get_level(Vf.ufl_domain())
    hierarchyc, levelc = get_level(Vc.ufl_domain())

    if hierarchyc != hierarchyf:
        raise ValueError("Can't map across hierarchies")

    hierarchy = hierarchyf
    increment = Fraction(1, hierarchyf.refinements_per_level)
    if levelc + increment != levelf:
        raise ValueError("Can't map between level %s and level %s" % (levelc, levelf))

    key = (entity_dofs_key(Vc.finat_element.entity_dofs())
           + entity_dofs_key(Vf.finat_element.entity_dofs())
           + (levelc, levelf))

    cache = mesh._shared_data_cache["hierarchy_coarse_node_to_fine_node_map"]
    def create_map_on_cpu():
        from pyop2.sequential import sequential_cpu_backend
        assert Vc.extruded == Vf.extruded
        if Vc.mesh().variable_layers or Vf.mesh().variable_layers:
            raise NotImplementedError("Not implemented for variable layers, sorry")
        if Vc.extruded and not ((Vf.mesh().layers - 1)/(Vc.mesh().layers - 1)).is_integer():
            raise ValueError("Coarse and fine meshes must have an integer ratio of layers")

        coarse_to_fine = hierarchy.coarse_to_fine_cells[levelc]
        coarse_to_fine_nodes = impl.coarse_to_fine_nodes(Vc, Vf, coarse_to_fine)
        return sequential_cpu_backend.Map(Vc.node_set, Vf.node_set,
                coarse_to_fine_nodes.shape[1], values=coarse_to_fine_nodes)

    return get_or_set_mg_hierarchy_map_cache(cache, key, create_map_on_cpu)

def coarse_cell_to_fine_node_map(Vc, Vf):
    if len(Vf) > 1:
        assert len(Vf) == len(Vc)
        return op2.compute_backend.MixedMap(coarse_cell_to_fine_node_map(f, c) for f, c in zip(Vf, Vc))
    mesh = Vc.mesh()
    assert hasattr(mesh, "_shared_data_cache")
    hierarchyf, levelf = get_level(Vf.ufl_domain())
    hierarchyc, levelc = get_level(Vc.ufl_domain())

    if hierarchyc != hierarchyf:
        raise ValueError("Can't map across hierarchies")

    hierarchy = hierarchyf
    increment = Fraction(1, hierarchyf.refinements_per_level)
    if levelc + increment != levelf:
        raise ValueError("Can't map between level %s and level %s" % (levelc, levelf))

    key = (entity_dofs_key(Vf.finat_element.entity_dofs()) + (levelc, levelf))
    cache = mesh._shared_data_cache["hierarchy_coarse_cell_to_fine_node_map"]

    def create_map_on_cpu():
        from pyop2.sequential import sequential_cpu_backend
        assert Vc.extruded == Vf.extruded
        if Vc.mesh().variable_layers or Vf.mesh().variable_layers:
            raise NotImplementedError("Not implemented for variable layers, sorry")
        if Vc.extruded and Vc.mesh().layers != Vf.mesh().layers:
            raise ValueError("Coarse and fine meshes must have same number of layers")

        coarse_to_fine = hierarchy.coarse_to_fine_cells[levelc]
        _, ncell = coarse_to_fine.shape
        iterset = Vc.mesh().cell_set
        arity = Vf.finat_element.space_dimension() * ncell
        coarse_to_fine_nodes = numpy.full((iterset.total_size, arity), -1, dtype=IntType)
        values = Vf.cell_node_map().values[coarse_to_fine, :].reshape(iterset.size, arity)

        coarse_to_fine_nodes[:Vc.mesh().cell_set.size, :] = values
        offset = Vf.offset
        if offset is not None:
            offset = numpy.tile(offset, ncell)
        return sequential_cpu_backend.Map(iterset, Vf.node_set, arity=arity,
                values=coarse_to_fine_nodes, offset=offset)

    return get_or_set_mg_hierarchy_map_cache(cache, key, create_map_on_cpu)


def physical_node_locations(V):
    #TODO: Do we need to cache this per backend as well?
    element = V.ufl_element()
    if element.value_shape():
        assert isinstance(element, (ufl.VectorElement, ufl.TensorElement))
        element = element.sub_elements()[0]
    mesh = V.mesh()
    cache = mesh._shared_data_cache["hierarchy_physical_node_locations"]
    key = element
    try:
        return cache[key]
    except KeyError:
        Vc = firedrake.FunctionSpace(mesh, ufl.VectorElement(element))
        locations = firedrake.interpolate(firedrake.SpatialCoordinate(mesh), Vc)
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
