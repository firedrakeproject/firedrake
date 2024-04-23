import numpy
from fractions import Fraction
import pyop3 as op3
from firedrake.utils import IntType
from firedrake.functionspacedata import entity_dofs_key
import finat.ufl
import firedrake
from firedrake.cython import mgimpl as impl
from pyrsistent import freeze


def coarse_to_fine_cell_map(coarse_mesh, fine_mesh, coarse_to_fine_data):
    connectivity = {
        freeze({coarse_mesh.name: coarse_mesh.cell_label}): [
            op3.TabulatedMapComponent(fine_mesh.name, fine_mesh.cell_label, coarse_to_fine_data)
        ]
    }
    return op3.Map(connectivity)


def fine_node_to_coarse_node_map(Vf, Vc):
    if len(Vf) > 1:
        assert len(Vf) == len(Vc)
        return op3.MixedMap(fine_node_to_coarse_node_map(f, c) for f, c in zip(Vf, Vc))
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

    key = (entity_dofs_key(Vc.finat_element.entity_dofs())
           + entity_dofs_key(Vf.finat_element.entity_dofs())
           + (levelc, levelf))

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

        axes = op3.AxisTree.from_iterable([Vf.nodes, fine_to_coarse_nodes.shape[1]])
        fine_to_coarse_node_dat = op3.HierarchicalArray(axes, data=fine_to_coarse_nodes)
        fine_to_coarse_node_map = op3.Map({
            freeze({"nodes": "XXX"}): [
                op3.TabulatedMapComponent("nodes", "XXX", fine_to_coarse_node_dat)
            ]
        })
        return cache.setdefault(key, fine_to_coarse_node_map)


def coarse_node_to_fine_node_map(Vc, Vf):
    if len(Vf) > 1:
        assert len(Vf) == len(Vc)
        return op3.MixedMap(coarse_node_to_fine_node_map(f, c) for f, c in zip(Vf, Vc))
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

    key = (entity_dofs_key(Vc.finat_element.entity_dofs())
           + entity_dofs_key(Vf.finat_element.entity_dofs())
           + (levelc, levelf))

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

        axes = op3.AxisTree.from_iterable([Vc.nodes, coarse_to_fine_nodes.shape[1]])
        coarse_to_fine_node_dat = op3.HierarchicalArray(axes, data=coarse_to_fine_nodes)
        coarse_to_fine_node_map = op3.Map({
            freeze({"nodes": "XXX"}): [
                op3.TabulatedMapComponent("nodes", "XXX", coarse_to_fine_node_dat)
            ]
        })
        return cache.setdefault(key, coarse_to_fine_node_map)


def coarse_cell_to_fine_node_map(Vc, Vf):
    if len(Vf) > 1:
        assert len(Vf) == len(Vc)
        return op3.MixedMap(coarse_cell_to_fine_node_map(f, c) for f, c in zip(Vf, Vc))
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

    key = (entity_dofs_key(Vf.finat_element.entity_dofs()) + (levelc, levelf))
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
        iterset = Vc.mesh().owned_cells
        arity = Vf.finat_element.space_dimension() * ncell
        coarse_to_fine_nodes = numpy.full((Vc.mesh().cells.size, arity*level_ratio), -1, dtype=IntType)
        values = Vf.owned_cell_node_list[coarse_to_fine, :].reshape(iterset.size, arity)

        if Vc.extruded:
            off = numpy.tile(Vf.offset, ncell)
            coarse_to_fine_nodes[:iterset.size, :] = numpy.hstack([values + off*i for i in range(level_ratio)])
        else:
            coarse_to_fine_nodes[:iterset.size, :] = values
        if Vf.extruded:
            offset = numpy.tile(Vf.offset*level_ratio, ncell*level_ratio)

        axes = op3.AxisTree.from_iterable([Vc.nodes, arity*level_ratio])
        coarse_cell_to_fine_node_dat = op3.HierarchicalArray(axes, data=coarse_to_fine_nodes)
        coarse_cell_to_fine_node_map = op3.Map({
            freeze({Vc.mesh().topology.name: Vc.mesh().cell_label}): [
                op3.TabulatedMapComponent("nodes", "XXX", coarse_cell_to_fine_node_dat)
            ]
        })
        return cache.setdefault(key, coarse_cell_to_fine_node_map)


def physical_node_locations(V):
    element = V.ufl_element()
    if element.value_shape:
        assert isinstance(element, (finat.ufl.VectorElement, finat.ufl.TensorElement))
        element = element.sub_elements[0]
    mesh = V.mesh()
    # This is a defaultdict, so the first time we access the key we
    # get a fresh dict for the cache.
    cache = mesh._geometric_shared_data_cache["hierarchy_physical_node_locations"]
    key = element
    try:
        return cache[key]
    except KeyError:
        Vc = firedrake.FunctionSpace(mesh, finat.ufl.VectorElement(element))
        # FIXME: This is unsafe for DG coordinates and CG target spaces.
        locations = firedrake.assemble(firedrake.Interpolate(firedrake.SpatialCoordinate(mesh), Vc))
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
