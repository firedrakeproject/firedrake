import numpy
from pyop2 import op2
from pyop2.datatypes import IntType
from firedrake.functionspacedata import entity_dofs_key
import ufl
import firedrake


def fine_node_to_coarse_node_map(Vf, Vc):
    if len(Vf) > 1:
        assert len(Vf) == len(Vc)
        return op2.MixedMap(fine_node_to_coarse_node_map(f, c) for f, c in zip(Vf, Vc))
    mesh = Vf.mesh()
    assert hasattr(mesh, "_shared_data_cache")
    hierarchyf, levelf = get_level(Vf.ufl_domain())
    hierarchyc, levelc = get_level(Vc.ufl_domain())

    if hierarchyc != hierarchyf:
        raise ValueError("Can't map across hierarchies")

    hierarchy = hierarchyf
    if levelc + 1 != levelf:
        raise ValueError("Can't map between level %s and level %s" % (levelc, levelf))

    key = (entity_dofs_key(Vc.finat_element.entity_dofs()) +
           entity_dofs_key(Vf.finat_element.entity_dofs()) +
           (levelc, levelf))

    cache = mesh._shared_data_cache["hierarchy_fine_node_to_coarse_node_map"]
    try:
        return cache[key]
    except KeyError:
        # XXX: Rewrite in cython.
        fine_to_coarse = hierarchy._fine_to_coarse[levelc+1]
        fine_map = Vf.cell_node_map()
        coarse_map = Vc.cell_node_map()
        fine_to_coarse_nodes = numpy.full((fine_map.toset.total_size,
                                           coarse_map.arity),
                                          -1,
                                          dtype=IntType)
        extruded = Vf.extruded
        assert extruded == Vc.extruded
        assert Vc.mesh().layers == Vf.mesh().layers
        for fcell, nodes in enumerate(fine_map.values):
            ccell = fine_to_coarse[fcell]
            if extruded:
                for l in range(Vf.mesh().layers - 1):
                    fine_to_coarse_nodes[nodes + Vf.offset*l, :] = coarse_map.values[ccell, :] + Vc.offset * l
            else:
                fine_to_coarse_nodes[nodes, :] = coarse_map.values[ccell, :]
        return cache.setdefault(key, op2.Map(Vf.node_set, Vc.node_set, coarse_map.arity,
                                             values=fine_to_coarse_nodes))


def coarse_node_to_fine_node_map(Vc, Vf):
    if len(Vf) > 1:
        assert len(Vf) == len(Vc)
        return op2.MixedMap(coarse_node_to_fine_node_map(f, c) for f, c in zip(Vf, Vc))
    mesh = Vc.mesh()
    assert hasattr(mesh, "_shared_data_cache")
    hierarchyf, levelf = get_level(Vf.ufl_domain())
    hierarchyc, levelc = get_level(Vc.ufl_domain())

    if hierarchyc != hierarchyf:
        raise ValueError("Can't map across hierarchies")

    hierarchy = hierarchyf
    if levelc + 1 != levelf:
        raise ValueError("Can't map between level %s and level %s" % (levelc, levelf))

    key = (entity_dofs_key(Vc.finat_element.entity_dofs()) +
           entity_dofs_key(Vf.finat_element.entity_dofs()) +
           (levelc, levelf))

    cache = mesh._shared_data_cache["hierarchy_coarse_node_to_fine_node_map"]
    try:
        return cache[key]
    except KeyError:
        # XXX: Rewrite in cython.
        coarse_to_fine = hierarchy._coarse_to_fine[levelc]
        fine_map = Vf.cell_node_map()
        coarse_map = Vc.cell_node_map()

        _, fcell_per_ccell = coarse_to_fine.shape

        coarse_to_fine_nodes = numpy.full((coarse_map.toset.total_size,
                                           fine_map.arity * fcell_per_ccell),
                                          -1,
                                          dtype=IntType)
        extruded = Vc.extruded
        assert extruded == Vf.extruded
        assert Vc.mesh().layers == Vf.mesh().layers
        for ccell, nodes in enumerate(coarse_map.values):
            fcells = coarse_to_fine[ccell]
            if extruded:
                for l in range(Vc.mesh().layers - 1):
                    coarse_to_fine_nodes[nodes + Vc.offset*l, :] = (fine_map.values[fcells, :].reshape(-1) +
                                                                    numpy.tile(Vf.offset*l, len(fcells)))
            else:
                coarse_to_fine_nodes[nodes, :] = fine_map.values[fcells, :].reshape(-1)

        return cache.setdefault(key, op2.Map(Vc.node_set, Vf.node_set,
                                             fine_map.arity * fcell_per_ccell,
                                             values=coarse_to_fine_nodes))


def physical_node_locations(V):
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


# def fine_node_to_reference_basis(V):
#     # XXX: procedure for computing physical space node locations of
#     # fine basis functions:
#     # 1. Determine reference element location for each node: hit
#     #    fiat/finat <- Not done now.
#     # 2. Evaluate *fine* coordinate field at this location.
#     # 3. Now we have physical location X_f
#     # 4. Now map to reference space on coarse cell.
#     if len(V) > 1:
#         # XXX: cache this
#         return op2.MixedDat(fine_node_to_reference_basis(V_) for V_ in V)
#     mesh = V.mesh()


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
