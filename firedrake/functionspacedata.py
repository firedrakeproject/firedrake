"""This module provides an object that encapsulates data that can be
shared between different :class:`~.FunctionSpace` objects.

The sharing is based on the idea of compatibility of function space
node layout.  The shared data is stored on the :func:`~.Mesh` the
function space is created on, since the created objects are
mesh-specific.  The sharing is done on an individual key basis.  So,
for example, Sets can be shared between all function spaces with the
same number of nodes per topological entity.  However, maps are
specific to the node *ordering*.

This means, for example, that function spaces with the same *node*
ordering, but different numbers of dofs per node (e.g. FiniteElement
vs VectorElement) can share the PyOP2 Set and Map data.
"""

import numpy
import finat.ufl
import finat
from decorator import decorator
from functools import partial

from tsfc.finatinterface import create_element as _create_element

from pyop2 import op2
from firedrake.utils import IntType
from pyop2.utils import as_tuple

from firedrake.cython import extrusion_numbering as extnum
from firedrake.cython import dmcommon
from firedrake import halo as halo_mod
from firedrake import mesh as mesh_mod
from firedrake import extrusion_utils as eutils
from firedrake.petsc import PETSc


__all__ = ("get_shared_data", )


@PETSc.Log.EventDecorator("FunctionSpaceData: CreateElement")
def create_element(ufl_element):
    finat_element = _create_element(ufl_element)
    if isinstance(finat_element, finat.TensorFiniteElement):
        # Retrieve scalar element
        finat_element = finat_element.base_element
    return finat_element


@decorator
def cached(f, mesh, key, *args, **kwargs):
    """Sui generis caching for a function whose data is
    associated with a mesh.

    :arg f: The function to cache.
    :arg mesh: The mesh to cache on (should have a
        ``_shared_data_cache`` object).
    :arg key: The key to the cache.
    :args args: Additional arguments to ``f``.
    :kwargs kwargs:  Additional keyword arguments to ``f``."""
    assert hasattr(mesh, "_shared_data_cache")
    cache = mesh._shared_data_cache[f.__name__]
    try:
        return cache[key]
    except KeyError:
        result = f(mesh, key, *args, **kwargs)
        cache[key] = result
        return result


@cached
def get_global_numbering(mesh, key, global_numbering=None):
    """Get a PETSc Section describing the global numbering.

    This numbering associates function space nodes with topological
    entities.

    :arg mesh: The mesh to use.
    :arg key: a (nodes_per_entity, real_tensorproduct, boundary_set) tuple where
        nodes_per_entity is a tuple of the number of nodes per topological
        entity; real_tensorproduct is True if the function space is a
        degenerate fs x Real tensorproduct; boundary_set is a set of boundary
        markers, indicating sub-domains a boundary condition is specified on.
    :returns: A new PETSc Section.
    """
    if global_numbering:
        return global_numbering
    nodes_per_entity, real_tensorproduct, boundary_set = key
    return mesh.create_section(nodes_per_entity, real_tensorproduct, boundary_set=boundary_set)


@cached
def get_node_set(mesh, key):
    """Get the :class:`node set <pyop2.Set>`.

    :arg mesh: The mesh to use.
    :arg key: a (nodes_per_entity, real_tensorproduct, boundary_set) tuple
        where nodes_per_entity is a tuple of the number of nodes per
        topological entity; real_tensorproduct is True if the function space is
        a degenerate fs x Real tensorproduct; boundary_set is a set of boundary
        markers, indicating sub-domains a boundary condition is specified on.
    :returns: A :class:`pyop2.Set` for the function space nodes.
    """
    nodes_per_entity, real_tensorproduct, _ = key
    global_numbering, constrained_size = get_global_numbering(mesh, key)
    node_classes = mesh.node_classes(nodes_per_entity, real_tensorproduct=real_tensorproduct)
    halo = halo_mod.Halo(mesh.topology_dm, global_numbering, comm=mesh.comm)
    node_set = op2.Set(node_classes, halo=halo, comm=mesh.comm, constrained_size=constrained_size)
    extruded = mesh.cell_set._extruded

    assert global_numbering.getStorageSize() == node_set.total_size
    if not extruded and node_set.total_size >= (1 << (IntType.itemsize * 8 - 4)):
        raise RuntimeError("Problems with more than %d nodes per process unsupported", (1 << (IntType.itemsize * 8 - 4)))
    return node_set


def get_cell_node_list(mesh, entity_dofs, entity_permutations, global_numbering, offsets):
    """Get the cell->node list for specified dof layout.

    :arg mesh: The mesh to use.
    :arg entity_dofs: The FInAT entity_dofs dict.
    :arg entity_permutations: The FInAT entity_permutations dict.
    :arg global_numbering: The PETSc Section describing node layout
        (see :func:`get_global_numbering`).
    :arg offsets: layer offsets for each entity (maybe ignored).
    :returns: A numpy array mapping mesh cells to function space
        nodes.
    """
    return mesh.make_cell_node_list(global_numbering, entity_dofs, entity_permutations, offsets)


def get_facet_node_list(mesh, kind, cell_node_list, offsets):
    """Get the facet->node list for specified dof layout.

    :arg mesh: The mesh to use.
    :arg kind: The facet kind (one of ``"interior_facets"`` or
        ``"exterior_facets"``).
    :arg cell_node_list: The map from mesh cells to function space
        nodes, see :func:`get_cell_node_list`.
    :arg offsets: layer offsets for each entity (maybe ignored).
    :returns: A numpy array mapping mesh facets to function space
        nodes.
    """
    assert kind in ["interior_facets", "exterior_facets"]
    if mesh.topology_dm.getStratumSize(kind, 1) > 0:
        return dmcommon.get_facet_nodes(mesh, cell_node_list, kind, offsets)
    else:
        return numpy.array([], dtype=IntType)


@cached
def get_entity_node_lists(mesh, key, entity_dofs, entity_permutations, global_numbering, offsets):
    """Get the map from mesh entity sets to function space nodes.

    :arg mesh: The mesh to use.
    :arg key: a (entity_dofs_key, real_tensorproduct, entity_permutations_key,
        boundary_set) tuple.
    :arg entity_dofs: FInAT entity dofs.
    :arg entity_permutations: FInAT entity permutations.
    :arg global_numbering: The PETSc Section describing node layout
        (see :func:`get_global_numbering`).
    :arg offsets: layer offsets for each entity (maybe ignored).
    :returns: A dict mapping mesh entity sets to numpy arrays of
        function space nodes.
    """
    # set->node lists are specific to the sorted entity_dofs.
    cell_node_list = get_cell_node_list(mesh, entity_dofs, entity_permutations, global_numbering, offsets)
    interior_facet_node_list = partial(get_facet_node_list, mesh, "interior_facets", cell_node_list, offsets)
    exterior_facet_node_list = partial(get_facet_node_list, mesh, "exterior_facets", cell_node_list, offsets)

    class magic(dict):
        def __missing__(self, key):
            if type(mesh.topology) is mesh_mod.VertexOnlyMeshTopology:
                return self.setdefault(key,
                                       {mesh.cell_set: lambda: cell_node_list}[key]())
            else:
                return self.setdefault(key,
                                       {mesh.cell_set: lambda: cell_node_list,
                                        mesh.interior_facets.set: interior_facet_node_list,
                                        mesh.exterior_facets.set: exterior_facet_node_list}[key]())

    return magic()


@cached
def get_map_cache(mesh, key):
    """Get the map cache for this mesh.

    :arg mesh: The mesh to use.
    :arg key: a (entity_dofs_key, real_tensorproduct, entity_permutations_key,
        boundary_set) tuple where entity_dofs is Canonicalised entity_dofs
        (see :func:`entity_dofs_key`); real_tensorproduct is True if the
        function space is a degenerate fs x Real tensorproduct; boundary_set is
        the set of subdomains a restricted function space is applied to, or
        None if using a regular function space.
    """
    if type(mesh.topology) is mesh_mod.VertexOnlyMeshTopology:
        return {mesh.cell_set: None}
    else:
        return {mesh.cell_set: None,
                mesh.interior_facets.set: None,
                mesh.exterior_facets.set: None,
                "boundary_node": None}


@cached
def get_boundary_masks(mesh, key, finat_element):
    """Get masks for facet dofs.

    :arg mesh: The mesh to use.
    :arg key: Canonicalised entity_dofs (see :func:`entity_dofs_key`).
    :arg finat_element: The FInAT element.
    :returns: ``None`` or a 3-tuple of a Section, an array of indices, and
        an array indicating which points in the Section correspond to
        the facets of the cell.  If section.getDof(p) is non-zero,
        then there are ndof basis functions topologically associated
        with points in the closure of point p.  The basis function
        indices are in the index array, starting at section.getOffset(p).
    """
    if not mesh.cell_set._extruded:
        return None
    _, kind = key
    assert kind in {"cell", "interior_facet"}
    dim = finat_element.cell.get_spatial_dimension()
    ecd = finat_element.entity_closure_dofs()
    # Number of entities on cell excepting the cell itself.
    chart = sum(map(len, ecd.values())) - 1
    closure_section = PETSc.Section().create(comm=PETSc.COMM_SELF)
    # Double up for interior facets.
    if kind == "cell":
        ncell = 1
    else:
        ncell = 2
    closure_section.setChart(0, ncell*chart)
    closure_indices = []
    facet_points = []
    p = 0

    offset = finat_element.space_dimension()
    for cell in range(ncell):
        for ent in sorted(ecd.keys()):
            # Never need closure of cell
            if sum(ent) == dim:
                continue
            for key in sorted(ecd[ent].keys()):
                closure_section.setDof(p, len(ecd[ent][key]))
                vals = numpy.asarray(sorted(ecd[ent][key]), dtype=IntType)
                closure_indices.extend(vals + cell*offset)
                if sum(ent) == dim - 1:
                    facet_points.append(p)
                p += 1
    closure_section.setUp()
    closure_indices = numpy.asarray(closure_indices, dtype=IntType)
    facet_points = numpy.asarray(facet_points, dtype=IntType)
    return (closure_section, closure_indices, facet_points)


@cached
def get_work_function_cache(mesh, ufl_element):
    """Get the cache for work functions.

    :arg mesh: The mesh to use.
    :arg ufl_element: The ufl element, used as a key.
    :returns: A dict.

    :class:`.FunctionSpace` objects sharing the same UFL element (and
    therefore comparing equal) share a work function cache.
    """
    return {}


@cached
def get_top_bottom_boundary_nodes(mesh, key, V):
    """Get top or bottom boundary nodes of an extruded function space.

    :arg mesh: The mesh to cache on.
    :arg key: A 3-tuple of ``(entity_dofs_key, sub_domain, boundary_set)`` key.
        Where sub_domain indicates top or bottom.
    :arg V: The FunctionSpace to select from.
    :arg entity_dofs: The flattened entity dofs.
    :returnsL: A numpy array of the (unique) boundary nodes.
    """
    _, sub_domain, boundary_set = key
    cell_node_list = V.cell_node_list
    offset = V.offset
    if mesh.variable_layers:
        return extnum.top_bottom_boundary_nodes(mesh, cell_node_list,
                                                V.cell_boundary_masks,
                                                offset,
                                                sub_domain)
    else:
        if mesh.extruded_periodic and sub_domain == "top":
            raise ValueError("Invalid subdomain 'top': 'top' boundary is identified as 'bottom' boundary in periodic extrusion")
        idx = {"bottom": -2, "top": -1}[sub_domain]
        section, indices, facet_points = V.cell_boundary_masks
        facet = facet_points[idx]
        dof = section.getDof(facet)
        off = section.getOffset(facet)
        mask = indices[off:off+dof]
        nodes = cell_node_list[..., mask]
        if sub_domain == "top":
            nodes = nodes + offset[mask]*(mesh.cell_set.layers - 2)
        return numpy.unique(nodes)


@cached
def get_facet_closure_nodes(mesh, key, V):
    """Function space nodes in the closure of facets with a given
    marker.
    :arg mesh: Mesh to cache on
    :arg key: (edofs, sub_domain, boundary_set) tuple
    :arg V: function space.
    :returns: numpy array of unique nodes in the closure of facets
       with provided markers (both interior and exterior)."""
    _, sub_domain, boundary_set = key
    if sub_domain not in {"on_boundary", "top", "bottom"}:
        valid = set(mesh.interior_facets.unique_markers)
        valid |= set(mesh.exterior_facets.unique_markers)
        invalid = set(sub_domain) - valid
        if invalid:
            raise LookupError(f"BC construction got invalid markers {invalid}. "
                              f"Valid markers are '{valid}'")
    return dmcommon.facet_closure_nodes(V, sub_domain)


def get_max_work_functions(V):
    """Get the maximum number of work functions.

    :arg V: The function space to get the number of work functions for.
    :returns: The maximum number of work functions.

    This number is shared between all function spaces with the same
    :meth:`~.FunctionSpace.ufl_element` and
    :meth:`~FunctionSpace.mesh`.

    The default is 25 work functions per function space.  This can be
    set using :func:`set_max_work_functions`.
    """
    mesh = V.mesh()
    assert hasattr(mesh, "_shared_data_cache")
    cache = mesh._shared_data_cache["max_work_functions"]
    return cache.get(V.ufl_element(), 25)


def set_max_work_functions(V, val):
    """Set the maximum number of work functions.

    :arg V: The function space to set the number of work functions
        for.
    :arg val: The new maximum number of work functions.

    This number is shared between all function spaces with the same
    :meth:`~.FunctionSpace.ufl_element` and
    :meth:`~FunctionSpace.mesh`.
    """
    mesh = V.mesh()
    assert hasattr(mesh, "_shared_data_cache")
    cache = mesh._shared_data_cache["max_work_functions"]
    cache[V.ufl_element()] = val


def entity_dofs_key(entity_dofs):
    """Provide a canonical key for an entity_dofs dict.

    :arg entity_dofs: The FInAT entity_dofs.
    :returns: A tuple of canonicalised entity_dofs (suitable for
        caching).
    """
    key = []
    for k in sorted(entity_dofs.keys()):
        sub_key = [k]
        for sk in sorted(entity_dofs[k]):
            sub_key.append(tuple(entity_dofs[k][sk]))
        key.append(tuple(sub_key))
    key = tuple(key)
    return key


def entity_permutations_key(entity_permutations):
    """Provide a canonical key for an entity_permutations dict.

    :arg entity_permutations: The FInAT entity_permutations.
    :returns: A tuple of canonicalised entity_permutations (suitable for
        caching).
    """
    key = []
    for k in sorted(entity_permutations.keys()):
        sub_key = [k]
        for sk in sorted(entity_permutations[k]):
            subsub_key = [sk]
            for ssk in sorted(entity_permutations[k][sk]):
                subsub_key.append((ssk, tuple(entity_permutations[k][sk][ssk])))
            sub_key.append(tuple(subsub_key))
        key.append(tuple(sub_key))
    key = tuple(key)
    return key


class FunctionSpaceData(object):
    """Function spaces with the same entity dofs share data.  This class
    stores that shared data.  It is cached on the mesh.

    :arg mesh: The mesh to share the data on.
    :arg ufl_element: The UFL element.
    :arg boundary_set: The set of subdomains that a Dirichlet boundary condition
        will act on. This is None if the function space is not a
        :class:`.RestrictedFunctionSpace`.
    """
    __slots__ = ("real_tensorproduct", "map_cache", "entity_node_lists",
                 "node_set", "cell_boundary_masks",
                 "interior_facet_boundary_masks", "offset", "offset_quotient",
                 "extruded", "mesh", "global_numbering", "boundary_set")

    @PETSc.Log.EventDecorator()
    def __init__(self, mesh, ufl_element, boundary_set=None):
        if type(ufl_element) is finat.ufl.MixedElement:
            raise ValueError("Can't create FunctionSpace for MixedElement")

        self.boundary_set = boundary_set

        finat_element = create_element(ufl_element)
        real_tensorproduct = eutils.is_real_tensor_product_element(finat_element)
        entity_dofs = finat_element.entity_dofs()
        nodes_per_entity = tuple(mesh.make_dofs_per_plex_entity(entity_dofs))
        try:
            entity_permutations = finat_element.entity_permutations
        except NotImplementedError:
            entity_permutations = None

        # Create the PetscSection mapping topological entities to functionspace nodes
        # For non-scalar valued function spaces, there are multiple dofs per node.
        key = (nodes_per_entity, real_tensorproduct, boundary_set)
        # These are keyed only on nodes per topological entity.
        global_numbering, constrained_size = get_global_numbering(mesh, key)
        node_set = get_node_set(mesh, key)

        edofs_key = entity_dofs_key(entity_dofs)
        # entity_permutations is None if not yet implemented
        eperm_key = entity_permutations_key(entity_permutations) if entity_permutations else None

        self.real_tensorproduct = real_tensorproduct
        # Empty map caches. This is a sui generis cache
        # implementation because of the need to support boundary
        # conditions.
        # Map caches are specific to a cell_node_list, which is keyed by entity_dof
        self.map_cache = get_map_cache(mesh, (edofs_key, real_tensorproduct, eperm_key, boundary_set))

        if isinstance(mesh, mesh_mod.ExtrudedMeshTopology):
            self.offset = eutils.calculate_dof_offset(finat_element)
        else:
            self.offset = None
        if isinstance(mesh, mesh_mod.ExtrudedMeshTopology) and mesh.extruded_periodic:
            self.offset_quotient = eutils.calculate_dof_offset_quotient(finat_element)
        else:
            self.offset_quotient = None

        self.entity_node_lists = get_entity_node_lists(mesh, (edofs_key, real_tensorproduct, eperm_key, boundary_set), entity_dofs, entity_permutations, global_numbering, self.offset)
        self.node_set = node_set
        self.cell_boundary_masks = get_boundary_masks(mesh, (edofs_key, "cell"), finat_element)
        self.interior_facet_boundary_masks = get_boundary_masks(mesh, (edofs_key, "interior_facet"), finat_element)
        self.extruded = mesh.cell_set._extruded
        self.mesh = mesh
        self.global_numbering = global_numbering

    def __eq__(self, other):
        if type(self) is not type(other):
            return False
        return all(getattr(self, s) is getattr(other, s) for s in
                   FunctionSpaceData.__slots__)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return "FunctionSpaceData(%r, %r)" % (self.mesh, self.node_set)

    def __str__(self):
        return "FunctionSpaceData(%s, %s)" % (self.mesh, self.node_set)

    @PETSc.Log.EventDecorator()
    def boundary_nodes(self, V, sub_domain):
        if sub_domain in ["bottom", "top"]:
            if not V.extruded:
                raise ValueError("Invalid subdomain '%s' for non-extruded mesh",
                                 sub_domain)
            entity_dofs = eutils.flat_entity_dofs(V.finat_element.entity_dofs())
            key = (entity_dofs_key(entity_dofs), sub_domain, V.boundary_set)
            return get_top_bottom_boundary_nodes(V.mesh(), key, V)
        else:
            if sub_domain == "on_boundary":
                sdkey = sub_domain
            else:
                sdkey = as_tuple(sub_domain)
            key = (entity_dofs_key(V.finat_element.entity_dofs()), sdkey, V.boundary_set)
            return get_facet_closure_nodes(V.mesh(), key, V)

    @PETSc.Log.EventDecorator()
    def get_map(self, V, entity_set, map_arity, name, offset, offset_quotient):
        """Return a :class:`pyop2.Map` from some topological entity to
        degrees of freedom.

        :arg V: The :class:`FunctionSpace` to create the map for.
        :arg entity_set: The :class:`pyop2.Set` of entities to map from.
        :arg map_arity: The arity of the resulting map.
        :arg name: A name for the resulting map.
        :arg offset: Map offset (for extruded).
        :arg offset_quotient: Map offset_quotient (for extruded)."""
        # V is only really used for error checking and "name".
        assert len(V) == 1, "get_map should not be called on MixedFunctionSpace"
        entity_node_list = self.entity_node_lists[entity_set]
        val = self.map_cache[entity_set]
        if val is None:
            val = op2.Map(entity_set, self.node_set,
                          map_arity,
                          entity_node_list,
                          ("%s_"+name) % (V.name),
                          offset=offset,
                          offset_quotient=offset_quotient)

            self.map_cache[entity_set] = val
        return val


@PETSc.Log.EventDecorator()
def get_shared_data(mesh, ufl_element, boundary_set=None):
    """Return the ``FunctionSpaceData`` for the given
    element.

    :arg mesh: The mesh to build the function space data on.
    :arg ufl_element: A UFL element.
    :arg boundary_set: A set of boundary markers, indicating the subdomains a
        boundary condition is specified on.
    :raises ValueError: if mesh or ufl_element are invalid.
    :returns: a ``FunctionSpaceData`` object with the shared
        data.
    """
    if not isinstance(mesh, mesh_mod.AbstractMeshTopology):
        raise ValueError("%s is not an AbstractMeshTopology" % mesh)
    if not isinstance(ufl_element, finat.ufl.finiteelement.FiniteElementBase):
        raise ValueError("Can't create function space data from a %s" %
                         type(ufl_element))
    return FunctionSpaceData(mesh, ufl_element, boundary_set)
