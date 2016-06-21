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

from __future__ import absolute_import

import numpy
import FIAT
from decorator import decorator


from FIAT.finite_element import entity_support_dofs

from coffee import base as ast

from pyop2 import op2

from firedrake import dmplex as dm_mod
from firedrake import halo as halo_mod
from firedrake import mesh as mesh_mod
from firedrake.petsc import PETSc


__all__ = ("get_shared_data", )


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
def get_global_numbering(mesh, nodes_per_entity):
    """Get a PETSc Section describing the global numbering.

    This numbering associates function space nodes with topological
    entities.

    :arg mesh: The mesh to use.
    :arg nodes_per_entity: a tuple of the number of nodes per
        topological entity.
    :returns: A new PETSc Section.
    """
    return mesh._plex.createSection([1], nodes_per_entity, perm=mesh._plex_renumbering)


@cached
def get_node_set(mesh, nodes_per_entity):
    """Get the :class:`node set <pyop2.Set>`.

    :arg mesh: The mesh to use.
    :arg nodes_per_entity: The number of function space nodes per
        topological entity.
    :returns: A :class:`pyop2.Set` for the function space nodes.
    """
    global_numbering = get_global_numbering(mesh, nodes_per_entity)
    # Use a DM to create the halo SFs
    dm = PETSc.DMShell().create(mesh.comm)
    dm.setPointSF(mesh._plex.getPointSF())
    dm.setDefaultSection(global_numbering)
    node_classes = tuple(numpy.dot(nodes_per_entity, mesh._entity_classes))
    node_set = op2.Set(node_classes, halo=halo_mod.Halo(dm), comm=mesh.comm)
    # Don't need it any more, explicitly destroy.
    dm.destroy()
    extruded = bool(mesh.layers)
    if extruded:
        node_set = op2.ExtrudedSet(node_set, layers=mesh.layers)

    assert global_numbering.getStorageSize() == node_set.total_size
    return node_set


def get_cell_node_list(mesh, entity_dofs, global_numbering):
    """Get the cell->node list for specified dof layout.

    :arg mesh: The mesh to use.
    :arg entity_dofs: The FIAT entity_dofs dict.
    :arg global_numbering: The PETSc Section describing node layout
        (see :func:`get_global_numbering`).
    :returns: A numpy array mapping mesh cells to function space
        nodes.
    """
    return mesh.make_cell_node_list(global_numbering, entity_dofs)


def get_facet_node_list(mesh, kind, cell_node_list):
    """Get the facet->node list for specified dof layout.

    :arg mesh: The mesh to use.
    :arg kind: The facet kind (one of ``"interior_facets"`` or
        ``"exterior_facets"``).
    :arg cell_node_list: The map from mesh cells to function space
        nodes, see :func:`get_cell_node_list`.
    :returns: A numpy array mapping mesh facets to function space
        nodes.
    """
    assert kind in ["interior_facets", "exterior_facets"]
    if mesh._plex.getStratumSize(kind, 1) > 0:
        facet = getattr(mesh, kind)
        return dm_mod.get_facet_nodes(facet.facet_cell, cell_node_list)
    else:
        return numpy.array([], dtype=numpy.int32)


@cached
def get_entity_node_lists(mesh, key, entity_dofs, global_numbering):
    """Get the map from mesh entity sets to function space nodes.

    :arg mesh: The mesh to use.
    :arg key: Canonicalised entity_dofs (see :func:`entity_dofs_key`).
    :arg entity_dofs: FIAT entity dofs.
    :arg global_numbering: The PETSc Section describing node layout
        (see :func:`get_global_numbering`).
    :returns: A dict mapping mesh entity sets to numpy arrays of
        function space nodes.
    """
    # set->node lists are specific to the sorted entity_dofs.
    cell_node_list = get_cell_node_list(mesh, entity_dofs, global_numbering)
    interior_facet_node_list = get_facet_node_list(mesh, "interior_facets", cell_node_list)
    exterior_facet_node_list = get_facet_node_list(mesh, "exterior_facets", cell_node_list)
    return {mesh.cell_set: cell_node_list,
            mesh.interior_facets.set: interior_facet_node_list,
            mesh.exterior_facets.set: exterior_facet_node_list}


@cached
def get_map_caches(mesh, entity_dofs):
    """Get the map caches for this mesh.

    :arg mesh: The mesh to use.
    :arg entity_dofs: Canonicalised entity_dofs (see
        :func:`entity_dofs_key`).
    """
    return {mesh.cell_set: {},
            mesh.interior_facets.set: {},
            mesh.exterior_facets.set: {},
            "boundary_node": {}}


@cached
def get_dof_offset(mesh, key, entity_dofs, ndof):
    """Get the dof offsets.

    :arg mesh: The mesh to use.
    :arg key: Canonicalised entity_dofs (see :func:`entity_dofs_key`).
    :arg entity_dofs: The FIAT entity_dofs dict.
    :arg ndof: The number of dofs (the FIAT space_dimension).
    :returns: A numpy array of dof offsets (extruded) or ``None``.
    """
    return mesh.make_offset(entity_dofs, ndof)


@cached
def get_bt_masks(mesh, key, fiat_element):
    """Get masks for top and bottom dofs.

    :arg mesh: The mesh to use.
    :arg key: Canonicalised entity_dofs (see :func:`entity_dofs_key`).
    :arg fiat_element: The FIAT element.
    :returns: A dict mapping ``"topological"`` and ``"geometric"``
        keys to bottom and top dofs (extruded) or ``None``.
    """
    if not bool(mesh.layers):
        return None
    bt_masks = {}
    # Compute the top and bottom masks to identify boundary dofs
    #
    # Sorting the keys of the closure entity dofs, the whole cell
    # comes last [-1], before that the horizontal facet [-2], before
    # that vertical facets [-3]. We need the horizontal facets here.
    closure_dofs = fiat_element.entity_closure_dofs()
    horiz_facet_dim = sorted(closure_dofs.keys())[-2]
    b_mask = closure_dofs[horiz_facet_dim][0]
    t_mask = closure_dofs[horiz_facet_dim][1]
    bt_masks["topological"] = (b_mask, t_mask)  # conversion to tuple
    # Geometric facet dofs
    facet_dofs = entity_support_dofs(fiat_element, horiz_facet_dim)
    bt_masks["geometric"] = (facet_dofs[0], facet_dofs[1])
    return bt_masks


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

    :arg entity_dofs: The FIAT entity_dofs.
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


class FunctionSpaceData(object):
    """Function spaces with the same entity dofs share data.  This class
    stores that shared data.  It is cached on the mesh.

    :arg mesh: The mesh to share the data on.
    :arg fiat_element: The FIAT describing how nodes are attached to
       topological entities.
    """
    __slots__ = ("map_caches", "entity_node_lists",
                 "node_set", "bt_masks", "offset",
                 "extruded", "mesh", "global_numbering")

    def __init__(self, mesh, fiat_element):
        entity_dofs = fiat_element.entity_dofs()
        nodes_per_entity = tuple(mesh.make_dofs_per_plex_entity(entity_dofs))

        # Create the PetscSection mapping topological entities to functionspace nodes
        # For non-scalar valued function spaces, there are multiple dofs per node.

        # These are keyed only on nodes per topological entity.
        global_numbering = get_global_numbering(mesh, nodes_per_entity)
        node_set = get_node_set(mesh, nodes_per_entity)

        edofs_key = entity_dofs_key(entity_dofs)

        # Empty map caches. This is a sui generis cache
        # implementation because of the need to support boundary
        # conditions.
        # Map caches are specific to a cell_node_list, which is keyed by entity_dof
        self.map_caches = get_map_caches(mesh, edofs_key)
        self.entity_node_lists = get_entity_node_lists(mesh, edofs_key, entity_dofs, global_numbering)
        self.node_set = node_set
        self.offset = get_dof_offset(mesh, edofs_key, entity_dofs, fiat_element.space_dimension())
        self.bt_masks = get_bt_masks(mesh, edofs_key, fiat_element)
        self.extruded = bool(mesh.layers)
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

    def exterior_facet_boundary_node_map(self, V, method):
        """Return the :class:`pyop2.Map` from exterior facets to nodes
        on the boundary.

        :arg V: The function space.
        :arg method:  The method for determining boundary nodes.  See
           :class:`~.DirichletBC` for details.
        """
        try:
            return self.map_caches["boundary_node"][method]
        except KeyError:
            pass
        el = V.fiat_element

        dim = self.mesh.facet_dimension()

        if method == "topological":
            boundary_dofs = el.entity_closure_dofs()[dim]
        elif method == "geometric":
            # This function is only called on extruded meshes when
            # asking for the nodes that live on the "vertical"
            # exterior facets.
            boundary_dofs = entity_support_dofs(el, dim)

        nodes_per_facet = \
            len(boundary_dofs[0])

        # HACK ALERT
        # The facet set does not have a halo associated with it, since
        # we only construct halos for DoF sets.  Fortunately, this
        # loop is direct and we already have all the correct
        # information available locally.  So We fake a set of the
        # correct size and carry out a direct loop
        facet_set = op2.Set(self.mesh.exterior_facets.set.total_size,
                            comm=self.mesh.comm)

        fs_dat = op2.Dat(facet_set**el.space_dimension(),
                         data=V.exterior_facet_node_map().values_with_halo.view())

        facet_dat = op2.Dat(facet_set**nodes_per_facet,
                            dtype=numpy.int32)

        # Ensure these come out in sorted order.
        local_facet_nodes = numpy.array(
            [boundary_dofs[e] for e in sorted(boundary_dofs.keys())])

        # Helper function to turn the inner index of an array into c
        # array literals.
        c_array = lambda xs: "{"+", ".join(map(str, xs))+"}"

        # AST for: l_nodes[facet[0]][n]
        rank_ast = ast.Symbol("l_nodes", rank=(ast.Symbol("facet", rank=(0,)), "n"))

        body = ast.Block([ast.Decl("int",
                                   ast.Symbol("l_nodes", (len(el.get_reference_element().topology[dim]),
                                                          nodes_per_facet)),
                                   init=ast.ArrayInit(c_array(map(c_array, local_facet_nodes))),
                                   qualifiers=["const"]),
                          ast.For(ast.Decl("int", "n", 0),
                                  ast.Less("n", nodes_per_facet),
                                  ast.Incr("n", 1),
                                  ast.Assign(ast.Symbol("facet_nodes", ("n",)),
                                             ast.Symbol("cell_nodes", (rank_ast, ))))
                          ])

        kernel = op2.Kernel(ast.FunDecl("void", "create_bc_node_map",
                                        [ast.Decl("int*", "cell_nodes"),
                                         ast.Decl("int*", "facet_nodes"),
                                         ast.Decl("unsigned int*", "facet")],
                                        body),
                            "create_bc_node_map")

        local_facet_dat = op2.Dat(facet_set ** self.mesh.exterior_facets._rank,
                                  self.mesh.exterior_facets.local_facet_dat.data_ro_with_halos,
                                  dtype=numpy.uintc)
        op2.par_loop(kernel, facet_set,
                     fs_dat(op2.READ),
                     facet_dat(op2.WRITE),
                     local_facet_dat(op2.READ))

        if self.extruded:
            offset = self.offset[boundary_dofs[0]]
        else:
            offset = None
        val = op2.Map(facet_set, self.node_set,
                      nodes_per_facet,
                      facet_dat.data_ro_with_halos,
                      name="exterior_facet_boundary_node",
                      offset=offset)
        self.map_caches["boundary_node"][method] = val
        return val

    def get_map(self, V, entity_set, map_arity, bcs, name, offset, parent):
        """Return a :class:`pyop2.Map` from some topological entity to
        degrees of freedom.

        :arg V: The :class:`FunctionSpace` to create the map for.
        :arg entity_set: The :class:`pyop2.Set` of entities to map from.
        :arg map_arity: The arity of the resulting map.
        :arg bcs: An iterable of :class:`~.DirichletBC` objects (may
            be ``None``.
        :arg name: A name for the resulting map.
        :arg offset: Map offset (for extruded).
        :arg parent: The parent map (used when bcs are provided)."""
        # V is only really used for error checking and "name".
        assert len(V) == 1, "get_map should not be called on MixedFunctionSpace"
        entity_node_list = self.entity_node_lists[entity_set]
        cache = self.map_caches[entity_set]

        if bcs is not None:
            # Separate explicit bcs (we just place negative entries in
            # the appropriate map values) from implicit ones (extruded
            # top and bottom) that require PyOP2 code gen.
            explicit_bcs = [bc for bc in bcs if bc.sub_domain not in ['top', 'bottom']]
            implicit_bcs = [(bc.sub_domain, bc.method) for bc in bcs if bc.sub_domain in ['top', 'bottom']]
            if len(explicit_bcs) == 0:
                # Implicit bcs are not part of the cache key for the
                # map (they only change the generated PyOP2 code),
                # hence rewrite bcs here.
                bcs = ()
            if len(implicit_bcs) == 0:
                implicit_bcs = None
        else:
            # Empty tuple if no bcs found.  This is so that matrix
            # assembly, which uses a set to keep track of the bcs
            # applied to matrix hits the cache when that set is
            # empty.  tuple(set([])) == tuple().
            bcs = ()
            implicit_bcs = None

        for bc in bcs:
            fs = bc.function_space()
            # Unwind proxies for ComponentFunctionSpace, but not
            # IndexedFunctionSpace.
            while fs.component is not None and fs.parent is not None:
                fs = fs.parent
            if fs.topological != V:
                raise RuntimeError("DirichletBC defined on a different FunctionSpace!")
        # Ensure bcs is a tuple in a canonical order for the hash key.
        lbcs = tuple(sorted(bcs, key=lambda bc: bc.__hash__()))

        cache = self.map_caches[entity_set]
        try:
            # Cache hit
            val = cache[lbcs]
            # In the implicit bc case, we decorate the cached map with
            # the list of implicit boundary conditions so PyOP2 knows
            # what to do.
            if implicit_bcs:
                val = op2.DecoratedMap(val, implicit_bcs=implicit_bcs)
            return val
        except KeyError:
            # Cache miss.
            # Any top and bottom bcs (for the extruded case) are handled elsewhere.
            nodes = [bc.nodes for bc in lbcs if bc.sub_domain not in ['top', 'bottom']]
            decorate = False
            if nodes:
                bcids = reduce(numpy.union1d, nodes)
                negids = numpy.copy(bcids)
                for bc in lbcs:
                    if bc.sub_domain in ["top", "bottom"]:
                        continue
                    # FunctionSpace with component is IndexedVFS
                    if bc.function_space().component is not None:
                        # For indexed VFS bcs, we encode the component
                        # in the high bits of the map value.
                        # That value is then negated to indicate to
                        # the generated code to discard the values
                        #
                        # So here we do:
                        #
                        # node = -(node + 2**(30-cmpt) + 1)
                        #
                        # And in the generated code we can then
                        # extract the information to discard the
                        # correct entries.
                        val = 2 ** (30 - bc.function_space().component)
                        # bcids is sorted, so use searchsorted to find indices
                        idx = numpy.searchsorted(bcids, bc.nodes)
                        negids[idx] += val
                        decorate = True
                node_list_bc = numpy.arange(self.node_set.total_size,
                                            dtype=numpy.int32)
                # Fix up for extruded, doesn't commute with indexedvfs
                # for now
                if self.extruded:
                    node_list_bc[bcids] = -10000000
                else:
                    node_list_bc[bcids] = -(negids + 1)
                new_entity_node_list = node_list_bc.take(entity_node_list)
            else:
                new_entity_node_list = entity_node_list

            val = op2.Map(entity_set, self.node_set,
                          map_arity,
                          new_entity_node_list,
                          ("%s_"+name) % (V.name),
                          offset=offset,
                          parent=parent,
                          bt_masks=self.bt_masks)

            if decorate:
                val = op2.DecoratedMap(val, vector_index=True)
            cache[lbcs] = val
            if implicit_bcs:
                return op2.DecoratedMap(val, implicit_bcs=implicit_bcs)
            return val


def get_shared_data(mesh, fiat_element):
    """Return the :class:`FunctionSpaceData` for the given
    element.

    :arg mesh: The mesh to build the function space data on.
    :arg fiat_element: A FIAT element.
    :raises ValueError: if mesh or fiat_element are invalid.
    :returns: a :class:`FunctionSpaceData` object with the shared
        data.
    """
    if not isinstance(mesh, mesh_mod.MeshTopology):
        raise ValueError("%s is not a MeshTopology" % mesh)
    if not isinstance(fiat_element, FIAT.finite_element.FiniteElement):
        raise ValueError("Can't create function space data from a %s" %
                         type(fiat_element))
    return FunctionSpaceData(mesh, fiat_element)
