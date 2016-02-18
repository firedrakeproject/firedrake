"""This module provides an object that encapsulates data that can be
shared between different :class:`~.FunctionSpace` objects.

The sharing is based on the idea of compatibility of function space
node layout.  The shared data is stored on the :func:`~.Mesh` the
function space is created on, since the created objects are
mesh-specific.  The key for this cache is a canonicalised version of
the function space's FIAT element's entity_dofs.

As such, this means that function spaces with the same *node*
ordering, but different numbers of dofs per node (e.g. FiniteElement
vs VectorElement) can share the PyOP2 Set and Map data.
"""

from __future__ import absolute_import

import numpy
import FIAT

from FIAT.finite_element import facet_support_dofs
from FIAT.tensor_product import horiz_facet_support_dofs, vert_facet_support_dofs

from coffee import base as ast

from pyop2 import op2

from firedrake import dmplex as dm_mod
from firedrake import halo as halo_mod
from firedrake import mesh as mesh_mod
from firedrake.petsc import PETSc


__all__ = ("get_shared_data", )


class FunctionSpaceData(object):
    """Function spaces with the same entity dofs share data.  This class
    stores that shared data.  It is cached on the mesh.

    :arg mesh: The mesh to share the data on.
    :arg fiat_element: The FIAT describing how nodes are attached to
       topological entities.
    """
    def __init__(self, mesh, fiat_element):
        entity_dofs = fiat_element.entity_dofs()
        nodes_per_entity = mesh.make_dofs_per_plex_entity(entity_dofs)
        plex = mesh._plex
        # Create the PetscSection mapping topological entities to functionspace nodes
        # For non-scalar valued function spaces, there are multiple dofs per node.
        global_numbering = plex.createSection([1], nodes_per_entity,
                                              perm=mesh._plex_renumbering)

        offset = mesh.make_offset(entity_dofs, fiat_element.space_dimension())

        cell_node_list = mesh.make_cell_node_list(global_numbering, entity_dofs)

        if plex.getStratumSize("interior_facets", 1) > 0:
            interior_facet_node_list = dm_mod.get_facet_nodes(mesh.interior_facets.facet_cell, cell_node_list)
        else:
            interior_facet_node_list = numpy.array([], dtype=numpy.int32)

        if plex.getStratumSize("exterior_facets", 1) > 0:
            exterior_facet_node_list = \
                dm_mod.get_facet_nodes(mesh.exterior_facets.facet_cell,
                                       cell_node_list)
        else:
            exterior_facet_node_list = numpy.array([], dtype=numpy.int32)

        # Use a DM to create the halo SFs
        dm = PETSc.DMShell().create()
        dm.setPointSF(plex.getPointSF())
        dm.setDefaultSection(global_numbering)
        node_classes = tuple(numpy.dot(nodes_per_entity, mesh._entity_classes))
        node_set = op2.Set(node_classes, halo=halo_mod.Halo(dm))
        # Don't need it any more, explicitly destroy.
        dm.destroy()

        extruded = bool(mesh.layers)
        if extruded:
            node_set = op2.ExtrudedSet(node_set, layers=mesh.layers)

        assert global_numbering.getStorageSize() == node_set.total_size

        bt_masks = None
        if extruded:
            bt_masks = {}
            # Compute the top and bottom masks to identify boundary dofs
            #
            # Sorting the keys of the closure entity dofs, the whole cell
            # comes last [-1], before that the horizontal facet [-2], before
            # that vertical facets [-3]. We need the horizontal facets here.
            closure_dofs = fiat_element.entity_closure_dofs()
            b_mask = closure_dofs[sorted(closure_dofs.keys())[-2]][0]
            t_mask = closure_dofs[sorted(closure_dofs.keys())[-2]][1]
            bt_masks["topological"] = (b_mask, t_mask)  # conversion to tuple
            # Geometric facet dofs
            facet_dofs = horiz_facet_support_dofs(fiat_element)
            bt_masks["geometric"] = (facet_dofs[0], facet_dofs[1])

        # Empty map caches. This is a sui generis cache
        # implementation because of the need to support boundary
        # conditions.
        self.map_caches = {mesh.cell_set: {},
                           mesh.interior_facets.set: {},
                           mesh.exterior_facets.set: {},
                           "boundary_node": {}}

        self.entity_node_lists = {mesh.cell_set: cell_node_list,
                                  mesh.interior_facets.set: interior_facet_node_list,
                                  mesh.exterior_facets.set: exterior_facet_node_list}

        self.node_set = node_set
        self.offset = offset
        self.bt_masks = bt_masks
        self.extruded = extruded
        self.mesh = mesh
        self.global_numbering = global_numbering

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
            if self.extruded:
                # This function is only called on extruded meshes when
                # asking for the nodes that live on the "vertical"
                # exterior facets.  Hence we don't need to worry about
                # horiz_facet_support_dofs as well.
                boundary_dofs = vert_facet_support_dofs(el)
            else:
                boundary_dofs = facet_support_dofs(el)

        nodes_per_facet = \
            len(boundary_dofs[0])

        # HACK ALERT
        # The facet set does not have a halo associated with it, since
        # we only construct halos for DoF sets.  Fortunately, this
        # loop is direct and we already have all the correct
        # information available locally.  So We fake a set of the
        # correct size and carry out a direct loop
        facet_set = op2.Set(self.mesh.exterior_facets.set.total_size)

        fs_dat = op2.Dat(facet_set**el.space_dimension(),
                         data=V.exterior_facet_node_map().values_with_halo)

        facet_dat = op2.Dat(facet_set**nodes_per_facet,
                            dtype=numpy.int32)

        # Ensure these come out in sorted order.
        local_facet_nodes = numpy.array(
            [boundary_dofs[e] for e in sorted(boundary_dofs.keys())])

        # Helper function to turn the inner index of an array into c
        # array literals.
        c_array = lambda xs: "{"+", ".join(map(str, xs))+"}"

        body = ast.Block([ast.Decl("int",
                                   ast.Symbol("l_nodes", (len(el.get_reference_element().topology[dim]),
                                                          nodes_per_facet)),
                                   init=ast.ArrayInit(c_array(map(c_array, local_facet_nodes))),
                                   qualifiers=["const"]),
                          ast.For(ast.Decl("int", "n", 0),
                                  ast.Less("n", nodes_per_facet),
                                  ast.Incr("n", 1),
                                  ast.Assign(ast.Symbol("facet_nodes", ("n",)),
                                             ast.Symbol("cell_nodes", ("l_nodes[facet[0]][n]",))))
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
    entity_dofs = fiat_element.entity_dofs()

    # Build key (sorted entity_dofs)
    key = []
    for k in sorted(entity_dofs.keys()):
        sub_key = [k]
        for sk in sorted(entity_dofs[k]):
            sub_key.append(tuple(entity_dofs[k][sk]))
        key.append(tuple(sub_key))
    key = tuple(key)

    try:
        return mesh._shared_data_cache[key]
    except KeyError:
        pass

    data = FunctionSpaceData(mesh, fiat_element)
    mesh._shared_data_cache[key] = data
    return data
