from __future__ import absolute_import
import numpy as np
import ufl
import weakref
from FIAT.finite_element import facet_support_dofs
from FIAT.tensor_finite_element import horiz_facet_support_dofs, vert_facet_support_dofs

import coffee.base as ast

from pyop2 import op2
from pyop2.caching import ObjectCached
from pyop2.utils import flatten

from firedrake.petsc import PETSc
from firedrake import dmplex
from firedrake import fiat_utils
import firedrake.mesh as mesh_t
from firedrake import halo
from firedrake import utils


__all__ = ['FunctionSpace', 'VectorFunctionSpace',
           'TensorFunctionSpace', 'MixedFunctionSpace',
           'IndexedFunctionSpace']


class FunctionSpaceMeta(type):
    """Metaclass for function spaces.

    All function space functionality requires only a mesh topology,
    the only exceptions are:
      .mesh()
      .ufl_element()
    :class:`WithGeometry` decorates a function space with a mesh
    geometry, overriding the above methods.  Since instance checks for
    various kinds of function spaces are very common, this metaclass
    makes sure they also work via the geometry decorator.
    """
    def __instancecheck__(self, other):
        if isinstance(other, WithGeometry):
            other = other.topological
        return super(FunctionSpaceMeta, self).__instancecheck__(other)


class FunctionSpaceBase(ObjectCached):
    """Base class for :class:`.FunctionSpace`, :class:`.VectorFunctionSpace` and
    :class:`.MixedFunctionSpace`.

    .. note ::

        Users should not directly create objects of this class, but one of its
        derived types.
    """

    __metaclass__ = FunctionSpaceMeta

    def __new__(cls, mesh, element, name=None, shape=()):
        """
        :param mesh: :class:`MeshTopology` to build this space on
        :param element: :class:`ufl.FiniteElementBase` to build this space from
        :param name: user-defined name for this space
        :param shape: shape of a :class:`.VectorFunctionSpace` or :class:`.TensorFunctionSpace`
        """

        assert mesh.ufl_cell() == element.cell()

        self = super(FunctionSpaceBase, cls).__new__(cls, mesh, element, name, shape)
        if self._initialized:
            return self

        self._mesh = mesh
        self._ufl_element = element
        self.name = name
        self._shape = shape

        # Compute the FIAT version of the UFL element above
        self.fiat_element = fiat_utils.fiat_from_ufl_element(element)

        entity_dofs = self.fiat_element.entity_dofs()
        dofs_per_entity = mesh.make_dofs_per_plex_entity(entity_dofs)

        self.extruded = bool(mesh.layers)
        self.offset = mesh.make_offset(entity_dofs,
                                       self.fiat_element.space_dimension())

        if mesh.layers:
            # Compute the top and bottom masks to identify boundary dofs
            #
            # Sorting the keys of the closure entity dofs, the whole cell
            # comes last [-1], before that the horizontal facet [-2], before
            # that vertical facets [-3]. We need the horizontal facets here.
            closure_dofs = self.fiat_element.entity_closure_dofs()
            b_mask = closure_dofs[sorted(closure_dofs.keys())[-2]][0]
            t_mask = closure_dofs[sorted(closure_dofs.keys())[-2]][1]
            self.bt_masks = {}
            self.bt_masks["topological"] = (b_mask, t_mask)  # conversion to tuple
            # Geometric facet dofs
            facet_dofs = horiz_facet_support_dofs(self.fiat_element)
            self.bt_masks["geometric"] = (facet_dofs[0], facet_dofs[1])
        else:
            self.bt_masks = None

        dm = PETSc.DMShell().create()
        dm.setAttr('__fs__', weakref.ref(self))
        dm.setPointSF(mesh._plex.getPointSF())
        # Create the PetscSection mapping topological entities to DoFs
        sec = mesh._plex.createSection([1], dofs_per_entity,
                                       perm=mesh._plex_renumbering)
        dm.setDefaultSection(sec)
        self._global_numbering = sec
        self._dm = dm
        self._ises = None
        self._halo = halo.Halo(dm)

        # Compute entity class offsets
        self.dof_classes = [0, 0, 0, 0]
        for d in range(mesh._plex.getDimension()+1):
            ndofs = dofs_per_entity[d]
            for i in range(4):
                self.dof_classes[i] += ndofs * mesh._entity_classes[d, i]

        # Tell the DM about the layout of the global vector
        with self.make_dat().vec_ro as v:
            self._dm.setGlobalVector(v.duplicate())

        self._node_count = self._global_numbering.getStorageSize()

        self.cell_node_list = mesh.make_cell_node_list(self._global_numbering,
                                                       entity_dofs)

        if mesh._plex.getStratumSize("interior_facets", 1) > 0:
            self.interior_facet_node_list = \
                dmplex.get_facet_nodes(mesh.interior_facets.facet_cell,
                                       self.cell_node_list)
        else:
            self.interior_facet_node_list = np.array([], dtype=np.int32)

        if mesh._plex.getStratumSize("exterior_facets", 1) > 0:
            self.exterior_facet_node_list = \
                dmplex.get_facet_nodes(mesh.exterior_facets.facet_cell,
                                       self.cell_node_list)
        else:
            self.exterior_facet_node_list = np.array([], dtype=np.int32)

        # Empty map caches. This is a sui generis cache
        # implementation because of the need to support boundary
        # conditions.
        self._cell_node_map_cache = {}
        self._exterior_facet_map_cache = {}
        self._interior_facet_map_cache = {}

        self._initialized = True
        return self

    @classmethod
    def _process_args(cls, *args, **kwargs):
        # Already processed
        return args, kwargs

    @classmethod
    def _cache_key(cls, element, name=None, shape=()):
        # Key on processed arguments
        return element, name, shape

    @property
    def index(self):
        """Position of this :class:`FunctionSpaceBase` in the
        :class:`.MixedFunctionSpace` it was extracted from."""
        return None

    @property
    def node_count(self):
        """The number of global nodes in the function space. For a
        plain :class:`.FunctionSpace` this is equal to
        :attr:`dof_count`, however for a :class:`.VectorFunctionSpace`,
        the :attr:`dof_count`, is :attr:`dim` times the
        :attr:`node_count`."""

        return self._node_count

    @property
    def dof_count(self):
        """The number of global degrees of freedom in the function
        space. Cf. :attr:`node_count`."""

        return self.node_count*self.dim

    @utils.cached_property
    def node_set(self):
        """A :class:`pyop2.Set` containing the nodes of this
        :class:`.FunctionSpace`. One or (for
        :class:`.VectorFunctionSpace`\s) more degrees of freedom are
        stored at each node.
        """

        name = "%s_nodes" % self.name
        if self._halo:
            s = op2.Set(self.dof_classes, name,
                        halo=self._halo)
            if self.extruded:
                return op2.ExtrudedSet(s, layers=self._mesh.layers)
            return s
        else:
            s = op2.Set(self.node_count, name)
            if self.extruded:
                return op2.ExtrudedSet(s, layers=self._mesh.layers)
            return s

    @utils.cached_property
    def dof_dset(self):
        """A :class:`pyop2.DataSet` containing the degrees of freedom of
        this :class:`.FunctionSpace`."""
        return op2.DataSet(self.node_set, self.shape or 1, name="%s_nodes_dset" % self.name)

    def make_dat(self, val=None, valuetype=None, name=None, uid=None):
        """Return a newly allocated :class:`pyop2.Dat` defined on the
        :attr:`dof_dset` of this :class:`.Function`."""
        return op2.Dat(self.dof_dset, val, valuetype, name, uid=uid)

    def cell_node_map(self, bcs=None):
        """Return the :class:`pyop2.Map` from interior facets to
        function space nodes. If present, bcs must be a tuple of
        :class:`.DirichletBC`\s. In this case, the facet_node_map will return
        negative node indices where boundary conditions should be
        applied. Where a PETSc matrix is employed, this will cause the
        corresponding values to be discarded during matrix assembly."""

        if bcs:
            parent = self.cell_node_map()
        else:
            parent = None

        return self._map_cache(self._cell_node_map_cache,
                               self._mesh.cell_set,
                               self.cell_node_list,
                               self.fiat_element.space_dimension(),
                               bcs,
                               "cell_node",
                               self.offset,
                               parent)

    def interior_facet_node_map(self, bcs=None):
        """Return the :class:`pyop2.Map` from interior facets to
        function space nodes. If present, bcs must be a tuple of
        :class:`.DirichletBC`\s. In this case, the facet_node_map will return
        negative node indices where boundary conditions should be
        applied. Where a PETSc matrix is employed, this will cause the
        corresponding values to be discarded during matrix assembly."""

        if bcs:
            parent = self.interior_facet_node_map()
        else:
            parent = None

        offset = self.cell_node_map().offset
        map = self._map_cache(self._interior_facet_map_cache,
                              self._mesh.interior_facets.set,
                              self.interior_facet_node_list,
                              2*self.fiat_element.space_dimension(),
                              bcs,
                              "interior_facet_node",
                              offset=np.append(offset, offset),
                              parent=parent)
        map.factors = (self._mesh.interior_facets.facet_cell_map,
                       self.cell_node_map())
        return map

    def exterior_facet_node_map(self, bcs=None):
        """Return the :class:`pyop2.Map` from exterior facets to
        function space nodes. If present, bcs must be a tuple of
        :class:`.DirichletBC`\s. In this case, the facet_node_map will return
        negative node indices where boundary conditions should be
        applied. Where a PETSc matrix is employed, this will cause the
        corresponding values to be discarded during matrix assembly."""

        if bcs:
            parent = self.exterior_facet_node_map()
        else:
            parent = None

        facet_set = self._mesh.exterior_facets.set
        if isinstance(self._mesh.topology, mesh_t.ExtrudedMeshTopology):
            name = "extruded_exterior_facet_node"
            offset = self.offset
        else:
            name = "exterior_facet_node"
            offset = None
        return self._map_cache(self._exterior_facet_map_cache,
                               facet_set,
                               self.exterior_facet_node_list,
                               self.fiat_element.space_dimension(),
                               bcs,
                               name,
                               parent=parent,
                               offset=offset)

    def bottom_nodes(self, method='topological'):
        """Return a list of the bottom boundary nodes of the extruded mesh.
        The bottom mask is applied to every bottom layer cell to get the
        dof ids."""
        try:
            mask = self.bt_masks[method][0]
        except KeyError:
            raise ValueError("Unknown boundary condition method %s" % method)
        return np.unique(self.cell_node_list[:, mask])

    def top_nodes(self, method='topological'):
        """Return a list of the top boundary nodes of the extruded mesh.
        The top mask is applied to every top layer cell to get the dof ids."""
        try:
            mask = self.bt_masks[method][1]
        except KeyError:
            raise ValueError("Unknown boundary condition method %s" % method)
        voffs = self.offset.take(mask)*(self._mesh.layers-2)
        return np.unique(self.cell_node_list[:, mask] + voffs)

    def _map_cache(self, cache, entity_set, entity_node_list, map_arity, bcs, name,
                   offset=None, parent=None):
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
                bcs = None
            if len(implicit_bcs) == 0:
                implicit_bcs = None
        else:
            implicit_bcs = None
        if bcs is None:
            # Empty tuple if no bcs found.  This is so that matrix
            # assembly, which uses a set to keep track of the bcs
            # applied to matrix hits the cache when that set is
            # empty.  tuple(set([])) == tuple().
            lbcs = tuple()
        else:
            for bc in bcs:
                fs = bc.function_space()
                if isinstance(fs, IndexedVFS):
                    fs = fs._parent
                if fs.topological != self:
                    raise RuntimeError("DirichletBC defined on a different FunctionSpace!")
            # Ensure bcs is a tuple in a canonical order for the hash key.
            lbcs = tuple(sorted(bcs, key=lambda bc: bc.__hash__()))
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
                bcids = reduce(np.union1d, nodes)
                negids = np.copy(bcids)
                for bc in lbcs:
                    if bc.sub_domain in ["top", "bottom"]:
                        continue
                    if isinstance(bc.function_space(), IndexedVFS):
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
                        val = 2 ** (30 - bc.function_space().index)
                        # bcids is sorted, so use searchsorted to find indices
                        idx = np.searchsorted(bcids, bc.nodes)
                        negids[idx] += val
                        decorate = True
                node_list_bc = np.arange(self.node_count, dtype=np.int32)
                # Fix up for extruded, doesn't commute with indexedvfs for now
                if isinstance(self.mesh().topology, mesh_t.ExtrudedMeshTopology):
                    node_list_bc[bcids] = -10000000
                else:
                    node_list_bc[bcids] = -(negids + 1)
                new_entity_node_list = node_list_bc.take(entity_node_list)
            else:
                new_entity_node_list = entity_node_list

            val = op2.Map(entity_set, self.node_set,
                          map_arity,
                          new_entity_node_list,
                          ("%s_"+name) % (self.name),
                          offset,
                          parent,
                          self.bt_masks)

            if decorate:
                val = op2.DecoratedMap(val, vector_index=True)
            cache[lbcs] = val
            if implicit_bcs:
                return op2.DecoratedMap(val, implicit_bcs=implicit_bcs)
            return val

    @utils.memoize
    def exterior_facet_boundary_node_map(self, method):
        '''The :class:`pyop2.Map` from exterior facets to the nodes on
        those facets. Note that this differs from
        :meth:`exterior_facet_node_map` in that only surface nodes
        are referenced, not all nodes in cells touching the surface.

        :arg method: The method for determining boundary nodes. See
            :class:`~.bcs.DirichletBC`.
        '''

        el = self.fiat_element

        dim = self._mesh.facet_dimension()

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
        facet_set = op2.Set(self._mesh.exterior_facets.set.total_size)

        fs_dat = op2.Dat(facet_set**el.space_dimension(),
                         data=self.exterior_facet_node_map().values_with_halo)

        facet_dat = op2.Dat(facet_set**nodes_per_facet,
                            dtype=np.int32)

        local_facet_nodes = np.array(
            [dofs for e, dofs in boundary_dofs.iteritems()])

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

        local_facet_dat = op2.Dat(facet_set ** self._mesh.exterior_facets._rank,
                                  self._mesh.exterior_facets.local_facet_dat.data_ro_with_halos,
                                  dtype=np.uintc)
        op2.par_loop(kernel, facet_set,
                     fs_dat(op2.READ),
                     facet_dat(op2.WRITE),
                     local_facet_dat(op2.READ))

        if isinstance(self._mesh.topology, mesh_t.ExtrudedMeshTopology):
            offset = self.offset[boundary_dofs[0]]
        else:
            offset = None
        return op2.Map(facet_set, self.node_set,
                       nodes_per_facet,
                       facet_dat.data_ro_with_halos,
                       name="exterior_facet_boundary_node",
                       offset=offset)

    @property
    def shape(self):
        return self._shape

    @property
    def rank(self):
        return len(self.shape)

    @property
    def dim(self):
        """The product of the :attr:`.dim` of the :class:`.FunctionSpace`."""
        return np.prod(self.shape, dtype=int)

    @property
    def topological(self):
        """Function space on a mesh topology."""
        return self

    def mesh(self):
        return self._mesh

    def ufl_element(self):
        return self._ufl_element

    def __len__(self):
        return 1

    def __iter__(self):
        yield self

    def __getitem__(self, i):
        """Return ``self`` if ``i`` is 0 or raise an exception."""
        if i != 0:
            raise IndexError("Only index 0 supported on a FunctionSpace")
        return self

    def __mul__(self, other):
        """Create a :class:`.MixedFunctionSpace` composed of this
        :class:`.FunctionSpace` and other"""
        return MixedFunctionSpace((self, other))


class WithGeometry(object):
    def __init__(self, function_space, mesh):
        function_space = function_space.topological
        assert mesh.topology is function_space.mesh()
        assert mesh.topology is not mesh

        self._topological = function_space
        self._mesh = mesh

        self._ufl_element = function_space.ufl_element().reconstruct(domain=mesh)

        if hasattr(function_space, '_parent'):
            self._parent = WithGeometry(function_space._parent, mesh)

        if hasattr(function_space, '_fs'):
            self._fs = WithGeometry(function_space._fs, mesh)

    def mesh(self):
        return self._mesh

    def ufl_element(self):
        return self._ufl_element

    def __eq__(self, other):
        return self._topological == other._topological and self._mesh is other._mesh

    def __ne__(self, other):
        return not self.__eq__(other)

    def __len__(self):
        return len(self._topological)

    def split(self):
        spaces = []
        for subspace in self._topological.split():
            spaces.append(WithGeometry(subspace, self._mesh))
        return spaces

    def __iter__(self):
        for subspace in self._topological:
            yield WithGeometry(subspace, self._mesh)

    def __getitem__(self, i):
        return WithGeometry(self._topological[i], self._mesh)

    def sub(self, i):
        return WithGeometry(self._topological.sub(i), self._mesh)

    def __mul__(self, other):
        """Create a :class:`.MixedFunctionSpace` composed of this
        :class:`.FunctionSpace` and other"""
        return MixedFunctionSpace((self, other))

    @property
    def topological(self):
        """Function space on a mesh topology."""
        return self._topological

    def __getattr__(self, name):
        return getattr(self._topological, name)


class FunctionSpace(FunctionSpaceBase):
    def __new__(cls, mesh, family, degree=None, name=None, vfamily=None, vdegree=None):
        """Create a function space

        :arg mesh: mesh to build the function space on
        :arg family: string describing function space family, or an
            :class:`~ufl.finiteelement.outerproductelement.OuterProductElement`
        :arg degree: degree of the function space
        :arg name: (optional) name of the function space
        :arg vfamily: family of function space in vertical dimension
            (extruded meshes only)
        :arg vdegree: degree of function space in vertical dimension
            (extruded meshes only)

        If the mesh is an extruded mesh, and the ``family`` argument is a
        :class:`~ufl.finiteelement.outerproductelement.OuterProductElement`,
        ``degree``, ``vfamily`` and ``vdegree`` are ignored, since the
        ``family`` provides all necessary information, otherwise a
        :class:`~ufl.finiteelement.outerproductelement.OuterProductElement`
        is built from the (``family``, ``degree``) and (``vfamily``,
        ``vdegree``) pair.  If the ``vfamily`` and ``vdegree`` are not
        provided, the vertical element defaults to the same as the
        (``family``, ``degree``) pair.

        If the mesh is not an extruded mesh, ``vfamily`` and
        ``vdegree`` are ignored.
        """
        mesh.init()
        mesh_t = mesh.topology

        # Two choices:
        # 1) Pass in mesh, family, degree to generate a simple function space.
        # 2) Set up the function space using FiniteElement, EnrichedElement,
        #    OuterProductElement and so on.
        if isinstance(family, ufl.FiniteElementBase):
            # Second case...
            element = family
        else:
            # First case...
            if isinstance(mesh_t.ufl_cell(), ufl.OuterProductCell) and vfamily is not None and vdegree is not None:
                # If OuterProductCell, make the OuterProductElement
                la = ufl.FiniteElement(family,
                                       domain=mesh_t._base_mesh.ufl_cell(),
                                       degree=degree)
                # If second element was passed in, use it
                lb = ufl.FiniteElement(vfamily,
                                       domain=ufl.interval,
                                       degree=vdegree)
                # Now make the OuterProductElement
                element = ufl.OuterProductElement(la, lb)
            else:
                # Otherwise, just make the element
                element = ufl.FiniteElement(family,
                                            domain=mesh_t.ufl_cell(),
                                            degree=degree)
        self = super(FunctionSpace, cls).__new__(cls, mesh_t, element, name=name)
        if mesh is not mesh_t:
            self = WithGeometry(self, mesh)
        return self


class VectorFunctionSpace(FunctionSpaceBase):
    """A vector finite element :class:`FunctionSpace`."""

    def __new__(cls, mesh, family, degree=None, dim=None, name=None, vfamily=None, vdegree=None):
        mesh.init()
        mesh_t = mesh.topology

        # VectorFunctionSpace dimension defaults to the geometric dimension of the mesh.
        dim = dim or mesh.ufl_cell().geometric_dimension()

        if isinstance(mesh_t.ufl_cell(), ufl.OuterProductCell) and isinstance(family, ufl.OuterProductElement):
            element = ufl.OuterProductVectorElement(family, dim=dim)
        elif isinstance(mesh_t.ufl_cell(), ufl.OuterProductCell) and vfamily is not None and vdegree is not None:
            la = ufl.FiniteElement(family,
                                   domain=mesh_t._base_mesh.ufl_cell(),
                                   degree=degree)
            lb = ufl.FiniteElement(vfamily,
                                   domain=ufl.interval,
                                   degree=vdegree)
            element = ufl.OuterProductVectorElement(la, lb, dim=dim)
        else:
            element = ufl.VectorElement(family,
                                        domain=mesh_t.ufl_cell(),
                                        degree=degree, dim=dim)

        self = super(VectorFunctionSpace, cls).__new__(cls, mesh_t, element, name=name, shape=(dim,))
        if mesh is not mesh_t:
            self = WithGeometry(self, mesh)
        return self

    def sub(self, i):
        """Return an :class:`IndexedVFS` for the requested component.

        This can be used to apply :class:`~.DirichletBC`\s to components
        of a :class:`VectorFunctionSpace`."""
        return IndexedVFS(self, i)


class TensorFunctionSpace(FunctionSpaceBase):
    """A tensor-valued :class:`FunctionSpace`."""
    def __new__(cls, mesh, family, degree=None, shape=None, symmetry=None, name=None, vfamily=None, vdegree=None):
        mesh.init()
        mesh_t = mesh.topology

        # TensorFunctionSpace shape defaults to the (gdim, gdim)
        shape = shape or (mesh.ufl_cell().geometric_dimension(),) * 2

        if isinstance(mesh_t.ufl_cell(), ufl.OuterProductCell):
            raise NotImplementedError("TensorFunctionSpace on extruded meshes not implemented")
        else:
            element = ufl.TensorElement(family, domain=mesh_t.ufl_cell(),
                                        degree=degree, shape=shape,
                                        symmetry=symmetry)

        self = super(TensorFunctionSpace, cls).__new__(cls, mesh_t, element, name=name, shape=shape)
        if mesh is not mesh_t:
            self = WithGeometry(self, mesh)
        return self


class MixedFunctionSpace(FunctionSpaceBase):
    """A mixed finite element :class:`FunctionSpace`."""

    def __new__(cls, spaces, name=None):
        """
        :param spaces: a list (or tuple) of :class:`FunctionSpace`\s

        The function space may be created as ::

            V = MixedFunctionSpace(spaces)

        ``spaces`` may consist of multiple occurances of the same space: ::

            P1  = FunctionSpace(mesh, "CG", 1)
            P2v = VectorFunctionSpace(mesh, "Lagrange", 2)

            ME  = MixedFunctionSpace([P2v, P1, P1, P1])
        """

        # Check that function spaces are on the same mesh
        meshes = [space.mesh() for space in spaces]
        for i in xrange(1, len(meshes)):
            if meshes[i] is not meshes[0]:
                raise ValueError("All function spaces must be defined on the same mesh!")

        # Select mesh
        mesh = meshes[0]

        # Get topological spaces
        spaces = flatten(spaces)
        if mesh is mesh.topology:
            spaces = tuple(spaces)
        else:
            spaces = tuple(space.topological for space in spaces)

        # Ask object from cache
        self = ObjectCached.__new__(cls, mesh, spaces, name)
        if not self._initialized:
            self._spaces = [IndexedFunctionSpace(s, i, self)
                            for i, s in enumerate(spaces)]
            self._mesh = mesh.topology
            self._ufl_element = ufl.MixedElement(*[fs.ufl_element() for fs in spaces])
            self.name = name or '_'.join(str(s.name) for s in spaces)
            self._initialized = True
            dm = PETSc.DMShell().create()
            with self.make_dat().vec_ro as v:
                dm.setGlobalVector(v.duplicate())
            dm.setAttr('__fs__', weakref.ref(self))
            dm.setCreateFieldDecomposition(self.create_field_decomp)
            dm.setCreateSubDM(self.create_subdm)
            self._dm = dm
            self._ises = self.dof_dset.field_ises
            self._subspaces = []

        if mesh is not mesh.topology:
            self = WithGeometry(self, mesh)
        return self

    @classmethod
    def _cache_key(cls, spaces, name):
        return spaces, name

    @classmethod
    def create_subdm(cls, dm, fields, *args, **kwargs):
        W = dm.getAttr('__fs__')()
        if len(fields) == 1:
            # Subspace is just a single FunctionSpace.
            subspace = W[fields[0]]
        else:
            # Need to build an MFS for the subspace
            subspace = MixedFunctionSpace([W[f] for f in fields])
        # Sub-DM is just the DM belonging to the subspace.
        subdm = subspace._dm
        # Keep hold of strong reference, to created subspace (given we
        # only hold a weakref in the shell DM)
        W._subspaces.append(subspace)
        # Index set mapping from W into subspace.
        iset = PETSc.IS().createGeneral(np.concatenate([W._ises[f].indices for f in fields]))
        return iset, subdm

    @classmethod
    def create_field_decomp(cls, dm, *args, **kwargs):
        W = dm.getAttr('__fs__')()
        # Don't pass split number if name is None (this way the
        # recursively created splits have the names you want)
        names = [s.name for s in W]
        dms = [V._dm for V in W]
        return names, W._ises, dms

    def split(self):
        """The list of :class:`FunctionSpace`\s of which this
        :class:`MixedFunctionSpace` is composed."""
        return self._spaces

    def sub(self, i):
        """Return the `i`th :class:`FunctionSpace` in this
        :class:`MixedFunctionSpace`."""
        return self[i]

    def num_sub_spaces(self):
        """Return the number of :class:`FunctionSpace`\s of which this
        :class:`MixedFunctionSpace` is composed."""
        return len(self)

    def __len__(self):
        """Return the number of :class:`FunctionSpace`\s of which this
        :class:`MixedFunctionSpace` is composed."""
        return len(self._spaces)

    def __getitem__(self, i):
        """Return the `i`th :class:`FunctionSpace` in this
        :class:`MixedFunctionSpace`."""
        return self._spaces[i]

    def __iter__(self):
        for s in self._spaces:
            yield s

    @property
    def dim(self):
        """Return the sum of the :attr:`FunctionSpace.dim`\s of the
        :class:`FunctionSpace`\s this :class:`MixedFunctionSpace` is
        composed of."""
        return sum(fs.dim for fs in self._spaces)

    @property
    def node_count(self):
        """Return a tuple of :attr:`FunctionSpace.node_count`\s of the
        :class:`FunctionSpace`\s of which this :class:`MixedFunctionSpace` is
        composed."""
        return tuple(fs.node_count for fs in self._spaces)

    @property
    def dof_count(self):
        """Return a tuple of :attr:`FunctionSpace.dof_count`\s of the
        :class:`FunctionSpace`\s of which this :class:`MixedFunctionSpace` is
        composed."""
        return tuple(fs.dof_count for fs in self._spaces)

    @utils.cached_property
    def node_set(self):
        """A :class:`pyop2.MixedSet` containing the nodes of this
        :class:`MixedFunctionSpace`. This is composed of the
        :attr:`FunctionSpace.node_set`\s of the underlying
        :class:`FunctionSpace`\s this :class:`MixedFunctionSpace` is
        composed of one or (for VectorFunctionSpaces) more degrees of freedom
        are stored at each node."""
        return op2.MixedSet(s.node_set for s in self._spaces)

    @utils.cached_property
    def dof_dset(self):
        """A :class:`pyop2.MixedDataSet` containing the degrees of freedom of
        this :class:`MixedFunctionSpace`. This is composed of the
        :attr:`FunctionSpace.dof_dset`\s of the underlying
        :class:`FunctionSpace`\s of which this :class:`MixedFunctionSpace` is
        composed."""
        return op2.MixedDataSet(s.dof_dset for s in self._spaces)

    def cell_node_map(self, bcs=None):
        """A :class:`pyop2.MixedMap` from the :attr:`Mesh.cell_set` of the
        underlying mesh to the :attr:`node_set` of this
        :class:`MixedFunctionSpace`. This is composed of the
        :attr:`FunctionSpace.cell_node_map`\s of the underlying
        :class:`FunctionSpace`\s of which this :class:`MixedFunctionSpace` is
        composed."""
        # FIXME: these want caching of sorts
        bc_list = [[] for _ in self]
        if bcs:
            for bc in bcs:
                bc_list[bc.function_space().index].append(bc)
        return op2.MixedMap(s.cell_node_map(bc_list[i])
                            for i, s in enumerate(self._spaces))

    def interior_facet_node_map(self, bcs=None):
        """Return the :class:`pyop2.MixedMap` from interior facets to
        function space nodes. If present, bcs must be a tuple of
        :class:`.DirichletBC`\s. In this case, the facet_node_map will return
        negative node indices where boundary conditions should be
        applied. Where a PETSc matrix is employed, this will cause the
        corresponding values to be discarded during matrix assembly."""
        # FIXME: these want caching of sorts
        bc_list = [[] for _ in self]
        if bcs:
            for bc in bcs:
                bc_list[bc.function_space().index].append(bc)
        return op2.MixedMap(s.interior_facet_node_map(bc_list[i])
                            for i, s in enumerate(self._spaces))

    def exterior_facet_node_map(self, bcs=None):
        """Return the :class:`pyop2.Map` from exterior facets to
        function space nodes. If present, bcs must be a tuple of
        :class:`.DirichletBC`\s. In this case, the facet_node_map will return
        negative node indices where boundary conditions should be
        applied. Where a PETSc matrix is employed, this will cause the
        corresponding values to be discarded during matrix assembly."""
        # FIXME: these want caching of sorts
        bc_list = [[] for _ in self]
        if bcs:
            for bc in bcs:
                bc_list[bc.function_space().index].append(bc)
        return op2.MixedMap(s.exterior_facet_node_map(bc_list[i])
                            for i, s in enumerate(self._spaces))

    @utils.cached_property
    def exterior_facet_boundary_node_map(self):
        '''The :class:`pyop2.MixedMap` from exterior facets to the nodes on
        those facets. Note that this differs from
        :meth:`exterior_facet_node_map` in that only surface nodes
        are referenced, not all nodes in cells touching the surface.'''
        return op2.MixedMap(s.exterior_facet_boundary_node_map for s in self._spaces)

    def make_dat(self, val=None, valuetype=None, name=None, uid=None):
        """Return a newly allocated :class:`pyop2.MixedDat` defined on the
        :attr:`dof_dset` of this :class:`MixedFunctionSpace`."""
        if val is not None:
            assert len(val) == len(self)
        else:
            val = [None for _ in self]
        return op2.MixedDat(s.make_dat(v, valuetype, "%s[cmpt-%d]" % (name, i), utils._new_uid())
                            for i, (s, v) in enumerate(zip(self._spaces, val)))


class IndexedVFS(FunctionSpaceBase):
    """A helper class used to keep track of indexing of a
    :class:`VectorFunctionSpace`.

    Users should not instantiate this by hand.  Instead call
    :meth:`VectorFunctionSpace.sub`."""
    def __new__(cls, parent, index):
        assert isinstance(parent, VectorFunctionSpace), "Only valid for VFS"
        assert 0 <= index < parent.dim, \
            "Invalid index %d, not in [0, %d)" % (index, parent.dim)
        if index > 2:
            raise NotImplementedError("Indexing VFS not implemented for index > 2")
        element = parent._ufl_element.sub_elements()[0]
        self = object.__new__(cls)
        self._delegate = FunctionSpace(parent.mesh(), element)
        self._parent = parent
        self._index = index
        self._fs = parent
        self.node_set
        self.dof_dset
        return self

    @property
    def index(self):
        """Position of this :class:`FunctionSpaceBase` in the
        :class:`.MixedFunctionSpace` it was extracted from."""
        return self._index

    @utils.cached_property
    def node_set(self):
        return self._parent.node_set

    def __getattr__(self, name):
        return getattr(self._delegate, name)


class IndexedFunctionSpace(FunctionSpaceBase):
    """A :class:`.FunctionSpaceBase` with an index to indicate which position
    it has as part of a :class:`MixedFunctionSpace`."""

    def __new__(cls, fs, index, parent):
        """
        :param fs: the :class:`.FunctionSpaceBase` that was extracted
        :param index: the position in the parent :class:`MixedFunctionSpace`
        :param parent: the parent :class:`MixedFunctionSpace`
        """
        self = object.__new__(cls)
        # If the function space was extracted from a mixed function space,
        # extract the underlying component space
        if isinstance(fs, IndexedFunctionSpace):
            fs = fs._fs
        # Override the __class__ to make instance checks on the type of the
        # wrapped function space work as expected
        self.__class__ = type(fs.__class__.__name__,
                              (self.__class__, fs.__class__), {})
        self._fs = fs
        self._index = index
        self._parent = parent
        return self

    @property
    def index(self):
        """Position of this :class:`FunctionSpaceBase` in the
        :class:`.MixedFunctionSpace` it was extracted from."""
        return self._index

    def __getattr__(self, name):
        return getattr(self._fs, name)

    def __repr__(self):
        return "<IndexedFunctionSpace: %r at %d>" % (FunctionSpaceBase.__repr__(self._fs), self._index)

    @property
    def node_set(self):
        """A :class:`pyop2.Set` containing the nodes of this
        :class:`FunctionSpace`. One or (for VectorFunctionSpaces) more degrees
        of freedom are stored at each node."""
        return self._fs.node_set

    @property
    def dof_dset(self):
        """A :class:`pyop2.DataSet` containing the degrees of freedom of
        this :class:`FunctionSpace`."""
        return self._fs.dof_dset

    @property
    def exterior_facet_boundary_node_map(self):
        '''The :class:`pyop2.Map` from exterior facets to the nodes on
        those facets. Note that this differs from
        :meth:`exterior_facet_node_map` in that only surface nodes
        are referenced, not all nodes in cells touching the surface.'''
        return self._fs.exterior_facet_boundary_node_map
