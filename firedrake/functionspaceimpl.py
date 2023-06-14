r"""
This module provides the implementations of :class:`~.FunctionSpace`
and :class:`~.MixedFunctionSpace` objects, along with some utility
classes for attaching extra information to instances of these.
"""

import abc
import collections
from collections import OrderedDict
import numbers

import numpy

import ufl

from pyop2 import op2, mpi

from firedrake import dmhooks, utils
from firedrake.functionspacedata import get_shared_data, create_element
from firedrake.mesh import MeshGeometry
from firedrake.petsc import PETSc


class AbstractFunctionSpace(ufl.FunctionSpace, abc.ABC, collections.abc.Sized, collections.abc.Iterable):
    r"""A function space.

    A :class:`FunctionSpace` associates degrees of freedom with
    topological mesh entities.  The degree of freedom mapping is
    determined from the provided element.

    :arg mesh: The :func:`~.Mesh` to build the function space on.
    :arg element: The :class:`~ufl.classes.FiniteElementBase` describing the
        degrees of freedom.
    :kwarg name: An optional name for this :class:`FunctionSpace`,
        useful for later identification.

    The element can be a essentially any
    :class:`~ufl.classes.FiniteElementBase`, except for a
    :class:`~ufl.classes.MixedElement`, for which one should use the
    :class:`MixedFunctionSpace` constructor.

    To determine whether the space is scalar-, vector- or
    tensor-valued, one should inspect the :attr:`rank` of the
    resulting object.  Note that function spaces created on
    *intrinsically* vector-valued finite elements (such as the
    Raviart-Thomas space) have ``rank`` 0.

    .. warning::

       Users should not build a :class:`FunctionSpace` directly, instead
       they should use the utility :func:`~.FunctionSpace` function,
       which provides extra error checking and argument sanitising.


    :arg mesh: The mesh with geometric information to use.
    :arg element: The UFL element.
    :arg component: The component of this space in a parent vector
        element space, or ``None``.
    """
    #FIXME docstring
    def __init__(self, mesh, element, component=None, parent=None, name=None):
        if not isinstance(mesh, MeshGeometry):
            raise TypeError("Mesh must be of type MeshGeometry")

        if component is not None and not isinstance(component, numbers.Integral):
            raise TypeError("component must either be None or an integer")
        if type(element) is ufl.MixedElement:
            raise ValueError("Can't create FunctionSpace for MixedElement")

        super().__init__(mesh, element)
        self.component = component
        self.parent = parent
        self.comm = mesh.comm
        self._comm = mpi.internal_comm(mesh.comm)

        sdata = get_shared_data(mesh.topology, element)

        self._shared_data = sdata
        self._mesh = mesh

        self.value_size = int(numpy.prod(self.shape, dtype=int))
        r"""The total number of degrees of freedom at each function
        space node."""
        self.name = name
        r"""The (optional) descriptive name for this space."""

        # Need to create finat element again as sdata does not
        # want to carry finat_element.
        self.finat_element = create_element(element)
        # Used for reconstruction of mixed/component spaces.
        # sdata carries real_tensorproduct.
        self.real_tensorproduct = sdata.real_tensorproduct
        self.extruded = sdata.extruded
        self.offset = sdata.offset
        self.offset_quotient = sdata.offset_quotient

    def __del__(self):
        if hasattr(self, "_comm"):
            mpi.decref(self._comm)

    # These properties are overridden in ProxyFunctionSpaces, but are
    # provided by FunctionSpace so that we don't have to special case.
    index = None
    r"""The position of this space in its parent
    :class:`MixedFunctionSpace`, or ``None``."""

    parent = None
    r"""The parent space if this space was extracted from one, or ``None``."""

    component = None
    r"""The component of this space in its parent VectorElement space, or
    ``None``."""

    @property
    def node_set(self):
        r"""A :class:`pyop2.types.set.Set` representing the function space nodes."""
        return self._shared_data.node_set

    @property
    def dof_dset(self):
        r"""A :class:`pyop2.types.dataset.DataSet` representing the function space
        degrees of freedom."""
        return self._shared_data.dof_dset

    @property
    def cell_boundary_masks(self):
        return self._shared_data.cell_boundary_masks

    @property
    def interior_facet_boundary_masks(self):
        return self._shared_data.interior_facet_boundary_masks

    @property
    def shape(self):
        return self._shared_data.shape

    @property
    def rank(self):
        r"""The rank of this :class:`FunctionSpace`.  Spaces where the
        element is scalar-valued (or intrinsically vector-valued) have
        rank zero.  Spaces built on :class:`~ufl.classes.VectorElement` or
        :class:`~ufl.classes.TensorElement` instances have rank equivalent to
        the number of components of their
        :meth:`~ufl.classes.FiniteElementBase.value_shape`."""
        return len(self.shape)

    def _ufl_signature_data_(self, *args, **kwargs):
        return (type(self), self.component,
                super()._ufl_signature_data_(*args, **kwargs))

    @utils.cached_property
    def subfunctions(self):
        """Split into a tuple of constituent spaces."""
        return (self,)

    mesh = ufl.FunctionSpace.ufl_domain

    @property
    def _ad_parent_space(self):
        return self.parent

    def ufl_function_space(self):
        r"""The :class:`~ufl.classes.FunctionSpace` this object represents."""
        return self

    def ufl_cell(self):
        r"""The :class:`~ufl.classes.Cell` this FunctionSpace is defined on."""
        return self.ufl_domain().ufl_cell()

    @PETSc.Log.EventDecorator()
    def split(self):
        import warnings
        warnings.warn("The .split() method is deprecated, please use the .subfunctions property instead", category=FutureWarning)
        return self.subfunctions

    @utils.cached_property
    def _components(self):
        if len(self) == 1:
            return tuple(ComponentFunctionSpace(self, i) for i in range(self.value_size))
        else:
            return self.subfunctions

    @PETSc.Log.EventDecorator()
    def sub(self, i):
        if len(self) == 1:
            bound = self.value_size
        else:
            bound = len(self)
        if i < 0 or i >= bound:
            raise IndexError("Invalid component %d, not in [0, %d)" % (i, bound))
        return self._components[i]

    @utils.cached_property
    def dm(self):
        dm = self._dm()
        dmhooks.set_function_space(dm, self)
        return dm

    @property
    def num_work_functions(self):
        r"""The number of checked out work functions."""
        from firedrake.functionspacedata import get_work_function_cache
        cache = get_work_function_cache(self.mesh(), self.ufl_element())
        return sum(cache.values())

    @property
    def max_work_functions(self):
        r"""The maximum number of work functions this :class:`FunctionSpace` supports.

        See :meth:`get_work_function` for obtaining work functions."""
        from firedrake.functionspacedata import get_max_work_functions
        return get_max_work_functions(self)

    @max_work_functions.setter
    def max_work_functions(self, val):
        r"""Set the number of work functions this :class:`FunctionSpace` supports.

        :arg val: The new maximum number of work functions.
        :raises ValueError: if the provided value is smaller than the
            number of currently checked out work functions.
            """
        # Clear cache
        from firedrake.functionspacedata import get_work_function_cache, set_max_work_functions
        cache = get_work_function_cache(self.mesh(), self.ufl_element())
        if val < len(cache):
            for k in list(cache.keys()):
                if not cache[k]:
                    del cache[k]
            if val < len(cache):
                raise ValueError("Can't set work function cache smaller (%d) than current checked out functions (%d)" %
                                 (val, len(cache)))
        set_max_work_functions(self, val)

    def get_work_function(self, zero=True):
        r"""Get a temporary work :class:`~.Function` on this :class:`FunctionSpace`.

        :arg zero: Should the :class:`~.Function` be guaranteed zero?
            If ``zero`` is ``False`` the returned function may or may
            not be zeroed, and the user is responsible for appropriate
            zeroing.

        :raises ValueError: if :attr:`max_work_functions` are already
            checked out.

        .. note ::

            This method is intended to be used for short-lived work
            functions, if you actually need a function for general
            usage use the :class:`~.Function` constructor.

            When you are finished with the work function, you should
            restore it to the pool of available functions with
            :meth:`restore_work_function`.

        """
        from firedrake.functionspacedata import get_work_function_cache
        cache = get_work_function_cache(self.mesh(), self.ufl_element())
        for function in cache.keys():
            # Check if we've got a free work function available
            out = cache[function]
            if not out:
                cache[function] = True
                if zero:
                    function.dat.zero()
                return function
        if len(cache) == self.max_work_functions:
            raise ValueError("Can't check out more than %d work functions." %
                             self.max_work_functions)
        from firedrake import Function
        function = Function(self)
        cache[function] = True
        return function

    def restore_work_function(self, function):
        r"""Restore a work function obtained with :meth:`get_work_function`.

        :arg function: The work function to restore
        :raises ValueError: if the provided function was not obtained
            with :meth:`get_work_function` or it has already been restored.

        .. warning::

           This does *not* invalidate the name in the calling scope,
           it is the user's responsibility not to use a work function
           after restoring it.
        """
        from firedrake.functionspacedata import get_work_function_cache
        cache = get_work_function_cache(self.mesh(), self.ufl_element())
        try:
            out = cache[function]
        except KeyError:
            raise ValueError("Function %s is not a work function" % function)

        if not out:
            raise ValueError("Function %s is not checked out, cannot restore" % function)
        cache[function] = False

    def __eq__(self, other):
        if not isinstance(other, FunctionSpace):
            return False
        # FIXME: Think harder about equality
        return self.mesh() is other.mesh() and \
            self.dof_dset is other.dof_dset and \
            self.ufl_element() == other.ufl_element() and \
            self.component == other.component

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((self.mesh(), self.dof_dset, self.ufl_element()))

    def __repr__(self):
        return f"FunctionSpace({self.mesh()!r}, {self.ufl_element()!r}, name={self.name!r})"

    def __str__(self):
        return f"FunctionSpace({self.mesh()}, {self.ufl_element()}, name={self.name})"

    def __getitem__(self, i):
        return self.subfunctions[i]

    def __mul__(self, other):
        r"""Create a :class:`.MixedFunctionSpace` composed of this
        :class:`.FunctionSpace` and other"""
        from firedrake.functionspace import MixedFunctionSpace
        return MixedFunctionSpace((self, other))

    def boundary_nodes(self, sub_domain):
        r"""Return the boundary nodes for this :class:`~.WithGeometry`.

        :arg sub_domain: the mesh marker selecting which subset of facets to consider.
        :returns: A numpy array of the unique function space nodes on
           the selected portion of the boundary.

        See also :class:`~.DirichletBC` for details of the arguments.
        """
        # Have to replicate the definition from FunctionSpace because
        # we want to access the DM on the WithGeometry object.
        return self._shared_data.boundary_nodes(self, sub_domain)

    def collapse(self):
        from firedrake import FunctionSpace
        return FunctionSpace(self.mesh(), self.ufl_element())

    def _dm(self):
        from firedrake.mg.utils import get_level
        dm = self.dof_dset.dm
        _, level = get_level(self.mesh())
        dmhooks.attach_hooks(dm, level=level,
                             sf=self.mesh().topology_dm.getPointSF(),
                             section=self._shared_data.global_numbering)
        # Remember the function space so we can get from DM back to FunctionSpace.
        dmhooks.set_function_space(dm, self)
        return dm

    @utils.cached_property
    def cell_node_list(self):
        r"""A numpy array mapping mesh cells to function space nodes."""
        return self._shared_data.entity_node_lists[self.mesh().cell_set]

    @utils.cached_property
    def _ises(self):
        return self.dof_dset.field_ises

    @property
    @abc.abstractmethod
    def node_count(self):
        pass

    @property
    @abc.abstractmethod
    def dof_count(self):
        pass

    @property
    @abc.abstractmethod
    def dim(self):
        pass

    @abc.abstractmethod
    def make_dat(self, *args, **kwargs):
        pass


class FunctionSpace(AbstractFunctionSpace):

    def __len__(self):
        return 1

    def __iter__(self):
        yield self

    def __getitem__(self, i):
        r"""Return the ith subspace."""
        if i != 0:
            raise IndexError("Only index 0 supported on a FunctionSpace")
        return self

    def sub(self, i):
        r"""Return a view into the ith component."""
        if self.rank == 0:
            assert i == 0
            return self
        return self._components[i]

    def __mul__(self, other):
        r"""Create a :class:`.MixedFunctionSpace` composed of this
        :class:`.FunctionSpace` and other"""
        from firedrake.functionspace import MixedFunctionSpace
        return MixedFunctionSpace((self, other))

    @utils.cached_property
    def node_count(self):
        r"""The number of nodes (includes halo nodes) of this function space on
        this process.  If the :class:`FunctionSpace` has :attr:`FunctionSpace.rank` 0, this
        is equal to the :attr:`FunctionSpace.dof_count`, otherwise the :attr:`FunctionSpace.dof_count` is
        :attr:`dim` times the :attr:`node_count`."""
        return self.node_set.total_size

    @utils.cached_property
    def dof_count(self):
        r"""The number of degrees of freedom (includes halo dofs) of this
        function space on this process. Cf. :attr:`FunctionSpace.node_count` ."""
        return self.node_count*self.value_size

    def dim(self):
        r"""The global number of degrees of freedom for this function space.

        See also :attr:`FunctionSpace.dof_count` and :attr:`FunctionSpace.node_count` ."""
        return self.dof_dset.layout_vec.getSize()

    def make_dat(self, val=None, valuetype=None, name=None):
        r"""Return a newly allocated :class:`pyop2.types.dat.Dat` defined on the
        :attr:`dof_dset` of this :class:`.Function`."""
        return op2.Dat(self.dof_dset, val, valuetype, name)

    def cell_node_map(self):
        r"""Return the :class:`pyop2.types.map.Map` from cells to
        function space nodes."""
        return self._shared_data.cell_node_map(self.name)

    def interior_facet_node_map(self):
        r"""Return the :class:`pyop2.types.map.Map` from interior facets to
        function space nodes."""
        sdata = self._shared_data
        offset = self.cell_node_map().offset
        if offset is not None:
            offset = numpy.append(offset, offset)
        offset_quotient = self.cell_node_map().offset_quotient
        if offset_quotient is not None:
            offset_quotient = numpy.append(offset_quotient, offset_quotient)
        return sdata.get_map(self,
                             self.mesh().interior_facets.set,
                             2*self.finat_element.space_dimension(),
                             "interior_facet_node",
                             offset,
                             offset_quotient)

    def exterior_facet_node_map(self):
        r"""Return the :class:`pyop2.types.map.Map` from exterior facets to
        function space nodes."""
        sdata = self._shared_data
        return sdata.get_map(self,
                             self.mesh().exterior_facets.set,
                             self.finat_element.space_dimension(),
                             "exterior_facet_node",
                             self.offset,
                             self.offset_quotient)

    def boundary_nodes(self, sub_domain):
        r"""Return the boundary nodes for this :class:`~.FunctionSpace`.

        :arg sub_domain: the mesh marker selecting which subset of facets to consider.
        :returns: A numpy array of the unique function space nodes on
           the selected portion of the boundary.

        See also :class:`~.DirichletBC` for details of the arguments.
        """
        return self._shared_data.boundary_nodes(self, sub_domain)

    @PETSc.Log.EventDecorator()
    def local_to_global_map(self, bcs, lgmap=None):
        r"""Return a map from process local dof numbering to global dof numbering.

        If BCs is provided, mask out those dofs which match the BC nodes."""
        # Caching these things is too complicated, since it depends
        # not just on the bcs, but also the parent space, and anything
        # this space has been recursively split out from [e.g. inside
        # fieldsplit]
        if bcs is None or len(bcs) == 0:
            return lgmap or self.dof_dset.lgmap
        for bc in bcs:
            fs = bc.function_space()
            while fs.component is not None and fs.parent is not None:
                fs = fs.parent
            if fs != self:
                raise RuntimeError("DirichletBC defined on a different FunctionSpace!")
        unblocked = any(bc.function_space().component is not None
                        for bc in bcs)
        if lgmap is None:
            lgmap = self.dof_dset.lgmap
            if unblocked:
                indices = lgmap.indices.copy()
                bsize = 1
            else:
                indices = lgmap.block_indices.copy()
                bsize = lgmap.getBlockSize()
                assert bsize == self.value_size
        else:
            # MatBlock case, LGMap is already unrolled.
            indices = lgmap.block_indices.copy()
            bsize = lgmap.getBlockSize()
            unblocked = True
        nodes = []
        for bc in bcs:
            if bc.function_space().component is not None:
                nodes.append(bc.nodes * self.value_size
                             + bc.function_space().component)
            elif unblocked:
                tmp = bc.nodes * self.value_size
                for i in range(self.value_size):
                    nodes.append(tmp + i)
            else:
                nodes.append(bc.nodes)
        nodes = numpy.unique(numpy.concatenate(nodes))
        indices[nodes] = -1
        return PETSc.LGMap().create(indices, bsize=bsize, comm=lgmap.comm)


class MixedFunctionSpace(AbstractFunctionSpace):
    r"""A function space on a mixed finite element.

    This is essentially just a bag of individual
    :class:`FunctionSpace` objects.

    :arg spaces: The constituent spaces.
    :kwarg name: An optional name for the mixed space.

    .. warning::

       Users should not build a :class:`MixedFunctionSpace` directly,
       but should instead use the functional interface provided by
       :func:`.MixedFunctionSpace`.
    """
    def __init__(self, spaces, name=None):
        mesh, = set(s.mesh() for s in spaces)
        super(ufl.FunctionSpace, self).__init__(mesh, ufl.MixedElement(*[s.ufl_element() for s in spaces]))

        self._spaces = tuple(
            IndexedFunctionSpace(i, s, self) for i, s in enumerate(spaces)
        )
        self.name = name or "_".join(str(s.name) for s in spaces)
        self._subspaces = {}
        self._mesh = mesh
        self.comm = self.node_set.comm

    # These properties are so a mixed space can behave like a normal FunctionSpace.
    index = None
    component = None
    parent = None
    rank = 1

    def mesh(self):
        return self._mesh

    def ufl_function_space(self):
        r"""The :class:`~ufl.classes.FunctionSpace` associated with this space."""
        return self

    def __eq__(self, other):
        if not isinstance(other, MixedFunctionSpace) or len(other) != len(self):
            return False
        return all(s == o for s, o in zip(self, other))

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(tuple(self))

    @utils.cached_property
    def subfunctions(self):
        r"""The list of :class:`FunctionSpace`\s of which this
        :class:`MixedFunctionSpace` is composed."""
        return self._spaces

    def split(self):
        import warnings
        warnings.warn("The .split() method is deprecated, please use the .subfunctions property instead", category=FutureWarning)
        return self.subfunctions

    def sub(self, i):
        r"""Return the `i`th :class:`FunctionSpace` in this
        :class:`MixedFunctionSpace`."""
        return self._spaces[i]

    def num_sub_spaces(self):
        r"""Return the number of :class:`FunctionSpace`\s of which this
        :class:`MixedFunctionSpace` is composed."""
        return len(self)

    def __len__(self):
        r"""Return the number of :class:`FunctionSpace`\s of which this
        :class:`MixedFunctionSpace` is composed."""
        return len(self._spaces)

    def __getitem__(self, i):
        r"""Return the `i`th :class:`FunctionSpace` in this
        :class:`MixedFunctionSpace`."""
        return self._spaces[i]

    def __iter__(self):
        return iter(self._spaces)

    def __repr__(self):
        return "MixedFunctionSpace(%s, name=%r)" % \
            (", ".join(repr(s) for s in self), self.name)

    def __str__(self):
        return "MixedFunctionSpace(%s)" % ", ".join(str(s) for s in self)

    @utils.cached_property
    def value_size(self):
        r"""Return the sum of the :attr:`FunctionSpace.value_size`\s of the
        :class:`FunctionSpace`\s this :class:`MixedFunctionSpace` is
        composed of."""
        return sum(fs.value_size for fs in self._spaces)

    @utils.cached_property
    def node_count(self):
        r"""Return a tuple of :attr:`FunctionSpace.node_count`\s of the
        :class:`FunctionSpace`\s of which this :class:`MixedFunctionSpace` is
        composed."""
        return tuple(fs.node_count for fs in self._spaces)

    @utils.cached_property
    def dof_count(self):
        r"""Return a tuple of :attr:`FunctionSpace.dof_count`\s of the
        :class:`FunctionSpace`\s of which this :class:`MixedFunctionSpace` is
        composed."""
        return tuple(fs.dof_count for fs in self._spaces)

    def dim(self):
        r"""The global number of degrees of freedom for this function space.

        See also :attr:`FunctionSpace.dof_count` and :attr:`FunctionSpace.node_count`."""
        return self.dof_dset.layout_vec.getSize()

    @utils.cached_property
    def node_set(self):
        r"""A :class:`pyop2.types.set.MixedSet` containing the nodes of this
        :class:`MixedFunctionSpace`. This is composed of the
        :attr:`FunctionSpace.node_set`\s of the underlying
        :class:`FunctionSpace`\s this :class:`MixedFunctionSpace` is
        composed of one or (for VectorFunctionSpaces) more degrees of freedom
        are stored at each node."""
        return op2.MixedSet(s.node_set for s in self._spaces)

    @utils.cached_property
    def dof_dset(self):
        r"""A :class:`pyop2.types.dataset.MixedDataSet` containing the degrees of freedom of
        this :class:`MixedFunctionSpace`. This is composed of the
        :attr:`FunctionSpace.dof_dset`\s of the underlying
        :class:`FunctionSpace`\s of which this :class:`MixedFunctionSpace` is
        composed."""
        return op2.MixedDataSet(s.dof_dset for s in self._spaces)

    def cell_node_map(self):
        r"""A :class:`pyop2.types.map.MixedMap` from the ``Mesh.cell_set`` of the
        underlying mesh to the :attr:`node_set` of this
        :class:`MixedFunctionSpace`. This is composed of the
        :attr:`FunctionSpace.cell_node_map`\s of the underlying
        :class:`FunctionSpace`\s of which this :class:`MixedFunctionSpace` is
        composed."""
        return op2.MixedMap(s.cell_node_map() for s in self._spaces)

    def interior_facet_node_map(self):
        r"""Return the :class:`pyop2.types.map.MixedMap` from interior facets to
        function space nodes."""
        return op2.MixedMap(s.interior_facet_node_map() for s in self)

    def exterior_facet_node_map(self):
        r"""Return the :class:`pyop2.types.map.Map` from exterior facets to
        function space nodes."""
        return op2.MixedMap(s.exterior_facet_node_map() for s in self)

    def local_to_global_map(self, bcs):
        r"""Return a map from process local dof numbering to global dof numbering.

        If BCs is provided, mask out those dofs which match the BC nodes."""
        raise NotImplementedError("Not for mixed maps right now sorry!")

    def make_dat(self, val=None, valuetype=None, name=None):
        r"""Return a newly allocated :class:`pyop2.types.dat.MixedDat` defined on the
        :attr:`dof_dset` of this :class:`MixedFunctionSpace`."""
        if val is not None:
            assert len(val) == len(self)
        else:
            val = [None for _ in self]
        return op2.MixedDat(s.make_dat(v, valuetype, "%s[cmpt-%d]" % (name, i))
                            for i, (s, v) in enumerate(zip(self._spaces, val)))

    @utils.cached_property
    def dm(self):
        r"""A PETSc DM describing the data layout for fieldsplit solvers."""
        dm = self._dm()
        dmhooks.set_function_space(dm, self)
        return dm

    def _dm(self):
        from firedrake.mg.utils import get_level
        dm = self.dof_dset.dm
        _, level = get_level(self.mesh())
        dmhooks.attach_hooks(dm, level=level)
        return dm

    @utils.cached_property
    def _ises(self):
        return self.dof_dset.field_ises


class ProxyFunctionSpace(FunctionSpace):
    r"""A :class:`FunctionSpace` that one can attach extra properties to.

    :arg mesh: The mesh to use.
    :arg element: The UFL element.
    :arg name: The name of the function space.

    .. warning::

       Users should not build a :class:`ProxyFunctionSpace` directly,
       it is mostly used as an internal implementation detail.
    """
    def __repr__(self):
        return "%sProxyFunctionSpace(%r, %r, name=%r, index=%r, component=%r)" % \
            (str(self.identifier).capitalize(),
             self.mesh(),
             self.ufl_element(),
             self.name,
             self.index,
             self.component)

    def __str__(self):
        return "%sProxyFunctionSpace(%s, %s, name=%s, index=%s, component=%s)" % \
            (str(self.identifier).capitalize(),
             self.mesh(),
             self.ufl_element(),
             self.name,
             self.index,
             self.component)

    identifier = None
    r"""An optional identifier, for debugging purposes."""

    no_dats = False
    r"""Can this proxy make :class:`pyop2.types.dat.Dat` objects"""

    def make_dat(self, *args, **kwargs):
        r"""Create a :class:`pyop2.types.dat.Dat`.

        :raises ValueError: if :attr:`no_dats` is ``True``.
        """
        if self.no_dats:
            raise ValueError("Can't build Function on %s function space" % self.identifier)
        return super(ProxyFunctionSpace, self).make_dat(*args, **kwargs)


def IndexedFunctionSpace(index, space, parent):
    r"""Build a new FunctionSpace that remembers it is a particular
    subspace of a :class:`MixedFunctionSpace`.

    :arg index: The index into the parent space.
    :arg space: The subspace to represent
    :arg parent: The parent mixed space.
    :returns: A new :class:`ProxyFunctionSpace` with index and parent
        set.
    """
    if space.ufl_element().family() == "Real":
        new = RealFunctionSpace(space.mesh(), space.ufl_element(), name=space.name)
    else:
        new = ProxyFunctionSpace(space.mesh(), space.ufl_element(), name=space.name)
    new.index = index
    new.parent = parent
    new.identifier = "indexed"
    return new


def ComponentFunctionSpace(parent, component):
    r"""Build a new FunctionSpace that remembers it represents a
    particular component.  Used for applying boundary conditions to
    components of a :func:`.VectorFunctionSpace` or :func:`.TensorFunctionSpace`.

    :arg parent: The parent space (a FunctionSpace with a
        VectorElement or TensorElement).
    :arg component: The component to represent.
    :returns: A new :class:`ProxyFunctionSpace` with the component set.
    """
    element = parent.ufl_element()
    assert type(element) in frozenset([ufl.VectorElement, ufl.TensorElement])
    if not (0 <= component < parent.value_size):
        raise IndexError("Invalid component %d. not in [0, %d)" %
                         (component, parent.value_size))
    new = ProxyFunctionSpace(parent.mesh(), element.sub_elements()[0], name=parent.name)
    new.identifier = "component"
    new.component = component
    new.parent = parent
    return new


class RealFunctionSpace(FunctionSpace):
    r""":class:`FunctionSpace` based on elements of family "Real". A
    :class`RealFunctionSpace` only has a single global value for the
    whole mesh.

    This class should not be directly instantiated by users. Instead,
    FunctionSpace objects will transform themselves into
    :class:`RealFunctionSpace` objects as appropriate.

    """

    finat_element = None
    rank = 0
    shape = ()
    value_size = 1

    def __init__(self, mesh, element, name):
        self._ufl_function_space = ufl.FunctionSpace(mesh.ufl_mesh(), element)
        self.name = name
        self.comm = mesh.comm
        self._mesh = mesh
        self.dof_dset = op2.GlobalDataSet(self.make_dat())
        self.node_set = self.dof_dset.set

    def __eq__(self, other):
        if not isinstance(other, RealFunctionSpace):
            return False
        # FIXME: Think harder about equality
        return self.mesh() is other.mesh() and \
            self.ufl_element() == other.ufl_element()

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.mesh(), self.ufl_element()))

    def _dm(self):
        from firedrake.mg.utils import get_level
        dm = self.dof_dset.dm
        _, level = get_level(self.mesh())
        dmhooks.attach_hooks(dm, level=level,
                             sf=self.mesh().topology_dm.getPointSF(),
                             section=None)
        # Remember the function space so we can get from DM back to FunctionSpace.
        dmhooks.set_function_space(dm, self)
        return dm

    def make_dat(self, val=None, valuetype=None, name=None):
        r"""Return a newly allocated :class:`pyop2.types.glob.Global` representing the
        data for a :class:`.Function` on this space."""
        return op2.Global(self.value_size, val, valuetype, name, self.comm)

    def cell_node_map(self, bcs=None):
        ":class:`RealFunctionSpace` objects have no cell node map."
        return None

    def interior_facet_node_map(self, bcs=None):
        ":class:`RealFunctionSpace` objects have no interior facet node map."
        return None

    def exterior_facet_node_map(self, bcs=None):
        ":class:`RealFunctionSpace` objects have no exterior facet node map."
        return None

    def bottom_nodes(self):
        ":class:`RealFunctionSpace` objects have no bottom nodes."
        return None

    def top_nodes(self):
        ":class:`RealFunctionSpace` objects have no bottom nodes."
        return None

    def dim(self):
        return 1

    def local_to_global_map(self, bcs, lgmap=None):
        assert len(bcs) == 0
        return None
