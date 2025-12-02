r"""
This module provides the implementations of :class:`~.FunctionSpace`
and :class:`~.MixedFunctionSpace` objects, along with some utility
classes for attaching extra information to instances of these.
"""
from __future__ import annotations

import collections
import dataclasses
import functools
import numbers
import warnings
import operator
from collections import OrderedDict, defaultdict
from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass
from functools import cached_property, reduce
from immutabledict import immutabledict as idict
from typing import Literal, Optional
from mpi4py import MPI

import finat.ufl
import numpy
import pyop3 as op3
import ufl
from pyop3 import mpi
from pyop3.utils import just_one, single_valued

from ufl.cell import CellSequence
from ufl.duals import is_dual, is_primal
from pyop3.pyop2_utils import as_tuple

from firedrake import dmhooks, utils, extrusion_utils as eutils
from firedrake.cython import dmcommon
from firedrake.extrusion_utils import is_real_tensor_product_element
from firedrake.cython import extrusion_numbering as extnum
from firedrake.functionspacedata import create_element
from firedrake.mesh import MeshTopology, ExtrudedMeshTopology, VertexOnlyMeshTopology
from firedrake.mesh import MeshGeometry, MeshSequenceTopology, MeshSequenceGeometry
from firedrake.petsc import PETSc
from firedrake.utils import IntType


def check_element(element, top=True):
    """Run some checks on the provided element.

    The :class:`finat.ufl.mixedelement.VectorElement` and
    :class:`finat.ufl.mixedelement.TensorElement` modifiers must be "outermost"
    for function space construction to work, excepting that they
    should not wrap a :class:`finat.ufl.mixedelement.MixedElement`.  Similarly,
    a base :class:`finat.ufl.mixedelement.MixedElement` must be outermost (it
    can contain :class:`finat.ufl.mixedelement.MixedElement` instances, provided
    they satisfy the other rules). This function checks that.

    Parameters
    ----------
    element :
        The :class:`UFL element
        <finat.ufl.finiteelementbase.FiniteElementBase>` to check.
    top : bool
        Are we at the top element (in which case the modifier is legal).

    Returns
    -------

    ``None`` if the element is legal.

    Raises
    ------
    ValueError
        If the element is illegal.
    """
    if isinstance(element.cell, CellSequence) and \
       type(element) is not finat.ufl.MixedElement:
        raise ValueError("MixedElement modifier must be outermost")
    if element.cell.cellname == "hexahedron" and \
       element.family() not in ["Q", "DQ", "Real"]:
        raise NotImplementedError("Currently can only use 'Q', 'DQ', and/or 'Real' elements on hexahedral meshes, not", element.family())
    if type(element) in (finat.ufl.BrokenElement, finat.ufl.RestrictedElement,
                         finat.ufl.HDivElement, finat.ufl.HCurlElement):
        inner = (element._element, )
    elif type(element) is finat.ufl.EnrichedElement:
        inner = element._elements
    elif type(element) is finat.ufl.TensorProductElement:
        inner = element.factor_elements
    elif isinstance(element, finat.ufl.MixedElement):
        if not top:
            raise ValueError(f"{type(element).__name__} modifier must be outermost")
        else:
            inner = element.sub_elements
    else:
        inner = ()
    for e in inner:
        check_element(e, top=False)


@functools.lru_cache()
def _num_entity_dofs(element):
    ndofs = {}
    for entity_key, entities in element.entity_dofs().items():
        ndofs[entity_key] = utils.single_valued(map(len, entities.values()))
    return ndofs


class WithGeometryBase:
    r"""Attach geometric information to a :class:`~.FunctionSpace`.

    Function spaces on meshes with different geometry but the same
    topology can share data, except for their UFL cell.  This class
    facilitates that.

    Users should not instantiate a :class:`WithGeometryBase` object
    explicitly except in a small number of cases.

    When instantiating a :class:`WithGeometryBase`, users should call
    :meth:`WithGeometryBase.create` rather than ``__init__``.

    :arg mesh: The mesh with geometric information to use.
    :arg element: The UFL element.
    :arg component: The component of this space in a parent vector
        element space, or ``None``.
    :arg cargo: :class:`FunctionSpaceCargo` instance carrying
        Firedrake-specific data that is not required for code
        generation.
    """
    node_label = "nodes"

    def __init__(self, mesh, element, component=None, cargo=None):
        if type(element) is finat.ufl.MixedElement:
            if not isinstance(mesh, MeshSequenceGeometry):
                raise TypeError(f"Can only use MixedElement with MeshSequenceGeometry: got {type(mesh)}")
        assert component is None or isinstance(component, tuple)
        assert cargo is None or isinstance(cargo, FunctionSpaceCargo)
        super().__init__(mesh, element, label=cargo.topological._label or "")
        self.component = component
        self.cargo = cargo
        self.comm = mesh.comm
        self._comm = mpi.internal_comm(mesh.comm, self)

    @classmethod
    def create(cls, function_space, mesh, parent=None):
        """Create a :class:`WithGeometry`.

        Parameters
        ----------
        function_space : FunctionSpace or MixedFunctionSpace
            Topological function space to attach geometry to.
        mesh : MeshGeometry
            Mesh with geometric information to use.
        parent : WithGeometry
            Parent geometric function space if exists.

        """
        if isinstance(function_space, MixedFunctionSpace):
            if not isinstance(mesh, MeshSequenceGeometry):
                raise TypeError(f"Can only use MixedFunctionSpace with MeshSequenceGeometry: got {type(mesh)}")
        function_space = function_space.topological
        assert mesh.topology == function_space.mesh()
        assert mesh.topology != mesh

        element = function_space.ufl_element().reconstruct(cell=mesh.ufl_cell())

        topological = function_space
        component = function_space.component

        if function_space.parent is not None:
            if parent is None:
                raise ValueError("Must pass parent if function_space.parent is not None")
        else:
            parent = None

        cargo = FunctionSpaceCargo(topological, parent)
        return cls(mesh, element, component=component, cargo=cargo)

    def _ufl_signature_data_(self, *args, **kwargs):
        return (type(self), self.component,
                super()._ufl_signature_data_(*args, **kwargs))

    @property
    def parent(self):
        return self.cargo.parent

    @parent.setter
    def parent(self, val):
        breakpoint()
        self.cargo.parent = val

    @property
    def topological(self):
        return self.cargo.topological

    @topological.setter
    def topological(self, val):
        self.cargo.topological = val

    @utils.cached_property
    def strong_subspaces(self):
        r"""Split into a tuple of constituent spaces."""
        return tuple(type(self).create(subspace, mesh, parent=self)
                     for mesh, subspace in zip(self.mesh(), self.topological.strong_subspaces, strict=True))

    @utils.cached_property
    def subspaces(self):
        r"""Split into a tuple of constituent spaces."""
        if isinstance(self.topological, MixedFunctionSpace):
            return tuple(
                type(self).create(subspace, mesh, parent=self)
                for mesh, subspace in zip(self.mesh(), self.topological.subspaces, strict=True)
            )
        else:
            return (self, )

    @property
    def subfunctions(self):
        import warnings
        warnings.warn("The 'subfunctions' property is deprecated for function spaces, please use the "
                      "'subspaces' property instead", category=FutureWarning)
        return self.subspaces

    mesh = ufl.FunctionSpace.ufl_domain

    @property
    def _ad_parent_space(self):
        return self.parent

    def ufl_function_space(self):
        r"""The :class:`~ufl.classes.FunctionSpace` this object represents."""
        return self

    def ufl_cell(self):
        r"""The :class:`~ufl.classes.Cell` this FunctionSpace is defined on."""
        return self.mesh().ufl_cell()

    @utils.cached_property
    def _strong_components(self):
        components = numpy.empty(self.shape, dtype=object)
        for ix in numpy.ndindex(self.shape):
            components[ix] = type(self).create(self.topological.sub(ix, weak=False), self.mesh())
        return utils.readonly(components)

    @utils.cached_property
    def _components(self):
        components = numpy.empty(self.shape, dtype=object)
        for ix in numpy.ndindex(self.shape):
            components[ix] = type(self).create(self.topological.sub(ix, weak=True), self.mesh(), parent=self)
        return utils.readonly(components)

    @PETSc.Log.EventDecorator()
    def sub(self, indices, *, weak: bool = True):
        if type(self.ufl_element()) is finat.ufl.MixedElement:
            if weak:
                return self.subspaces[indices]
            else:
                return self.strong_subspaces[indices]
        else:
            indices = parse_component_indices(indices, self.shape)
            if weak:
                return self._components[indices]
            else:
                return self._strong_components[indices]

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
        if is_primal(self) != is_primal(other) or \
                is_dual(self) != is_dual(other):
            return False
        try:
            return self.topological == other.topological and \
                self.mesh() == other.mesh()
        except AttributeError:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.mesh(), self.topological))

    def __len__(self):
        return len(self.topological)

    def __repr__(self):
        return "%s(%r, %r)" % (self.__class__.__name__, self.topological, self.mesh())

    def __str__(self):
        return "%s(%s, %s)" % (self.__class__.__name__, self.topological, self.mesh())

    def __iter__(self):
        return iter(self.subspaces)

    def __getitem__(self, i):
        return self.subspaces[i]

    def __mul__(self, other):
        r"""Create a :class:`.MixedFunctionSpace` composed of this
        :class:`.FunctionSpace` and other"""
        from firedrake.functionspace import MixedFunctionSpace
        return MixedFunctionSpace((self, other))

    def __getattr__(self, name):
        val = getattr(self.topological, name)
        setattr(self, name, val)
        return val

    def __dir__(self):
        current = super().__dir__()
        return list(OrderedDict.fromkeys(dir(self.topological) + current))

    def boundary_nodes(self, sub_domain):
        r"""Return the boundary nodes for this :class:`~.FunctionSpace`.

        :arg sub_domain: the mesh marker selecting which subset of facets to consider.
        :returns: A numpy array of the unique function space nodes on
           the selected portion of the boundary.

        See also :class:`~.DirichletBC` for details of the arguments.
        """
        r"""Return the boundary nodes for this :class:`~.FunctionSpace`.

        :arg sub_domain: the mesh marker selecting which subset of facets to consider.
        :returns: A numpy array of the unique function space nodes on
           the selected portion of the boundary.

        See also :class:`~.DirichletBC` for details of the arguments.
        """
        V = self  # fixme
        if sub_domain in ["bottom", "top"]:
            if not V.extruded:
                raise ValueError("Invalid subdomain '%s' for non-extruded mesh",
                                 sub_domain)
            entity_dofs = eutils.flat_entity_dofs(V.finat_element.entity_dofs())
            key = (entity_dofs_key(entity_dofs), sub_domain, V.boundary_set)
            return self.get_top_bottom_boundary_nodes(V.mesh(), key)
        else:
            if sub_domain == "on_boundary":
                sdkey = sub_domain
            else:
                sdkey = as_tuple(sub_domain)
            key = (entity_dofs_key(V.finat_element.entity_dofs()), sdkey, V.boundary_set)
            return self.get_facet_closure_nodes(V.mesh(), key)

    # TODO: cache on the mesh
    def get_facet_closure_nodes(self, mesh, key):
        """Function space nodes in the closure of facets with a given
        marker.
        :arg mesh: Mesh to cache on
        :arg key: (edofs, sub_domain, boundary_set) tuple
        :arg V: function space.
        :returns: numpy array of unique nodes in the closure of facets
           with provided markers (both interior and exterior)."""
        _, sub_domain, boundary_set = key
        if sub_domain not in {"on_boundary", "top", "bottom"}:
            valid = set(self._mesh.facet_markers)
            invalid = set(sub_domain) - valid
            if invalid:
                raise LookupError(f"BC construction got invalid markers {invalid}. "
                                  f"Valid markers are '{valid}'")
        return dmcommon.facet_closure_nodes(self, sub_domain)

    def collapse(self):
        return type(self).create(self.topological.collapse(), self.mesh())

    @classmethod
    def make_function_space(cls, mesh, element, name=None, **kwargs):
        r"""Factory method for :class:`.WithGeometryBase`."""
        topology = mesh.topology
        # Create a new abstract (Mixed/Real)FunctionSpace, these are neither primal nor dual.
        if type(element) is finat.ufl.MixedElement:
            if isinstance(mesh, MeshGeometry):
                mesh = MeshSequenceGeometry([mesh for _ in element.sub_elements])
                topology = mesh.topology
            else:
                if not isinstance(mesh, MeshSequenceGeometry):
                    raise TypeError(f"mesh must be MeshSequenceGeometry: got {mesh}")
            spaces = [cls.make_function_space(topo, e) for topo, e in zip(topology, element.sub_elements, strict=True)]
            new = MixedFunctionSpace(spaces, topology, name=name)
        else:
            if isinstance(mesh, MeshSequenceGeometry):
                raise TypeError(f"mesh must not be MeshSequenceGeometry: got {mesh}")
            # Check that any Vector/Tensor/Mixed modifiers are outermost.
            check_element(element)
            if element.family() == "Real":
                new = RealFunctionSpace(topology, element, name=name, **kwargs)
            else:
                new = FunctionSpace(topology, element, name=name, **kwargs)
        # Skip this if we are just building subspaces of an abstract MixedFunctionSpace
        if mesh is not topology:
            # Create a concrete WithGeometry or FiredrakeDualSpace on this mesh
            new = cls.create(new, mesh)
        return new

    def reconstruct(
        self,
        mesh: MeshGeometry | None = None,
        element: finat.ufl.FiniteElement | None = None,
        name: str | None = None,
        **kwargs,
    ) -> "WithGeometryBase":
        """Return a new function space with modified fields.

        Parameters
        ----------
        mesh :
            The mesh (defaults to same mesh).
        element :
            The finite element (defaults to same element).
        name :
            The name (defaults to `None`).

        Returns
        -------
        WithGeometryBase :
            The new function space of the same class as ``self``.

        Any extra kwargs are used to reconstruct the finite element.
        For details see `finat.ufl.finiteelement.FiniteElement.reconstruct`.

        """
        from firedrake.bcs import restricted_function_space
        V_parent = self

        # Deal with ProxyFunctionSpace
        indices = []
        while True:
            if V_parent.index is not None:
                indices.append(V_parent.index)
            if V_parent.component is not None:
                indices.append(V_parent.component)
            if V_parent.parent is not None:
                V_parent = V_parent.parent
            else:
                break

        if mesh is None:
            mesh = V_parent.mesh()
        if element is None:
            element = V_parent.ufl_element()

        cell = mesh.topology.ufl_cell()
        if len(kwargs) > 0 or element.cell != cell:
            element = element.reconstruct(cell=cell, **kwargs)

        # Reconstruct the parent space
        V = type(self).make_function_space(mesh, element, name=name)

        # Deal with RestrictedFunctionSpace
        boundary_sets = [V_.boundary_set for V_ in V_parent]
        if any(boundary_sets):
            V = restricted_function_space(V, boundary_sets)

        for i in reversed(indices):
            V = V.sub(i)
        return V


class WithGeometry(WithGeometryBase, ufl.FunctionSpace):

    def __init__(self, mesh, element, component=None, cargo=None):
        super(WithGeometry, self).__init__(mesh, element,
                                           component=component,
                                           cargo=cargo)

    def dual(self):
        parent = None if self.parent is None else self.parent.dual()
        return FiredrakeDualSpace.create(self.topological, self.mesh(), parent=parent)


class FiredrakeDualSpace(WithGeometryBase, ufl.functionspace.DualSpace):

    def __init__(self, mesh, element, component=None, cargo=None):
        super(FiredrakeDualSpace, self).__init__(mesh, element,
                                                 component=component,
                                                 cargo=cargo)

    def dual(self):
        parent = None if self.parent is None else self.parent.dual()
        return WithGeometry.create(self.topological, self.mesh(), parent=parent)


@dataclass(frozen=True)
class AxisConstraint:
    axis: op3.Axis
    within_axes: Mapping[str, str] = dataclasses.field(default_factory=idict)

    def with_constraint(self, constraint) -> AxisConstraint:
        return type(self)(self.axis, self.within_axes | constraint)


class FunctionSpace:
    r"""A representation of a function space.

    A :class:`FunctionSpace` associates degrees of freedom with
    topological mesh entities.  The degree of freedom mapping is
    determined from the provided element.

    :arg mesh: The :func:`~.Mesh` to build the function space on.
    :arg element: The :class:`finat.ufl.finiteelementbase.FiniteElementBase` describing the
        degrees of freedom.
    :kwarg name: An optional name for this :class:`FunctionSpace`,
        useful for later identification.

    The element can be a essentially any
    :class:`finat.ufl.finiteelementbase.FiniteElementBase`, except for a
    :class:`finat.ufl.mixedelement.MixedElement`, for which one should use the
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

    """

    boundary_set = frozenset()

    @PETSc.Log.EventDecorator()
    def __init__(self, mesh, element, name=None, *, layout=None):
        super(FunctionSpace, self).__init__()
        if type(element) is finat.ufl.MixedElement:
            raise ValueError("Can't create FunctionSpace for MixedElement")

        # The function space shape is the number of dofs per node,
        # hence it is not always the value_shape.  Vector and Tensor
        # element modifiers *must* live on the outside!
        if type(element) in {finat.ufl.TensorElement, finat.ufl.VectorElement} \
           or (isinstance(element, finat.ufl.WithMapping)
               and type(element.wrapee) in {finat.ufl.TensorElement, finat.ufl.VectorElement}):
            # The number of "free" dofs is given by reference_value_shape,
            # not value_shape due to symmetry specifications
            rvs = element.reference_value_shape
            # This requires that the sub element is not itself a
            # tensor element (which is checked by the top level
            # constructor of function spaces)
            shape_element = element
            if isinstance(element, finat.ufl.WithMapping):
                shape_element = element.wrapee
            sub = shape_element.sub_elements[0].reference_value_shape
            self.shape = rvs[:len(rvs) - len(sub)]
        else:
            self.shape = ()
        self._label = ""
        self._ufl_function_space = ufl.FunctionSpace(mesh.ufl_mesh(), element, label=self._label)
        self._mesh = mesh

        self.value_size = self._ufl_function_space.value_size
        r"""The number of scalar components of this :class:`FunctionSpace`."""

        self.rank = len(self.shape)
        r"""The rank of this :class:`FunctionSpace`.  Spaces where the
        element is scalar-valued (or intrinsically vector-valued) have rank
        zero.  Spaces built on :class:`finat.ufl.mixedelement.VectorElement` or
        :class:`finat.ufl.mixedelement.TensorElement` have rank 1 and 2
        respectively."""

        self.name = name
        r"""The (optional) descriptive name for this space."""

        # User comm
        self.comm = mesh.comm
        # Internal comm
        self._comm = mpi.internal_comm(self.comm, self)

        self.element = element
        self.finat_element = create_element(element)

        entity_dofs = self.finat_element.entity_dofs()
        nodes_per_entity = tuple(len(entity_dofs[d][0]) for d in sorted(entity_dofs))
        real_tensor_product = is_real_tensor_product_element(self.finat_element)
        key = (nodes_per_entity, real_tensor_product, self.shape)

        if layout is None:
            # NOTE: Indicates bad inheritance
            if isinstance(self, RealFunctionSpace):
                layout = ("dof",)
            else:
                layout = ("mesh", "dof") + tuple(f"dim{i}" for i in range(self.rank))

        self.layout = layout
        self.extruded = isinstance(mesh, ExtrudedMeshTopology)

    @cached_property
    def offset(self):
        if isinstance(self.mesh(), ExtrudedMeshTopology):
            return eutils.calculate_dof_offset(self.finat_element)
        else:
            return None

    @cached_property
    def cell_boundary_masks(self):
        edofs_key = entity_dofs_key(self.finat_element.entity_dofs())
        return self.get_boundary_masks(self.mesh(), (edofs_key, "cell"), self.finat_element)

    def get_boundary_masks(self, mesh, key, finat_element):
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
        if not isinstance(mesh.topology, ExtrudedMeshTopology):
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

    # TODO:
    # @cached_on(mesh)?
    @cached_property
    def layout_axes(self) -> AxisTree:
        # idea is to define this for this and mixed function space etc - this is the
        # *data layout* which is different to .axes (which is always the same for a
        # given space regardless of the data layout).
        # We can build this up dynamically by do a pre-order traversal of some tree spec
        # and building things up.
        # E.g.: ["mesh", {0: ["dof", {"XXX": "dim"}], 1: ...] and so on. Go down this tree
        # thing and attach axes on the way.
        # This could also just be ["mesh", "dof", "dim", "dof", "dim", "dof", "dim"] and
        # could just pop things off - but that's quite unclear...
        return layout_from_spec(self.layout, self.axis_constraints)

    @cached_property
    def axis_constraints(self) -> tuple[AxisConstraint]:
        from firedrake.cython import dmcommon
        import pyop3.extras.debug

        mesh_axis = self._mesh.flat_points
        num_points = mesh_axis.local_size
        plex = self._mesh.topology_dm

        constraints = [AxisConstraint(mesh_axis)]

        pyop3.extras.debug.warn_todo("Do in Cython")

        # identify constrained points
        constrained_points = set()
        if self.boundary_set:
            for marker in self.boundary_set:
                if marker == "on_boundary":
                    label = "exterior_facets"
                    marker = 1
                else:
                    label = dmcommon.FACE_SETS_LABEL
                n = plex.getStratumSize(label, marker)
                if n == 0:
                    continue
                points = plex.getStratumIS(label, marker).indices
                for i in range(n):
                    p = points[i]
                    if p not in constrained_points:
                        constrained_points.add(p)

        num_constrained_points = len(constrained_points)
        num_unconstrained_points = num_points - num_constrained_points

        num_unconstrained_dofs = numpy.empty(num_points, dtype=IntType)
        num_constrained_dofs = numpy.empty_like(num_unconstrained_dofs)
        for old_pt in range(mesh_axis.local_size):
            if self._mesh._is_renumbered:
                new_pt = self._mesh._old_to_new_point_renumbering.indices[old_pt]
            else:
                new_pt = old_pt

            ndofs = self.local_section.getDof(old_pt)

            if new_pt not in constrained_points:
                num_unconstrained_dofs[new_pt] = ndofs
                num_constrained_dofs[new_pt] = 0
            else:
                num_unconstrained_dofs[new_pt] = 0
                num_constrained_dofs[new_pt] = ndofs

        unconstrained_dofs_dat = op3.Dat(mesh_axis, data=num_unconstrained_dofs)
        constrained_dofs_dat = op3.Dat(mesh_axis, data=num_constrained_dofs)
        unconstrained_dofs_expr = op3.as_linear_buffer_expression(unconstrained_dofs_dat)

        if self.boundary_set:
            constrained_dofs_expr = op3.as_linear_buffer_expression(constrained_dofs_dat)
            regions = [
                op3.AxisComponentRegion(unconstrained_dofs_expr, "unconstrained"),
                op3.AxisComponentRegion(constrained_dofs_expr, "constrained"),
            ]
        else:
            regions = [
                op3.AxisComponentRegion(unconstrained_dofs_expr),
            ]

        component = op3.AxisComponent(regions, "XXX")
        dof_axis = op3.Axis(component, "dof")

        constraint = AxisConstraint(
            dof_axis,
            idict({mesh_axis.label: mesh_axis.component.label})
        )
        constraints.append(constraint)

        for i, dim in enumerate(self.shape):
            shape_axis = op3.Axis({"XXX": dim}, f"dim{i}")
            constraint = AxisConstraint(shape_axis)
            constraints.append(constraint)

        return tuple(constraints)

    @cached_property
    def axes(self) -> op3.AxisForest:
        return op3.AxisForest([self.plex_axes, self.nodal_axes])

    @cached_property
    def plex_axes(self) -> op3.IndexedAxisTree:
        strata_slice = self._mesh._strata_slice
        index_tree = op3.IndexTree(strata_slice)
        for slice_component in strata_slice.components:
            path = {strata_slice.label: slice_component.label}

            dim = slice_component.label
            ndofs = single_valued(len(v) for v in self.finat_element.entity_dofs()[dim].values())
            subslice = op3.Slice("dof", [op3.AffineSliceComponent("XXX", stop=ndofs, label="XXX")], label=f"dof{slice_component.label}")
            index_tree = index_tree.add_node(path, subslice)

            # same as in parloops.py
            if self.shape:
                shape_slices = op3.IndexTree.from_iterable([
                    op3.Slice(f"dim{i}", [op3.AffineSliceComponent("XXX", label="XXX")], label=f"dim{i}")
                    for i, dim in enumerate(self.shape)
                ])

                index_tree = index_tree.add_subtree(path | {subslice.label: "XXX"}, shape_slices)
        return self.layout_axes[index_tree]

    @cached_property
    def nodes(self) -> op3.Axis:
        scalar_axis_tree = self.layout_axes.blocked(self.shape)
        num_nodes = scalar_axis_tree.local_size
        return op3.Axis([op3.AxisComponent(num_nodes, sf=scalar_axis_tree.sf)], "nodes")

    @cached_property
    def nodal_axes(self) -> op3.IndexedAxisTree:
        # Create the axis tree without index information
        axis_tree = self.nodes.as_tree()
        for i, dim in enumerate(self.shape):
            axis_tree = axis_tree.add_axis(axis_tree.leaf_path, op3.Axis({"XXX": dim}, f"dim{i}"))
        assert axis_tree.sf == self.layout_axes.sf

        # Now determine the targets mapping the nodes back to mesh
        # points and DoFs which constitute the 'true' layout axis tree. This
        # means we have to determine the mapping:
        #
        #   n0 -> (p0, d0)
        #   n1 -> (p0, d1)
        #   n2 -> (p1, d0)
        #   ...
        #
        # We realise this by computing the pair of mappings:
        #
        #   n0 -> p0, n1 -> p0, n2 -> p1, ...
        #
        # and
        #
        #   n0 -> d0, n1 -> d1, n2 -> d0, ...
        #
        # The excessive tabulations should not impose a performance penalty
        # because the mappings are compressed during compilation.
        op3.extras.debug.warn_todo("Cythonize")

        node_point_map_array = numpy.full(self.nodes.local_size, -1, dtype=IntType)
        node_dof_map_array = node_point_map_array.copy()

        dof_axis = utils.just_one(axis for axis in self.layout_axes.nodes if axis.label == "dof")
        ndofs = dof_axis.component.size.buffer.buffer.data_ro  # this sucks

        node = 0
        for point, ndof in enumerate(ndofs):
            for dof in range(ndof):
                node_point_map_array[node] = point
                node_dof_map_array[node] = dof
                node += 1
        assert (node_point_map_array >= 0).all() and (node_dof_map_array >= 0).all()

        node_point_map_dat = op3.Dat(self.nodes, data=node_point_map_array)
        node_dof_map_dat = op3.Dat(self.nodes, data=node_dof_map_array)

        node_point_map_expr = op3.as_linear_buffer_expression(node_point_map_dat)
        node_dof_map_expr = op3.as_linear_buffer_expression(node_dof_map_dat)

        # We have the two mappings as expressions, now we have to plug them
        # into the indexed axis tree in the right way.
        target = {}
        orig_target: idict[ConcretePathT, tuple[AxisTarget, ...]] = utils.just_one(axis_tree.targets)
        for source_path, axis_targets in orig_target.items():
            target[source_path] = []
            for axis_target in axis_targets:
                if axis_target.axis == "nodes":
                    mesh_target = op3.AxisTarget("mesh", "mylabel", node_point_map_expr)
                    dof_target = op3.AxisTarget("dof", "XXX", node_dof_map_expr)
                    target[source_path].extend([mesh_target, dof_target])
                else:
                    # All other axes (e.g. 'dim0') map directly to the layout axes
                    # and do not require modification
                    target[source_path].append(axis_target)
        target = utils.freeze(target)

        return op3.IndexedAxisTree(
            axis_tree,
            unindexed=self.layout_axes,
            targets=(target,),
        )

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

    def __eq__(self, other):
        if not isinstance(other, FunctionSpace):
            return False
        # FIXME: Think harder about equality
        return self.mesh() == other.mesh() and \
            self.ufl_element() == other.ufl_element() and \
            self.component == other.component

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.mesh(), self.axes, self.ufl_element()))

    @utils.cached_property
    def _ad_parent_space(self):
        return self.parent

    @property
    def block_shape(self) -> tuple[IntType, ...]:
        return self.shape

    @property
    def block_size(self) -> IntType:
        """The total number of degrees of freedom at each function space node."""
        return numpy.prod(self.shape, dtype=IntType)

    @utils.cached_property
    def dm(self):
        r"""A PETSc DM describing the data layout for this FunctionSpace."""
        dm = self._dm()
        dmhooks.set_function_space(dm, self)
        return dm

    def _dm(self):
        from firedrake.mg.utils import get_level
        dm = PETSc.DMShell().create(comm=self.comm)
        dm.setLocalSection(self.local_section)
        dm.setGlobalVector(self.template_vec)
        _, level = get_level(self.mesh())
        dmhooks.attach_hooks(dm, level=level,
                             sf=self.mesh().topology_dm.getPointSF())
        # Remember the function space so we can get from DM back to FunctionSpace.
        dmhooks.set_function_space(dm, self)
        return dm

    @utils.cached_property
    def template_vec(self):
        """Dummy PETSc Vec of the right size for this set of axes."""
        vec = PETSc.Vec().create(comm=self.comm)
        vec.setSizes(
            (self.layout_axes.owned.local_size, self.layout_axes.global_size),
            bsize=self.block_size,
        )
        vec.setUp()
        return vec

    @cached_property
    @utils.deprecated("field_ises")
    def _ises(self):
        """A list of PETSc ISes defining the global indices for each set in
        the DataSet.

        Used when extracting blocks from matrices for solvers."""
        return self.field_ises

    @cached_property
    def field_ises(self) -> tuple[PETSc.IS]:
        """A list of PETSc ISes defining the global indices for each set in
        the DataSet.

        Used when extracting blocks from matrices for solvers."""
        size = self.axes.owned.local_size
        start = self.comm.exscan(size) or 0
        is_ = PETSc.IS().createStride(size, first=start, comm=self.comm)
        is_.setBlockSize(self.block_size)
        return (is_,)

    @cached_property
    def local_ises(self) -> tuple[PETSc.IS]:
        is_ = PETSc.IS().createStride(self.axes.local_size, comm=MPI.COMM_SELF)
        is_.setBlockSize(self.block_size)
        return (is_,)

    # TODO: cythonize
    @utils.cached_property
    def local_section(self):
        dm = self._mesh.topology_dm
        section = PETSc.Section().create(comm=self.comm)
        section.setChart(0, self._mesh.num_points)

        if self._mesh._new_to_old_point_renumbering is not None:
            section.setPermutation(self._mesh._new_to_old_point_renumbering)

        entity_dofs = _num_entity_dofs(self.finat_element)

        if type(self._mesh.topology) is MeshTopology:
            stratum_ranges = {
            }
            for dim in range(dm.getDimension()+1):
                ndofs = entity_dofs[dim]
                for pt in range(*dm.getDepthStratum(dim)):
                    section.setDof(pt, ndofs)
        elif type(self._mesh.topology) is VertexOnlyMeshTopology:
            stratum_ranges = {0: (0, dm.getLocalSize())}
            ndofs = entity_dofs[0]
            for pt in range(dm.getLocalSize()):
                section.setDof(pt, ndofs)
        else:
            assert self.extruded

            dim_label = dm.getLabel("depth")
            base_dim_label = dm.getLabel("base_dim")
            for pt in range(*dm.getChart()):
                dim = dim_label.getValue(pt)
                base_dim = base_dim_label.getValue(pt)
                if base_dim == dim:
                    # vertex
                    ndofs = entity_dofs[base_dim, 0]
                else:
                    # edge
                    ndofs = entity_dofs[base_dim, 1]
                section.setDof(pt, ndofs)

        if self._ufl_function_space.ufl_element().family() == "Real":
            p_start, p_end = section.getChart()
            for p in range(p_start, p_end):
                section.setOffset(p, 0)
        else:
            section.setUp()

        return section

    @utils.cached_property
    def cell_node_map(self) -> op3.Map:
        try:
            cells_axis, node_axis = self.cell_node_dat.axes.nodes
        except:
            breakpoint()
        return op3.Map(
            {
                idict({cells_axis.label: cells_axis.component.label}): [
                    [op3.TabulatedMapComponent("nodes", None, self.cell_node_dat)]
                ],
            }, 
            # TODO: This is only here so labels resolve, ideally we would relabel to make this fine
            name=node_axis.label
        )

    # IMPORTANT: This is only for the subspace - if addressing a subfunction with this an offset is needed
    @utils.cached_property
    def cell_node_list(self) -> np.ndarray:
        r"""A numpy array mapping mesh cells to function space nodes."""
        nodes = self.cell_node_dat.data_ro.reshape((self._mesh.cells.owned.size, -1))
        if self.extruded:
            return nodes[::self.mesh().layers-1]
        else:
            return nodes

    @cached_property
    def cell_node_dat(self) -> op3.Dat:
        # internal detail really, do not expose in pyop3/__init__.py
        from pyop3.expr.visitors import loopified_shape, get_shape
        from firedrake.parloops import transform_packed_cell_closure_dat

        mesh = self.mesh()

        indices_axes = self.axes.blocked(self.shape)
        indices_array = numpy.arange(indices_axes.size, dtype=IntType)
        indices_dat = op3.Dat(indices_axes, data=indices_array)

        cell_index = self._mesh.cells.owned.iter()

        scalar_space = self
        if self.shape:
            scalar_space = self.sub(0)

        map_expr = transform_packed_cell_closure_dat(indices_dat[mesh.closure(cell_index)], scalar_space, cell_index)
        map_axes = op3.AxisTree(self._mesh.cells.owned.root)
        map_axes = map_axes.add_subtree(map_axes.leaf_path, get_shape(map_expr)[0])
        map_dat = op3.Dat.full(map_axes, -1, dtype=IntType, prefix="map")
        op3.loop(cell_index, map_dat[cell_index].assign(map_expr), eager=True)

        # now reshape things because we want to have 2 axes: cells and nodes
        ncells = mesh.cells.owned.local_size
        arity = map_axes.local_size // ncells
        cells_axis = map_axes.root
        node_axis = op3.Axis(arity)
        map_axes = op3.AxisTree.from_iterable([cells_axis, node_axis])
        map_dat = op3.Dat(map_axes, buffer=map_dat.buffer, prefix="map")

        return map_dat

    @cached_property
    def extr_cell_node_list(self):
        assert self.extruded
        if self.mesh().variable_layers:
            raise NotImplementedError
        return self.cell_node_dat.data_ro.reshape((self._mesh.cells.owned.size, -1))

    @utils.cached_property
    def topological(self):
        r"""Function space on a mesh topology."""
        return self

    def mesh(self):
        return self._mesh

    def ufl_element(self):
        r"""The :class:`finat.ufl.finiteelementbase.FiniteElementBase` associated
        with this space."""
        return self.ufl_function_space().ufl_element()

    def ufl_function_space(self):
        r"""The :class:`~ufl.classes.FunctionSpace` associated with this space."""
        return self._ufl_function_space

    def label(self):
        return self._label

    def __len__(self):
        return 1

    def __iter__(self):
        yield self

    def __repr__(self):
        return "FunctionSpace(%r, %r, name=%r)" % (self.mesh(),
                                                   self.ufl_element(),
                                                   self.name)

    def __str__(self):
        return self.__repr__()

    @utils.cached_property
    def subspaces(self):
        """Split into a tuple of constituent spaces."""
        return (self,)

    strong_subspaces = property(lambda self: self.subspaces)

    @property
    def subfunctions(self):
        import warnings
        warnings.warn("The 'subfunctions' property is deprecated for function spaces, please use the "
                      "'subspaces' property instead", category=FutureWarning)
        return self.subspaces

    def __getitem__(self, i):
        r"""Return the ith subspace."""
        if i != 0:
            raise IndexError("Only index 0 supported on a FunctionSpace")
        return self

    @utils.cached_property
    def _strong_components(self):
        if self.rank == 0:
            return self.strong_subspaces
        else:
            components = numpy.empty(self.shape, dtype=object)
            for ix in numpy.ndindex(self.shape):
                components[ix] = ComponentFunctionSpace(self, ix, weak=False)
            return utils.readonly(components)

    @utils.cached_property
    def _components(self):
        if self.rank == 0:
            return self.subspaces
        else:
            components = numpy.empty(self.shape, dtype=object)
            for ix in numpy.ndindex(self.shape):
                components[ix] = ComponentFunctionSpace(self, ix)
            return utils.readonly(components)

    def sub(self, indices, *, weak: bool = True):
        r"""Return a view into the ith component."""
        indices = parse_component_indices(indices, self.shape)
        if weak:
            return self._components[indices or 0]
        else:
            return self._strong_components[indices or 0]

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
        if self.boundary_set:
            raise NotImplementedError
        return self.nodal_axes.size
        constrained_node_set = set()
        for sub_domain in self.boundary_set:
            constrained_node_set.update(self._shared_data.boundary_nodes(self, sub_domain))
        return self.node_set.total_size - len(constrained_node_set)

    @utils.cached_property
    def dof_count(self):
        r"""The number of degrees of freedom (includes halo dofs) of this
        function space on this process. Cf. :attr:`FunctionSpace.node_count` ."""
        return self.axes.size

    def dim(self):
        r"""The global number of degrees of freedom for this function space.

        See also :attr:`FunctionSpace.dof_count` and :attr:`FunctionSpace.node_count` ."""
        return self.template_vec.getSize()

    def make_dat(self, val=None, valuetype=None, name=None):
        """Return a new Dat storing DoFs for the function space."""
        if val is not None:
            if isinstance(val, numpy.ndarray):
                if valuetype is not None:
                    assert val.dtype == valuetype
                data = val
            else:
                data = numpy.asarray(val, dtype=valuetype)
            return op3.Dat(self.axes, data=data.flatten(), name=name)
        else:
            return op3.Dat.zeros(self.axes, dtype=valuetype, name=name)

    # this is redundant
    def cell_closure_map(self, cell):
        """Return a map from cells to cell closures."""
        return self.mesh()._fiat_closure(cell)

    # NOTE: This shouldn't ever really be used, doesn't do the right permutations..
    def entity_node_map(self, iteration_spec):
        r"""Return entity node map rebased on ``source_mesh``.

        Parameters
        ----------
        source_mesh : MeshTopology
            Source (base) mesh topology.
        source_integral_type : str
            Integral type on source_mesh.
        source_subdomain_id : int
            Subdomain ID on source_mesh.
        source_all_integer_subdomain_ids : dict
            All integer subdomain ids on source_mesh.

        Returns
        -------
        pyop2.types.map.Map or None
            Entity node map.

        """
        mesh = self.mesh()
        if iteration_spec.mesh.topology is mesh.topology:
            composed_map = None
            target_integral_type = iteration_spec.integral_type
        elif mesh.submesh_youngest_common_ancester(iteration_spec.mesh):
            composed_map, target_integral_type = self.mesh().trans_mesh_entity_map(iteration_spec)
        else:
            # No shared topology, must be using a vertex-only mesh
            composed_map = mesh.cell_parent_cell_map(iteration_spec.loop_index)
            target_integral_type = "cell"

        if target_integral_type == "cell":
            def self_map(index):
                return self.mesh().closure(index)
        elif target_integral_type == "exterior_facet_top":
            def self_map(index):
                return self.mesh().closure(index)
        elif target_integral_type == "exterior_facet_bottom":
            def self_map(index):
                return self.mesh().closure(index)
        elif target_integral_type == "interior_facet_horiz":
            def self_map(index):
                return self.mesh().closure(index)
        elif target_integral_type == "exterior_facet":
            def self_map(index):
                return self.mesh().closure(self.mesh().support(index))
        elif target_integral_type == "exterior_facet_vert":
            def self_map(index):
                return mesh.closure(mesh.support(index))
        elif target_integral_type == "interior_facet":
            def self_map(index):
                return mesh.closure(mesh.support(index))
        elif target_integral_type == "interior_facet_vert":
            raise NotImplementedError
            self_map = self.interior_facet_node_map()
        else:
            raise ValueError(f"Unknown integral_type: {target_integral_type}")

        if not composed_map:
            return self_map(iteration_spec.loop_index)
        else:
            return self_map(composed_map)

    # def cell_node_map(self):
    #     r"""Return the :class:`pyop2.types.map.Map` from cels to
    #     function space nodes."""
    #     sdata = self._shared_data
    #     return sdata.get_map(self,
    #                          self.mesh().cell_set,
    #                          self.finat_element.space_dimension(),
    #                          "cell_node",
    #                          self.offset,
    #                          self.offset_quotient)

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
        V = self  # fixme
        if sub_domain in ["bottom", "top"]:
            if not V.extruded:
                raise ValueError("Invalid subdomain '%s' for non-extruded mesh",
                                 sub_domain)
            entity_dofs = eutils.flat_entity_dofs(V.finat_element.entity_dofs())
            key = (entity_dofs_key(entity_dofs), sub_domain, V.boundary_set)
            return self.get_top_bottom_boundary_nodes(V.mesh(), key, V)
        else:
            if sub_domain == "on_boundary":
                sdkey = sub_domain
            else:
                sdkey = as_tuple(sub_domain)
            key = (entity_dofs_key(V.finat_element.entity_dofs()), sdkey, V.boundary_set)
            return get_facet_closure_nodes(V.mesh(), key, V)

    def get_top_bottom_boundary_nodes(self, mesh, key):
        """Get top or bottom boundary nodes of an extruded function space.

        :arg mesh: The mesh to cache on.
        :arg key: A 3-tuple of ``(entity_dofs_key, sub_domain, boundary_set)`` key.
            Where sub_domain indicates top or bottom.
        :arg V: The FunctionSpace to select from.
        :arg entity_dofs: The flattened entity dofs.
        :returnsL: A numpy array of the (unique) boundary nodes.
        """
        V = self # fixme
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
                nodes = nodes + offset[mask]*(mesh.layers - 2)
            return numpy.unique(nodes)

    @utils.cached_property
    def _lgmap(self) -> PETSc.LGMap:
        """Return the mapping from process-local to global DoF numbering."""
        indices = self.block_axes.global_numbering
        return PETSc.LGMap().create(indices, bsize=self.value_size, comm=self.comm)

    @utils.cached_property
    def _unblocked_lgmap(self) -> PETSc.LGMap:
        """Return the local-to-global mapping with a block size of 1."""
        if self.value_size == 1:
            return self._lgmap
        else:
            indices = self.axes.global_numbering
            return PETSc.LGMap().create(indices, bsize=1, comm=self.comm)

    def collapse(self):
        return type(self)(self.mesh(), self.ufl_element())


class RestrictedFunctionSpace(FunctionSpace):
    r"""A representation of a function space, with additional information
    about where boundary conditions are to be applied.

    If a :class:`FunctionSpace` is represented as V, we can decompose V into
    V = V0 + VΓ, where V0 contains functions in the basis of V that vanish on
    the boundary where a boundary condition is applied, and VΓ contains all
    other basis functions. The :class:`RestrictedFunctionSpace`
    corresponding to V takes functions only from V0 when solving problems, or
    when creating a TestFunction and TrialFunction. The values on the boundary
    set will remain constant when solving, but are present in the
    output of the solver.

    :arg function_space: The :class:`FunctionSpace` to restrict.
    :kwarg boundary_set: A set of subdomains on which a DirichletBC will be applied.
    :kwarg name: An optional name for this :class:`RestrictedFunctionSpace`,
        useful for later identification.

    Notes
    -----
    If using this class to solve or similar, a list of DirichletBCs will still
    need to be specified on this space and passed into the function.
    """
    def __init__(self, function_space, boundary_set=frozenset(), name=None):
        label = ""
        boundary_set_ = []
        # NOTE: boundary_set must be deterministically ordered here to ensure
        # that the label is consistent between ranks.
        for boundary_domain in sorted(boundary_set, key=str):
            if isinstance(boundary_domain, str):
                boundary_set_.append(boundary_domain)
            else:
                # Currently, can not handle intersection of boundaries;
                # e.g., boundary_set = [(1, 2)], which is different from [1, 2].
                bd, = as_tuple(boundary_domain)
                boundary_set_.append(bd)
        boundary_set = boundary_set_
        for boundary_domain in boundary_set:
            label += str(boundary_domain)
            label += "_"
        self.boundary_set = frozenset(boundary_set)
        super().__init__(function_space._mesh.topology,
                         function_space.ufl_element(), function_space.name)
        self._label = label
        self._ufl_function_space = ufl.FunctionSpace(function_space._mesh.ufl_mesh(),
                                                     function_space.ufl_element(),
                                                     label=self._label)
        self.function_space = function_space
        self.name = name or function_space.name

    # def set_shared_data(self):
    #     sdata = get_shared_data(self._mesh, self.ufl_element(), self.boundary_set)
    #     self._shared_data = sdata
    #     self.node_set = sdata.node_set
    #     r"""A :class:`pyop2.types.set.Set` representing the function space nodes."""
    #     self.dof_dset = op2.DataSet(self.node_set, self.shape or 1,
    #                                 name="%s_nodes_dset" % self.name,
    #                                 apply_local_global_filter=sdata.extruded)
    #     r"""A :class:`pyop2.types.dataset.DataSet` representing the function space
    #     degrees of freedom."""
    #
    #     # check not all degrees of freedom are constrained
    #     unconstrained_dofs = self.dof_dset.size - self.dof_dset.constrained_size
    #     if self.comm.allreduce(unconstrained_dofs) == 0:
    #         raise ValueError("All degrees of freedom are constrained.")
    #     self.finat_element = create_element(self.ufl_element())
    #     # Used for reconstruction of mixed/component spaces.
    #     # sdata carries real_tensorproduct.
    #     self.real_tensorproduct = sdata.real_tensorproduct
    #     self.extruded = sdata.extruded
    #     self.offset = sdata.offset
    #     self.offset_quotient = sdata.offset_quotient
    #     self.cell_boundary_masks = sdata.cell_boundary_masks
    #     self.interior_facet_boundary_masks = sdata.interior_facet_boundary_masks
    #     self.global_numbering = sdata.global_numbering

    def __eq__(self, other):
        if not isinstance(other, RestrictedFunctionSpace):
            return False
        return self.function_space == other.function_space and \
            self.boundary_set == other.boundary_set

    def __repr__(self):
        return self.__class__.__name__ + "(%r, name=%r, boundary_set=%r)" % (
            str(self.function_space), self.name, self.boundary_set)

    def __hash__(self):
        return hash((self.mesh(), self.layout_axes, self.ufl_element(),
                     self.boundary_set))

    def local_to_global_map(self, bcs, lgmap=None, mat_type=None):
        return lgmap or self.dof_dset.lgmap

    def collapse(self):
        return type(self)(self.function_space.collapse(), boundary_set=self.boundary_set)

    @cached_property
    def nodal_axes(self) -> op3.IndexedAxisTree:
        scalar_axis_tree = self.axes.blocked(self.shape)
        num_nodes = scalar_axis_tree.size

        node_axis = op3.Axis([op3.AxisComponent(num_nodes, sf=scalar_axis_tree.sf)], "nodes")
        axis_tree = op3.AxisTree(node_axis)
        for i, dim in enumerate(self.shape):
            axis_tree = axis_tree.add_axis(axis_tree.leaf_path, op3.Axis({"XXX": dim}, f"dim{i}"))

        # Reuse the targets from the unconstrained space as they do not affect
        # the layout functions.
        targets = self.function_space.nodal_axes.targets

        return op3.IndexedAxisTree(
            axis_tree,
            unindexed=self.layout_axes,
            targets=targets,
        )


class MixedFunctionSpace:
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
    def __init__(self, spaces, mesh, name=None, *, layout=None):
        super(MixedFunctionSpace, self).__init__()
        if not isinstance(mesh, MeshSequenceTopology):
            raise TypeError(f"mesh must be MeshSequenceTopology: got {mesh}")
        if len(mesh) != len(spaces):
            raise RuntimeError(f"len(mesh) ({len(mesh)}) != len(spaces) ({len(spaces)})")

        # If any of 'spaces' is an indexed function space (i.e. from an already
        # mixed space) then recover the original space to use here.
        orig_spaces = []
        for space in spaces:
            if isinstance(space, ProxyFunctionSpace) and space.index is not None:
                space = space.parent._orig_spaces[space.index]
            orig_spaces.append(space)
        spaces = orig_spaces

        # If 'layout' isn't provided then build from the subspaces
        if layout is None:
            layout = ("field", tuple(subspace.layout for subspace in spaces))

        self._orig_spaces = spaces
        self.layout = layout
        self._strong_spaces = tuple(IndexedFunctionSpace(i, s, self, weak=False)
                             for i, s in enumerate(spaces))
        self._spaces = tuple(IndexedFunctionSpace(i, s, self)
                             for i, s in enumerate(spaces))
        self._ufl_function_space = ufl.FunctionSpace(mesh.ufl_mesh(),
                                                     finat.ufl.MixedElement(*[s.ufl_element() for s in spaces]))
        self.name = name or "_".join(str(s.name) for s in spaces)
        label = ""
        for s in spaces:
            label += "(" + s._label + ")_"
        self._label = label
        self.boundary_set = frozenset()
        self._subspaces = {}
        self._mesh = mesh

        self.comm = mesh.comm
        self._comm = mpi.internal_comm(mesh.comm, self)

    # These properties are so a mixed space can behave like a normal FunctionSpace.
    index = None
    component = None
    parent = None
    rank = 1

    # TODO:
    # @cached_on(mesh)?
    @cached_property
    def layout_axes(self) -> op3.AxisTree:
        return layout_from_spec(self.layout, self.axis_constraints)

    @cached_property
    def axes(self) -> op3.AxisForest:
        return op3.AxisForest([self.plex_axes, self.nodal_axes])

    @cached_property
    def plex_axes(self) -> op3.IndexedAxisTree:
        return self._make_axes("plex")

    @cached_property
    def nodal_axes(self) -> op3.IndexedAxisTree:
        return self._make_axes("nodal")

    def _make_axes(self, mode: Literal["plex", "nodal"]) -> op3.IndexedAxisTree:
        field_axis = utils.single_valued((
            axis for axis in self.layout_axes.nodes if axis.label == "field"
        ))
        axis_tree = op3.AxisTree(field_axis)
        targets = utils.StrictlyUniqueDefaultDict(list)
        for field_component, subspace in zip(
            field_axis.components, self._orig_spaces, strict=True
        ):
            if mode == "plex":
                subaxes = subspace.plex_axes
            else:
                assert mode == "nodal"
                subaxes = subspace.nodal_axes

            leaf_path = idict({field_axis.label: field_component.label})
            axis_tree = axis_tree.add_subtree(
                leaf_path, subaxes.materialize()
            )

            # Target a full slice of the 'field' component
            # FIXME: should be list of lists
            targets[leaf_path] = [
                op3.AxisTarget(
                    field_axis.label,
                    field_component.label,
                    op3.AxisVar(field_axis.linearize(field_component.label)),
                ),
            ]
            for subtarget in subaxes.targets:
                for subpath, subaxis_targets in subtarget.items():
                    # FIXME: should be list of lists
                    targets[leaf_path | subpath].extend(subaxis_targets)
                break  # stop here because we only care about the first target for the moment

        targets = utils.freeze(targets)
        # FIXME: This is intactably slow! Cannot allow this expansion to occur
        # targets = op3.utils.expand_collection_of_iterables(targets)
        targets = (targets,)
        return op3.IndexedAxisTree(
            axis_tree, unindexed=self.layout_axes, targets=targets,
        )

    @cached_property
    def axis_constraints(self) -> tuple[AxisConstraint]:
        field_axis = op3.Axis(
            [op3.AxisComponent(1, space.index) for space in self._spaces],
            "field",
        )
        return merge_axis_constraints(
            field_axis,
            [space.axis_constraints for space in self._spaces],
        )

    def mesh(self):
        return self._mesh

    @property
    def topological(self):
        r"""Function space on a mesh topology."""
        return self

    def ufl_element(self):
        r"""The :class:`finat.ufl.mixedelement.MixedElement` associated with this space."""
        return self.ufl_function_space().ufl_element()

    def ufl_function_space(self):
        r"""The :class:`~ufl.classes.FunctionSpace` associated with this space."""
        return self._ufl_function_space

    def __eq__(self, other):
        if not isinstance(other, MixedFunctionSpace) or len(other) != len(self):
            return False
        return all(s == o for s, o in zip(self, other))

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(tuple(self))

    @property
    def strong_subspaces(self):
        r"""The list of :class:`FunctionSpace`\s of which this
        :class:`MixedFunctionSpace` is composed."""
        return self._strong_spaces

    @property
    def subspaces(self):
        r"""The list of :class:`FunctionSpace`\s of which this
        :class:`MixedFunctionSpace` is composed."""
        return self._spaces

    @property
    def subfunctions(self):
        import warnings
        warnings.warn("The 'subfunctions' property is deprecated for function spaces, please use the "
                      "'subspaces' property instead", category=FutureWarning)
        return self.subspaces

    def sub(self, i, *, weak=True):
        r"""Return the `i`th :class:`FunctionSpace` in this
        :class:`MixedFunctionSpace`."""
        if weak:
            return self._spaces[i]
        else:
            return self._strong_spaces[i]

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

    @property
    def block_shape(self) -> tuple:
        return ()

    @property
    def block_size(self) -> IntType:
        return IntType.type(1)

    @utils.cached_property
    def node_count(self):
        r"""Return a tuple of :attr:`FunctionSpace.node_count`\s of the
        :class:`FunctionSpace`\s of which this :class:`MixedFunctionSpace` is
        composed."""
        return tuple(fs.node_count for fs in self._spaces)

    @cached_property
    def dof_count(self):
        r"""Return a tuple of :attr:`FunctionSpace.dof_count`\s of the
        :class:`FunctionSpace`\s of which this :class:`MixedFunctionSpace` is
        composed."""
        return tuple(fs.dof_count for fs in self._spaces)

    def dim(self):
        r"""The global number of degrees of freedom for this function space.

        See also :attr:`FunctionSpace.dof_count` and :attr:`FunctionSpace.node_count`."""
        return self.template_vec.getSize()

    @cached_property
    def field_ises(self) -> tuple[PETSc.IS, ...]:
        """A list of PETSc ISes defining the global indices for each set in
        the DataSet.

        Used when extracting blocks from matrices for solvers."""
        ises = []
        start = self._comm.exscan(self.axes.owned.local_size) or 0
        for subspace in self:
            size = subspace.axes.owned.local_size
            is_ = PETSc.IS().createStride(size, first=start, comm=self.comm)
            is_.setBlockSize(subspace.block_size)
            ises.append(is_)
            start += size
        return tuple(ises)

    @property
    @utils.deprecated("field_ises")
    def _ises(self):
        """A list of PETSc ISes defining the global indices for each set in
        the DataSet.

        Used when extracting blocks from matrices for solvers.

        """
        return self.field_ises

    @cached_property
    def local_ises(self) -> tuple[PETSc.IS, ...]:
        """A list of PETSc ISes defining the local indices for each set in
        the DataSet.

        Used when extracting blocks from matrices for solvers.

        """
        ises = []
        start = 0
        for subspace in self:
            size = subspace.axes.local_size
            is_ = PETSc.IS().createStride(size, first=start, comm=MPI.COMM_SELF)
            is_.setBlockSize(subspace.block_size)
            ises.append(is_)
            start += size
        return tuple(ises)

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

    def local_to_global_map(self, bcs, lgmap=None, mat_type=None):
        r"""Return a map from process local dof numbering to global dof numbering.

        If BCs is provided, mask out those dofs which match the BC nodes."""
        raise NotImplementedError("Not for mixed maps right now sorry!")

    # NOTE: This function is exactly the same as make_dat for a non-mixed space
    def make_dat(self, val=None, valuetype=None, name=None):
        r"""Return a newly allocated :class:`pyop2.types.dat.MixedDat` defined on the
        :attr:`dof_dset` of this :class:`MixedFunctionSpace`."""
        if val is not None and val.size != self.axes.size:
            raise ValueError("Provided array has the wrong number of entries")

        if val is not None:
            if valuetype is not None:
                assert val.dtype == valuetype
            return op3.Dat(self.axes, data=val.flatten(), name=name)
        else:
            return op3.Dat.zeros(self.axes, dtype=valuetype, name=name)

    @utils.cached_property
    def dm(self):
        r"""A PETSc DM describing the data layout for fieldsplit solvers."""
        dm = self._dm()
        dmhooks.set_function_space(dm, self)
        return dm

    def _dm(self):
        from firedrake.mg.utils import get_level

        dm = PETSc.DMShell().create(comm=self.comm)
        # Not implemented for mixed function spaces...
        # dm.setLocalSection(self.local_section)
        dm.setGlobalVector(self.template_vec)
        # TODO: Think harder about which mesh should be used
        _, level = get_level(self.mesh()[0])
        dmhooks.attach_hooks(dm, level=level)
        return dm

    # NOTE: this is the same as for the non-mixed case, should rejig types
    @cached_property
    def template_vec(self):
        """Dummy PETSc Vec of the right size for this function space."""
        vec = PETSc.Vec().create(comm=self.comm)
        vec.setSizes(
            (self.layout_axes.owned.local_size, self.layout_axes.global_size),
            bsize=self.block_size,
        )
        vec.setUp()
        return vec

    # FIXME: This loses information for proxy spaces
    def collapse(self):
        return type(self)([V_ for V_ in self], self.mesh())


class ProxyFunctionSpace(FunctionSpace):
    r"""A :class:`FunctionSpace` that one can attach extra properties to.

    :arg mesh: The mesh to use.
    :arg element: The UFL element.
    :arg name: The name of the function space.

    .. warning::

       Users should not build a :class:`ProxyFunctionSpace` directly,
       it is mostly used as an internal implementation detail.
    """
    def __new__(cls, mesh, element, name=None):
        topology = mesh.topology
        self = super(ProxyFunctionSpace, cls).__new__(cls)
        if mesh is not topology:
            return WithGeometry.create(self, mesh)
        else:
            return self

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


class ProxyRestrictedFunctionSpace(RestrictedFunctionSpace):
    r"""A :class:`RestrictedFunctionSpace` that one can attach extra properties to.

    :arg function_space: The function space to be restricted.
    :kwarg boundary_set: The boundary domains on which boundary conditions will
       be specified
    :kwarg name: The name of the restricted function space.

    .. warning::

       Users should not build a :class:`ProxyRestrictedFunctionSpace` directly,
       it is mostly used as an internal implementation detail.
    """
    def __new__(cls, function_space, boundary_set=frozenset(), name=None):
        topology = function_space._mesh.topology
        self = super(ProxyRestrictedFunctionSpace, cls).__new__(cls)
        if function_space._mesh is not topology:
            return WithGeometry.create(self, function_space._mesh)
        else:
            return self

    def __repr__(self):
        return "%sProxyRestrictedFunctionSpace(%r, name=%r,  boundary_set=%r, index=%r, component=%r)" % \
            (str(self.identifier).capitalize(),
             str(self.function_space),
             self.name,
             self.boundary_set,
             self.index,
             self.component)

    def __str__(self):
        return self.__repr__()

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
        return super(ProxyRestrictedFunctionSpace, self).make_dat(*args, **kwargs)


def IndexedFunctionSpace(index, space, parent, *, weak: bool = True):
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
    elif len(space.boundary_set) > 0:
        new = ProxyRestrictedFunctionSpace(space.function_space, name=space.name, boundary_set=space.boundary_set)
    else:
        new = ProxyFunctionSpace(space.mesh(), space.ufl_element(), name=space.name)
    new.index = index
    new.parent = parent
    new.identifier = "indexed"

    return new


def ComponentFunctionSpace(parent, component, *, weak: bool = True):
    r"""Build a new FunctionSpace that remembers it represents a
    particular component.  Used for applying boundary conditions to
    components of a :func:`.VectorFunctionSpace` or :func:`.TensorFunctionSpace`.

    :arg parent: The parent space (a FunctionSpace with a
        VectorElement or TensorElement).
    :arg component: The component to represent.
    :returns: A new :class:`ProxyFunctionSpace` with the component set.
    """
    element = parent.ufl_element()
    assert type(element) in frozenset([finat.ufl.VectorElement, finat.ufl.TensorElement])
    if component not in numpy.ndindex(parent.shape):
        raise IndexError(f"Invalid component 'component' not in '{parent.shape}'")
    new = ProxyFunctionSpace(parent.mesh(), element.sub_elements[0], name=parent.name)
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

    @cached_property
    def axis_constraints(self) -> tuple[AxisConstraint]:
        # Get the number of DoFs per cell, it is illegal to have DoFs on
        # other entities.
        ndofs = None
        for dim, dim_ndofs in _num_entity_dofs(self.finat_element).items():
            if dim == self.mesh().cell_label:
                ndofs = dim_ndofs
            else:
                assert dim_ndofs == 0
        assert ndofs is not None

        dof_axis = op3.Axis(
            op3.AxisComponent(ndofs, "XXX", sf=op3.single_star_sf(self._comm, ndofs)),
            "dof"
        )
        constraints = [AxisConstraint(dof_axis)]
        for i, dim in enumerate(self.shape):
            shape_axis = op3.Axis({"XXX": dim}, f"dim{i}")
            constraint = AxisConstraint(shape_axis)
            constraints.append(constraint)
        return tuple(constraints)

    @cached_property
    def plex_axes(self) -> op3.IndexedAxisTree:
        return self._make_axes("plex")

    @cached_property
    def nodal_axes(self) -> op3.IndexedAxisTree:
        return self._make_axes("nodal")

    def _make_axes(self, mode: Literal["plex", "nodal"]) -> op3.IndexedAxisTree:
        # For real function spaces the mesh is conceptually non-existent as all
        # cells map to the same globally-defined DoFs. We can trick pyop3 into
        # pretending that a mesh axis exists though by careful construction of
        # an indexed axis tree. With this trick no special-casing of real spaces
        # should be necessary anywhere else.

        # Create the pretend axis tree that includes the mesh axis. This is
        # just a DG0 function.
        dg_space = FunctionSpace(self._mesh, self.element.reconstruct(family="DG"))
        if mode == "plex":
            fake_axes = dg_space.plex_axes.materialize()
        else:
            assert mode == "nodal"
            fake_axes = dg_space.nodal_axes.materialize()

        # Now map the mesh-aware axis tree back to the actual one. For the 'plex'
        # case this means mapping all of the mesh points to nothing, and the
        # (single) cell DoF to 0. For the 'nodal' case we have to map all the node
        # points to 0.
        #
        # Other elements of the tree (i.e. tensor shape) are the same and
        # can be left unchanged.
        targets = utils.StrictlyUniqueDefaultDict(list)
        for path, axis_targets in utils.just_one(fake_axes.targets).items():
            if mode == "plex":
                if path.keys() == {self._mesh.name}:
                    targets[path] = ()
                else:
                    for axis_target in axis_targets:
                        if axis_target.axis == f"dof{self._mesh.cell_label}":
                            axis_target = op3.AxisTarget("dof", "XXX", 0)
                        elif axis_target.axis.startswith("dof"):
                            axis_target = op3.AxisTarget("dof", "XXX", op3.NAN)
                        targets[path].append(axis_target)
            else:
                assert mode == "nodal"
                for axis_target in axis_targets:
                    if axis_target.axis == "nodes":
                        axis_target = op3.AxisTarget("dof", "XXX", 0)
                    targets[path].append(axis_target)
        targets = utils.freeze(targets)

        # FIXME: This is intractably slow! Cannot allow this expansion to occur
        # targets = op3.utils.expand_collection_of_iterables(targets)
        targets = (targets,)

        return op3.IndexedAxisTree(
            fake_axes, unindexed=self.layout_axes, targets=targets,
        )


    # used?
    global_numbering = None

    def __eq__(self, other):
        if not isinstance(other, RealFunctionSpace):
            return False
        # FIXME: Think harder about equality
        return self.mesh() == other.mesh() and \
            self.ufl_element() == other.ufl_element()

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.mesh(), self.ufl_element()))

    # def set_shared_data(self):
    #     pass

    # def make_dof_dset(self):
    #     raise NotImplementedError
    #     return op2.GlobalDataSet(self.make_dat())

    # def entity_node_map(self, source_mesh, source_integral_type, source_subdomain_id, source_all_integer_subdomain_ids):
    #     return None
    #
    # def cell_node_map(self, bcs=None):
    #     ":class:`RealFunctionSpace` objects have no cell node map."
    #     return None
    #
    # def interior_facet_node_map(self, bcs=None):
    #     ":class:`RealFunctionSpace` objects have no interior facet node map."
    #     return None
    #
    # def exterior_facet_node_map(self, bcs=None):
    #     ":class:`RealFunctionSpace` objects have no exterior facet node map."
    #     return None
    #
    # def bottom_nodes(self):
    #     ":class:`RealFunctionSpace` objects have no bottom nodes."
    #     return None
    #
    # def top_nodes(self):
    #     ":class:`RealFunctionSpace` objects have no bottom nodes."
    #     return None
    #
    # def local_to_global_map(self, bcs, lgmap=None, mat_type=None):
    #     assert len(bcs) == 0
    #     return None


@dataclass
class FunctionSpaceCargo:
    """Helper class carrying data for a :class:`WithGeometryBase`.

    It is required because it permits Firedrake to have stripped forms
    that still know Firedrake-specific information (e.g. that they are a
    component of a parent function space).
    """

    topological: FunctionSpace
    parent: Optional[WithGeometryBase]


class InvalidFunctionSpaceLayoutException(Exception):
    pass


@functools.singledispatch
def layout_from_spec(layout_spec: Any, axis_constraints: Sequence) -> op3.AxisTree:
    visited_axes = frozenset()
    axis_nest = _parse_layout_spec(layout_spec, axis_constraints, visited_axes)
    return op3.AxisTree.from_nest(axis_nest)


def _parse_layout_spec(layout_spec: Sequence[str], axis_specs: Sequence, visited_axes) -> idict:
    if len(layout_spec) == 0:
        return _axis_nest_from_constraints(axis_specs, visited_axes)

    axis_label = layout_spec[0]

    candidate_axis_specs = frozenset(
        axis_spec
        for axis_spec in axis_specs
        if axis_spec.axis.label == axis_label
    )
    try:
        selected_axis_spec = utils.just_one(
            axis_spec
            for axis_spec in candidate_axis_specs
            if axis_spec.within_axes.items() <= visited_axes
        )
    except ValueError:
        raise InvalidFunctionSpaceLayoutException(
            "Cannot construct a valid function space layout from the provided spec"
        )
    selected_axis = selected_axis_spec.axis

    # filter out axis specs that match the current axis so they can't get
    # reused further down
    axis_specs = tuple(
        axis_spec
        for axis_spec in axis_specs
        if axis_spec not in candidate_axis_specs
    )

    if axis_specs:  # are there any remaining axes to attach?
        axis_nest = {selected_axis: []}
        if len(layout_spec) > 1:
            # 'sub_layout_specs' can either be flat (e.g. '["axis1", "axis2"]')
            # or nested (e.g. '[["axis1", ["axis2"]], ["axis3"]]'). If the former
            # then the spec is broadcasted to all components. Otherwise we assume
            # that the spec is per-component.
            # Using the example from above, '["axis1", "axis2"]' gets exploded to
            # [["axis1", "axis2"], ["axis1", "axis2"]] (assuming there are two
            # components for the current axis).
            if isinstance(layout_spec[1], str):  # flat case
                sub_layout_specs = [layout_spec[1:]] * len(selected_axis.components)
            else:  # nested
                assert len(layout_spec) == 2
                sub_layout_specs = layout_spec[1]

            # NOTE: This is exactly the same as the nested case except for broadcasting
            for component, sub_layout_spec in zip(
                selected_axis.components, sub_layout_specs, strict=True
            ):
                # prune axis specs that go down different branches
                axis_specs_ = tuple(
                    axis_spec
                    for axis_spec in axis_specs
                    if selected_axis.label not in axis_spec.within_axes
                    or (selected_axis.label, component.label) in axis_spec.within_axes.items()
                )

                # FIXME: Not doing anything with visited_axes
                visited_axes_ = visited_axes | {(selected_axis.label, component.label)}

                sub_axis_nest = _parse_layout_spec(sub_layout_spec, axis_specs_, visited_axes_)
                axis_nest[selected_axis].append(sub_axis_nest)
        else:
            # at the bottom of the provided layout spec, populate the axis tree
            # with the remaining axes
            for component in selected_axis.components:
                axis_specs_ = tuple(
                    axis_spec
                    for axis_spec in axis_specs
                    if selected_axis.label not in axis_spec.within_axes
                    or (selected_axis.label, component.label) in axis_spec.within_axes.items()
                )

                # FIXME: Not doing anything with visited_axes
                visited_axes_ = visited_axes | {(selected_axis.label, component.label)}
                sub_axis_nest = _axis_nest_from_constraints(axis_specs_, visited_axes_)
                axis_nest[selected_axis].append(sub_axis_nest)

        return idict(axis_nest)
    else:
        assert not layout_spec[1:], "More layout information provided than available axes"
        return selected_axis


def _axis_nest_from_constraints(axis_constraints: Sequence[AxisConstraint], visited_axes: Set[tuple[str, str]]) -> idict | op3.Axis:
    constraint, *subconstraints = axis_constraints
    axis = constraint.axis

    # filter out axis specs that match the current axis so they can't get reused further down
    axis_constraints = tuple(axis_spec for axis_spec in axis_constraints if axis_spec.axis != axis)

    axis_nest = collections.defaultdict(list)

    for component in axis.components:
        subconstraints_ = tuple(
            subconstraint
            for subconstraint in subconstraints
            if axis.label not in subconstraint.within_axes
            or (axis.label, component.label) in subconstraint.within_axes.items()
        )
        if subconstraints_:
            # FIXME: Not doing anything with visited_axes
            subnest = _axis_nest_from_constraints(subconstraints_, visited_axes)
            axis_nest[axis].append(subnest)

    return idict(axis_nest) if axis_nest else axis



def merge_axis_constraints(root_axis: op3.Axis, axis_constraintss: Sequence[Sequence[AxisConstraint]]) -> tuple[AxisConstraint]:
    # start by collecting like axes
    axis_info: defaultdict[op3.Axis, dict[op3.ComponentLabelT, idict]] = defaultdict(dict)
    for root_component, constraints in zip(root_axis.components, axis_constraintss, strict=True):
        for constraint in constraints:
            axis_info[constraint.axis][root_component.label] = constraint.within_axes

    # Now build the new set of constraints. To do this we inspect the
    # per-component constraints for each axis: if the constraints are all the
    # same then it is not necessary to specialise by component, otherwise an
    # extra constraint is needed. For example:
    #
    # * Consider the "dof" axis for a mixed space with identical subspaces:
    #
    #     {dof_axis: {0: {"mesh": None}, 1: {"mesh": None}}}
    #
    #   Here this is saying that 'dof_axis' exists under root components 0 and 1
    #   and each time must satisfy the constraint of having '{"mesh": None}'
    #   above them.
    #
    #   Since the constraints are identical for all components they do not need
    #   to be specialised. The final constraint is thus:
    #
    #     AxisConstraint(dof_axis, {"mesh": None})
    #
    # * Alternatively consider a mixed space of CG1 x Real:
    #
    #     {mesh_axis: {0: {}}}
    #
    #   The "mesh" axis only exists for the CG1 subspace and so a new constraint
    #   is needed:
    #
    #     AxisConstraint(mesh_axis, {"field": 0})
    constraints = [AxisConstraint(root_axis)]
    for axis, per_component_info in axis_info.items():
        if (
            per_component_info.keys() == set(root_axis.component_labels)
            and utils.is_single_valued(per_component_info.values())
        ):
            # Axis present for all components and constraints match: use as is
            within_axes = utils.single_valued(per_component_info.values())
            constraints.append(AxisConstraint(axis, within_axes))
        else:
            # Constraint mismatch: need to specialise by component
            for component_label, orig_within_axes in per_component_info.items():
                within_axes = orig_within_axes | {root_axis.label: component_label}
                constraints.append(AxisConstraint(axis, within_axes))
    return tuple(constraints)


@functools.singledispatch
def parse_component_indices(indices: Any, shape: tuple[int, ...]) -> tuple[int, ...]:
    raise TypeError


@parse_component_indices.register(tuple)
def _(indices: tuple[int, ...], shape: tuple[int, ...]) -> tuple[int, ...]:
    return indices


@parse_component_indices.register(int)
def _(index: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    # Historically tensor-valued spaces would be addressed using a flat index
    # instead of a tuple. Here we convert the old-style flat index to a
    # nested one. Eventually we should be able to remove this and simply cast
    # an integer index to a tuple (e.g. '3' to '(3,)').
    if len(shape) > 1:
        warnings.warn(
            "Scalar indexing of a tensor-valued space is no longer recommended "
            "practice, please pass a tuple instead",
            FutureWarning,
        )
    return list(numpy.ndindex(shape))[index]


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


# TODO: make return type and arg consistent
def mask_lgmap(lgmap: LGMap, bcs, idxs=()) -> PETSc.LGMap:
    """Return a map from process-local to global DoF numbering.

    # update this#

    Parameters
    ----------
    bcs
        Optional iterable of boundary conditions. If provided these DoFs
        are masked out (set to -1) in the returned map.

    Returns
    -------
    PETSc.LGMap
        The local-to-global mapping.

    """
    if not bcs:
        return lgmap.as_petsc_lgmap()

    # Set constrained values in the lgmap to -1

    unblocked = any(bc.function_space().component is not None for bc in bcs)
    if unblocked:
        indices = lgmap.unblocked_indices
        axes = lgmap.axes
        block_size = 1
    else:
        indices = lgmap.indices
        axes = lgmap.axes.blocked(lgmap.block_shape)
        block_size = lgmap.block_size

    indices = indices.copy()
    dat = op3.Dat(axes, data=indices)
    for bc in bcs:
        if unblocked:
            component = bc.function_space().component
            if component is None:
                component = Ellipsis
            dat[*idxs][bc.node_set, *component].assign(-1, eager=True)
        else:
            dat[*idxs][bc.node_set].assign(-1, eager=True)

    return PETSc.LGMap().create(dat.data_ro, bsize=block_size, comm=lgmap.comm)
