import functools
import numpy as np

from ufl.form import Form
from ufl.corealg.traversal import unique_pre_traversal
from ufl.algorithms.traversal import iter_expressions

import firedrake
from firedrake import functionspaceimpl
from firedrake.function import Function, CoordinatelessFunction
from firedrake.constant import Constant
#from firedrake.utils import IntType, RealType, ScalarType

from pyop2 import op2
from pyop2.datatypes import ScalarType, IntType, as_ctypes
from pyop2.utils import as_tuple

from finat.point_set import PointSet
from finat.quadrature import QuadratureRule
from tsfc.finatinterface import create_element

import gem
from gem.node import MemoizerArg
from gem.optimise import filtered_replace_indices


__all__ = ['ScalarSubspace', 'RotatedSubspace', 'Subspaces']


class Subspace(object):
    r"""Wrapper base for Firedrake subspaces.

    :arg function_space: The :class:`~.functionspaceimpl.WithGeometry`.
    :arg val: The subspace values that are multiplied to basis functions.
    :arg subdomain: The subdomain(s) on which values are set.
    The constructor mimics that of :class:`~DirichletBC`.
    """

    _globalcount = 0

    def __init__(self, function_space, val=None, subdomain=None, name=None, dtype=ScalarType, count=None):

        self._count = count or Subspace._globalcount
        if self._count >= Subspace._globalcount:
            Subspace._globalcount = self._count + 1

        V = function_space
        if isinstance(V, Function):
            V = V.function_space()
        elif not isinstance(V, functionspaceimpl.WithGeometry):
            raise NotImplementedError("Can't make a Subspace defined on a "
                                      + str(type(function_space)))

        if subdomain:
            if not val:
                raise RuntimeError("Must provide val if providing subdomain.")
            if not isinstance(subdomain, op2.Subset):
               # Turn subdomain into op2.Subset.
               subdomain = V.boundary_node_subset(subdomain) 
            val = Function(V).assign(val, subset=subdomain)
            self._data = val.topological
        else:
            if isinstance(val, (Function, CoordinatelessFunction)):
                val = val.topological
                if val.function_space() != V.topological:
                    raise ValueError("Function values have wrong function space.")
                self._data = val
            else:
                self._data = CoordinatelessFunction(V.topological,
                                                    val=val, name=name, dtype=dtype)
        self._function_space = V
        self.parent = None
        self.index = None

        self._repr = "Subspace(%s, %s)" % (repr(self._function_space), repr(self._count))

    def function_space(self):
        r"""Return the :class:`.FunctionSpace`, or :class:`.MixedFunctionSpace`
            that this :class:`Subspace` is a subspace of.
        """
        return self._function_space

    def ufl_element(self):
        return self.function_space().ufl_element()

    def __getattr__(self, name):
        val = getattr(self._data, name)
        setattr(self, name, val)
        return val
    
    def __hash__(self):
        return hash(repr(self))

    def __str__(self):
        count = str(self._count)
        if len(count) == 1:
            return "s_%s" % count
        else:
            return "s_{%s}" % count

    def __repr__(self):
        return self._repr

    #@utils.cached_property
    @property
    def complement(self):
        return ComplementSubspace(self)

    def transform(self, expressions, subspace_expr, i_dummy, i, elem, dtype):
        r"""Apply linear transformation.

        :arg elem: UFL element: `self.ufl_function_space().ufl_element`
            or its subelement (in case of `MixedElement`).
        :arg expression: GEM expression representing local subspace data array
            associated with elem.
        :arg dtype: data type (= KernelBuilder.scalar_type).

        Classical implementation of functions/function spaces.
        Linear combination of basis:
        
        u = \sum [ u_i * \phi_i ]
              i
        
        u     : function
        u_i   : ith coefficient
        \phi_i: ith basis
        """
        raise NotImplementedError("Must implement `transform` method.")


class IndexedSubspace(object):
    def __init__(self, parent, index):
        self.parent = parent
        self.index = index

    def function_space(self):
        return self.parent.function_space().split()[self.index]

    def ufl_element(self):
        return self.function_space().ufl_element()

    def transform(self, expressions, subspace_expr, i_dummy, i, elem, dtype):
        return self.parent.transform(expressions, subspace_expr, i_dummy, i, elem, dtype)

    def __eq__(self, other):
        return self.parent is other.parent and \
               self.index == other.index
    
    def __hash__(self):
        return hash(repr(self))

    def __str__(self):
        return "%s[%s]" % (self.parent, self.index)

    def __repr__(self):
        return "IndexedSubspace(%s, %s)" % (repr(self.parent), repr(self.index))


class ScalarSubspace(Subspace):
    def __init__(self, V, val=None, subdomain=None, name=None, dtype=ScalarType):
        Subspace.__init__(self, V, val=val, subdomain=subdomain, name=name, dtype=dtype)

    def transform(self, expressions, subspace_expr, i_dummy, i, elem, dtype):
        r"""Basic subspace.

        Linear combination of weighted basis:

        u = \sum [ u_i * (w_i * \phi_i) ]
              i

        u     : function
        u_i   : ith coefficient
        \phi_i: ith basis
        w_i   : ith weight (stored in the subspace object)
                w_i = 0 to deselect the associated basis.
                w_i = 1 to select.
        """
        substitution = tuple(zip(i_dummy, i))
        mapper = MemoizerArg(filtered_replace_indices)
        expressions = tuple(mapper(expression, substitution) for expression in expressions)
        return tuple(gem.Product(gem.Indexed(subspace_expr, i), expression) for expression in expressions)


class RotatedSubspace(Subspace):
    def __init__(self, V, val=None, subdomain=None, name=None, dtype=ScalarType):
        Subspace.__init__(self, V, val=val, subdomain=subdomain, name=name, dtype=dtype)

    def transform(self, expressions, subspace_expr, i_dummy, i, elem, dtype):
        r"""Rotation subspace.

        u = \sum [ u_i * \sum [ \psi(e)_i * \sum [ \psi(e)_k * \phi(e)_k ] ] ]
              i            e                  k

        u       : function
        u_i     : ith coefficient
        \phi(e) : basis vector whose elements not associated with
                  topological entity e are set zero.
        \psi(e) : rotation vector whose elements not associated with
                  topological entity e are set zero.
        """
        shape = subspace_expr.shape
        finat_element = create_element(elem)
        if len(shape) == 1:
            entity_dofs = finat_element.entity_dofs()
        else:
            entity_dofs = finat_element.base_element.entity_dofs()
        _expressions = []
        for expression in expressions:
            _expression = gem.Zero()
            for dim in entity_dofs:
                for _, dofs in entity_dofs[dim].items():
                    if len(dofs) == 0 or (len(dofs) == 1 and len(shape) == 1):
                        continue
                    ind = np.zeros(shape, dtype=dtype)
                    for dof in dofs:
                        for ndind in np.ndindex(shape[1:]):
                            ind[(dof, ) + ndind] = 1.
                    temp = gem.IndexSum(gem.Product(gem.Product(gem.Literal(ind)[i_dummy], subspace_expr[i_dummy]), expression), i_dummy)
                    _expression = gem.Sum(_expression, gem.Product(gem.Product(gem.Literal(ind)[i], subspace_expr[i]), temp))
            _expressions.append(_expression)
        return tuple(_expressions)


class Subspaces(object):
    r"""Bag of :class:`.Subspace`s.

    :arg subspaces: :class:`.Subspace` objects.
    """

    def __init__(self, *subspaces):
        self._components = tuple(subspaces)

    def __iter__(self):
        return iter(self._components)

    def __len__(self):
        return len(self._components)

    #@utils.cached_property
    @property
    def components(self):
        return self._components

    #@utils.cached_property
    @property
    def complement(self):
        return ComplementSubspace(self)


class ComplementSubspace(object):
    r"""Complement of :class:`.Subspace` or :class:`.Subspaces`."""

    def __init__(self, subspace):
        if not isinstance(subspace, (Subspace, Subspaces)):
            raise TypeError("Expecting `Subspace` or `Subspaces`,"
                            " not %s." % subspace.__class__.__name__)
        self._subspace = subspace

    #@utils.cached_property
    @property
    def complement(self):
        return self._subspace


def make_subspace_numbers_and_parts(subspaces, original_subspaces):
    # -- subspace_numbers_: which subspaces are used in this TSFCIntegralData.
    # -- subspace_parts_  : which components are used if mixed (otherwise None).
    subspaces_and_parts_dict = {}
    for subspace in subspaces:
        if subspace.parent:
            subspaces_and_parts_dict.setdefault(subspace.parent, set()).update((subspace.index, ))
        else:
            subspaces_and_parts_dict[subspace] = None
    subspace_numbers = []
    subspace_parts = []
    for i, subspace in enumerate(original_subspaces):
        if subspace in subspaces_and_parts_dict:
            subspace_numbers.append(i)
            parts = subspaces_and_parts_dict[subspace]
            if parts is None:
                subspace_parts.append(None)
            else:
                parts = sorted(parts)
                subspace_parts.append(parts)
    subspaces = sort_indexed_subspaces(subspaces)
    return subspaces, subspace_numbers, subspace_parts


def sort_indexed_subspaces(subspaces):
    return sorted(subspaces, key=lambda s: (s.parent.count() if s.parent else s.count(), 
                                            -1 if s.index is None else s.index))


def extract_subspaces(a, cls=object):
    subspaces_and_objects = extract_indexed_subspaces(a, cls=cls)
    subspaces = set(s if s.parent is None else s.parent for s, o in subspaces_and_objects)
    return tuple(sorted(subspaces, key=lambda x: x.count()))


def extract_indexed_subspaces(a, cls=object):
    from firedrake.projected import FiredrakeProjected
    from firedrake.slate.slate import TensorBase, Tensor
    if isinstance(a, TensorBase):
        if isinstance(a, Tensor):
            return extract_indexed_subspaces(a.form, cls=cls)
        _set = set()
        for op in a.operands:
            _set.update(extract_indexed_subspaces(op, cls=cls))
        return _set
    elif isinstance(a, Form):        
        return set((o.subspace(), o.ufl_operands[0])
                    for e in iter_expressions(a)
                    for o in unique_pre_traversal(e)
                    if isinstance(o, FiredrakeProjected) and isinstance(o.ufl_operands[0], cls))
    else:
        raise TypeError("Unexpected type: %s" % str(type(a)))
