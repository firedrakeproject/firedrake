import itertools
import functools

import abc

import numpy as np

from ufl.form import Form
from ufl.corealg.traversal import unique_pre_traversal
from ufl.algorithms.traversal import iter_expressions

from firedrake import functionspaceimpl, utils
from firedrake.function import Function, CoordinatelessFunction

from pyop2.datatypes import RealType, ScalarType

import gem
from gem.node import MemoizerArg
from gem.optimise import filtered_replace_indices

from tsfc.finatinterface import create_element

__all__ = ['ScalarSubspace', 'RotatedSubspace', 'ComplementSubspace']


class AbstractSubspace(object, metaclass=abc.ABCMeta):
    """A representation of an abstract mesh topology without a concrete
        PETSc DM implementation"""

    def __init__(self, V, nonzero_indices=None):
        self._function_space = V
        if len(V) == 1:
            assert nonzero_indices is None, "nonzero_indices must be None for non-mixed function space."
        else:
            if nonzero_indices is None:
                nonzero_indices = tuple(range(len(V)))
            else:
                assert isinstance(nonzero_indices, tuple), "nonzero_indices must be tuple for mixed function space."
                nonzero_indices = tuple(nonzero_indices)
        self.nonzero_indices = nonzero_indices

    parent = None
    """Parent of an indexed mixed function subspace."""

    index = None
    """Index of an indexed mixed function subspace."""

    def function_space(self):
        """The base FunctionSpace of this Subspace."""
        return self._function_space

    def ufl_element(self):
        """The ufl element of the function space."""
        return self.function_space().ufl_element()

    @abc.abstractmethod
    def subspaces(self):
        """The typle of subspaces that actually carry data used to define this subspace."""
        pass

    @abc.abstractmethod
    def transform(self, expressions, subspace_expr, i_dummy, i, dtype):
        """Apply linear transformation.

        :arg expressions: a tuple of gem expressions written in terms of i_dummy.
        :arg subspace_expr: GEM expression representing local subspace data array
            associated with finat_element.
        :arg i_dummy: the multiindex of the expressions.
        :arg i: the multiindex of the return variable.
        :arg dtype: data type (= KernelBuilder.scalar_type).

        A non-projected (default) function is written as a
        linear combination of basis functions:

        .. math::

            u = \\sum_i [ u_i * \\phi_i ]

            u      : function
            u_i    : ith coefficient
            \\phi_i: ith basis
        """
        pass

    @abc.abstractmethod
    def is_zero(self, index):
        pass

    @abc.abstractmethod
    def is_identity(self, index):
        pass

    @abc.abstractmethod
    def __eq__(self):
        pass


# -- Base subspaces that carry data.


class Subspace(AbstractSubspace):
    """Abstract class for Firedrake subspaces.

    :arg V: the :class:`~.functionspaceimpl.WithGeometry`.
    :arg val: the subspace values.

    This class to some extent mimics :class:`ufl.Coefficient`.
    """

    _globalcount = 0

    def __init__(self, V, val=None, name=None, dtype=ScalarType, count=None, nonzero_indices=None):
        self._count = count or Subspace._globalcount
        if self._count >= Subspace._globalcount:
            Subspace._globalcount = self._count + 1
        if not isinstance(V, functionspaceimpl.WithGeometry):
            raise NotImplementedError("Can't make a Subspace defined on a "
                                      + str(type(V)))
        if isinstance(val, (Function, CoordinatelessFunction)):
            val = val.topological
            if val.function_space() != V.topological:
                raise ValueError("Function values have wrong function space.")
            self._data = val
        else:
            self._data = CoordinatelessFunction(V.topological,
                                                val=val, name=name, dtype=dtype)
        AbstractSubspace.__init__(self, V, nonzero_indices=nonzero_indices)
        self._repr = "Subspace(%s, %s)" % (repr(self._function_space), repr(self._count))

    def count(self):
        return self._count

    @property
    def topological(self):
        r"""The underlying coordinateless function."""
        return self._data

    @utils.cached_property
    def _split(self):
        return tuple(type(self)(V, val)
                     for (V, val) in zip(self.function_space(), self.topological.split()))

    def split(self):
        r"""Extract any sub :class:`Function`\s defined on the component spaces
        of this this :class:`Function`'s :class:`.FunctionSpace`."""
        return self._split

    @utils.cached_property
    def _components(self):
        if self.function_space().value_size == 1:
            return (self, )
        else:
            return tuple(type(self)(self.function_space().sub(i), self.topological.sub(i))
                         for i in range(self.function_space().value_size))

    def sub(self, i):
        r"""Extract the ith sub :class:`Function` of this :class:`Function`.

        :arg i: the index to extract

        See also :meth:`split`.

        If the :class:`Function` is defined on a
        :class:`~.VectorFunctionSpace` or :class:`~.TensorFunctiionSpace` this returns a proxy object
        indexing the ith component of the space, suitable for use in
        boundary condition application."""
        if len(self.function_space()) == 1:
            return self._components[i]
        return self._split[i]

    def subspaces(self):
        return (self, )

    def is_zero(self, index):
        if self.nonzero_indices is None:
            return False
        else:
            return index not in self.nonzero_indices

    def is_identity(self, index):
        return False

    def __eq__(self, other):
        if other is self:
            return True
        elif type(other) is not type(self):
            return False
        else:
            return other._function_space == self._function_space and other._count == self._count

    def __str__(self):
        count = str(self._count)
        if len(count) == 1:
            return "s_%s" % count
        else:
            return "s_{%s}" % count

    def __repr__(self):
        return self._repr

    def __hash__(self):
        return hash(repr(self))

    def __getattr__(self, name):
        val = getattr(self._data, name)
        setattr(self, name, val)
        return val


class ScalarSubspace(Subspace):
    def transform(self, expressions, subspace_expr, i_dummy, i, dtype):
        """Basic subspace.

        Linear combination of weighted basis:

        .. math::

            u = \\sum [ u_i * (w_i * \\phi_i) ]
                   i

            u      : function
            u_i    : ith coefficient
            \\phi_i: ith basis
            w_i    : ith weight (stored in the subspace object)
                    w_i = 0 to deselect the associated basis.
                    w_i = 1 to select.
        """
        subspace_expr, = subspace_expr
        substitution = tuple(zip(i_dummy, i))
        mapper = MemoizerArg(filtered_replace_indices)
        expressions = tuple(mapper(expression, substitution) for expression in expressions)
        return tuple(gem.Product(gem.Indexed(subspace_expr, i), expression) for expression in expressions)


class RotatedSubspace(Subspace):
    def transform(self, expressions, subspace_expr, i_dummy, i, dtype):
        """Rotation subspace.

        .. math::

            u = \\sum [ u_i * \\sum [ \\psi(e)_i * \\sum [ \\psi(e)_k * \\phi(e)_k ] ] ]
                   i             e                    k

            u        : function
            u_i      : ith coefficient
            \\phi(e) : basis vector whose elements not associated with
                       topological entity e are set zero.
            \\psi(e) : rotation vector whose elements not associated with
                       topological entity e are set zero.
        """
        subspace_expr, = subspace_expr
        shape = subspace_expr.shape
        finat_element = create_element(self.ufl_element())
        if len(shape) != 2:
            raise TypeError(f"{type(self)} is only for VectorElements, not for {self.ufl_element()}.")
        entity_dofs = finat_element.base_element.entity_dofs()
        _expressions = []
        for expression in expressions:
            _expression = gem.Zero()
            for dim in entity_dofs:
                for _, dofs in entity_dofs[dim].items():
                    if len(dofs) == 0:
                        continue
                    # Avoid pytools/persistent_dict.py TypeError: unsupported type for persistent hash keying: <class 'complex'>
                    #ind = np.zeros(shape, dtype=dtype)
                    ind = np.zeros(shape, dtype=RealType)
                    for dof in dofs:
                        for ndind in np.ndindex(shape[1:]):
                            ind[(dof, ) + ndind] = 1.
                    temp = gem.IndexSum(gem.Product(gem.Product(gem.Literal(ind)[i_dummy], subspace_expr[i_dummy]), expression), i_dummy)
                    _expression = gem.Sum(_expression, gem.Product(gem.Product(gem.Literal(ind)[i], subspace_expr[i]), temp))
            _expressions.append(_expression)
        return tuple(_expressions)


class NodalHermiteSubspace(Subspace):
    def transform(self, expressions, subspace_expr, i_dummy, i, dtype):
        """Rotation subspace.

        .. math::

            u = \\sum [ u_i * \\sum [ \\psi(e)_i * \\sum [ \\psi(e)_k * \\phi(e)_k ] ] ]
                   i             e                    k

            u        : function
            u_i      : ith coefficient
            \\phi(e) : basis vector whose elements not associated with
                       topological entity e are set zero.
            \\psi(e) : rotation vector whose elements not associated with
                       topological entity e are set zero.
        """
        subspace_expr, = subspace_expr
        shape = subspace_expr.shape
        finat_element = create_element(self.ufl_element())
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
                    # Avoid pytools/persistent_dict.py TypeError: unsupported type for persistent hash keying: <class 'complex'>
                    #ind = np.zeros(shape, dtype=dtype)
                    ind = np.zeros(shape, dtype=RealType)
                    for dof in dofs:
                        for ndind in np.ndindex(shape[1:]):
                            ind[(dof, ) + ndind] = 1.
                    temp = gem.IndexSum(gem.Product(gem.Product(gem.Literal(ind)[i_dummy], subspace_expr[i_dummy]), expression), i_dummy)
                    _expression = gem.Sum(_expression, gem.Product(gem.Product(gem.Literal(ind)[i], subspace_expr[i]), temp))
            _expressions.append(_expression)
        return tuple(_expressions)


# -- Wrapper subspaces.


class IndexedSubspace(AbstractSubspace):
    """Representation of indexed subspace.

    Convenient when splitting a form according to indices;
    see `split_form`.
    """
    def __init__(self, parent, index):
        AbstractSubspace.__init__(self, parent.function_space().split()[index])
        self.parent = parent
        self.index = index

    def split(self):
        raise RuntimeError("Unable to split an IndexSubspace.")

    def transform(self, expressions, subspace_expr, i_dummy, i, dtype):
        return self.parent.split()[self.index].transform(expressions, subspace_expr, i_dummy, i, dtype)

    def subspaces(self):
        return self.parent.subspaces()

    def is_zero(self, index):
        raise RuntimeError("is_zero method is not to be called on IndexedSubspace.")

    def is_identity(self, index):
        raise RuntimeError("is_identity method is not to be called on IndexedSubspace.")

    def __eq__(self, other):
        return self.parent == other.parent and self.index == other.index

    def __str__(self):
        return "%s[%s]" % (self.parent, self.index)

    def __repr__(self):
        return "IndexedSubspace(%s, %s)" % (repr(self.parent), repr(self.index))

    def __hash__(self):
        return hash(repr(self))


class ComplementSubspace(AbstractSubspace):
    def __init__(self, subspace):
        AbstractSubspace.__init__(self, subspace.function_space(), nonzero_indices=subspace.nonzero_indices)
        self._subspace = subspace

    @utils.cached_property
    def _split(self):
        return tuple(type(self)(s) for s in self._subspace.split())

    def split(self):
        r"""Extract any sub :class:`Function`\s defined on the component spaces
        of this this :class:`Function`'s :class:`.FunctionSpace`."""
        return self._split

    def transform(self, expressions, subspace_expr, i_dummy, i, dtype):
        substitution = tuple(zip(i_dummy, i))
        mapper = MemoizerArg(filtered_replace_indices)
        _expressions_base = tuple(mapper(expression, substitution) for expression in expressions)
        _expressions_cmpl = self._subspace.transform(expressions, subspace_expr, i_dummy, i, dtype)
        return tuple(gem.Sum(_base, gem.Product(gem.Literal(-1.), _cmpl)) for _base, _cmpl in zip(_expressions_base, _expressions_cmpl))

    def subspaces(self):
        return self._subspace.subspaces()

    def is_zero(self, index):
        return False

    def is_identity(self, index):
        return index not in self.nonzero_indices

    def __eq__(self, other):
        if other is self:
            return True
        elif type(other) is not type(self):
            return False
        else:
            return other._subspace == self._subspace

    def __str__(self):
        return "ComplementSubspace[%s]" % self._subspace

    def __repr__(self):
        return "ComplementSubspace(%s)" % repr(self._subspace)

    def __hash__(self):
        return hash(repr(self))


class DirectSumSubspace(AbstractSubspace):
    """Bag of :class:`.Subspace`s.

    :arg subspaces: the :class:`.Subspace`s.
    """

    def __init__(self, *subspaces):
        self._subspaces = tuple(subspaces)
        V, = set(s.function_space() for s in subspaces)
        nonzero_indices, = set(s.nonzero_indices for s in subspaces)
        AbstractSubspace.__init__(self, V, nonzero_indices=nonzero_indices)

    @utils.cached_property
    def _split(self):
        return tuple(type(self)(*ss) for ss in zip(*(s.split() for s in self._subspaces)))

    def split(self):
        r"""Extract any sub :class:`Function`\s defined on the component spaces
        of this this :class:`Function`'s :class:`.FunctionSpace`."""
        return self._split

    def transform(self, expressions, subspace_exprs, i_dummy, i, dtype):
        assert len(subspace_exprs) == len(self._subspaces)
        _expressions_list = []
        for subspace, subspace_expr in zip(self._subspaces, subspace_exprs):
            _expressions = subspace.transform(expressions, (subspace_expr, ), i_dummy, i, dtype)
            _expressions_list.append(_expressions)
        return tuple(functools.reduce(lambda a, b: gem.Sum(a, b), _exprs) for _exprs in zip(*_expressions_list))

    def subspaces(self):
        return self._subspaces

    def is_zero(self, index):
        if self.nonzero_indices is None:
            return False
        else:
            return index not in self.nonzero_indices

    def is_identity(self, index):
        return False

    def __eq__(self, other):
        if other is self:
            return True
        elif type(other) is not type(self):
            return False
        else:
            return other._subspaces == self._subspaces

    def __str__(self):
        return "DirectSumSubspace[%s]" % "+".join([str(s) for s in self._subspaces])

    def __repr__(self):
        return "DirectSumSubspace(%s)" % ", ".join([repr(s) for s in self._subspaces])

    def __hash__(self):
        return hash(repr(self))


# -- Helper functions


def extract_subspaces(form, cls=object):
    """Extract `Subspace`s from form.

    This compares to form.coefficients().
    """
    subspaces_and_objects = extract_indexed_subspaces(form, cls=cls)
    #subspaces = set(s if s.parent is None else s.parent for s, o in subspaces_and_objects)
    subspaces = set(itertools.chain(*(s.subspaces() for s, _ in subspaces_and_objects)))
    return tuple(sorted(subspaces, key=lambda x: x.count()))


def extract_indexed_subspaces(form, cls=object):
    from firedrake.projected import FiredrakeProjected
    from firedrake.slate.slate import TensorBase, Tensor
    if isinstance(form, TensorBase):
        if isinstance(form, Tensor):
            return extract_indexed_subspaces(form.form, cls=cls)
        _set = set()
        for op in form.operands:
            _set.update(extract_indexed_subspaces(op, cls=cls))
        return _set
    elif isinstance(form, Form):
        return set((o.subspace(), o.ufl_operands[0])
                   for e in iter_expressions(form)
                   for o in unique_pre_traversal(e)
                   if isinstance(o, FiredrakeProjected) and isinstance(o.ufl_operands[0], cls))
    else:
        raise TypeError("Unexpected type: %s" % str(type(form)))
