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


__all__ = ['ScalarSubspace', 'RotatedSubspace', 'Subspaces']


class Subspace(object):
    """Abstract class for Firedrake subspaces.

    :arg V: the :class:`~.functionspaceimpl.WithGeometry`.
    :arg val: the subspace values.

    This class to some extent mimics :class:`ufl.Coefficient`.
    """

    _globalcount = 0

    def __init__(self, V, val=None, name=None, dtype=ScalarType, count=None):
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
        self._function_space = V
        self.parent = None
        self.index = None
        self._repr = "Subspace(%s, %s)" % (repr(self._function_space), repr(self._count))

    def count(self):
        return self._count

    def function_space(self):
        return self._function_space

    def ufl_element(self):
        return self.function_space().ufl_element()

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

    def transform(self, expressions, subspace_expr, i_dummy, i, finat_element, dtype):
        """Apply linear transformation.

        :arg expressions: a tuple of gem expressions written in terms of i_dummy.
        :arg subspace_expr: GEM expression representing local subspace data array
            associated with finat_element.
        :arg i_dummy: the multiindex of the expressions.
        :arg i: the multiindex of the return variable.
        :arg finat_element: FInAT element.
        :arg dtype: data type (= KernelBuilder.scalar_type).

        A non-projected (default) function is written as a
        linear combination of basis functions:

        .. math::

            u = \\sum_i [ u_i * \\phi_i ]

            u      : function
            u_i    : ith coefficient
            \\phi_i: ith basis
        """
        raise NotImplementedError("Subclasses must implement `transform` method.")

    def subspaces(self):
        return (self, )

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


class IndexedSubspace(object):
    """Representation of indexed subspace.

    Convenient when splitting a form according to indices;
    see `split_form`.
    """
    def __init__(self, parent, index):
        self.parent = parent
        self.index = index

    def function_space(self):
        return self.parent.function_space().split()[self.index]

    def ufl_element(self):
        return self.function_space().ufl_element()

    def transform(self, expressions, subspace_expr, i_dummy, i, finat_element, dtype):
        return self.parent.transform(expressions, subspace_expr, i_dummy, i, finat_element, dtype)

    def subspaces(self):
        return self.parent.subspaces()

    def __eq__(self, other):
        return self.parent == other.parent and self.index == other.index

    def __str__(self):
        return "%s[%s]" % (self.parent, self.index)

    def __repr__(self):
        return "IndexedSubspace(%s, %s)" % (repr(self.parent), repr(self.index))

    def __hash__(self):
        return hash(repr(self))


class ScalarSubspace(Subspace):
    def __init__(self, V, val=None, name=None, dtype=ScalarType):
        Subspace.__init__(self, V, val=val, name=name, dtype=dtype)

    def transform(self, expressions, subspace_expr, i_dummy, i, finat_element, dtype):
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
        substitution = tuple(zip(i_dummy, i))
        mapper = MemoizerArg(filtered_replace_indices)
        expressions = tuple(mapper(expression, substitution) for expression in expressions)
        return tuple(gem.Product(gem.Indexed(subspace_expr, i), expression) for expression in expressions)


class RotatedSubspace(Subspace):
    def __init__(self, V, val=None, name=None, dtype=ScalarType):
        Subspace.__init__(self, V, val=val, name=name, dtype=dtype)

    def transform(self, expressions, subspace_expr, i_dummy, i, finat_element, dtype):
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
        shape = subspace_expr.shape
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


class Subspaces(object):
    """Bag of :class:`.Subspace`s.

    :arg subspaces: the :class:`.Subspace`s.
    """

    def __init__(self, *subspaces):
        self._components = tuple(subspaces)

    def __iter__(self):
        return iter(self._components)

    def __len__(self):
        return len(self._components)

    def __getitem__(self, i):
        return self._components[i]

    def components(self):
        return self._components


# -- Helper functions


def extract_subspaces(form, cls=object):
    """Extract `Subspace`s from form.

    This compares to form.coefficients().
    """
    subspaces_and_objects = extract_indexed_subspaces(form, cls=cls)
    subspaces = set(s if s.parent is None else s.parent for s, o in subspaces_and_objects)
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


def sort_indexed_subspaces(subspaces):
    return sorted(subspaces, key=lambda s: (s.parent.count() if s.parent else s.count(),
                                            -1 if s.index is None else s.index))


def make_subspace_numbers_and_parts(subspaces, original_subspaces):
    """Sort subspaces and make subspace_numbers and subspace_parts.

    :arg subspaces: a set of `(Indexed)Subspace`s found in the TSFCIntegralData.
    :arg original_subspaces: a tuple of sorted original subspaces found in the TSFCFormData.
    :returns: a tuple of sorted subspaces, subspace_numbers, and subspace_parts, where:
        subspace_numbers: which `Subspace`s are used in the TSFCIntegralData;
        this compares to `tsfc.Kernel.coefficient_numbers`.
        subspace_parts  : which components are used if mixed (otherwise None);
        this compares to `tsfc.Kernel.coefficient_parts` (, which will be introduced soon).
    """
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
