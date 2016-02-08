"""GEM is the intermediate language of TSFC for describing
tensor-valued mathematical expressions and tensor operations.
It is similar to Einstein's notation.

Its design was heavily inspired by UFL, with some major differences:
 - GEM has got nothing FEM-specific.
 - In UFL free indices are just unrolled shape, thus UFL is very
   restrictive about operations on expressions with different sets of
   free indices. GEM is much more relaxed about free indices.

Similarly to UFL, all GEM nodes have 'shape' and 'free_indices'
attributes / properties. Unlike UFL, however, index extents live on
the Index objects in GEM, not on all the nodes that have those free
indices.
"""

from __future__ import absolute_import

from itertools import chain
from numpy import asarray, unique

from tsfc.node import Node as NodeBase


class NodeMeta(type):
    """Metaclass of GEM nodes.

    When a GEM node is constructed, this metaclass automatically
    collects its free indices if 'free_indices' has not been set yet.
    """

    def __call__(self, *args, **kwargs):
        # Create and initialise object
        obj = super(NodeMeta, self).__call__(*args, **kwargs)

        # Set free_indices if not set already
        if not hasattr(obj, 'free_indices'):
            cfi = list(chain(*[c.free_indices for c in obj.children]))
            obj.free_indices = tuple(unique(cfi))

        return obj


class Node(NodeBase):
    """Abstract GEM node class."""

    __metaclass__ = NodeMeta

    __slots__ = ('free_indices')

    def is_equal(self, other):
        """Common subexpression eliminating equality predicate.

        When two (sub)expressions are equal, the children of one
        object are reassigned to the children of the other, so some
        duplicated subexpressions are eliminated.
        """
        result = NodeBase.is_equal(self, other)
        if result:
            self.children = other.children
        return result


class Terminal(Node):
    """Abstract class for terminal GEM nodes."""

    __slots__ = ()

    children = ()

    is_equal = NodeBase.is_equal


class Scalar(Node):
    """Abstract class for scalar-valued GEM nodes."""

    __slots__ = ()

    shape = ()


class Zero(Terminal):
    """Symbolic zero tensor"""

    __slots__ = ('shape',)
    __front__ = ('shape',)

    def __init__(self, shape=()):
        self.shape = shape

    @property
    def value(self):
        assert not self.shape
        return 0.0


class Literal(Terminal):
    """Tensor-valued constant"""

    __slots__ = ('array',)
    __front__ = ('array',)

    def __new__(cls, array):
        array = asarray(array)
        if (array == 0).all():
            # All zeros, make symbolic zero
            return Zero(array.shape)
        else:
            return super(Literal, cls).__new__(cls)

    def __init__(self, array):
        self.array = asarray(array, dtype=float)

    def is_equal(self, other):
        if type(self) != type(other):
            return False
        if self.shape != other.shape:
            return False
        return tuple(self.array.flat) == tuple(other.array.flat)

    def get_hash(self):
        return hash((type(self), self.shape, tuple(self.array.flat)))

    @property
    def value(self):
        return float(self.array)

    @property
    def shape(self):
        return self.array.shape


class Variable(Terminal):
    """Symbolic variable tensor"""

    __slots__ = ('name', 'shape')
    __front__ = ('name', 'shape')

    def __init__(self, name, shape):
        self.name = name
        self.shape = shape


class Sum(Scalar):
    __slots__ = ('children',)

    def __new__(cls, a, b):
        assert not a.shape
        assert not b.shape

        # Zero folding
        if isinstance(a, Zero):
            return b
        elif isinstance(b, Zero):
            return a

        self = super(Sum, cls).__new__(cls)
        self.children = a, b
        return self


class Product(Scalar):
    __slots__ = ('children',)

    def __new__(cls, a, b):
        assert not a.shape
        assert not b.shape

        # Zero folding
        if isinstance(a, Zero) or isinstance(b, Zero):
            return Zero()

        self = super(Product, cls).__new__(cls)
        self.children = a, b
        return self


class Division(Scalar):
    __slots__ = ('children',)

    def __new__(cls, a, b):
        assert not a.shape
        assert not b.shape

        # Zero folding
        if isinstance(b, Zero):
            raise ValueError("division by zero")
        if isinstance(a, Zero):
            return Zero()

        self = super(Division, cls).__new__(cls)
        self.children = a, b
        return self


class Power(Scalar):
    __slots__ = ('children',)

    def __new__(cls, base, exponent):
        assert not base.shape
        assert not exponent.shape

        # Zero folding
        if isinstance(base, Zero):
            if isinstance(exponent, Zero):
                raise ValueError("cannot solve 0^0")
            return Zero()
        elif isinstance(exponent, Zero):
            return Literal(1)

        self = super(Power, cls).__new__(cls)
        self.children = base, exponent
        return self


class MathFunction(Scalar):
    __slots__ = ('name', 'children')
    __front__ = ('name',)

    def __init__(self, name, argument):
        assert isinstance(name, str)
        assert not argument.shape

        self.name = name
        self.children = argument,


class MinValue(Scalar):
    __slots__ = ('children',)

    def __init__(self, a, b):
        assert not a.shape
        assert not b.shape

        self.children = a, b


class MaxValue(Scalar):
    __slots__ = ('children',)

    def __init__(self, a, b):
        assert not a.shape
        assert not b.shape

        self.children = a, b


class Comparison(Scalar):
    __slots__ = ('operator', 'children')
    __front__ = ('operator',)

    def __init__(self, op, a, b):
        assert not a.shape
        assert not b.shape

        if op not in [">", ">=", "==", "!=", "<", "<="]:
            raise ValueError("invalid operator")

        self.operator = op
        self.children = a, b


class LogicalNot(Scalar):
    __slots__ = ('children',)

    def __init__(self, expression):
        assert not expression.shape

        self.children = expression,


class LogicalAnd(Scalar):
    __slots__ = ('children',)

    def __init__(self, a, b):
        assert not a.shape
        assert not b.shape

        self.children = a, b


class LogicalOr(Scalar):
    __slots__ = ('children',)

    def __init__(self, a, b):
        assert not a.shape
        assert not b.shape

        self.children = a, b


class Conditional(Node):
    __slots__ = ('children', 'shape')

    def __init__(self, condition, then, else_):
        assert not condition.shape
        assert then.shape == else_.shape

        self.children = condition, then, else_
        self.shape = then.shape


class Index(object):
    """Free index"""

    # Not true object count, just for naming purposes
    count = 0

    __slots__ = ('name', 'extent')

    def __init__(self, name=None):
        if name is None:
            Index.count += 1
            name = "i_%d" % Index.count
        self.name = name

        # Initialise with indefinite extent
        self.extent = None

    def set_extent(self, value):
        # Set extent, check for consistency
        if self.extent is None:
            self.extent = value
        elif self.extent != value:
            raise ValueError("Inconsistent index extents!")

    def __str__(self):
        return self.name

    def __repr__(self):
        return "Index(%r)" % self.name


class VariableIndex(object):
    def __init__(self, name):
        self.name = name


class Indexed(Scalar):
    __slots__ = ('children', 'multiindex')
    __back__ = ('multiindex',)

    def __new__(cls, aggregate, multiindex):
        # Set index extents from shape
        assert len(aggregate.shape) == len(multiindex)
        for index, extent in zip(multiindex, aggregate.shape):
            if isinstance(index, Index):
                index.set_extent(extent)

        # Zero folding
        if isinstance(aggregate, Zero):
            return Zero()

        # All indices fixed
        if all(isinstance(i, int) for i in multiindex):
            if isinstance(aggregate, Literal):
                return Literal(aggregate.array[multiindex])
            elif isinstance(aggregate, ListTensor):
                return aggregate.array[multiindex]

        self = super(Indexed, cls).__new__(cls)
        self.children = (aggregate,)
        self.multiindex = multiindex

        new_indices = tuple(i for i in multiindex if isinstance(i, Index))
        self.free_indices = tuple(unique(aggregate.free_indices + new_indices))

        return self


class ComponentTensor(Node):
    __slots__ = ('children', 'multiindex', 'shape')
    __back__ = ('multiindex',)

    def __new__(cls, expression, multiindex):
        assert not expression.shape

        # Collect shape
        shape = tuple(index.extent for index in multiindex)
        assert all(shape)

        # Zero folding
        if isinstance(expression, Zero):
            return Zero(shape)

        self = super(ComponentTensor, cls).__new__(cls)
        self.children = (expression,)
        self.multiindex = multiindex
        self.shape = shape

        # Collect free indices
        assert set(multiindex) <= set(expression.free_indices)
        self.free_indices = tuple(set(expression.free_indices) - set(multiindex))

        return self


class IndexSum(Scalar):
    __slots__ = ('children', 'index')
    __back__ = ('index',)

    def __new__(cls, summand, index):
        # Sum zeros
        assert not summand.shape
        if isinstance(summand, Zero):
            return summand

        # Sum a single expression
        if index.extent == 1:
            return Indexed(ComponentTensor(summand, (index,)), (0,))

        self = super(IndexSum, cls).__new__(cls)
        self.children = (summand,)
        self.index = index

        # Collect shape and free indices
        assert index in summand.free_indices
        self.free_indices = tuple(set(summand.free_indices) - {index})

        return self


class ListTensor(Node):
    __slots__ = ('array',)

    def __new__(cls, array):
        array = asarray(array)

        # Zero folding
        if all(isinstance(elem, Zero) for elem in array.flat):
            assert all(elem.shape == () for elem in array.flat)
            return Zero(array.shape)

        self = super(ListTensor, cls).__new__(cls)
        self.array = array
        return self

    @property
    def children(self):
        return tuple(self.array.flat)

    @property
    def shape(self):
        return self.array.shape

    def reconstruct(self, *args):
        return ListTensor(asarray(args).reshape(self.array.shape))

    def __repr__(self):
        return "ListTensor(%r)" % self.array.tolist()

    def is_equal(self, other):
        """Common subexpression eliminating equality predicate."""
        if type(self) != type(other):
            return False
        if (self.array == other.array).all():
            self.array = other.array
            return True
        return False

    def get_hash(self):
        return hash((type(self), self.shape, self.children))
