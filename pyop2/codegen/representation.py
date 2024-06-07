import numbers
import itertools
from functools import partial
from collections import defaultdict
from pyop2.utils import cached_property
import numpy
from abc import ABCMeta
from pyop2.codegen.node import Node as NodeBase


class InstructionLabel(object):
    def __init__(self, within_inames=()):
        self.within_inames = tuple(w for w in within_inames if isinstance(w, Node))


class PackInst(InstructionLabel):
    pass


class UnpackInst(InstructionLabel):
    pass


class PreUnpackInst(InstructionLabel):
    pass


class KernelInst(InstructionLabel):
    pass


class Node(NodeBase):

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
    __slots__ = ()
    children = ()
    is_equal = NodeBase.is_equal


class Scalar(Node):
    __slots__ = ()

    shape = ()


class Constant(Terminal):
    __slots__ = ()


class DTypeMixin(object):

    @cached_property
    def dtype(self):
        dtype, = set(c.dtype for c in self.children)
        return dtype


class Zero(Constant):
    __slots__ = ("shape", "dtype")
    __front__ = ("shape", "dtype")

    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype


class IndexBase(metaclass=ABCMeta):
    pass


class Index(Terminal, Scalar):
    _count = itertools.count()
    __slots__ = ("extent", "merge", "name")
    __front__ = ("extent", "merge", "name")

    def __init__(self, extent=None, merge=True, name=None):
        self.name = name or "i%d" % next(Index._count)
        self.extent = None
        self.set_extent(extent)
        self.merge = merge

    def set_extent(self, value):
        if self.extent is None:
            if isinstance(value, numbers.Integral):
                value = int(value)
            self.extent = value
        elif self.extent != value:
            raise ValueError("Inconsistent index extents")

    dtype = numpy.int32


class FixedIndex(Terminal, Scalar):
    __slots__ = ("value", )
    __front__ = ("value", )

    extent = 1

    def __init__(self, value):
        assert isinstance(value, numbers.Integral)
        self.value = numpy.int32(value)

    dtype = numpy.int32


class RuntimeIndex(Scalar):
    _count = itertools.count()
    __slots__ = ("children", "name")
    __back__ = ("name", )

    def __init__(self, lo, hi, constraint, name):
        assert name is not None, "runtime indices need a name"
        self.name = name
        self.children = lo, hi, constraint

    @cached_property
    def extents(self):
        return self.children[:2]

    @cached_property
    def dtype(self):
        a, b, c = self.children
        assert a.dtype == b.dtype
        return a.dtype


IndexBase.register(FixedIndex)
IndexBase.register(Index)
IndexBase.register(RuntimeIndex)


class MultiIndex(Node):
    __slots__ = ("children", )

    def __init__(self, *indices):
        self.children = indices

    def __iter__(self):
        return iter(self.children)

    def __len__(self):
        return len(self.children)


class Extent(Scalar):
    __slots__ = ("children", )

    def __init__(self, multiindex):
        assert all(isinstance(i, (Index, FixedIndex)) for i in multiindex.children)
        self.children = multiindex,


class Symbol(Terminal):
    __slots__ = ("name", )
    __front__ = ("name", )

    def __init__(self, name):
        self.name = name


class Argument(Terminal):
    _count = defaultdict(partial(itertools.count))

    __slots__ = ("shape", "dtype", "_name", "prefix", "_gen_name")
    __front__ = ("shape", "dtype", "_name", "prefix")

    def __init__(self, shape, dtype, name=None, pfx=None):
        self.dtype = dtype
        self.shape = shape
        self._name = name
        pfx = pfx or "v"
        self.prefix = pfx
        self._gen_name = name or "%s%d" % (pfx, next(Argument._count[pfx]))

    def get_hash(self):
        return hash((type(self),) + self._cons_args(self.children) + (self.name,))

    @property
    def name(self):
        return self._name or self._gen_name


class Literal(Terminal, Scalar):
    __slots__ = ("value", )
    __front__ = ("value", )
    shape = ()

    def __new__(cls, value, casting=True):
        assert value.shape == ()
        assert isinstance(value, numpy.number)
        if value == 0:
            # All zeros, make symbolic zero
            return Zero((), value.dtype)
        else:
            return super().__new__(cls)

    def __init__(self, value, casting=True):
        self.value = value
        self.casting = casting

    def is_equal(self, other):
        if type(self) != type(other):
            return False
        return self.value == other.value

    def get_hash(self):
        return hash((type(self), self.value))

    @cached_property
    def dtype(self):
        return self.value.dtype


class NamedLiteral(Terminal):
    __slots__ = ("value", "parent", "suffix")
    __front__ = ("value", "parent", "suffix")

    def __init__(self, value, parent, suffix):
        self.value = value
        self.parent = parent
        self.suffix = suffix

    def is_equal(self, other):
        if type(self) != type(other):
            return False
        if self.shape != other.shape:
            return False
        if self.parent != other.parent:
            return False
        if self.suffix != other.suffix:
            return False
        return tuple(self.value.flat) == tuple(other.value.flat)

    def get_hash(self):
        return hash((type(self), self.shape, tuple(self.value.flat)))

    @cached_property
    def shape(self):
        return self.value.shape

    @cached_property
    def dtype(self):
        return self.value.dtype

    @property
    def name(self):
        return f"{self.parent.name}_{self.suffix}"


class Min(Scalar):
    __slots__ = ("children", )

    def __init__(self, a, b):
        assert not a.shape
        assert not b.shape
        self.children = a, b

    @cached_property
    def dtype(self):
        a, b = self.children
        return a.dtype


class Max(Scalar):
    __slots__ = ("children", )

    def __init__(self, a, b):
        assert not a.shape
        assert not b.shape
        self.children = a, b

    @cached_property
    def dtype(self):
        a, b = self.children
        return numpy.result_type(a.dtype, b.dtype)


class Sum(Scalar):
    __slots__ = ("children", )

    def __init__(self, a, b):
        assert not a.shape
        assert not b.shape
        self.children = a, b

    @cached_property
    def dtype(self):
        a, b = self.children
        return numpy.result_type(a.dtype, b.dtype)


class Product(Scalar):
    __slots__ = ("children", )

    def __init__(self, a, b):
        assert not a.shape
        assert not b.shape
        self.children = a, b

    @cached_property
    def dtype(self):
        a, b = self.children
        return numpy.result_type(a.dtype, b.dtype)


class QuotientBase(Scalar):
    __slots__ = ("children", )

    def __init__(self, a, b):
        assert not a.shape
        assert not b.shape
        self.children = a, b

    @cached_property
    def dtype(self):
        a, b = self.children
        return numpy.result_type(a.dtype, b.dtype)


class Quotient(QuotientBase):
    pass


class FloorDiv(QuotientBase):
    pass


class Remainder(QuotientBase):
    pass


class Indexed(Scalar):
    __slots__ = ("children", )

    def __new__(cls, aggregate, multiindex):
        multiindex = MultiIndex(*(int(i) if isinstance(i, numbers.Integral) else i
                                  for i in multiindex))
        assert len(aggregate.shape) == len(multiindex)
        for index, extent in zip(multiindex, aggregate.shape):
            if isinstance(index, Index):
                index.set_extent(extent)

        self = super().__new__(cls)
        self.children = (aggregate, multiindex)
        return self

    def index_ordering(self):
        _, multiindex = self.children
        return tuple(i for i in self.multiindex if isinstance(i, Index))

    @cached_property
    def dtype(self):
        return self.aggregate.dtype

    @cached_property
    def aggregate(self):
        return self.children[0]

    @cached_property
    def multiindex(self):
        return self.children[1]


class When(Node):
    __slots__ = ("children", )

    def __init__(self, condition, expr):
        self.children = condition, expr

    @cached_property
    def dtype(self):
        return self.children[1].dtype


class Materialise(Node):
    _count = itertools.count()
    __slots__ = ("children", "name", "label")
    __front__ = ("label",)

    def __init__(self, label, init, indices, *expressions_and_indices):
        assert all(isinstance(i, (Index, FixedIndex)) for i in indices)
        assert len(expressions_and_indices) % 2 == 0
        assert isinstance(label, InstructionLabel)
        self.label = label
        self.children = (init, indices) + tuple(expressions_and_indices)
        self.name = "t%d" % next(Materialise._count)

    def reconstruct(self, *args):
        new = type(self)(*self._cons_args(args))
        new.name = self.name
        return new

    @cached_property
    def shape(self):
        indices = self.children[1]
        return tuple(i.extent for i in indices)

    @cached_property
    def dtype(self):
        expr = self.children[0]
        return expr.dtype


class Variable(Terminal):
    __slots__ = ("name", "shape", "dtype")
    __front__ = ("name", "shape", "dtype")

    def __init__(self, name, shape, dtype):
        self.name = name
        self.shape = shape
        self.dtype = dtype


class DummyInstruction(Node):
    __slots__ = ("children", "label")
    __front__ = ("label",)

    def __init__(self, label, *children):
        self.children = children
        self.label = label


class Accumulate(Node):
    __slots__ = ("children", "label")
    __front__ = ("label",)

    def __init__(self, label, lvalue, rvalue):
        self.children = (lvalue, rvalue)
        self.label = label


class FunctionCall(Node):
    __slots__ = ("name", "access", "free_indices", "label", "children")
    __front__ = ("name", "label", "access", "free_indices")

    def __init__(self, name, label, access, free_indices, *arguments):
        self.children = tuple(arguments)
        self.access = tuple(access)
        self.free_indices = free_indices
        self.name = name
        self.label = label
        assert len(self.access) == len(self.children)


class Conditional(Scalar):
    __slots__ = ("children", )

    def __init__(self, condition, then, else_):
        assert not condition.shape
        assert not then.shape
        assert then.shape == else_.shape
        assert then.dtype == else_.dtype
        self.children = condition, then, else_
        self.shape = then.shape

    @cached_property
    def dtype(self):
        return self.children[1].dtype


class Comparison(Scalar):
    __slots__ = ("operator", "children")
    __front__ = ("operator", )

    def __init__(self, op, a, b):
        assert not a.shape
        assert not b.shape
        if op not in {">", ">=", "==", "!=", "<", "<="}:
            raise ValueError("invalid operator")

        self.operator = op
        self.children = a, b


class LogicalNot(Scalar, DTypeMixin):
    __slots__ = ("children", )

    def __init__(self, expression):
        assert not expression.shape
        self.children = expression,


class LogicalAnd(Scalar, DTypeMixin):
    __slots__ = ("children", )

    def __init__(self, a, b):
        assert not a.shape
        assert not b.shape
        self.children = a, b


class LogicalOr(Scalar, DTypeMixin):
    __slots__ = ("children", )

    def __init__(self, a, b):
        assert not a.shape
        assert not b.shape
        self.children = a, b


class BitwiseNot(Scalar, DTypeMixin):
    __slots__ = ("children", )

    def __init__(self, expression):
        assert not expression.shape
        self.children = expression,


class BitwiseAnd(Scalar, DTypeMixin):
    __slots__ = ("children", )

    def __init__(self, a, b):
        assert not a.shape
        assert not b.shape
        self.children = a, b


class BitwiseOr(Scalar, DTypeMixin):
    __slots__ = ("children", )

    def __init__(self, a, b):
        assert not a.shape
        assert not b.shape
        self.children = a, b


class BitShift(Scalar, DTypeMixin):
    __slots__ = ("direction", "children", )
    __front__ = ("direction", )

    def __init__(self, direction, expr, shift):
        assert direction in {"<<", ">>"}
        self.direction = direction
        self.children = expr, shift
