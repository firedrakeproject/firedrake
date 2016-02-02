from __future__ import absolute_import

import collections
import numpy

from singledispatch import singledispatch

import ufl

from tsfc.node import Node as node_Node, traversal


class NodeMeta(type):
    def __call__(self, *args, **kwargs):
        # Create and initialise object
        obj = super(NodeMeta, self).__call__(*args, **kwargs)

        # Set free_indices if not set already
        if not hasattr(obj, 'free_indices'):
            free_indices = set()
            for child in obj.children:
                free_indices |= set(child.free_indices)
            obj.free_indices = tuple(free_indices)

        return obj


class Node(node_Node):
    __metaclass__ = NodeMeta

    __slots__ = ('free_indices')


class Scalar(Node):
    __slots__ = ()

    shape = ()


class Zero(Node):
    __slots__ = ('shape',)
    __front__ = ('shape',)

    def __init__(self, shape=()):
        self.shape = shape

    children = ()

    @property
    def value(self):
        assert not self.shape
        return 0.0


class Literal(Node):
    __slots__ = ('array',)
    __front__ = ('array',)

    def __new__(cls, array):
        array = numpy.asarray(array)
        if (array == 0).all():
            return Zero(array.shape)
        else:
            return super(Literal, cls).__new__(cls)

    def __init__(self, array):
        self.array = numpy.asarray(array, dtype=float)

    children = ()

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


class Variable(Node):
    __slots__ = ('name', 'shape')
    __front__ = ('name', 'shape')

    def __init__(self, name, shape):
        self.name = name
        self.shape = shape

    children = ()


class Sum(Scalar):
    __slots__ = ('children',)

    def __new__(cls, a, b):
        assert not a.shape
        assert not b.shape

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
    __slots__ = ('extent')

    def __init__(self):
        self.extent = None

    def set_extent(self, value):
        if self.extent is None:
            self.extent = value
        elif self.extent != value:
            raise ValueError("Inconsistent index extents!")


class VariableIndex(object):
    def __init__(self, name):
        self.name = name


class Indexed(Scalar):
    __slots__ = ('children', 'multiindex')
    __back__ = ('multiindex',)

    def __new__(cls, aggregate, multiindex):
        assert len(aggregate.shape) == len(multiindex)
        for index, extent in zip(multiindex, aggregate.shape):
            if isinstance(index, Index):
                index.set_extent(extent)

        if isinstance(aggregate, Zero):
            return Zero()
        else:
            return super(Indexed, cls).__new__(cls)

    def __init__(self, aggregate, multiindex):
        self.children = (aggregate,)
        self.multiindex = multiindex

        new_indices = set(i for i in multiindex if isinstance(i, Index))
        self.free_indices = tuple(set(aggregate.free_indices) | new_indices)


class ComponentTensor(Node):
    __slots__ = ('children', 'multiindex', 'shape')
    __back__ = ('multiindex',)

    def __init__(self, expression, multiindex):
        assert not expression.shape
        # assert set(multiindex) <= set(expression.free_indices)
        assert all(index.extent for index in multiindex)

        self.children = (expression,)
        self.multiindex = multiindex

        self.free_indices = tuple(set(expression.free_indices) - set(multiindex))
        self.shape = tuple(index.extent for index in multiindex)


class IndexSum(Scalar):
    __slots__ = ('children', 'index')
    __back__ = ('index',)

    def __new__(cls, summand, index):
        assert not summand.shape
        if isinstance(summand, Zero):
            return summand

        self = super(IndexSum, cls).__new__(cls)
        self.children = (summand,)
        self.index = index

        assert index in summand.free_indices
        self.free_indices = tuple(set(summand.free_indices) - {index})

        return self


class ListTensor(Node):
    __slots__ = ('array',)

    def __new__(cls, array):
        array = numpy.asarray(array)

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
        return ListTensor(numpy.asarray(args).reshape(self.array.shape))

    def __repr__(self):
        return "ListTensor(%r)" % self.array.tolist()

    def is_equal(self, other):
        if type(self) != type(other):
            return False
        if self.shape != other.shape:
            return False
        return self.children == other.children

    def get_hash(self):
        return hash((type(self), self.shape, self.children))


class FromUFLMixin(object):
    def __init__(self):
        self.index_map = collections.defaultdict(Index)

    def scalar_value(self, o):
        return Literal(o.value())

    def identity(self, o):
        return Literal(numpy.eye(*o.ufl_shape))

    def zero(self, o):
        return Zero(o.ufl_shape)

    def sum(self, o, *ops):
        if o.ufl_shape:
            indices = tuple(Index() for i in range(len(o.ufl_shape)))
            return ComponentTensor(Sum(*[Indexed(op, indices) for op in ops]), indices)
        else:
            return Sum(*ops)

    def product(self, o, *ops):
        assert o.ufl_shape == ()
        return Product(*ops)

    def division(self, o, numerator, denominator):
        return Division(numerator, denominator)

    def abs(self, o, expr):
        if o.ufl_shape:
            indices = tuple(Index() for i in range(len(o.ufl_shape)))
            return ComponentTensor(MathFunction('abs', Indexed(expr, indices)), indices)
        else:
            return MathFunction('abs', expr)

    def power(self, o, base, exponent):
        return Power(base, exponent)

    def math_function(self, o, expr):
        return MathFunction(o._name, expr)

    def min_value(self, o, *ops):
        return MinValue(*ops)

    def max_value(self, o, *ops):
        return MaxValue(*ops)

    def binary_condition(self, o, left, right):
        return Comparison(o._name, left, right)

    def not_condition(self, o, expr):
        return LogicalNot(expr)

    def and_condition(self, o, *ops):
        return LogicalAnd(*ops)

    def or_condition(self, o, *ops):
        return LogicalOr(*ops)

    def conditional(self, o, condition, then, else_):
        assert o.ufl_shape == ()  # TODO
        return Conditional(condition, then, else_)

    def multi_index(self, o):
        indices = []
        for i in o:
            if isinstance(i, ufl.classes.FixedIndex):
                indices.append(int(i))
            elif isinstance(i, ufl.classes.Index):
                indices.append(self.index_map[i.count()])
        return tuple(indices)

    def indexed(self, o, aggregate, index):
        return Indexed(aggregate, index)

    def list_tensor(self, o, *ops):
        nesting = [isinstance(op, ListTensor) for op in ops]
        if all(nesting):
            return ListTensor(numpy.array([op.array for op in ops]))
        elif len(o.ufl_shape) > 1:
            children = []
            for op in ops:
                child = numpy.zeros(o.ufl_shape[1:], dtype=object)
                for multiindex in numpy.ndindex(child.shape):
                    child[multiindex] = Indexed(op, multiindex)
                children.append(child)
            return ListTensor(numpy.array(children))
        else:
            return ListTensor(numpy.array(ops))

    def component_tensor(self, o, expression, index):
        return ComponentTensor(expression, index)

    def index_sum(self, o, summand, indices):
        index, = indices

        if o.ufl_shape:
            indices = tuple(Index() for i in range(len(o.ufl_shape)))
            return ComponentTensor(IndexSum(Indexed(summand, indices), index), indices)
        else:
            return IndexSum(summand, index)

    def variable(self, o, expression, label):
        """Only used by UFL AD, at this point, the bare expression is what we want."""
        return expression

    def label(self, o):
        """Only used by UFL AD, don't need it at this point."""
        pass


def inline_indices(expression, result_cache):
    def cached_handle(node, subst):
        cache_key = (node, tuple(sorted(subst.items())))
        try:
            return result_cache[cache_key]
        except KeyError:
            result = handle(node, subst)
            result_cache[cache_key] = result
            return result

    @singledispatch
    def handle(node, subst):
        raise AssertionError("Cannot handle foreign type: %s" % type(node))

    @handle.register(Node)  # noqa: Not actually redefinition
    def _(node, subst):
        new_children = [cached_handle(child, subst) for child in node.children]
        if all(nc == c for nc, c in zip(new_children, node.children)):
            return node
        else:
            return node.reconstruct(*new_children)

    @handle.register(Indexed)  # noqa: Not actually redefinition
    def _(node, subst):
        child, = node.children
        multiindex = tuple(subst.get(i, i) for i in node.multiindex)
        if isinstance(child, ComponentTensor):
            new_subst = dict(zip(child.multiindex, multiindex))
            composed_subst = {k: new_subst.get(v, v) for k, v in subst.items()}
            composed_subst.update(new_subst)
            filtered_subst = {k: v for k, v in composed_subst.items() if k in child.children[0].free_indices}
            return cached_handle(child.children[0], filtered_subst)
        elif isinstance(child, ListTensor) and all(isinstance(i, int) for i in multiindex):
            return cached_handle(child.array[multiindex], subst)
        else:
            new_child = cached_handle(child, subst)
            if new_child == child and multiindex == node.multiindex:
                return node
            else:
                return Indexed(new_child, multiindex)

    return cached_handle(expression, {})


def collect_index_extents(expression):
    result = collections.OrderedDict()

    for node in traversal([expression]):
        if isinstance(node, Indexed):
            assert len(node.multiindex) == len(node.children[0].shape)
            for index, extent in zip(node.multiindex, node.children[0].shape):
                if isinstance(index, Index):
                    if index not in result:
                        result[index] = extent
                    elif result[index] != extent:
                        raise AssertionError("Inconsistent index extents!")

    return result
