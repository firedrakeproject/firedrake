from __future__ import absolute_import

import collections
import numpy

from singledispatch import singledispatch

import ufl
from ufl.corealg.multifunction import MultiFunction

from firedrake.fc.node import Node as node_Node, traversal


class Node(node_Node):
    __slots__ = ()


def as_node(node):
    if isinstance(node, Node):
        return node
    elif isinstance(node, (int, float, numpy.float64)):
        return Literal(node)
    else:
        raise ValueError("do not know how to make node from " + repr(node))


class Literal(Node):
    __slots__ = ('value',)
    __front__ = ('value',)

    def __init__(self, value):
        self.value = value

    children = ()


class Variable(Node):
    __slots__ = ('name', 'shape')
    __front__ = ('name', 'shape')

    def __init__(self, name, shape):
        self.name = name
        self.shape = shape

    children = ()


class Sum(Node):
    __slots__ = ('children',)

    def __init__(self, a, b):
        self.children = a, b


class Product(Node):
    __slots__ = ('children',)

    def __init__(self, a, b):
        self.children = a, b


class Division(Node):
    __slots__ = ('children',)

    def __init__(self, a, b):
        self.children = a, b


class Power(Node):
    __slots__ = ('children',)

    def __init__(self, base, exponent):
        self.children = base, exponent


class MathFunction(Node):
    __slots__ = ('name', 'children')
    __front__ = ('name',)

    def __init__(self, name, argument):
        self.name = name
        self.children = argument,


class MinValue(Node):
    __slots__ = ('children',)

    def __init__(self, a, b):
        self.children = a, b


class MaxValue(Node):
    __slots__ = ('children',)

    def __init__(self, a, b):
        self.children = a, b


class Comparison(Node):
    __slots__ = ('operator', 'children')
    __front__ = ('operator',)

    def __init__(self, a, op, b):
        if op not in [">", ">=", "==", "!=", "<", "<="]:
            raise ValueError("invalid operator")

        self.operator = op
        self.children = a, b


class LogicalNot(Node):
    __slots__ = ('children',)

    def __init__(self, expression):
        self.children = expression,


class LogicalAnd(Node):
    __slots__ = ('children',)

    def __init__(self, a, b):
        self.children = a, b


class LogicalOr(Node):
    __slots__ = ('children',)

    def __init__(self, a, b):
        self.children = a, b


class Conditional(Node):
    __slots__ = ('children',)

    def __init__(self, condition, then, else_):
        self.children = condition, then, else_


class Index(object):
    pass


class VariableIndex(object):
    def __init__(self, name):
        self.name = name


class Indexed(Node):
    __slots__ = ('children', 'multiindex')
    __back__ = ('multiindex',)

    def __init__(self, aggregate, multiindex):
        self.children = (aggregate,)
        self.multiindex = multiindex


class ComponentTensor(Node):
    __slots__ = ('children', 'multiindex')
    __back__ = ('multiindex',)

    def __init__(self, expression, multiindex):
        self.children = (expression,)
        self.multiindex = multiindex


class IndexSum(Node):
    __slots__ = ('children', 'index')
    __back__ = ('index',)

    def __init__(self, summand, index):
        self.children = (summand,)
        self.index = index


class ListTensor(Node):
    __slots__ = ('array',)

    def __init__(self, array):
        self.array = numpy.asarray(array, dtype=object)
        for multiindex, val in numpy.ndenumerate(self.array):
            self.array[multiindex] = as_node(val)

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
        return id(self.array) == id(other.array)

    def get_hash(self):
        return hash((type(self), id(self.array)))


class FromUFLMixin(object):
    def __init__(self):
        self.index_map = collections.defaultdict(Index)

    def scalar_value(self, o):
        return Literal(o.value())

    identity = MultiFunction.undefined  # TODO

    def zero(self, o):
        if o.ufl_shape:
            # TODO: tensor-valued literal? special shaped zero?
            return ListTensor(numpy.zeros(o.ufl_shape))
        else:
            return Literal(0)

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
        assert o.ufl_shape == ()
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
        return Comparison(left, o._name, right)

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


def inline_indices(expression):
    result_cache = {}

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

    @handle.register(Node)
    def _(node, subst):
        new_children = [cached_handle(child, subst) for child in node.children]
        if all(nc == c for nc, c in zip(new_children, node.children)):
            return node
        else:
            return node.reconstruct(*new_children)

    @handle.register(Indexed)
    def _(node, subst):
        child, = node.children
        multiindex = tuple(subst.get(i, i) for i in node.multiindex)
        if isinstance(child, ComponentTensor):
            new_subst = dict(zip(child.multiindex, multiindex))
            composed_subst = {k: new_subst.get(v, v) for k, v in subst.items()}
            composed_subst.update(new_subst)
            return cached_handle(child.children[0], composed_subst)
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
    result = {}

    for node in traversal(expression):
        if isinstance(node, Indexed):
            for index, extent in zip(node.multiindex, node.children[0].shape):
                if isinstance(index, Index):
                    if index not in result:
                        result[index] = extent
                    elif result[index] != extent:
                        raise AssertionError("Inconsistent index extents!")

    return result
