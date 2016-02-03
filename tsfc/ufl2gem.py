from __future__ import absolute_import

import collections
import numpy
import ufl

from tsfc.gem import (Literal, Zero, Sum, Product, Division, Power,
                      MathFunction, MinValue, MaxValue, Comparison,
                      LogicalNot, LogicalAnd, LogicalOr, Conditional,
                      Index, Indexed, ComponentTensor, IndexSum,
                      ListTensor)


class Mixin(object):
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
