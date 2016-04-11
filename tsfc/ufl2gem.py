"""Translation of UFL tensor-algebra into GEM tensor-algebra."""

from __future__ import absolute_import

import collections
import numpy
import ufl

from gem import (Literal, Zero, Identity, Sum, Product, Division,
                 Power, MathFunction, MinValue, MaxValue, Comparison,
                 LogicalNot, LogicalAnd, LogicalOr, Conditional,
                 Index, Indexed, ComponentTensor, IndexSum,
                 ListTensor)


class Mixin(object):
    """A mixin to be used with a UFL MultiFunction to translate UFL
    algebra into GEM tensor-algebra.  This node types translate pretty
    straightforwardly to GEM.  Other node types are not handled in
    this mixin."""

    def __init__(self):
        self.index_map = collections.defaultdict(Index)
        """A map for translating UFL free indices into GEM free
        indices."""

    def scalar_value(self, o):
        return Literal(o.value())

    def identity(self, o):
        return Identity(o._dim)

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
        assert condition.shape == ()
        if o.ufl_shape:
            indices = tuple(Index() for i in range(len(o.ufl_shape)))
            return ComponentTensor(Conditional(condition, Indexed(then, indices),
                                               Indexed(else_, indices)),
                                   indices)
        else:
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
        # UFL and GEM have a few semantical differences when it comes
        # to ListTensor.  In UFL, a ListTensor increases the rank by
        # one with respect to its constituents.  So to build a matrix
        # from scalars, one must build a ListTensor of ListTensors in
        # UFL, while a GEM ListTensor can directly construct rank two
        # tensor from scalars because it has an explicitly specified
        # shape.
        nesting = [isinstance(op, ListTensor) for op in ops]
        if all(nesting):
            # ListTensor of ListTensors in UFL, build a ListTensor of
            # higher rank in GEM.
            return ListTensor(numpy.array([op.array for op in ops]))
        elif len(o.ufl_shape) > 1:
            # On the other hand, TSFC only builds ListTensors from
            # scalars, while the constituents of a UFL ListTensor can
            # be non-scalar despite not being ListTensors themselves
            # (e.g. ComponentTensor, Zero).
            #
            # In this case we currently break up the tensor-valued
            # constituents into scalars by generating fixed indices to
            # access to every single element.
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
        # ufl.IndexSum technically has a MultiIndex, but it must have
        # exactly one index in it.
        index, = indices

        if o.ufl_shape:
            indices = tuple(Index() for i in range(len(o.ufl_shape)))
            return ComponentTensor(IndexSum(Indexed(summand, indices), index), indices)
        else:
            return IndexSum(summand, index)

    def variable(self, o, expression, label):
        # Only used by UFL AD, at this point, the bare expression is
        # what we want.
        return expression

    def label(self, o):
        # Only used by UFL AD, don't need it at this point.
        pass
