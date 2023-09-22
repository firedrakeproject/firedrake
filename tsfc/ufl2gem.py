"""Translation of UFL tensor-algebra into GEM tensor-algebra."""

import collections
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

    def real(self, o, expr):
        if o.ufl_shape:
            indices = tuple(Index() for i in range(len(o.ufl_shape)))
            return ComponentTensor(MathFunction('real', Indexed(expr, indices)), indices)
        else:
            return MathFunction('real', expr)

    def imag(self, o, expr):
        if o.ufl_shape:
            indices = tuple(Index() for i in range(len(o.ufl_shape)))
            return ComponentTensor(MathFunction('imag', Indexed(expr, indices)), indices)
        else:
            return MathFunction('imag', expr)

    def conj(self, o, expr):
        if o.ufl_shape:
            indices = tuple(Index() for i in range(len(o.ufl_shape)))
            return ComponentTensor(MathFunction('conj', Indexed(expr, indices)), indices)
        else:
            return MathFunction('conj', expr)

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

    def atan2(self, o, y, x):
        return MathFunction("atan2", y, x)

    def bessel_i(self, o, nu, arg):
        return MathFunction(o._name, nu, arg)

    def bessel_j(self, o, nu, arg):
        return MathFunction(o._name, nu, arg)

    def bessel_k(self, o, nu, arg):
        return MathFunction(o._name, nu, arg)

    def bessel_y(self, o, nu, arg):
        return MathFunction(o._name, nu, arg)

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
        return ListTensor(ops)

    def component_tensor(self, o, expression, index):
        return ComponentTensor(expression, index)

    def index_sum(self, o, summand, indices):
        # ufl.IndexSum technically has a MultiIndex, but it must have
        # exactly one index in it.
        index, = indices

        if o.ufl_shape:
            indices = tuple(Index() for i in range(len(o.ufl_shape)))
            return ComponentTensor(IndexSum(Indexed(summand, indices), (index,)), indices)
        else:
            return IndexSum(summand, (index,))

    def variable(self, o, expression, label):
        # Only used by UFL AD, at this point, the bare expression is
        # what we want.
        return expression

    def label(self, o):
        # Only used by UFL AD, don't need it at this point.
        pass
