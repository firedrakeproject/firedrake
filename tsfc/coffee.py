from __future__ import absolute_import

from math import isnan

import numpy
from singledispatch import singledispatch
from collections import defaultdict
import itertools

import coffee.base as coffee

from tsfc import gem as ein, impero as imp
from tsfc.constants import NUMPY_TYPE, SCALAR_TYPE, PRECISION


class Bunch(object):
    pass


def generate(temporaries, code, indices, declare):
    parameters = Bunch()
    parameters.declare = declare
    parameters.indices = indices
    parameters.names = {}
    counter = itertools.count()
    parameters.index_names = defaultdict(lambda: "i_%d" % next(counter))

    for i, temp in enumerate(temporaries):
        parameters.names[temp] = "t%d" % i

    return arabica(code, parameters)


def _coffee_symbol(symbol, rank=()):
    """Build a coffee Symbol, concatenating rank.

    :arg symbol: Either a symbol name, or else an existing coffee Symbol.
    :arg rank: The ``rank`` argument to the coffee Symbol constructor.

    If symbol is a symbol, then the returned symbol has rank
    ``symbol.rank + rank``."""
    if isinstance(symbol, coffee.Symbol):
        rank = symbol.rank + rank
        symbol = symbol.symbol
    else:
        assert isinstance(symbol, str)
    return coffee.Symbol(symbol, rank=rank)


def _decl_symbol(expr, parameters):
    multiindex = parameters.indices[expr]
    rank = tuple(index.extent for index in multiindex)
    if hasattr(expr, 'shape'):
        rank += expr.shape
    return _coffee_symbol(parameters.names[expr], rank=rank)


def _index_name(index, parameters):
    if index.name is None:
        return parameters.index_names[index]
    return index.name


def _ref_symbol(expr, parameters):
    multiindex = parameters.indices[expr]
    rank = tuple(_index_name(index, parameters) for index in multiindex)
    return _coffee_symbol(parameters.names[expr], rank=tuple(rank))


@singledispatch
def arabica(tree, parameters):
    raise AssertionError("cannot generate COFFEE from %s" % type(tree))


@arabica.register(imp.Block)  # noqa: Not actually redefinition
def _(tree, parameters):
    statements = [arabica(child, parameters) for child in tree.children]
    declares = []
    for expr in parameters.declare[tree]:
        declares.append(coffee.Decl(SCALAR_TYPE, _decl_symbol(expr, parameters)))
    return coffee.Block(declares + statements, open_scope=True)


@arabica.register(imp.For)  # noqa: Not actually redefinition
def _(tree, parameters):
    extent = tree.index.extent
    assert extent
    i = _coffee_symbol(_index_name(tree.index, parameters))
    # TODO: symbolic constant for "int"
    return coffee.For(coffee.Decl("int", i, init=0),
                      coffee.Less(i, extent),
                      coffee.Incr(i, 1),
                      arabica(tree.children[0], parameters))


@arabica.register(imp.Initialise)  # noqa: Not actually redefinition
def _(leaf, parameters):
    if parameters.declare[leaf]:
        return coffee.Decl(SCALAR_TYPE, _decl_symbol(leaf.indexsum, parameters), 0.0)
    else:
        return coffee.Assign(_ref_symbol(leaf.indexsum, parameters), 0.0)


@arabica.register(imp.Accumulate)  # noqa: Not actually redefinition
def _(leaf, parameters):
    return coffee.Incr(_ref_symbol(leaf.indexsum, parameters),
                       expression(leaf.indexsum.children[0], parameters))


@arabica.register(imp.Return)  # noqa: Not actually redefinition
def _(leaf, parameters):
    return coffee.Incr(expression(leaf.variable, parameters),
                       expression(leaf.expression, parameters))


@arabica.register(imp.ReturnAccumulate)  # noqa: Not actually redefinition
def _(leaf, parameters):
    return coffee.Incr(expression(leaf.variable, parameters),
                       expression(leaf.indexsum.children[0], parameters))


@arabica.register(imp.Evaluate)  # noqa: Not actually redefinition
def _(leaf, parameters):
    expr = leaf.expression
    if isinstance(expr, ein.ListTensor):
        # TODO: remove constant float branch.
        if parameters.declare[leaf]:
            values = numpy.array([expression(v, parameters) for v in expr.array.flat], dtype=object)
            if all(isinstance(value, float) for value in values):
                qualifiers = ["static", "const"]
                values = numpy.array(values, dtype=NUMPY_TYPE)
            else:
                qualifiers = []
            values = values.reshape(expr.shape)
            return coffee.Decl(SCALAR_TYPE,
                               _decl_symbol(expr, parameters),
                               coffee.ArrayInit(values, precision=PRECISION),
                               qualifiers=qualifiers)
        else:
            ops = []
            for multiindex, value in numpy.ndenumerate(expr.array):
                coffee_sym = _coffee_symbol(_ref_symbol(expr, parameters), rank=multiindex)
                ops.append(coffee.Assign(coffee_sym, expression(value, parameters)))
            return coffee.Block(ops, open_scope=False)
    elif isinstance(expr, ein.Literal):
        assert parameters.declare[leaf]
        return coffee.Decl(SCALAR_TYPE,
                           _decl_symbol(expr, parameters),
                           coffee.ArrayInit(expr.array, precision=PRECISION),
                           qualifiers=["static", "const"])
    else:
        code = expression(expr, parameters, top=True)
        if parameters.declare[leaf]:
            return coffee.Decl(SCALAR_TYPE, _decl_symbol(expr, parameters), code)
        else:
            return coffee.Assign(_ref_symbol(expr, parameters), code)


def expression(expr, parameters, top=False):
    if not top and expr in parameters.names:
        return _ref_symbol(expr, parameters)
    else:
        return handle(expr, parameters)


@singledispatch
def handle(expr, parameters):
    raise AssertionError("cannot generate COFFEE from %s" % type(expr))


@handle.register(ein.Product)  # noqa: Not actually redefinition
def _(expr, parameters):
    return coffee.Prod(*[expression(c, parameters)
                         for c in expr.children])


@handle.register(ein.Sum)  # noqa: Not actually redefinition
def _(expr, parameters):
    return coffee.Sum(*[expression(c, parameters)
                        for c in expr.children])


@handle.register(ein.Division)  # noqa: Not actually redefinition
def _(expr, parameters):
    return coffee.Div(*[expression(c, parameters)
                        for c in expr.children])


@handle.register(ein.Power)  # noqa: Not actually redefinition
def _(expr, parameters):
    base, exponent = expr.children
    return coffee.FunCall("pow", expression(base, parameters), expression(exponent, parameters))


@handle.register(ein.MathFunction)  # noqa: Not actually redefinition
def _(expr, parameters):
    name_map = {'abs': 'fabs', 'ln': 'log'}
    name = name_map.get(expr.name, expr.name)
    return coffee.FunCall(name, expression(expr.children[0], parameters))


@handle.register(ein.Comparison)  # noqa: Not actually redefinition
def _(expr, parameters):
    type_map = {">": coffee.Greater,
                ">=": coffee.GreaterEq,
                "==": coffee.Eq,
                "!=": coffee.NEq,
                "<": coffee.Less,
                "<=": coffee.LessEq}
    return type_map[expr.operator](*[expression(c, parameters) for c in expr.children])


@handle.register(ein.LogicalNot)  # noqa: Not actually redefinition
def _(expr, parameters):
    return coffee.Not(*[expression(c, parameters) for c in expr.children])


@handle.register(ein.LogicalAnd)  # noqa: Not actually redefinition
def _(expr, parameters):
    return coffee.And(*[expression(c, parameters) for c in expr.children])


@handle.register(ein.LogicalOr)  # noqa: Not actually redefinition
def _(expr, parameters):
    return coffee.Or(*[expression(c, parameters) for c in expr.children])


@handle.register(ein.Conditional)  # noqa: Not actually redefinition
def _(expr, parameters):
    return coffee.Ternary(*[expression(c, parameters) for c in expr.children])


@handle.register(ein.Literal)  # noqa: Not actually redefinition
@handle.register(ein.Zero)
def _(expr, parameters):
    assert not expr.shape
    if isnan(expr.value):
        return coffee.Symbol("NAN")
    else:
        return coffee.Symbol(("%%.%dg" % (PRECISION - 1)) % expr.value)


@handle.register(ein.Variable)  # noqa: Not actually redefinition
def _(expr, parameters):
    return _coffee_symbol(expr.name)


@handle.register(ein.Indexed)  # noqa: Not actually redefinition
def _(expr, parameters):
    rank = []
    for index in expr.multiindex:
        if isinstance(index, ein.Index):
            rank.append(_index_name(index, parameters))
        elif isinstance(index, ein.VariableIndex):
            rank.append(index.name)
        else:
            rank.append(index)
    return _coffee_symbol(expression(expr.children[0], parameters),
                          rank=tuple(rank))
