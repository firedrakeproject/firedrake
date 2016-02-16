from __future__ import absolute_import

from collections import defaultdict
from math import isnan
import itertools

import numpy
from singledispatch import singledispatch

import coffee.base as coffee

from tsfc import gem as ein, impero as imp
from tsfc.constants import SCALAR_TYPE, PRECISION


class Bunch(object):
    pass


def generate(impero_c, index_names):
    parameters = Bunch()
    parameters.declare = impero_c.declare
    parameters.indices = impero_c.indices

    parameters.names = {}
    for i, temp in enumerate(impero_c.temporaries):
        parameters.names[temp] = "t%d" % i

    counter = itertools.count()
    parameters.index_names = defaultdict(lambda: "i_%d" % next(counter))
    parameters.index_names.update(index_names)

    return statement(impero_c.tree, parameters)


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
    rank = tuple(index.extent for index in multiindex) + expr.shape
    return _coffee_symbol(parameters.names[expr], rank=rank)


def _ref_symbol(expr, parameters):
    multiindex = parameters.indices[expr]
    rank = tuple(parameters.index_names[index] for index in multiindex)
    return _coffee_symbol(parameters.names[expr], rank=tuple(rank))


@singledispatch
def statement(tree, parameters):
    raise AssertionError("cannot generate COFFEE from %s" % type(tree))


@statement.register(imp.Block)
def statement_block(tree, parameters):
    statements = [statement(child, parameters) for child in tree.children]
    declares = []
    for expr in parameters.declare[tree]:
        declares.append(coffee.Decl(SCALAR_TYPE, _decl_symbol(expr, parameters)))
    return coffee.Block(declares + statements, open_scope=True)


@statement.register(imp.For)
def statement_for(tree, parameters):
    extent = tree.index.extent
    assert extent
    i = _coffee_symbol(parameters.index_names[tree.index])
    # TODO: symbolic constant for "int"
    return coffee.For(coffee.Decl("int", i, init=0),
                      coffee.Less(i, extent),
                      coffee.Incr(i, 1),
                      statement(tree.children[0], parameters))


@statement.register(imp.Initialise)
def statement_initialise(leaf, parameters):
    if parameters.declare[leaf]:
        return coffee.Decl(SCALAR_TYPE, _decl_symbol(leaf.indexsum, parameters), 0.0)
    else:
        return coffee.Assign(_ref_symbol(leaf.indexsum, parameters), 0.0)


@statement.register(imp.Accumulate)
def statement_accumulate(leaf, parameters):
    return coffee.Incr(_ref_symbol(leaf.indexsum, parameters),
                       expression(leaf.indexsum.children[0], parameters))


@statement.register(imp.Return)
def statement_return(leaf, parameters):
    return coffee.Incr(expression(leaf.variable, parameters),
                       expression(leaf.expression, parameters))


@statement.register(imp.ReturnAccumulate)
def statement_returnaccumulate(leaf, parameters):
    return coffee.Incr(expression(leaf.variable, parameters),
                       expression(leaf.indexsum.children[0], parameters))


@statement.register(imp.Evaluate)
def statement_evaluate(leaf, parameters):
    expr = leaf.expression
    if isinstance(expr, ein.ListTensor):
        if parameters.declare[leaf]:
            array_expression = numpy.vectorize(lambda v: expression(v, parameters))
            return coffee.Decl(SCALAR_TYPE,
                               _decl_symbol(expr, parameters),
                               coffee.ArrayInit(array_expression(expr.array),
                                                precision=PRECISION))
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
        return _expression(expr, parameters)


@singledispatch
def _expression(expr, parameters):
    raise AssertionError("cannot generate COFFEE from %s" % type(expr))


@_expression.register(ein.Product)
def _expression_product(expr, parameters):
    return coffee.Prod(*[expression(c, parameters)
                         for c in expr.children])


@_expression.register(ein.Sum)
def _expression_sum(expr, parameters):
    return coffee.Sum(*[expression(c, parameters)
                        for c in expr.children])


@_expression.register(ein.Division)
def _expression_division(expr, parameters):
    return coffee.Div(*[expression(c, parameters)
                        for c in expr.children])


@_expression.register(ein.Power)
def _expression_power(expr, parameters):
    base, exponent = expr.children
    return coffee.FunCall("pow", expression(base, parameters), expression(exponent, parameters))


@_expression.register(ein.MathFunction)
def _expression_mathfunction(expr, parameters):
    name_map = {'abs': 'fabs', 'ln': 'log'}
    name = name_map.get(expr.name, expr.name)
    return coffee.FunCall(name, expression(expr.children[0], parameters))


@_expression.register(ein.Comparison)
def _expression_comparison(expr, parameters):
    type_map = {">": coffee.Greater,
                ">=": coffee.GreaterEq,
                "==": coffee.Eq,
                "!=": coffee.NEq,
                "<": coffee.Less,
                "<=": coffee.LessEq}
    return type_map[expr.operator](*[expression(c, parameters) for c in expr.children])


@_expression.register(ein.LogicalNot)
def _expression_logicalnot(expr, parameters):
    return coffee.Not(*[expression(c, parameters) for c in expr.children])


@_expression.register(ein.LogicalAnd)
def _expression_logicaland(expr, parameters):
    return coffee.And(*[expression(c, parameters) for c in expr.children])


@_expression.register(ein.LogicalOr)
def _expression_logicalor(expr, parameters):
    return coffee.Or(*[expression(c, parameters) for c in expr.children])


@_expression.register(ein.Conditional)
def _expression_conditional(expr, parameters):
    return coffee.Ternary(*[expression(c, parameters) for c in expr.children])


@_expression.register(ein.Literal)
@_expression.register(ein.Zero)
def _expression_scalar(expr, parameters):
    assert not expr.shape
    if isnan(expr.value):
        return coffee.Symbol("NAN")
    else:
        return coffee.Symbol(("%%.%dg" % (PRECISION - 1)) % expr.value)


@_expression.register(ein.Variable)
def _expression_variable(expr, parameters):
    return _coffee_symbol(expr.name)


@_expression.register(ein.Indexed)
def _expression_indexed(expr, parameters):
    rank = []
    for index in expr.multiindex:
        if isinstance(index, ein.Index):
            rank.append(parameters.index_names[index])
        elif isinstance(index, ein.VariableIndex):
            rank.append(index.name)
        else:
            rank.append(index)
    return _coffee_symbol(expression(expr.children[0], parameters),
                          rank=tuple(rank))
