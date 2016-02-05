"""Generate COFFEE AST from ImperoC tuple data.

This is the final stage of code generation in TSFC."""

from __future__ import absolute_import

from collections import defaultdict
from math import isnan
import itertools

import numpy
from singledispatch import singledispatch

import coffee.base as coffee

import gem
import gem.impero as imp

from tsfc.constants import SCALAR_TYPE, PRECISION


class Bunch(object):
    pass


def generate(impero_c, index_names, roots=()):
    """Generates COFFEE code.

    :arg impero_c: ImperoC tuple with Impero AST and other data
    :arg index_names: pre-assigned index names
    :arg roots: list of expression DAG roots for attaching
        #pragma coffee expression
    :returns: COFFEE function body
    """
    parameters = Bunch()
    parameters.declare = impero_c.declare
    parameters.indices = impero_c.indices
    parameters.roots = roots

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
    """Build a COFFEE Symbol for declaration."""
    multiindex = parameters.indices[expr]
    rank = tuple(index.extent for index in multiindex) + expr.shape
    return _coffee_symbol(parameters.names[expr], rank=rank)


def _ref_symbol(expr, parameters):
    """Build a COFFEE Symbol for referencing a value."""
    multiindex = parameters.indices[expr]
    rank = tuple(parameters.index_names[index] for index in multiindex)
    return _coffee_symbol(parameters.names[expr], rank=tuple(rank))


def _root_pragma(expr, parameters):
    """Decides whether to annonate the expression with
    #pragma coffee expression"""
    if expr in parameters.roots:
        return "#pragma coffee expression"
    else:
        return None


@singledispatch
def statement(tree, parameters):
    """Translates an Impero (sub)tree into a COFFEE AST corresponding
    to a C statement.

    :arg tree: Impero (sub)tree
    :arg parameters: miscellaneous code generation data
    :returns: COFFEE AST
    """
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
    pragma = _root_pragma(leaf.indexsum, parameters)
    return coffee.Incr(_ref_symbol(leaf.indexsum, parameters),
                       expression(leaf.indexsum.children[0], parameters),
                       pragma=pragma)


@statement.register(imp.Return)
def statement_return(leaf, parameters):
    pragma = _root_pragma(leaf.expression, parameters)
    return coffee.Incr(expression(leaf.variable, parameters),
                       expression(leaf.expression, parameters),
                       pragma=pragma)


@statement.register(imp.ReturnAccumulate)
def statement_returnaccumulate(leaf, parameters):
    pragma = _root_pragma(leaf.indexsum, parameters)
    return coffee.Incr(expression(leaf.variable, parameters),
                       expression(leaf.indexsum.children[0], parameters),
                       pragma=pragma)


@statement.register(imp.Evaluate)
def statement_evaluate(leaf, parameters):
    expr = leaf.expression
    if isinstance(expr, gem.ListTensor):
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
    elif isinstance(expr, gem.Literal):
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
    """Translates GEM expression into a COFFEE snippet, stopping at
    temporaries.

    :arg expr: GEM expression
    :arg parameters: miscellaneous code generation data
    :arg top: do not generate temporary reference for the root node
    :returns: COFFEE expression
    """
    if not top and expr in parameters.names:
        return _ref_symbol(expr, parameters)
    else:
        return _expression(expr, parameters)


@singledispatch
def _expression(expr, parameters):
    raise AssertionError("cannot generate COFFEE from %s" % type(expr))


@_expression.register(gem.Product)
def _expression_product(expr, parameters):
    return coffee.Prod(*[expression(c, parameters)
                         for c in expr.children])


@_expression.register(gem.Sum)
def _expression_sum(expr, parameters):
    return coffee.Sum(*[expression(c, parameters)
                        for c in expr.children])


@_expression.register(gem.Division)
def _expression_division(expr, parameters):
    return coffee.Div(*[expression(c, parameters)
                        for c in expr.children])


@_expression.register(gem.Power)
def _expression_power(expr, parameters):
    base, exponent = expr.children
    return coffee.FunCall("pow", expression(base, parameters), expression(exponent, parameters))


@_expression.register(gem.MathFunction)
def _expression_mathfunction(expr, parameters):
    name_map = {'abs': 'fabs', 'ln': 'log'}
    name = name_map.get(expr.name, expr.name)
    return coffee.FunCall(name, expression(expr.children[0], parameters))


@_expression.register(gem.MinValue)
def _expression_minvalue(expr, parameters):
    return coffee.FunCall('fmin', *[expression(c, parameters) for c in expr.children])


@_expression.register(gem.MaxValue)
def _expression_maxvalue(expr, parameters):
    return coffee.FunCall('fmax', *[expression(c, parameters) for c in expr.children])


@_expression.register(gem.Comparison)
def _expression_comparison(expr, parameters):
    type_map = {">": coffee.Greater,
                ">=": coffee.GreaterEq,
                "==": coffee.Eq,
                "!=": coffee.NEq,
                "<": coffee.Less,
                "<=": coffee.LessEq}
    return type_map[expr.operator](*[expression(c, parameters) for c in expr.children])


@_expression.register(gem.LogicalNot)
def _expression_logicalnot(expr, parameters):
    return coffee.Not(*[expression(c, parameters) for c in expr.children])


@_expression.register(gem.LogicalAnd)
def _expression_logicaland(expr, parameters):
    return coffee.And(*[expression(c, parameters) for c in expr.children])


@_expression.register(gem.LogicalOr)
def _expression_logicalor(expr, parameters):
    return coffee.Or(*[expression(c, parameters) for c in expr.children])


@_expression.register(gem.Conditional)
def _expression_conditional(expr, parameters):
    return coffee.Ternary(*[expression(c, parameters) for c in expr.children])


@_expression.register(gem.Literal)
@_expression.register(gem.Zero)
def _expression_scalar(expr, parameters):
    assert not expr.shape
    if isnan(expr.value):
        return coffee.Symbol("NAN")
    else:
        return coffee.Symbol(("%%.%dg" % (PRECISION - 1)) % expr.value)


@_expression.register(gem.Variable)
def _expression_variable(expr, parameters):
    return _coffee_symbol(expr.name)


@_expression.register(gem.Indexed)
def _expression_indexed(expr, parameters):
    rank = []
    for index in expr.multiindex:
        if isinstance(index, gem.Index):
            rank.append(parameters.index_names[index])
        elif isinstance(index, gem.VariableIndex):
            rank.append(expression(index.expression, parameters).gencode())
        else:
            rank.append(index)
    return _coffee_symbol(expression(expr.children[0], parameters),
                          rank=tuple(rank))
