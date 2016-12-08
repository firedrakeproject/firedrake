"""Generate COFFEE AST from ImperoC tuple data.

This is the final stage of code generation in TSFC."""

from __future__ import absolute_import, print_function, division

from collections import defaultdict
from functools import reduce
from math import isnan
import itertools

import numpy
from singledispatch import singledispatch

import coffee.base as coffee

from gem import gem, impero as imp

from tsfc.parameters import SCALAR_TYPE


class Bunch(object):
    pass


def generate(impero_c, index_names, precision, roots=(), argument_indices=()):
    """Generates COFFEE code.

    :arg impero_c: ImperoC tuple with Impero AST and other data
    :arg index_names: pre-assigned index names
    :arg precision: floating-point precision for printing
    :arg roots: list of expression DAG roots for attaching
        #pragma coffee expression
    :arg argument_indices: argument indices for attaching
        #pragma coffee linear loop
        to the argument loops
    :returns: COFFEE function body
    """
    params = Bunch()
    params.declare = impero_c.declare
    params.indices = impero_c.indices
    params.precision = precision
    params.epsilon = 10.0 * eval("1e-%d" % precision)
    params.roots = roots
    params.argument_indices = argument_indices

    params.names = {}
    for i, temp in enumerate(impero_c.temporaries):
        params.names[temp] = "t%d" % i

    counter = itertools.count()
    params.index_names = defaultdict(lambda: "i_%d" % next(counter))
    params.index_names.update(index_names)

    return statement(impero_c.tree, params)


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
    if tree.index in parameters.argument_indices:
        pragma = "#pragma coffee linear loop"
    else:
        pragma = None
    extent = tree.index.extent
    assert extent
    i = _coffee_symbol(parameters.index_names[tree.index])
    # TODO: symbolic constant for "int"
    return coffee.For(coffee.Decl("int", i, init=0),
                      coffee.Less(i, extent),
                      coffee.Incr(i, 1),
                      statement(tree.children[0], parameters),
                      pragma=pragma)


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
                                                precision=parameters.precision))
        else:
            ops = []
            for multiindex, value in numpy.ndenumerate(expr.array):
                coffee_sym = _coffee_symbol(_ref_symbol(expr, parameters), rank=multiindex)
                ops.append(coffee.Assign(coffee_sym, expression(value, parameters)))
            return coffee.Block(ops, open_scope=False)
    elif isinstance(expr, gem.Constant):
        assert parameters.declare[leaf]
        return coffee.Decl(SCALAR_TYPE,
                           _decl_symbol(expr, parameters),
                           coffee.ArrayInit(expr.array, parameters.precision),
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


@_expression.register(gem.Failure)
def _expression_failure(expr, parameters):
    raise expr.exception


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


@_expression.register(gem.Constant)
def _expression_scalar(expr, parameters):
    assert not expr.shape
    if isnan(expr.value):
        return coffee.Symbol("NAN")
    else:
        v = expr.value
        r = round(v, 1)
        if r and abs(v - r) < parameters.epsilon:
            v = r  # round to nonzero
        return coffee.Symbol(("%%.%dg" % parameters.precision) % v)


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


@_expression.register(gem.FlexiblyIndexed)
def _expression_flexiblyindexed(expr, parameters):
    var = expression(expr.children[0], parameters)
    assert isinstance(var, coffee.Symbol)
    assert not var.rank
    assert not var.offset

    rank = []
    offset = []
    for off, idxs in expr.dim2idxs:
        if idxs:
            indices, strides = zip(*idxs)
            strides = cumulative_strides(strides)
        else:
            indices = ()
            strides = ()

        iss = []
        for i, s in zip(indices, strides):
            if isinstance(i, int):
                off += i * s
            elif isinstance(i, gem.Index):
                iss.append((i, s))
            else:
                raise AssertionError("Unexpected index type!")

        if len(iss) == 0:
            rank.append(off)
            offset.append((1, 0))
        elif len(iss) == 1:
            (i, s), = iss
            rank.append(parameters.index_names[i])
            offset.append((s, off))
        else:
            parts = []
            if off:
                parts += [coffee.Symbol(str(off))]
            for i, s in iss:
                index_sym = coffee.Symbol(parameters.index_names[i])
                assert s
                if s == 1:
                    parts += [index_sym]
                else:
                    parts += [coffee.Prod(index_sym, coffee.Symbol(str(s)))]
            assert parts
            rank.append(reduce(coffee.Sum, parts))
            offset.append((1, 0))

    return coffee.Symbol(var.symbol, rank=tuple(rank), offset=tuple(offset))
