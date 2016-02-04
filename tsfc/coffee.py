from __future__ import absolute_import

import collections
import itertools
from math import isnan

import numpy
from singledispatch import singledispatch

import coffee.base as coffee

from tsfc import gem as ein, impero as imp
from tsfc.constants import NUMPY_TYPE, SCALAR_TYPE, PRECISION


class Bunch(object):
    pass


class OrderedCounter(collections.Counter, collections.OrderedDict):
    """A Counter object that has deterministic iteration order."""
    pass


def generate(indexed_ops, temporaries, shape_map, apply_ordering, index_extents, index_names):
    temporaries_set = set(temporaries)
    ops = [op for indices, op in indexed_ops]

    code = make_loop_tree(indexed_ops)

    reference_count = collections.Counter()
    for op in ops:
        reference_count.update(count_references(temporaries_set, op))
    assert temporaries_set == set(reference_count)

    indices, declare = place_declarations(code, reference_count, shape_map, apply_ordering, ops)

    parameters = Bunch()
    parameters.index_extents = index_extents
    parameters.declare = declare
    parameters.indices = indices
    parameters.names = {}

    for i, temp in enumerate(temporaries):
        parameters.names[temp] = "t%d" % i

    for index, name in index_names:
        parameters.names[index] = name

    index_counter = 0
    for index in index_extents:
        if index not in parameters.names:
            index_counter += 1
            parameters.names[index] = "i_%d" % index_counter

    return arabica(code, parameters)


def make_loop_tree(indexed_ops, level=0):
    keyfunc = lambda (indices, op): indices[level:level+1]
    statements = []
    for first_index, op_group in itertools.groupby(indexed_ops, keyfunc):
        if first_index:
            inner_block = make_loop_tree(op_group, level+1)
            statements.append(imp.For(first_index[0], inner_block))
        else:
            statements.extend(op for indices, op in op_group)
    return imp.Block(statements)


def count_references(temporaries, op):
    counter = collections.Counter()

    def recurse(o, top=False):
        if o in temporaries:
            counter[o] += 1

        if top or o not in temporaries:
            if isinstance(o, (ein.Literal, ein.Variable)):
                pass
            elif isinstance(o, ein.Indexed):
                recurse(o.children[0])
            else:
                for c in o.children:
                    recurse(c)

    if isinstance(op, imp.Evaluate):
        recurse(op.expression, top=True)
    elif isinstance(op, imp.Initialise):
        counter[op.indexsum] += 1
    elif isinstance(op, imp.Return):
        recurse(op.expression)
    elif isinstance(op, imp.Accumulate):
        counter[op.indexsum] += 1
        recurse(op.indexsum.children[0])
    else:
        raise AssertionError("unhandled operation: %s" % type(op))

    return counter


def place_declarations(tree, reference_count, shape_map, apply_ordering, operations):
    temporaries = set(reference_count)
    indices = {}
    # We later iterate over declare keys, so need this to be ordered
    declare = collections.OrderedDict()

    def recurse(expr, loop_indices):
        if isinstance(expr, imp.Block):
            declare[expr] = []
            # Need to iterate over counter in given order
            counter = OrderedCounter()
            for statement in expr.children:
                counter.update(recurse(statement, loop_indices))
            for e, count in counter.items():
                if count == reference_count[e]:
                    indices[e] = apply_ordering(set(shape_map(e)) - loop_indices)
                    if indices[e]:
                        declare[expr].append(e)
                    del counter[e]
            return counter
        elif isinstance(expr, imp.For):
            return recurse(expr.children[0], loop_indices | {expr.index})
        else:
            return count_references(temporaries, expr)

    remainder = recurse(tree, set())
    assert not remainder

    for op in operations:
        declare[op] = False
        if isinstance(op, imp.Evaluate):
            e = op.expression
        elif isinstance(op, imp.Initialise):
            e = op.indexsum
        else:
            continue

        if len(indices[e]) == 0:
            declare[op] = True

    return indices, declare


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
    rank = tuple(parameters.index_extents[index] for index in multiindex)
    if hasattr(expr, 'shape'):
        rank += expr.shape
    return _coffee_symbol(parameters.names[expr], rank=rank)


def _ref_symbol(expr, parameters):
    multiindex = parameters.indices[expr]
    rank = tuple(parameters.names[index] for index in multiindex)
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
    i = _coffee_symbol(parameters.names[tree.index])
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
        return coffee.Symbol(("%%.%dg" % PRECISION) % expr.value)


@handle.register(ein.Variable)  # noqa: Not actually redefinition
def _(expr, parameters):
    return _coffee_symbol(expr.name)


@handle.register(ein.Indexed)  # noqa: Not actually redefinition
def _(expr, parameters):
    rank = []
    for index in expr.multiindex:
        if isinstance(index, ein.Index):
            rank.append(parameters.names[index])
        elif isinstance(index, ein.VariableIndex):
            rank.append(index.name)
        else:
            rank.append(index)
    return _coffee_symbol(expression(expr.children[0], parameters),
                          rank=tuple(rank))
