from __future__ import absolute_import

import collections
import itertools

from tsfc import gem, impero as imp


class OrderedCounter(collections.Counter, collections.OrderedDict):
    """A Counter object that has deterministic iteration order."""
    pass


def process(indexed_ops, temporaries, shape_map, apply_ordering):
    temporaries_set = set(temporaries)
    ops = [op for indices, op in indexed_ops]

    code = make_loop_tree(indexed_ops)

    reference_count = collections.Counter()
    for op in ops:
        reference_count.update(count_references(temporaries_set, op))
    assert temporaries_set == set(reference_count)

    indices, declare = place_declarations(code, reference_count, shape_map, apply_ordering, ops)

    return code, indices, declare


def make_loop_tree(indexed_ops, level=0):
    """Creates an Impero AST with loops from a list of operations and
    their respective free indices.

    :arg indexed_ops: A list of (free indices, operation) pairs, each
                      operation must be an Impero terminal node.
    :arg level: depth of loop nesting
    :returns: Impero AST with loops, without declarations
    """
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
            if isinstance(o, (gem.Literal, gem.Variable)):
                pass
            elif isinstance(o, gem.Indexed):
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
    elif isinstance(op, imp.ReturnAccumulate):
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
