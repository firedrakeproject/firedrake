"""Utilities for building an Impero AST from an ordered list of
terminal Impero operations, and for building any additional data
required for straightforward C code generation.

What this module does is independent of whether we eventually generate
C code or a COFFEE AST.
"""

from __future__ import absolute_import

import collections
import itertools

from singledispatch import singledispatch

from tsfc.node import traversal, collect_refcount
from tsfc import impero as imp


# ImperoC is named tuple for C code generation.
#
# Attributes:
#     tree        - Impero AST describing the loop structure and operations
#     temporaries - List of GEM expressions which have assigned temporaries
#     declare     - Where to declare temporaries to get correct C code
#     indices     - Indices for declarations and referencing values
ImperoC = collections.namedtuple('ImperoC', ['tree', 'temporaries', 'declare', 'indices'])


def inline_temporaries(expressions, ops, coffee_licm=False):
    """Inline temporaries which could be inlined without blowing up
    the code.

    :arg expressions: a multi-root GEM expression DAG, used for
                      reference counting
    :arg ops: ordered list of Impero terminals
    :arg coffee_licm: Trust COFFEE to do LICM. If enabled, inlining
                      can move calculations inside inner loops.
    :returns: a filtered ``ops``, without the unnecessary
              :class:`impero.Evaluate`s
    """
    refcount = collect_refcount(expressions)

    candidates = set()  # candidates for inlining
    for op in ops:
        if isinstance(op, imp.Evaluate):
            expr = op.expression
            if expr.shape == () and refcount[expr] == 1:
                candidates.add(expr)

    if not coffee_licm:
        # Prevent inlining that pulls expressions into inner loops
        for node in traversal(expressions):
            for child in node.children:
                if child in candidates and set(child.free_indices) != set(node.free_indices):
                    candidates.remove(child)

    # Filter out candidates
    return [op for op in ops if not (isinstance(op, imp.Evaluate) and op.expression in candidates)]


def process(ops, get_indices):
    """Process the scheduling of operations, builds an Impero AST +
    other data which are directly used for C / COFFEE code generation.

    :arg ops: a list of Impero terminal nodes
    :arg get_indices: callable mapping from GEM nodes to an ordering
                      of free indices
    :returns: ImperoC named tuple
    """
    # Build Impero AST
    tree = make_loop_tree(ops, get_indices)

    # Collect temporaries
    temporaries = collect_temporaries(ops)

    # Determine declarations
    declare, indices = place_declarations(ops, tree, temporaries, get_indices)

    return ImperoC(tree, temporaries, declare, indices)


def collect_temporaries(ops):
    """Collects GEM expressions to assign to temporaries from a list
    of Impero terminals."""
    result = []
    for op in ops:
        # IndexSum temporaries should be added either at Initialise or
        # at Accumulate.  The difference is only in ordering
        # (numbering).  We chose Accumulate here.
        if isinstance(op, (imp.Initialise, imp.Return, imp.ReturnAccumulate)):
            pass
        elif isinstance(op, imp.Accumulate):
            result.append(op.indexsum)
        elif isinstance(op, imp.Evaluate):
            result.append(op.expression)
        else:
            raise AssertionError("unhandled operation: %s" % type(op))
    return result


def make_loop_tree(ops, get_indices, level=0):
    """Creates an Impero AST with loops from a list of operations and
    their respective free indices.

    :arg ops: a list of Impero terminal nodes
    :arg get_indices: callable mapping from GEM nodes to an ordering
                      of free indices
    :arg level: depth of loop nesting
    :returns: Impero AST with loops, without declarations
    """
    keyfunc = lambda op: op.loop_shape(get_indices)[level:level+1]
    statements = []
    for first_index, op_group in itertools.groupby(ops, keyfunc):
        if first_index:
            inner_block = make_loop_tree(op_group, get_indices, level+1)
            statements.append(imp.For(first_index[0], inner_block))
        else:
            statements.extend(op_group)
    return imp.Block(statements)


def place_declarations(ops, tree, temporaries, get_indices):
    """Determines where and how to declare temporaries for an Impero AST.

    :arg ops: terminals of ``tree``
    :arg tree: Impero AST to determine the declarations for
    :arg temporaries: list of GEM expressions which are assigned to
                      temporaries
    :arg get_indices: callable mapping from GEM nodes to an ordering
                      of free indices
    """

    temporaries_set = set(temporaries)
    assert len(temporaries_set) == len(temporaries)

    # Collect the total number of temporary references
    total_refcount = collections.Counter()
    for op in ops:
        total_refcount.update(temp_refcount(temporaries_set, op))
    assert temporaries_set == set(total_refcount)

    # Result
    declare = {}
    indices = {}

    @singledispatch
    def recurse(expr, loop_indices):
        """Visit an Impero AST to collect declarations.

        :arg expr: Impero tree node
        :arg loop_indices: loop indices (in order) from the outer
                           loops surrounding ``expr``
        :returns: :class:`collections.Counter` with the reference
                  counts for each temporary in the subtree whose root
                  is ``expr``
        """
        return AssertionError("unsupported expression type %s" % type(expr))

    @recurse.register(imp.Terminal)
    def recurse_terminal(expr, loop_indices):
        return temp_refcount(temporaries_set, expr)

    @recurse.register(imp.For)
    def recurse_for(expr, loop_indices):
        return recurse(expr.children[0], loop_indices + (expr.index,))

    @recurse.register(imp.Block)
    def recurse_block(expr, loop_indices):
        # Temporaries declared at the beginning of the block are
        # collected here
        declare[expr] = []

        # Collect reference counts for the block
        refcount = collections.Counter()
        for statement in expr.children:
            refcount.update(recurse(statement, loop_indices))

        # Visit :class:`collections.Counter` in deterministic order
        for e in sorted(refcount.keys(), key=temporaries.index):
            if refcount[e] == total_refcount[e]:
                # If all references are within this block, then this
                # block is the right place to declare the temporary.
                assert loop_indices == get_indices(e)[:len(loop_indices)]
                indices[e] = get_indices(e)[len(loop_indices):]
                if indices[e]:
                    # Scalar-valued temporaries are not declared until
                    # their value is assigned.  This does not really
                    # matter, but produces a more compact and nicer to
                    # read C code.
                    declare[expr].append(e)
                # Remove expression from the ``refcount`` so it will
                # not be declared again.
                del refcount[e]
        return refcount

    # Populate result
    remainder = recurse(tree, ())
    assert not remainder

    # Set in ``declare`` for Impero terminals whether they should
    # declare the temporary that they are writing to.
    for op in ops:
        declare[op] = False
        if isinstance(op, imp.Evaluate):
            e = op.expression
        elif isinstance(op, imp.Initialise):
            e = op.indexsum
        elif isinstance(op, (imp.Accumulate, imp.Return, imp.ReturnAccumulate)):
            continue
        else:
            raise AssertionError("unhandled operation type %s" % type(op))

        if len(indices[e]) == 0:
            declare[op] = True

    return declare, indices


def temp_refcount(temporaries, op):
    """Collects the number of times temporaries are referenced when
    generating code for an Impero terminal.

    :arg temporaries: set of temporaries
    :arg op: Impero terminal
    :returns: :class:`collections.Counter` object mapping some of
               elements from ``temporaries`` to the number of times
               they will referenced from ``op``
    """
    counter = collections.Counter()

    def recurse(o):
        """Traverses expression until reaching temporaries, counting
        temporary references."""
        if o in temporaries:
            counter[o] += 1
        else:
            for c in o.children:
                recurse(c)

    def recurse_top(o):
        """Traverses expression until reaching temporaries, counting
        temporary references. Always descends into children at least
        once, even when the root is a temporary."""
        if o in temporaries:
            counter[o] += 1
        for c in o.children:
            recurse(c)

    if isinstance(op, imp.Initialise):
        counter[op.indexsum] += 1
    elif isinstance(op, imp.Accumulate):
        recurse_top(op.indexsum)
    elif isinstance(op, imp.Evaluate):
        recurse_top(op.expression)
    elif isinstance(op, imp.Return):
        recurse(op.expression)
    elif isinstance(op, imp.ReturnAccumulate):
        recurse(op.indexsum.children[0])
    else:
        raise AssertionError("unhandled operation: %s" % type(op))

    return counter
