"""Impero to Impero passes on the parallel loops of a kernel, which every GEM kernel runs.

The iterations of a parallel loop (the items) are independent, apart from
increments of kernel arguments. Only the outermost parallel loop runs its
items concurrently, as in :class:`pyop3.compile.mlir.MLIRBuilder`.
"""

from __future__ import annotations

import itertools
import numbers
from collections.abc import Callable

from gem import gem, impero as imp
from gem.impero_utils import ImperoC
from gem.node import pre_traversal

from pyop3 import utils
from pyop3.compile.passes.nests import (
    TEMPORARY_DATA, GemRewriter, LinearIndex, assign_value, flat_entry, gem_nodes, is_temporary, new_temporary,
    rewrite_terminal,
)


def flatten_item_temporaries(impero: ImperoC) -> ImperoC:
    """Give each item its own copy of the per-item temporaries.

    pyop3 declares a temporary of a loop body with one entry per item, such as
    ``t[item, j]``, so that the GEM scheduler keeps its accesses inside the
    parallel loop. Each item only accesses its own entries, so ``t[item, j]``
    becomes the flat ``t[j]``, and ``t[item]`` becomes ``t[0]``. Code
    generation then allocates the temporary once per item.
    """
    items = {n.index for n in pre_traversal([impero.tree]) if isinstance(n, imp.For) and n.is_parallel}

    def flatten(variable: gem.Variable, multiindex: tuple) -> gem.Node | None:
        rest = tuple(itertools.dropwhile(items.__contains__, multiindex))
        if not is_temporary(variable) or len(rest) == len(multiindex):
            return None
        shape = variable.shape[len(multiindex) - len(rest):]
        size = shape[0] if shape else 1
        if len(shape) > 1 or not isinstance(size, numbers.Integral):
            raise NotImplementedError(f"Temporary {variable.name} must have one static dimension besides the items")
        return gem.Indexed(gem.Variable(variable.name, (int(size),), dtype=variable.dtype, data=TEMPORARY_DATA), rest or (0,))

    rewrite = GemRewriter(replace_access=flatten)
    return impero._replace(
        tree=map_terminals(impero.tree, lambda loops, terminal: rewrite_terminal(terminal, rewrite)),
        temporaries=[rewrite(e) for e in impero.temporaries],
        indices={rewrite(e): indices for e, indices in impero.indices.items()},
    )


def assign_crossing_values(impero: ImperoC) -> imp.Block:
    """Store each value of an item that one loop computes and another loop reads in a temporary.

    ``impero.indices`` gives the loop indices of each such value. Its evaluation,
    or the initialisation and accumulation of an index sum, become assignments to a
    new temporary with one entry per point of those indices, and its reads become
    loads from the temporary. A value that reads no variable (see `_is_stateless`) is
    computed again at each use instead. Values outside the parallel loops do not depend on the item,
    and each item that uses them computes them again, so they are not stored.
    """
    items = [n.statement for n in pre_traversal([impero.tree]) if isinstance(n, imp.For) and n.is_parallel]
    names = utils.UniqueNameGenerator(
        forced_prefix="", existing_names={e.name for e in gem_nodes(impero.tree) if isinstance(e, gem.Variable)}
    )
    temporaries = {}
    for terminal in pre_traversal(items):
        match terminal:
            case imp.Evaluate(expression=value) if not _is_stateless(value):
                pass
            case imp.Initialise(indexsum=value) | imp.Accumulate(indexsum=value):
                pass
            case _:
                continue
        if value not in temporaries and (indices := impero.indices.get(value)):
            temporaries[value] = new_temporary(names("t"), indices, value.dtype)

    replace = GemRewriter(temporaries)

    def assign(loops, terminal: imp.Terminal) -> imp.Terminal:
        assignment = assign_value(terminal, temporaries, replace)
        return assignment if assignment is not None else rewrite_terminal(terminal, replace)

    return map_terminals(impero.tree, assign)


def _is_stateless(expr: gem.Node) -> bool:
    """Whether ``expr`` reads no variable, so that it has the same value at every use.

    These are tensor-valued values, which are read from their global buffers,
    indices, constants, and entries of constants at loop indices, such as
    quadrature weights. Other values are stored where they are computed, because
    a variable that they read may be written before a later use.
    """
    if isinstance(expr, gem.Indexed) and isinstance(expr.aggregate, gem.Constant):
        return all(isinstance(index, (gem.Index, numbers.Integral)) for index in expr.multiindex)
    return bool(expr.shape) or isinstance(expr, (gem.Index, gem.Constant))


def mark_atomics(tree: imp.Block) -> imp.Block:
    """Make each increment ``"atomic"`` if another item of its parallel loop may increment the same entry."""
    def mark(loops: tuple[imp.For, ...], terminal: imp.Terminal) -> imp.Terminal:
        items = tuple(loop.index for loop in loops if loop.is_parallel)[:1]
        if items and isinstance(terminal, imp.Assignment) and is_atomic(terminal, items):
            return imp.Assignment(terminal.assignee, terminal.expression, "atomic")
        return terminal

    return map_terminals(tree, mark)


def is_atomic(assignment: imp.Assignment, threads: tuple[gem.Index, ...]) -> bool:
    """Whether ``assignment`` increments a kernel argument at an entry that another thread may increment.

    ``threads`` are the indices whose points run on different threads. A
    ``"write"`` to a kernel argument stays a plain store: concurrent writers
    either store the same value or write through an injective map.
    """
    if assignment.mode != "inc" or is_temporary(assignment.assignee.aggregate):
        return False
    linear = LinearIndex.from_gem(flat_entry(assignment.assignee.multiindex))
    return linear is None or not linear.is_injective(threads)


def map_terminals(node: imp.Node, function: Callable, loops: tuple[imp.For, ...] = ()) -> imp.Node:
    """Rebuild ``node`` with ``function(loops, terminal)`` in place of each terminal, with ``loops`` the enclosing loops."""
    match node:
        case imp.Block():
            return imp.Block(map_terminals(child, function, loops) for child in node.children)
        case imp.For():
            return imp.For(node.index, map_terminals(node.statement, function, (*loops, node)), node.is_parallel)
        case imp.Terminal():
            return function(loops, node)
    utils.raise_missing_dispatch_handler(node)
