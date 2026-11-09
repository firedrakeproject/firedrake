"""Impero to Impero passes on the parallel loops of a kernel, which every GEM kernel runs.

The iterations of a parallel loop (the items) are independent, apart from
increments of kernel arguments. Only the outermost parallel loop runs its
items concurrently, as in `pyop3.compile.mlir.MLIRBuilder`.
"""

from __future__ import annotations

import functools
import itertools
import numbers
from collections.abc import Callable, Mapping

from gem import gem, impero as imp, replace
from gem.impero_utils import ImperoC
from gem.node import pre_traversal

from pyop3 import utils
from pyop3.compile.passes.impero.nests import is_atomic, new_temporary
from pyop3.compile.passes.impero.utils import (
    assign_value, gem_nodes, is_item_loop, is_temporary, name_generator, temporary,
)

ITEMS_PER_BLOCK = 128
"""The default number of threads in a block for single-cell-per-thread parallelism"""

def flatten_item_temporaries(impero: ImperoC) -> ImperoC:
    """Give each item its own copy of the per-item temporaries.

    pyop3 declares a temporary of a loop body with one entry per item, such as
    `t[item, j]`, so that the GEM scheduler keeps its accesses inside the
    parallel loop. Each item only accesses its own entries, so `t[item, j]`
    becomes the flat `t[j]`, and `t[item]` becomes `t[0]`. Code
    generation then allocates the temporary once per item.

    I am unsure if temporaries will continue being blown up to t[item, j].
    It provides clarity on the parallel nature but, in-practice, only t[j].
    
    As a result, if temporaries are no longer blown up, we can simply remove this pass.
    It acts as a source of truth to the technique. 
    """

    # Set of indices of the item loops  
    items = {n.index for n in pre_traversal([impero.tree]) if is_item_loop(n)}

    # NOTE: This flatten removes all appearances of indices in `items` (i.e. t[item, j] -> t[j])
    def flatten(access: gem.Indexed) -> gem.Node | None:
        variable, multiindex = access.aggregate, access.multiindex
        # Tuple of the indices are not item indices (i.e. t[item, j] -> (j, ))
        rest = tuple(itertools.dropwhile(items.__contains__, multiindex))
        
        # Return if variable is not a temporary or if not a item-level temporary
        if not is_temporary(variable) or len(rest) == len(multiindex):
            return None

        shape = variable.shape[len(multiindex) - len(rest):]
        size = shape[0] if shape else 1  

        if len(shape) > 1 or not isinstance(size, numbers.Integral):
            raise NotImplementedError(f"Temporary {variable.name} must have one static dimension besides the items")
        
        return gem.Indexed(temporary(variable.name, (int(size),), variable.dtype), rest or (0,))

    # NOTE: flattened access keeps any temporary in its own indices unflattened, rewrite bottom-up if one appears
    mapping = {
        n: new for n in pre_traversal([*gem_nodes(impero.tree), *impero.temporaries, *impero.indices])
        if isinstance(n, gem.Indexed) and (new := flatten(n)) is not None
    }

    return impero._replace(
        tree=map_terminals(impero.tree, lambda loops, terminal: rewrite_terminal(terminal, mapping)),
        temporaries=[replace(e, mapping) for e in impero.temporaries],
        indices={replace(e, mapping): indices for e, indices in impero.indices.items()},
    )


def assign_crossing_values(impero: ImperoC) -> imp.Block:
    """Store each value of an item that one loop computes and another loop reads in a temporary.

    `impero.indices` gives the loop indices of each such value. Its evaluation,
    or the initialisation and accumulation of an index sum, become assignments to a
    new temporary with one entry per point of those indices, and its reads become
    loads from the temporary. Values outside the parallel loops do not depend on the item,
    and each item that uses them computes them again, so they are not stored.
    """
    items = [n.statement for n in pre_traversal([impero.tree]) if is_item_loop(n)]
    names = name_generator(impero.tree)
    temporaries = {}
    for terminal in pre_traversal(items):
        match terminal:
            case imp.Evaluate(expression=value) | imp.Initialise(indexsum=value) | imp.Accumulate(indexsum=value):
                pass
            case _:
                continue
        if value not in temporaries and (indices := impero.indices.get(value)):
            temporaries[value] = new_temporary(names("t"), indices, value.dtype)

    def assign(loops, terminal: imp.Terminal) -> imp.Terminal:
        assignment = assign_value(terminal, temporaries)
        return assignment if assignment is not None else rewrite_terminal(terminal, temporaries)

    return map_terminals(impero.tree, assign)


def mark_atomics(tree: imp.Block) -> imp.Block:
    """Make each increment `"atomic"` if another item of its parallel loop may increment the same kernel argument"""

    def mark(loops, terminal):
        items = tuple(loop.index for loop in loops if loop.is_parallel)[:1]
        if items and isinstance(terminal, imp.Assignment) and is_atomic(terminal, items):
            return imp.Assignment(terminal.assignee, terminal.expression, "atomic")
        return terminal

    return map_terminals(tree, mark)

def batch_items(tree: imp.Block) -> imp.Block:
    """Run each top-level parallel loop of `tree` on a GPU with one thread per item.

    This is single-cell-per-thread (SCPT) parallelism, which achieves better performance for lower
    order elements (where memory per thread is less) 

    The loop becomes an `gem.impero.ItemBatch` with one
    `gem.impero.ThreadLoop` over the item only, whose body is the loop body.
    """

    # Confirming that there is only a single item-loop in the Impero tree.
    # I think that extruded meshes could cause issues in this scenario??
    others = (child for child in tree.children if not is_item_loop(child))
    if any(map(is_item_loop, pre_traversal(others))):
        raise NotImplementedError("Only a top-level parallel loop can run on a GPU")

    def batch(node: imp.Node) -> imp.Node:
        if not is_item_loop(node):
            return node
        local = gem.Index(name="local", extent=ITEMS_PER_BLOCK)
        loop = imp.ThreadLoop((), node.statement)
        return imp.ItemBatch(node.index, local, batch_size=ITEMS_PER_BLOCK, block_size=ITEMS_PER_BLOCK, statement=imp.Block([loop]))

    return imp.Block(map(batch, tree.children))


def rewrite_terminal(terminal: imp.Terminal, mapping: Mapping) -> imp.Terminal:
    """Return `terminal` with `gem.replace` by `mapping` applied to each of its GEM expressions"""
    def rewrite(value):
        return replace(value, mapping) if isinstance(value, gem.Node) else value

    return type(terminal)(*(rewrite(getattr(terminal, slot)) for slot in terminal.__slots__))


@functools.singledispatch
def map_terminals(node: imp.Node, function: Callable, loops: tuple[imp.For, ...] = ()) -> imp.Node:
    """Rebuild `node` with `function(loops, terminal)` in place of each terminal, with `loops` the enclosing loops"""
    utils.raise_missing_dispatch_handler(node)


@map_terminals.register
def _(node: imp.Block, function: Callable, loops: tuple[imp.For, ...] = ()) -> imp.Node:
    return imp.Block(map_terminals(child, function, loops) for child in node.children)


@map_terminals.register
def _(node: imp.For, function: Callable, loops: tuple[imp.For, ...] = ()) -> imp.Node:
    return imp.For(node.index, map_terminals(node.statement, function, (*loops, node)), node.is_parallel)


@map_terminals.register
def _(node: imp.Terminal, function: Callable, loops: tuple[imp.For, ...] = ()) -> imp.Node:
    return function(loops, node)
