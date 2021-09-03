from gem.node import MemoizerArg
from functools import singledispatch
from itertools import repeat
import firedrake.slate.slate as sl


def optimise(expression):
    """Optimises a Slate expression, e.g. by pushing blocks inside the expression.

    :arg expression: A (potentially unoptimised) Slate expression.

    Returns: An optimised Slate expression
    """
    return push_block(expression)


def push_block(expression):
    """Executes a Slate compiler optimisation pass.
    The optimisation is achieved by pushing blocks from the outside to the inside of an expression.
    Without the optimisation the local TSFC kernels are assembled first
    and then the result of the assembly kernel gets indexed in the Slate kernel
    (and further linear algebra operations maybe done on it).
    The optimisation pass essentially changes the order of assembly and indexing.

    :arg expression: A (potentially unoptimised) Slate expression.

    Returns: An optimised Slate expression, where Blocks are terminal whereever possible.
    """
    mapper = MemoizerArg(_push_block)
    ret = mapper(expression, ())
    return ret


@singledispatch
def _push_block(expr, self, indices):
    raise AssertionError("Cannot handle terminal type: %s" % type(expr))


@_push_block.register(sl.Transpose)
def _push_block_transpose(expr, self, indices):
    """Indices of the Blocks are transposed if Block is pushed into a Transpose."""
    return sl.Transpose(*map(self, expr.children, repeat(indices[::-1]))) if indices else expr


@_push_block.register(sl.Add)
@_push_block.register(sl.Negative)
def _push_block_distributive(expr, self, indices):
    """Distributes Blocks for these nodes"""
    return type(expr)(*map(self, expr.children, repeat(indices))) if indices else expr


@_push_block.register(sl.Factorization)
@_push_block.register(sl.Inverse)
@_push_block.register(sl.Solve)
@_push_block.register(sl.Mul)
def _push_block_stop(expr, self, indices):
    """Blocks cannot be pushed further into this set of nodes."""
    return sl.Block(expr, indices) if indices else expr


@_push_block.register(sl.Tensor)
def _push_block_tensor(expr, self, indices):
    """Turns a Block on a Tensor into a Tensor of an indexed form."""
    return sl.Tensor(sl.Block(expr, indices).form) if indices else expr


@_push_block.register(sl.AssembledVector)
def _push_block_assembled_vector(expr, self, indices):
    """Turns a Block on an AssembledVector into the  specialized node BlockAssembledVector."""
    return sl.BlockAssembledVector(expr._function, sl.Block(expr, indices).form, indices) if indices else expr


@_push_block.register(sl.Block)
def _push_block_block(expr, self, indices):
    """Inlines Blocks into each other.
    Note that the indices are inlined from the ouside.
    Example: If we have got the Slate expression A.blocks[:3,:3].blocks[0,0], we encounter the (0,0)-blocks first.
    The first time round the indices are empty and the (0,0) are in expr._indices.
    The second time round, (0,0) is stored in indices and the slices are in expr._indices.
    So in the following line we basically say indices = ((0,1,2)[0], (0,1,2)[0])
    """
    reindexed = tuple(big[slice(small[0], small[-1]+1)] for big, small in zip(expr._indices, indices))
    indices = expr._indices if not indices else reindexed
    block, = map(self, expr.children, repeat(indices))
    return block
