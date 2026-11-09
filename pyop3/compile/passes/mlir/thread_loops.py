"""An xDSL pass that lowers the thread loops of a GPU kernel.

A thread loop is an `scf.parallel` that is marked with `THREAD_LOOP`, directly
in the `gpu.launch` of its batch. Its upper bounds are constants. An SCPT kernel
has one thread loop, over the items of its batch. A thread-transposed kernel has
one per phase, over the batch size then the extents of the phase domain.
`LowerThreadLoopsPass`, in each `gpu.launch`:

* puts a `gpu.barrier` before each thread loop that has a RAW, WAR or WAW
  dependence on workgroup memory since the last barrier, then

The barriers come first because the thread loops are found by a marking attribute,
(see THREAD_LOOP) which the distribution removes. The pass changes only the marked loops, so it
has no effect on other kernels.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from xdsl.context import Context
from xdsl.dialects import arith, gpu, memref, scf
from xdsl.dialects.builtin import (
    DenseArrayBase, IndexType, IntegerAttr, MemRefType, ModuleOp, UnregisteredOp, i64,
)
from xdsl.ir import Block, Operation, Region, SSAValue
from xdsl.passes import ModulePass
from xdsl.rewriter import InsertPoint, Rewriter
from xdsl.transforms.canonicalization_patterns.utils import const_evaluate_operand

THREAD_LOOP = "pyop3.thread_loop"
"""The attribute that marks an `scf.parallel` as a thread loop."""

WORKGROUP_ADDRESS_SPACE = IntegerAttr(3, i64)
"""The address space of shared (workgroup) memory, in both NVVM and ROCDL."""


def thread_loops(launch: gpu.LaunchOp) -> list[scf.ParallelOp]:
    """Return the thread loops in the body of `launch`, in order."""
    return [op for op in launch.body.block.ops if isinstance(op, scf.ParallelOp) and THREAD_LOOP in op.attributes]


@dataclasses.dataclass(frozen=True)
class LowerThreadLoopsPass(ModulePass):
    """Insert the barriers between the thread loops of each `gpu.launch`, then distribute the loops over its threads.

    The block size of the `gpu.launch` and the upper bounds of the thread
    loops must be constants.
    """

    name = "pyop3-lower-thread-loops"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        for launch in [o for o in op.walk() if isinstance(o, gpu.LaunchOp)]:
            if not (loops := thread_loops(launch)):
                continue
            _insert_barriers(loops)
            block_size = _constant(launch.blockSizeX)
            thread_id = launch.body.block.args[3]
            for loop in loops:
                _distribute(loop, thread_id, block_size)


def _insert_barriers(loops: list[scf.ParallelOp]) -> None:
    """Insert a `gpu.barrier` before each of `loops` that depends on workgroup memory of an earlier loop.

    A dependence is a read of a memref that an earlier loop wrote (RAW), or a
    write to a memref that an earlier loop read or wrote (WAR, WAW). Only the
    loops after the last barrier are earlier loops.
    """
    written, read = set(), set()
    for loop in loops:
        reads, writes = _workgroup_effects(loop)
        # If RAW or WA(R/W) then we place barrier
        if reads & written or writes & (written | read):
            Rewriter.insert_op(gpu.BarrierOp(), InsertPoint.before(loop))
            written, read = set(), set()
        written |= writes
        read |= reads


def _workgroup_effects(loop: Operation) -> tuple[set[SSAValue], set[SSAValue]]:
    """Return the workgroup memrefs that `loop` reads and writes."""
    reads, writes = set(), set()
    for op in loop.walk():
        if isinstance(op, (memref.LoadOp, memref.StoreOp, memref.AtomicRMWOp)) and _is_workgroup(op.memref):
            if not isinstance(op, memref.StoreOp):
                reads.add(op.memref)
            if not isinstance(op, memref.LoadOp):
                writes.add(op.memref)
    return reads, writes


def _is_workgroup(value: SSAValue) -> bool:
    # Check if memref and using shared memory
    return isinstance(value.type, MemRefType) and value.type.memory_space == WORKGROUP_ADDRESS_SPACE


def _distribute(loop: scf.ParallelOp, thread_id: SSAValue, block_size: int) -> None:
    """Replace `loop` with an `scf.for` over its flat points, strided by `block_size`.

    This is the component where distribute threads for thread transposition
    """
    extents = [_constant(ub) for ub in loop.upperBound]
    num_points, step = _index_constant(int(np.prod(extents))), _index_constant(block_size)
    body = Block(arg_types=[IndexType()])
    Rewriter.insert_op([num_points, step, scf.ForOp(thread_id, num_points, step, [], Region(body))], InsertPoint.before(loop))

    # Without an outer bound, so the first index is the quotient by the other extents
    split = UnregisteredOp.with_name("affine.delinearize_index").create(
        operands=body.args,
        result_types=[IndexType()] * len(extents),
        properties={"static_basis": DenseArrayBase.from_list(i64, extents[1:])},
    )
    body.add_op(split)

    Rewriter.inline_block(loop.body.block, InsertPoint.at_end(body), split.results)
    # Always assuming empty scf.reduce. This may lead to a bug in future.
    Rewriter.erase_op(body.last_op)   
    body.add_op(scf.YieldOp())
    Rewriter.erase_op(loop)

# Also have a similar function used in mlir.py but I don't know how to deal with the separation
def _index_constant(value: int) -> arith.ConstantOp:
    return arith.ConstantOp(IntegerAttr(value, IndexType()))


def _constant(value: SSAValue) -> int:
    # xDSL helper to obtain the integer of an SSA value 
    if (constant := const_evaluate_operand(value)) is None:
        raise ValueError(f"Expected a constant, got {value}")
    return constant
