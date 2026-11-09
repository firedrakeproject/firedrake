"""An xDSL pass that changes the ordering of atomic adds on GPU

Monotonic means that the ordering is not consistent but we can guarantee
that an operations to the same address will not overlap
"""

from __future__ import annotations

import dataclasses

from xdsl.context import Context
from xdsl.dialects import arith, llvm, memref
from xdsl.dialects.builtin import (
    DenseArrayBase,
    IntegerAttr,
    ModuleOp,
    NoneAttr,
    StringAttr,
    UnrealizedConversionCastOp,
    UnitAttr,
    UnregisteredOp,
    i64,
)
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

# The values of `memref.atomic_rmw`'s `kind` property that add, from MLIR's `arith.AtomicRMWKind`.
# They are the same values that `MLIRBuilder.process_assignment` emits.
ATOMIC_RMW_KIND_ADDF = 0
ATOMIC_RMW_KIND_ADDI = 1

# The values of the `bin_op` and `ordering` properties of `llvm.atomicrmw`, from LLVM's
# `AtomicBinOp` and `AtomicOrdering` enumerations. "Monotonic" is LLVM's name for the
# relaxed ordering of C++.
LLVM_ATOMIC_BIN_OP = {ATOMIC_RMW_KIND_ADDF: 11, ATOMIC_RMW_KIND_ADDI: 1}
LLVM_ATOMIC_ORDERING_MONOTONIC = 2


@dataclasses.dataclass(frozen=True)
class RelaxAtomicRMW(RewritePattern):
    """Replace an atomic add into a memref with a relaxed atomic add.

    The standard lowering of `memref.atomic_rmw` gives an `llvm.atomicrmw` with the
    acquire-release ordering. This is very slow and we can (safely) obtain better performance
    with relaxed (LLVM's monotonic) ordering.

    This pattern makes the `llvm.atomicrmw` itself, with the
    monotonic (relaxed) ordering. It takes the pointer from the memref descriptor, as the
    standard lowering does. 

    The conversion casts are removed by `--reconcile-unrealized-casts` 
    and the backend still knows where the memory is.

    Only adds into a rank-1 memref with the identity layout are rewritten, which is how
    pyop3 passes arrays to kernels. Other atomics are left for the standard lowering.

    `syncscope` is the LLVM synchronisation scope, or the system scope if it is empty.
    `no_fine_grained_memory` is additional property that we have to add for AMD GPU
    to promise that the memory is not fine-grained.
    """

    syncscope: str = ""
    no_fine_grained_memory: bool = False

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.AtomicRMWOp, rewriter: PatternRewriter) -> None:
        kind = op.kind.value.data
        memref_type = op.memref.type
        if (
            kind not in LLVM_ATOMIC_BIN_OP
            or memref_type.get_num_dims() != 1
            or not isinstance(memref_type.layout, NoneAttr)
        ):
            return

        # The descriptor of a rank-1 memref is (allocated pointer, aligned pointer,
        # offset, sizes, strides).
        size_array = llvm.LLVMArrayType(1, i64)
        descriptor_type = llvm.LLVMStructType.from_type_list(
            [llvm.LLVMPointerType(), llvm.LLVMPointerType(), i64, size_array, size_array]
        )
        descriptor = UnrealizedConversionCastOp.get([op.memref], [descriptor_type])
        aligned = llvm.ExtractValueOp(
            DenseArrayBase.from_list(i64, [1]), descriptor.results[0], llvm.LLVMPointerType()
        )
        index = arith.IndexCastOp(op.indices[0], i64)
        address = llvm.GEPOp.from_mixed_indices(
            aligned.results[0], [index.result], pointee_type=op.value.type
        )
        properties = {
            "bin_op": IntegerAttr(LLVM_ATOMIC_BIN_OP[kind], i64),
            "ordering": IntegerAttr(LLVM_ATOMIC_ORDERING_MONOTONIC, i64),
        }
        if self.syncscope:
            properties["syncscope"] = StringAttr(self.syncscope)
        atomic = UnregisteredOp.with_name("llvm.atomicrmw").create(
            operands=[address.result, op.value],
            result_types=[op.value.type],
            properties=properties,
            attributes={"rocdl.no_fine_grained_memory": UnitAttr()} if self.no_fine_grained_memory else {},
        )
        # The atomic is the last new operation, so its result replaces the old result.
        rewriter.replace(op, [descriptor, aligned, index, address, atomic])


@dataclasses.dataclass(frozen=True)
class RelaxAtomicRMWPass(ModulePass):
    """Make the atomic adds into memrefs relaxed instead of acquire-release.

    pyop3 kernels use atomic adds to increment their output from many threads. A
    relaxed atomic is enough for this. The kernel only increments the output, nothing
    reads the output while the kernel runs, and the end of the kernel orders the writes
    before any later read.
    """

    name = "pyop3-relax-atomicrmw"

    syncscope: str = ""
    no_fine_grained_memory: bool = False

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        pattern = RelaxAtomicRMW(self.syncscope, self.no_fine_grained_memory)
        PatternRewriteWalker(pattern).rewrite_module(op)
