"""xDSL passes that pyop3 applies to the MLIR that it generates, before and around `mlir-opt`"""

from __future__ import annotations

from xdsl.context import Context
from xdsl.dialects import get_all_dialects

from pyop3.compile.passes.mlir.atomics import RelaxAtomicRMWPass
from pyop3.compile.passes.mlir.thread_loops import DistributeThreadLoopsPass, InsertBarriersPass


def mlir_context() -> Context:
    """Return an xDSL context that can parse the MLIR that pyop3 generates and lowers.

    Every xDSL dialect is registered. Operations that xDSL does not define, such as
    `llvm.mlir.poison` or `llvm.atomicrmw`, are kept as generic operations, so that
    they are printed again without change.
    """
    ctx = Context(allow_unregistered=True)
    for name, factory in get_all_dialects().items():
        ctx.register_dialect(name, factory)
    return ctx


PYOP3_PASSES = {
    pass_type.name: pass_type
    for pass_type in (RelaxAtomicRMWPass, InsertBarriersPass, DistributeThreadLoopsPass)
}
"""The pyop3 xDSL passes by name, for the per-device lists in pyop3.cc.DEVICE_XDSL_PASSES."""
