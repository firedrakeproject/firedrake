"""The passes that pyop3 applies to a kernel.

* `pyop3.compile.passes.impero`: Impero to Impero passes, before the kernel is lowered to MLIR.
* `pyop3.compile.passes.mlir`: the lowering of the MLIR to the LLVM dialect, with the xDSL passes and `mlir-opt`.
"""

from pyop3.compile.passes.mlir import build_lowering_pipeline, lower, ptxas_logs  # noqa: F401
