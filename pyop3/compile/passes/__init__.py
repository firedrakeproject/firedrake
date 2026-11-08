"""The passes that pyop3 applies to a kernel.

* `pyop3.compile.passes.impero`: Impero to Impero passes, before the kernel is lowered to MLIR.
* `pyop3.compile.passes.mlir`: xDSL passes on the MLIR, before and around `mlir-opt`.
"""
