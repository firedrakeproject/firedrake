"""The lowering of the MLIR that pyop3 generates to the LLVM dialect.

The pyop3 xDSL passes of the device run first, then `mlir-opt` runs the
`mlir-opt` passes of the device.
"""

from __future__ import annotations

import re
import subprocess
from io import StringIO

from xdsl.context import Context
from xdsl.dialects import get_all_dialects
from xdsl.parser import Parser
from xdsl.passes import PassPipeline
from xdsl.printer import Printer
from xdsl.transforms.mlir_opt import MLIROptPass
from xdsl.utils.exceptions import DiagnosticException, ParseError

import pyop3.config
from pyop3.compile.passes.mlir.atomics import RelaxAtomicRMWPass
from pyop3.compile.passes.mlir.gpu_ext import AddressSpaceAttr, DynamicSharedMemoryOp
from pyop3.compile.passes.mlir.thread_loops import LowerThreadLoopsPass
from pyop3.device import AMDGPU, CPU, CUDAGPU, Device
from pyop3.exceptions import CompilationException


def _gpu_mlir_opt_passes(attach_target: str, *convert_gpu: str) -> tuple[str, ...]:
    """Return the `mlir-opt` passes of a GPU, given the passes of its vendor.

    Parameters
    ----------
    attach_target
        The pass that attaches the target (the chip) to the GPU modules.
    convert_gpu
        The passes that lower the `gpu` dialect to the LLVM dialect of the vendor.
    """
    return (
        "--canonicalize",
        "--cse",
        "--gpu-launch-sink-index-computations",
        "--gpu-kernel-outlining",
        attach_target,
        "--convert-scf-to-cf",
        "--expand-strided-metadata",
        "--lower-affine",
        *convert_gpu,
        "--convert-index-to-llvm",
        "--convert-arith-to-llvm",
        "--convert-cf-to-llvm",
        "--finalize-memref-to-llvm",
        "--convert-func-to-llvm",
        "--gpu-to-llvm",
        "--reconcile-unrealized-casts",
        "--gpu-module-to-binary",
    )


MLIR_OPT_PASSES = {
    CPU: (
        "--canonicalize",
        "--cse",
        "--symbol-dce",
        "--buffer-deallocation-pipeline",
        "--scf-parallel-for-to-nested-fors",
        "--buffer-loop-hoisting",
        "--convert-scf-to-cf",
        "--canonicalize",
        "--convert-math-to-llvm",
        "--convert-math-to-libm",
        "--convert-cf-to-llvm",
        "--convert-func-to-llvm",
        "--finalize-memref-to-llvm",
        "--convert-arith-to-llvm",
        "--convert-index-to-llvm",
        "--reconcile-unrealized-casts",
    ),
    CUDAGPU: _gpu_mlir_opt_passes(
        "--nvvm-attach-target=chip={arch} features=+ptx80 O=3", "--convert-gpu-to-nvvm", "--convert-nvvm-to-llvm",
    ),
    AMDGPU: _gpu_mlir_opt_passes(
        "--rocdl-attach-target=chip={arch} O=3", "--convert-gpu-to-rocdl=chipset={arch} runtime=HIP",
    ),
}
"""The `mlir-opt` passes of each device. `{arch}` is replaced with the architecture of the device."""

# The CPU passes run --buffer-deallocation-pipeline, which rejects llvm.atomicrmw because
# that operation does not declare its memory effects. So the CPU kernels keep the
# standard lowering of their atomics.
XDSL_PASSES = {
    CPU: (),
    CUDAGPU: (LowerThreadLoopsPass(), RelaxAtomicRMWPass()),
    AMDGPU: (LowerThreadLoopsPass(), RelaxAtomicRMWPass(syncscope="agent", no_fine_grained_memory=True)),
}
"""The pyop3 xDSL passes of each device, which run before `mlir-opt`."""


def mlir_context() -> Context:
    """Return an xDSL context that can parse the MLIR that pyop3 generates and lowers.

    Every xDSL dialect is registered, and the `gpu` additions of
    `pyop3.compile.passes.mlir.gpu_ext`. Operations that xDSL does not define, such as
    `llvm.mlir.poison` or `llvm.atomicrmw`, are kept as generic operations, so that
    they are printed again without change.
    """
    ctx = Context(allow_unregistered=True)
    for name, factory in get_all_dialects().items():
        ctx.register_dialect(name, factory)
    ctx.load_op(DynamicSharedMemoryOp)
    ctx.load_attr_or_type(AddressSpaceAttr)
    return ctx


def build_lowering_pipeline(device: Device, mlir_opt: str) -> PassPipeline:
    """Return the pass pipeline that lowers the MLIR of a kernel for `device` to the LLVM dialect.

    The passes are frozen dataclasses, so the passes of the pipeline can be part of a cache key.

    Parameters
    ----------
    device
        The device that the kernel runs on.
    mlir_opt
        The `mlir-opt` executable.

    Returns
    -------
    PassPipeline
        The xDSL passes of `device`, then an `mlir-opt` pass. `mlir-opt` prints its
        output in the generic form, so that xDSL can parse it again.
    """
    flags = (flag.format(arch=device.arch) for flag in MLIR_OPT_PASSES[type(device)])
    if pyop3.config.ptxas_verbose:
        # mlir-opt then stores the output of the PTX compiler in the binary, where ptxas_logs finds it
        flags = (
            f"{flag} collect-compiler-diagnostics ptxas-cmd-options=-v" if flag.startswith("--nvvm-attach-target") else flag
            for flag in flags
        )
    arguments = (*flags, *getattr(pyop3.config, "extra_mlir_opt_flags", ()), "--mlir-print-op-generic")
    return PassPipeline((*XDSL_PASSES[type(device)], MLIROptPass(executable=mlir_opt, arguments=arguments)))


def lower(code: str, pipeline: PassPipeline) -> str:
    """Return the MLIR `code` after the passes of `pipeline`, in the generic form.

    Raises
    ------
    subprocess.CalledProcessError
        If `mlir-opt` fails. Its error output is in the `stderr` of the exception.
    CompilationException
        If xDSL cannot parse or rewrite the module.
    """
    ctx = mlir_context()
    try:
        module = Parser(ctx, code).parse_module()
        pipeline.apply(ctx, module)
    except (DiagnosticException, ParseError) as e:
        if isinstance(e.__cause__, subprocess.CalledProcessError):
            raise e.__cause__
        raise CompilationException(str(e)) from e
    lowered = StringIO()
    Printer(print_generic_format=True, stream=lowered).print_op(module)
    return lowered.getvalue()


def ptxas_logs(lowered: str) -> list[str]:
    """Return the output of the PTX compiler that `mlir-opt` stores in `lowered` if `pyop3.config.ptxas_verbose` is set."""
    return [
        re.sub(r"\\([0-9A-Fa-f]{2})", lambda m: chr(int(m[1], 16)), log)
        for log in re.findall(r'ISACompilerLog = "((?:[^"\\]|\\.)*)"', lowered)
    ]
