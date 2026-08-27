from __future__ import annotations

import abc
import contextlib
import functools
import numbers
import os
from typing import Any

import loopy as lp
import numpy as np
import pymbolic as pym
from petsc4py import PETSc

import pyop3.axis_tree
import pyop3.buffer
import pyop3.cache
import pyop3.config
import pyop3.constants
import pyop3.dtypes
import pyop3.expr
from pyop3 import mpi, utils
from pyop3.axis_tree.tree import (
    UNIT_AXIS_TREE,
    IndexedAxisTree,
)
from pyop3.buffer import (
    AbstractBuffer,
    NullBuffer,
    PetscMatBuffer,
)
from pyop3.constants import INC, MAX_RW, MAX_WRITE, MIN_RW, MIN_WRITE, READ, RW, WRITE
from pyop3.dtypes import IntType
from pyop3.insn.base import (
    Exscan,
    InstructionList,
    Loop,
    NonEmptyArrayAssignment,
    NullInstruction,
    StandaloneCalledFunction,
)

from pyop3.lower.loopy import LoopyCodegenContext

# TODO: import other way around?
from pyop3.lower.transform import (
    with_attach_debugger,
    with_likwid_markers,
    with_petsc_event,
    _collect_temporary_shapes
)

def _compile_static_hashkey(op: PreprocessedOperation, compiler_parameters: ParsedCompilerParameters) -> Hashable:
    return (op.disk_cache_key, compiler_parameters, pyop3.config)

@pyop3.cache.memory_and_disk_cache(
    hashkey=_compile_static_hashkey,
    get_comm=lambda op, *a, **kw: op.comm,
)
def _compile_static(op: InstructionExecutionContext, compiler_parameters: ParsedCompilerParameters) -> tuple:
    """Compile the operation without regard for specific data values.

    This function is therefore suitable for disk caching.
    Function passes compilation process to compiler-specified backend 

    Returns
    -------
    TU
    datamap

    """
    insn = op.preprocess()
    function_name = "pyop3_loop"  # TODO: Provide as kwarg

    if isinstance(insn, InstructionList):
        cs_expr = insn.instructions
    else:
        cs_expr = (insn,)

    if compiler_parameters.backend == "loopy": 
        make_context = LoopyCodegenContext
    elif compiler_parameters.backend == "mlir": 
        raise NotImplementedError("Class is still being implemented.") 

    context = make_context(
        propagate_negatives=compiler_parameters.propagate_negatives,
        mask_array_accesses=compiler_parameters.mask_array_accesses,
    )
    # NOTE: so I think LoopCollection is a better abstraction here - don't want to be
    # explicitly dealing with contexts at this point. Can always sniff them out again.
    # for context, ex in cs_expr:
    for ex in cs_expr:
        # ex = expand_implicit_pack_unpack(ex)

        # add external loop indices as kernel arguments
        # FIXME: removed because cs_expr needs to sniff the context now
        loop_indices = {}

        for e in pyop3.collections.as_tuple(ex): # TODO: get rid of this loop
            # context manager?
            context.set_temporary_shapes(_collect_temporary_shapes(e))
            _compile(e, loop_indices, context)

    if not context.buffer_intents:
        raise pyop3.exceptions.EffectlessComputationException(
            "The generated kernel does not modify any global data, this may indicate that something has gone wrong"
        )

    translation_unit = context.finalize_kernel(function_name, compiler_parameters)

    # Extra information needed by the code executor
    kernel_name_to_buffer_info = utils.invert_mapping(context.kernel_names)
    buffer_intents = context.buffer_intents

    # Replace buffers with their indices, dropping any temporaries. Also
    # match the calling order for the kernel.
    kernel_name_to_global_buffer_info = {}
    global_buffer_intents = {}
    for kernel_arg in translation_unit.default_entrypoint.args:
        buf_view = kernel_name_to_buffer_info[kernel_arg.name]
        buf_index = op.preprocessed_buffers.index(buf_view.buffer)
        kernel_name_to_global_buffer_info[kernel_arg.name] = (buf_index, buf_view.nest_indices)
        global_buffer_intents[buf_index] = buffer_intents[buf_view.buffer]

    return translation_unit, kernel_name_to_global_buffer_info, global_buffer_intents

@functools.singledispatch
def _compile(expr: Any, loop_indices: Dict, codegen_context: CodegenContext) -> None:
    raise TypeError(f"No handler defined for {type(expr).__name__}")

@_compile.register(NullInstruction)
def _(null, *args, **kwargs): 
    pass

@_compile.register(InstructionList)
def _(
    insn_list, 
    loop_indices, 
    codegen_context
) -> None:
    for insn in insn_list: 
        _compile(insn, loop_indices, codegen_context)

@_compile.register(Loop)
def _(
    loop, 
    loop_indices, 
    codegen_context
) -> None:
    codegen_context._parse_loop_properly_this_time(
        loop, 
        loop.index.iterset, 
        loop_indices, 
    )

@_compile.register(StandaloneCalledFunction)
def _(call, loop_indices, codegen_context):
    codegen_context.compile_standalone_function(call, loop_indices)

@_compile.register(NonEmptyArrayAssignment)
def parse_assignment(assignment: NonEmptyArrayAssignment, loop_indices, codegen_context: CodegenContext):
    if any(isinstance(arg, pyop3.expr.MatPetscMatBufferExpression) for arg in assignment.arguments):
        codegen_context.compile_petsc_mat(assignment, loop_indices)
    else:
        codegen_context._compile_array_assignment(
            assignment,
            loop_indices,
            assignment.axis_trees,
        )

@_compile.register(Exscan)
def _(exscan, loop_indices, codegen_context):
    codegen_context.compile_exscan(exscan, loop_indices)
