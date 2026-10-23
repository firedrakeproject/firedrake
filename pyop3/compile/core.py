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
from immutabledict import immutabledict as idict
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
    AbstractAssignment,
    AssignmentType,
    Exscan,
    InstructionList,
    Loop,
    NonEmptyArrayAssignment,
    NullInstruction,
    StandaloneCalledFunction,
    assignment_type_as_intent,
)


def _compile_static_hashkey(op: PreprocessedOperation, compiler_parameters: ParsedCompilerParameters, cc_options) -> Hashable:
    return (op.disk_cache_key, compiler_parameters, cc_options, pyop3.config)

# FIXME: can't currently pickle things
# @pyop3.cache.memory_and_disk_cache(
@pyop3.cache.memory_cache(
    hashkey=_compile_static_hashkey,
    get_comm=lambda op, *a, **kw: op.comm,
)
# TODO: put cc_options inside compiler_parameters
def _compile_static(op: InstructionExecutionContext, compiler_parameters: ParsedCompilerParameters, cc_options) -> tuple:
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
        from pyop3.compile.loopy import LoopyCompiler

        compiler = LoopyCompiler(
            named_terminal_buffer_intents=op.named_terminal_buffer_intents,
            comm=op.comm,
            propagate_negatives=compiler_parameters.propagate_negatives,
            mask_array_accesses=compiler_parameters.mask_array_accesses,
        )
        # NOTE: so I think LoopCollection is a better abstraction here - don't want to be
        # explicitly dealing with contexts at this point. Can always sniff them out again.
        # for context, ex in cs_expr:
        for ex in cs_expr:
            for e in pyop3.collections.as_tuple(ex): # TODO: get rid of this loop
                # context manager?
                compiler.set_temporary_shapes(_collect_temporary_shapes(e))
                compiler(e)

    else:
        from pyop3.compile.gem import GemCompiler

        assert compiler_parameters.backend == "gem"
        compiler = GemCompiler(
            named_terminal_buffer_intents=op.named_terminal_buffer_intents,
            comm=op.comm,
            propagate_negatives=compiler_parameters.propagate_negatives,
            mask_array_accesses=compiler_parameters.mask_array_accesses,
        )
        # TODO: This should really just live inside GemCompiler
        for ex in cs_expr:
            for e in pyop3.collections.as_tuple(ex): # TODO: get rid of this loop

                compiler.set_temporary_shapes(_collect_temporary_shapes(e))
                # TODO: have a top-level .call() or similar. There is a difference in what
                # we want to do at the top-level vs recursively
                compiler(e)

    executable, arguments, buffer_intents = compiler.finalize_kernel(function_name, compiler_parameters, cc_options)
    del compiler  # the compiler is done, don't touch it again

    # Replace buffers with their indices, dropping any temporaries. Also
    # match the calling order for the kernel.
    # NOTE: The kernel_name_to_buffer_info attr can be figured out by the context at finalisation
    kernel_name_to_buffer_info = {}
    buffer_intents_by_index = {}
    for kernel_arg_name in arguments.keys():
        # hack
        if kernel_arg_name.startswith("t_"):
            continue
        buf_view = arguments[kernel_arg_name]
        buf_index = op.preprocessed_buffers.index(buf_view.buffer)

        kernel_name_to_buffer_info[kernel_arg_name] = (buf_index, buf_view.nest_indices)
        buffer_intents_by_index[buf_index] = buffer_intents[buf_view.buffer]

    return executable, kernel_name_to_buffer_info, buffer_intents_by_index




# NOTE: Make this overloaded function into class?
@functools.singledispatch
def _collect_temporary_shapes(expr):
    raise TypeError(f"No handler defined for {type(expr).__name__}")

@_collect_temporary_shapes.register(InstructionList)
def _(insn_list):
    return utils.merge_dicts(_collect_temporary_shapes(insn) for insn in insn_list)

@_collect_temporary_shapes.register(Loop)
def _(loop):
    shapes = {}
    for stmt in loop.statements:
        for temp, shape in _collect_temporary_shapes(stmt).items():
            if shape is None:
                continue
            if temp in shapes:
                assert shapes[temp] == shape
            else:
                shapes[temp] = shape
    return shapes

@_collect_temporary_shapes.register(AbstractAssignment)
@_collect_temporary_shapes.register(NullInstruction)
@_collect_temporary_shapes.register(Exscan)
def _(assignment: AbstractAssignment, /) -> idict:
    return idict()

@_collect_temporary_shapes.register
def _(call: StandaloneCalledFunction):
    import loopy as lp

    if isinstance(call.function.code, lp.TranslationUnit):
        return idict(
            {
                arg.buffer: lp_arg.shape
                for lp_arg, arg in zip(
                    call.function.code.default_entrypoint.args, call.arguments, strict=True
                )
                if isinstance(lp_arg, lp.ArrayArg)
            }
        )
    else:
        return idict()

def _petsc_mat_load(assignment, mat_name, array_name, nrow, ncol, irow, icol, blocked):
    if blocked:
        return f"MatGetValuesBlockedLocal({mat_name}, {nrow}, &({irow}), {ncol}, &({icol}), &({array_name}[0]));"
    else:
        return f"MatGetValuesLocal({mat_name}, {nrow}, &({irow}), {ncol}, &({icol}), &({array_name}[0]));"


def _petsc_mat_store(assignment, mat_name, array_name, nrow, ncol, irow, icol, blocked):
    if blocked:
        return f"MatSetValuesBlockedLocal({mat_name}, {nrow}, &({irow}), {ncol}, &({icol}), &({array_name}[0]), INSERT_VALUES);"
    else:
        return f"MatSetValuesLocal({mat_name}, {nrow}, &({irow}), {ncol}, &({icol}), &({array_name}[0]), INSERT_VALUES);"


def _petsc_mat_add(assignment, mat_name, array_name, nrow, ncol, irow, icol, blocked):
    if blocked:
        return f"MatSetValuesBlockedLocal({mat_name}, {nrow}, &({irow}), {ncol}, &({icol}), &({array_name}[0]), ADD_VALUES);"
    else:
        return f"MatSetValuesLocal({mat_name}, {nrow}, &({irow}), {ncol}, &({icol}), &({array_name}[0]), ADD_VALUES);"
