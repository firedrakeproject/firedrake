import abc
import collections
import contextlib
import ctypes
import dataclasses
import enum
import functools
import os
import numbers
import textwrap
import warnings
import weakref
from collections.abc import Mapping
from functools import cached_property
from typing import Any
from weakref import WeakValueDictionary

# NOTE: Some of this code is not specific to loopy, could be refactored
# This is generally a bit nasty and abstraction breaking because it relies on attrs
# of the InstructionExecutionContext
@pyop3.cache.memory_and_disk_cache(
    hashkey=_compile_static_hashkey,
    get_comm=lambda op, *args, **kwargs: op.comm,
)
def _compile_static(op: InstructionExecutionContext, compiler_parameters: ParsedCompilerParameters) -> tuple:
    """Compile the operation without regard for specific data values.

    This function is therefore suitable for disk caching.

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

    context = LoopyCodegenContext(check_negatives=compiler_parameters.check_negatives)
    # NOTE: so I think LoopCollection is a better abstraction here - don't want to be
    # explicitly dealing with contexts at this point. Can always sniff them out again.
    # for context, ex in cs_expr:
    for ex in cs_expr:
        # ex = expand_implicit_pack_unpack(ex)

        # add external loop indices as kernel arguments
        # FIXME: removed because cs_expr needs to sniff the context now
        loop_indices = {}

        for e in utils.as_tuple(ex): # TODO: get rid of this loop
            # context manager?
            context.set_temporary_shapes(_collect_temporary_shapes(e))
            _compile(e, loop_indices, context)

    if not context.global_buffers:
        raise pyop3.exceptions.EffectlessComputationException(
            "The generated kernel does not modify any global data, this may indicate that something has gone wrong"
        )

    # add a no-op instruction touching all of the kernel arguments so they are
    # not silently dropped
    noop = lp.CInstruction(
        (),
        "",
        read_variables=frozenset({a.name for a in context.arguments}),
        within_inames=frozenset(),
        within_inames_is_final=True,
        depends_on=context._depends_on,
    )
    context._instructions.append(noop)

    preambles = [
        ("20_debug", "#include <stdio.h>"),  # dont always inject
        ("30_petsc", "#include <petsc.h>"),  # perhaps only if petsc callable used?
    ]

    translation_unit = lp.make_kernel(
        context.domains,
        context.instructions,
        context.arguments,
        name=function_name,
        target=LOOPY_TARGET,
        lang_version=LOOPY_LANG_VERSION,
        preambles=preambles,
    )
    translation_unit = lp.merge((translation_unit, *context.subkernels))

    entrypoint = translation_unit.default_entrypoint
    if compiler_parameters.add_likwid_markers:
        entrypoint = with_likwid_markers(entrypoint)
    if compiler_parameters.add_petsc_event:
        entrypoint = with_petsc_event(entrypoint)
    if compiler_parameters.attach_debugger:
        entrypoint = with_attach_debugger(entrypoint)
    translation_unit = translation_unit.with_kernel(entrypoint)

    kernel_to_buffer_names = utils.invert_mapping(context._kernel_names)
    buffer_index_map = {}
    for kernel_arg in entrypoint.args:
        buffer_key = kernel_to_buffer_names[kernel_arg.name]
        buffer_ref = context.global_buffers[buffer_key]
        buffer_index = op.preprocessed_buffers.index(buffer_ref)
        intent = context.global_buffer_intents[buffer_key]
        buffer_index_map[kernel_arg.name] = (buffer_index, buffer_ref.nest_indices, intent)

    return translation_unit, buffer_index_map
