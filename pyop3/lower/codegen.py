from __future__ import annotations

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

from cachetools import cachedmethod
from petsc4py import PETSc

import loopy as lp
import numpy as np
import pymbolic as pym
from immutabledict import immutabledict as idict

import pyop3.axis_tree
import pyop3.cache
import pyop3.config
import pyop3.dtypes
import pyop3.expr
from pyop3 import utils, mpi
from pyop3.cache import memory_and_disk_cache
from pyop3.expr import NonlinearDatBufferExpression
from pyop3.expr.visitors import collect_axis_vars, replace
from pyop3.axis_tree.tree import UNIT_AXIS_TREE, IndexedAxisTree, AxisComponent, relabel_path
from pyop3.buffer import AbstractBuffer, ConcreteBuffer, PetscMatBuffer, ArrayBuffer, NullBuffer
from pyop3.dtypes import IntType
from pyop3.lower.transform import with_likwid_markers, with_petsc_event, with_attach_debugger
from pyop3.lower.context import CodegenContext
from pyop3.lower.mlir import MLIRCodegenContext # to remove
from pyop3.lower.loopy import LoopyCodegenContext
from pyop3.insn.base import (
    Intent,
    INC,
    MAX_RW,
    MAX_WRITE,
    MIN_RW,
    MIN_WRITE,
    READ,
    RW,
    AbstractAssignment,
    Exscan,
    NullInstruction,
    assignment_type_as_intent,
    WRITE,
    AssignmentType,
    ConcretizedNonEmptyArrayAssignment,
    StandaloneCalledFunction,
    Loop,
    InstructionList,
)
# TODO: import other way around?
from pyop3.insn.exec import parse_compiler_parameters

# def _compile_static_hashkey(op: PreprocessedOperation, compiler_parameters: ParsedCompilerParameters) -> Hashable:
#     # NOTE: is config valid to include here?
#     return (op.disk_cache_key, compiler_parameters, pyop3.config)

# @pyop3.cache.memory_and_disk_cache(
#     hashkey=_compile_static_hashkey,
#     get_comm=lambda op, *args, **kwargs: op.comm,
# )
def _compile_static(op, compiler_parameters) -> Tuple:
    """Compile the operation without regard for specific data values.

    This function is therefore suitable for disk caching.

    Returns
    -------
    TU
    datamap

    """

    insn = op.preprocess()
    function_name = "pyop3_loop"

    if isinstance(insn, InstructionList):
        cs_expr = insn.instructions
    else:
        cs_expr = (insn,)

    # Default to loopy codegen backend
    target = getattr(compiler_parameters, 'codegen', 'loopy')
    if target == "loopy":
        codegen_context = LoopyCodegenContext(check_negatives=compiler_parameters.check_negatives)
    elif target == "mlir":
        codegen_context = MLIRCodegenContext(check_negatives=compiler_parameters.check_negatives)
    
    # NOTE: so I think LoopCollection is a better abstraction here - don't want to be
    # explicitly dealing with codegen_contexts at this point. Can always sniff them out again.
    # for codegen_context, ex in cs_expr:
    for ex in cs_expr:
        # ex = expand_implicit_pack_unpack(ex)

        # add external loop indices as kernel arguments
        # FIXME: removed because cs_expr needs to sniff the codegen_context now
        loop_indices = {}

        for e in utils.as_tuple(ex): # TODO: get rid of this loop
            # codegen_context manager?
            codegen_context.set_temporary_shapes(_collect_temporary_shapes(e))
            _compile(e, loop_indices, codegen_context)

    if not codegen_context.global_buffers:
        import pyop3.exceptions
        raise pyop3.exceptions.EffectlessComputationException(
            "The generated kernel does not modify any global data, this may indicate that something has gone wrong"
        )

    translation_unit, final_context = codegen_context.finalize_kernel(function_name, compiler_parameters)
    
    kernel_to_buffer_names = utils.invert_mapping(final_context._kernel_names)
    buffer_index_map = {}
    for kernel_arg in translation_unit.default_entrypoint.args:
        buffer_key = kernel_to_buffer_names[kernel_arg.name]
        buffer_ref = final_context.global_buffers[buffer_key]
        buffer_index = op.preprocessed_buffers.index(buffer_ref)
        intent = final_context.global_buffer_intents[buffer_key]
        buffer_index_map[kernel_arg.name] = (buffer_index, buffer_ref.nest_indices, intent)
            
    return translation_unit, buffer_index_map

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
def _(assignment):
    return idict()

@_collect_temporary_shapes.register(StandaloneCalledFunction)
def _(call):
    import loopy as lp # TODO: Remove once StandaloneCalledFunction integrated with MLIR 
    return idict(
        {
            (arg.buffer.name, arg.buffer.nest_indices): lp_arg.shape
            for lp_arg, arg in zip(
                call.function.code.default_entrypoint.args, call.arguments, strict=True
            )
            if isinstance(lp_arg, lp.ArrayArg)
        }
    )


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
    _parse_loop_properly_this_time(
        loop, 
        loop.index.iterset, 
        loop_indices, 
        codegen_context
    )

def _parse_loop_properly_this_time(
    loop, 
    axis_tree, 
    loop_indices, 
    codegen_context, 
    axis=None, 
    path=None, 
    iname_map=None
) -> None:
    if axis_tree is UNIT_AXIS_TREE:
        for stmt in loop.statements: 
            _compile(
                stmt, 
                loop_indices, 
                codegen_context
            )
        return

    if utils.strictly_all(x is None for x in {axis, path, iname_map}):
        axis = axis_tree.root
        path = idict()
        iname_map = idict()

    for component in axis.components:
        path_ = path | {axis.label: component.label}
        if axis_tree.linearize(path_, partial=True).size == 0: continue
        
        if component.local_size != 1:
            iname = codegen_context.unique_name("i")
            domain_var = codegen_context.register_extent(component.local_size, iname_map, loop_indices)
            codegen_context.add_domain(iname, domain_var)
            iname_replace_map_ = iname_map | {axis.label: pym.var(iname)}
            within = frozenset({iname})
        else:
            iname_replace_map_ = iname_map | {axis.label: 0}
            within = set()

        with codegen_context.within_inames(within):
            if subaxis := axis_tree.node_map[path_]:
                _parse_loop_properly_this_time(loop, axis_tree, loop_indices, codegen_context, axis=subaxis, path=path_, iname_map=iname_replace_map_)
            else:
                loop_indices |= idict({
                    (loop.index.id, axis_label): iname
                    for axis_label, iname in iname_replace_map_.items()
                })
                for stmt in loop.statements: _compile(stmt, loop_indices, codegen_context)

@_compile.register(StandaloneCalledFunction)
def _(call, loop_indices, codegen_context):
    codegen_context.compile_standalone_function(call, loop_indices)

@_compile.register(ConcretizedNonEmptyArrayAssignment)
def _(assignment, loop_indices, codegen_context):
    if any(isinstance(arg, pyop3.expr.MatPetscMatBufferExpression) for arg in assignment.arguments):
        codegen_context.compile_petsc_mat(assignment, loop_indices)
    else:
        _compile_array_assignment(assignment, loop_indices, codegen_context, assignment.axis_trees)

def _compile_array_assignment(
        assignment, 
        loop_indices, 
        codegen_context, 
        axis_trees, 
        *,
        iname_replace_maps=None, 
        # TODO document these under "Other Parameters"
        axis_tree=None, 
        paths=None
):
    if paths is None:
        paths = []
    if iname_replace_maps is None: 
        iname_replace_maps = []

    if axis_tree is None:
        axis_tree, *axis_trees = axis_trees

        paths += [idict()]
        iname_replace_maps += [idict()]
        
        if axis_tree.is_empty or axis_tree is UNIT_AXIS_TREE or isinstance(axis_tree, IndexedAxisTree):
            if axis_trees: 
                raise NotImplementedError("Refactor needed")

            codegen_context.add_leaf_assignment(
                assignment, 
                paths, 
                iname_replace_maps, 
                loop_indices
            )
            return

    axis = axis_tree.node_map[paths[-1]]
    for component in axis.components:
        new_paths = paths.copy()
        new_paths[-1] = paths[-1] | {axis.label: component.label}
        
        if axis_tree.linearize(new_paths[-1], partial=True).size == 0: 
            continue
        
        if component.local_size != 1:
            iname = codegen_context.unique_name("i")
            ext = codegen_context.register_extent(
                component.local_size, 
                iname_replace_maps[-1], 
                loop_indices
            )
            codegen_context.add_domain(iname, ext)
            new_maps = iname_replace_maps.copy()
            new_maps[-1] = iname_replace_maps[-1] | {axis.label: pym.var(iname)}
            within_inames = {iname}
        else:
            new_maps = iname_replace_maps.copy()
            new_maps[-1] = iname_replace_maps[-1] | {axis.label: 0}
            within_inames = set()

        with codegen_context.within_inames(within_inames):
            if axis_tree.node_map[new_paths[-1]]:
                _compile_array_assignment(
                    assignment, 
                    loop_indices, 
                    codegen_context, 
                    axis_trees, 
                    iname_replace_maps=new_maps, 
                    axis_tree=axis_tree, 
                    paths=new_paths
                )
            elif axis_trees:
                _compile_array_assignment(
                    assignment, 
                    loop_indices, 
                    codegen_context, 
                    axis_trees, 
                    iname_replace_maps=new_maps, 
                    axis_tree=None, 
                    paths=new_paths
                )
            else:
                codegen_context.add_leaf_assignment(
                    assignment, 
                    new_paths, 
                    new_maps, 
                    loop_indices
                )

@_compile.register(Exscan)
def _(exscan, loop_indices, codegen_context):
    codegen_context.compile_exscan(exscan, loop_indices)
