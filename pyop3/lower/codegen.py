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

from pyop3.lower.loopy import LoopyCodegenContext

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
        raise NotImplementedError("MLIR code generation is still being implemented.") 

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
    _compile_loop(
        loop, 
        loop.index.iterset, 
        loop_indices, 
        codegen_context,
    )

@_compile.register(StandaloneCalledFunction)
def _(call, loop_indices, codegen_context):
    args = [(arg, spec) for arg, spec in zip(call.arguments, call.argspec, strict=True)]
    codegen_context.add_function_call(call.function.code, args)
    subkernel = call.function.code.with_entrypoints(frozenset())
    codegen_context.add_subkernel(subkernel)

@_compile.register(NonEmptyArrayAssignment)
def parse_assignment(assignment: NonEmptyArrayAssignment, loop_indices, codegen_context: CodegenContext):
    if any(isinstance(arg, pyop3.expr.MatPetscMatBufferExpression) for arg in assignment.arguments):
        _compile_petsc_mat(assignment, loop_indices, codegen_context)
    else:
        _compile_array_assignment(
            assignment,
            loop_indices,
            assignment.axis_trees,
            codegen_context,
        )

def _compile_petsc_mat(
    assignment,
    loop_indices,
    codegen_context,
):
    if not isinstance(codegen_context, LoopyCodegenContext):
        raise NotImplementedError("Only supported for Loopy")
    # We need to know whether the matrix is the assignee or not because we need
    # to know whether to put MatGetValues or MatSetValues
    if isinstance(assignment.assignee.buffer_view.buffer, PetscMatBuffer):
        mat = assignment.assignee
        expr = assignment.expression
        setting_mat_values = True
    else:
        mat = assignment.expression
        expr = assignment.assignee
        setting_mat_values = False

    row_axis_tree, column_axis_tree = assignment.axis_trees

    assert isinstance(expr, pyop3.expr.BufferExpression)

    # now emit the right line of code, this should properly be a lp.ScalarCallable
    # https://petsc.org/release/manualpages/Mat/MatGetValuesLocal/
    mat_name = codegen_context.add_buffer(mat.buffer_view, assignment_type_as_intent(assignment.assignment_type))
    array_name = codegen_context.add_buffer(expr.buffer_view, READ)

    rsize = row_axis_tree.size
    csize = column_axis_tree.size

    # these sizes can be expressions that need evaluating
    rsize_var = codegen_context.register_extent(
        rsize,
        {},
        loop_indices,
    )

    csize_var = codegen_context.register_extent(
        csize,
        {},
        loop_indices,
    )

    # convert the generic expressions to 
    # for example:
    #
    #   map0[3*i0 + i1]
    #   map0[3*i0 + i2 + 3]
    #
    # to the shared top-level layout:
    #
    #   map0[3*i0]
    #
    # which is what Mat{Get,Set}Values() needs.
    layout_exprs = []
    for layout in [mat.row_layout, mat.column_layout]:
        subst_sublayout = layout.layouts[idict()]
        subst_layout = pyop3.expr.LinearDatBufferExpression(layout.buffer, subst_sublayout)
        layout_expr = codegen_context.lower_expr(subst_layout, ((),), loop_indices)
        layout_exprs.append(layout_expr)
    irow, icol = layout_exprs

    # FIXME:
    blocked = False

    # hacky
    myargs = [
        assignment, mat_name, array_name, rsize_var, csize_var, irow, icol, blocked
    ]
    if setting_mat_values:
        match assignment.assignment_type:
            case AssignmentType.WRITE:
                call_str = _petsc_mat_store(*myargs)
            case AssignmentType.INC:
                call_str = _petsc_mat_add(*myargs)
            case _:
                raise AssertionError
    else:
        call_str = _petsc_mat_load(*myargs)

    codegen_context.add_cinstruction(call_str)


def _compile_array_assignment(
        assignment, 
        loop_indices, 
        axis_trees, 
        codegen_context,
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
                assignment.assignee,
                assignment.expression,
                assignment.assignment_type,
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
        
        elif component.size != 1:
            iname = codegen_context.unique_name("i")
            ext = codegen_context.register_extent(
                component.size, 
                iname_replace_maps[-1], 
                loop_indices
            )
            codegen_context.add_domain(iname, ext)
            new_maps = iname_replace_maps.copy()
            new_maps[-1] = iname_replace_maps[-1] | {axis.label: codegen_context.var(iname)}
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
                    axis_trees, 
                    codegen_context,
                    iname_replace_maps=new_maps, 
                    axis_tree=axis_tree, 
                    paths=new_paths
                )
            elif axis_trees:
                _compile_array_assignment(
                    assignment, 
                    loop_indices, 
                    axis_trees, 
                    codegen_context,
                    iname_replace_maps=new_maps, 
                    axis_tree=None, 
                    paths=new_paths
                )
            else:
                codegen_context.add_leaf_assignment(
                    assignment.assignee,
                    assignment.expression,
                    assignment.assignment_type,
                    new_paths, 
                    new_maps, 
                    loop_indices
                )

def _compile_loop(
        loop,
        axis_tree,
        loop_indices,
        codegen_context,
        *,
        axis=None,
        path=None,
        iname_map=None,
) -> None:
    if axis_tree is UNIT_AXIS_TREE:
        # NOTE: might need an expression here sometimes
        for statement in loop.statements:
            _compile(
                statement,
                # loop_indices | dict(loop_exprs),
                loop_indices,
                codegen_context,
            )
        return

    if utils.strictly_all(x is None for x in {axis, path, iname_map}):
        axis = axis_tree.root
        path = idict()
        iname_map = idict()

    for component in axis.components:
        path_ = path | {axis.label: component.label}

        if axis_tree.linearize(path_, partial=True).size == 0:
            continue
        elif component.size != 1:
            iname = codegen_context.unique_name("i")
            domain_var = codegen_context.register_extent(
                component.size,
                iname_map,
                loop_indices
            )
            codegen_context.add_domain(iname, domain_var)
            iname_replace_map_ = iname_map | {axis.label: codegen_context.var(iname)}
            within_inames = frozenset({iname})
        else:
            iname_replace_map_ = iname_map | {axis.label: 0}
            within_inames = set()

        with codegen_context.within_inames(within_inames):
            if subaxis := axis_tree.node_map[path_]:
                _compile_loop(
                    loop,
                    axis_tree,
                    loop_indices,
                    codegen_context,
                    axis=subaxis,
                    path=path_,
                    iname_map=iname_replace_map_
                )
            else:
                loop_indices |= idict({
                    (loop.index.id, axis_label): iname
                    for axis_label, iname in iname_replace_map_.items()
                })
                for statement in loop.statements:
                    _compile(
                        statement,
                        loop_indices,
                        codegen_context,
                    )

@_compile.register(Exscan)
def _(exscan, loop_indices, codegen_context):
    if not isinstance(codegen_context, LoopyCodegenContext):
        raise NotImplementedError("Only supported for Loopy")

    if exscan.scan_type != "+": 
        raise NotImplementedError 
 
    domain_var = codegen_context.register_extent(
        exscan.extent,
        {},
        loop_indices,
    )

    iname = codegen_context.unique_name("i")
    codegen_context.add_domain(iname, domain_var)

    iname_var = codegen_context.var(iname)
    iname_map = {exscan.scan_axis.label: codegen_context.var(iname)}

    lexpr = codegen_context.lower_expr(exscan.assignee, [iname_map], loop_indices, intent=RW)
    rexpr = lexpr + codegen_context.lower_expr(exscan.expression, [iname_map], loop_indices)

    lexpr = pym.substitute(lexpr, {iname: iname_var+1})
    codegen_context.add_assignment(lexpr, rexpr)

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
    return idict(
        {
            arg.buffer: lp_arg.shape
            for lp_arg, arg in zip(
                call.function.code.default_entrypoint.args, call.arguments, strict=True
            )
            if isinstance(lp_arg, lp.ArrayArg)
        }
    )

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
