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
import dataclasses
import importlib

import pymbolic as pym
from pyop3.lang import (
    Intent,
    INC,
    MAX_RW,
    MAX_WRITE,
    MIN_RW,
    MIN_WRITE,
    READ,
    WRITE,
    RW,
)
from pyop3.axtree.tree import UNIT_AXIS_TREE, Add, AxisVar, IndexedAxisTree, Mul, AxisComponent, relabel_path
from pyop3.itree.tree import AffineSliceComponent, LoopIndexVar, Slice, IndexTree
from pyop3.tensor import LinearDatBufferExpression, NonlinearDatBufferExpression, Scalar
from pyop3.tensor.dat import LinearMatBufferExpression, NonlinearMatBufferExpression, MatBufferExpression, BufferExpression
from pyop3.utils import (
    just_one,
)

def lower_expr(expr, iname_maps, loop_indices, ctx, *, intent=READ, paths=None, shape=None) -> str:
    return _lower_cp_expr(expr, iname_maps, loop_indices, ctx, intent=intent, paths=paths, shape=shape)


@functools.singledispatch
def _lower_cp_expr(obj: Any, /, *args, **kwargs) -> str:
    raise TypeError(f"No handler defined for {type(obj).__name__}")


@_lower_cp_expr.register(numbers.Number)
def _(num: numbers.Number, /, *args, **kwargs) -> numbers.Number:
    return str(num)


@_lower_cp_expr.register(Add)
def _(add: Add, /, *args, **kwargs) -> str:
    return f"{_lower_cp_expr(add.a, *args, **kwargs)} + {_lower_cp_expr(add.b, *args, **kwargs)}"


@_lower_cp_expr.register(Mul)
def _(mul: Mul, /, *args, **kwargs) -> str:
    return f"{_lower_cp_expr(mul.a, *args, **kwargs)} * {_lower_cp_expr(mul.b, *args, **kwargs)}"


@_lower_cp_expr.register(AxisVar)
def _(axis_var: AxisVar, /, iname_maps, *args, **kwargs) -> str:
    return just_one(iname_maps)[axis_var.axis_label]


@_lower_cp_expr.register(LoopIndexVar)
def _(loop_var: LoopIndexVar, /, iname_maps, loop_indices, *args, **kwargs) -> str:
    return loop_indices[(loop_var.loop_id, loop_var.axis_label)]


@_lower_cp_expr.register(LinearDatBufferExpression)
def _(expr: LinearDatBufferExpression, /, iname_maps, loop_indices, context, *, intent, **kwargs) -> str:
    return lower_buffer_access(expr.buffer, [expr.layout], iname_maps, loop_indices, context, intent=intent)


@_lower_cp_expr.register(NonlinearDatBufferExpression)
def _(expr: NonlinearDatBufferExpression, /, iname_maps, loop_indices, context, *, intent, paths, **kwargs) -> str:
    path = just_one(paths)
    return lower_buffer_access(expr.buffer, [expr.layouts[path]], iname_maps, loop_indices, context, intent=intent)


@_lower_cp_expr.register(LinearMatBufferExpression)
def _(expr: LinearMatBufferExpression, /, iname_maps, loop_indices, context, *, intent, paths, shape) -> str:
    layouts = (expr.row_layout, expr.column_layout)
    return lower_buffer_access(expr.buffer, layouts, iname_maps, loop_indices, context, intent=intent, shape=shape)

@_lower_cp_expr.register(NonlinearMatBufferExpression)
def _(expr: NonlinearMatBufferExpression, /, iname_maps, loop_indices, context, *, intent, paths, shape) -> str:
    row_path, column_path = paths
    layouts = (expr.row_layouts[row_path], expr.column_layouts[column_path])
    return lower_buffer_access(expr.buffer, layouts, iname_maps, loop_indices, context, intent=intent, shape=shape)


def lower_buffer_access(buffer: AbstractBuffer, layouts, iname_maps, loop_indices, context, *, intent, shape=None) -> str:
    name_in_kernel = context.add_buffer(buffer, intent)

    offset_expr = ""
    strides = utils.strides(shape) if shape else (1,)
    for stride, layout, iname_map in zip(strides, layouts, iname_maps, strict=True):
        offset_expr += stride * lower_expr(layout, [iname_map], loop_indices, context)
    indices = maybe_multiindex(buffer, offset_expr, context)
    # TODO way to check indices is a map vs a single indexLw
    if intent == INC:
        return f"cpx.scatter_add({name_in_kernel}, {indices}, {{rhs}})"
    if intent == READ:
        return f"cp.take({name_in_kernel}, {indices})"
    if intent == WRITE:
        return f"{name_in_kernel}[{indices}] = {{rhs}}"
    raise NotImplementedError(f"Intent {intent} does not have a handler")


def maybe_multiindex(buffer_ref, offset_expr, context):
    # hack to handle the facbuffer.t that temporaries can have shape but we want to
    # linearly index it here
    buffer_key = (buffer_ref.buffer.name, buffer_ref.nest_indices)
    if buffer_key in context._temporary_shapes:
        shape = context._temporary_shapes[buffer_key]
        rank = len(shape)
        extra_indices = (0,) * (rank - 1)
        if len(extra_indices) > 1:
            raise NotImplementedError("Handing shape TODO")

        # also has to be a scalar, not an expression
        temp_offset_name = context.add_temporary("j")
        temp_offset_var = temp_offset_name
        context.add_assignment(temp_offset_var, offset_expr)
        indices = extra_indices + (temp_offset_var,)
    else:
        indices = f"{offset_expr}"

    return indices


class CuPyTranslationUnit():

    def __init__(self, domains, instructions, arguments, temporaries, function_name):
        self.filename = "temp_cupy_file"
        self._domains = domains
        self._instructions = instructions
        self._arguments = arguments
        self._temporaries = temporaries
        self.function_name = function_name
        self._entrypoint = CuPyEntrypoint(function_name, arguments)

    @property
    def default_entrypoint(self) -> CuPyEntrypoint:
        return self._entrypoint
        
    def compile(self, preambles):
        code_lines = preambles
        indent = 0
        args = ",".join([a.name for a in self._arguments]) #add types?
        code_lines += ["\t"*indent + f"def {self.function_name}({args}):"]
        indent += 1
        for temp in self._temporaries:
            if temp.initializer is not None:
                code_lines += [ "\t"*indent + f"{temp.name} : {temp.dtype} = {initialiser}"]
            else:
                code_lines += [ "\t"*indent + f"{temp.name} : {temp.dtype}"]
        #code_lines += ["\t"*indent + "breakpoint()"]

        current_inames = set()
        for insn in self._instructions:
            # exit completed loops 
            for i in range(len(current_inames - insn.inames)):
                indent -= 1 
            while insn.inames > current_inames:
                dom  = self._domains.pop(0)
                code_lines += ["\t"*indent + f"{dom[0]}:"]
                current_inames.add(dom[1])
                indent += 1
            code_lines += ["\t"*indent + f"{insn.insn_str}"]

        self.code_string = "\n".join(code_lines)

    def construct(self):
        with open(f"./{self.filename}.py", "w") as file:
            file.write(self.code_string)
        temp_file = importlib.reload(__import__(f"{self.filename}"))
        return getattr(temp_file, self._entrypoint.name)

@dataclasses.dataclass
class CuPyEntrypoint:
    name : str
    args : list

@dataclasses.dataclass
class CuPyArgument:
    name: str
    dtype: str

@dataclasses.dataclass
class CuPyTemporary:
    name: str
    dtype: str
    initializer: cp.ndarray = None

@dataclasses.dataclass
class CuPyInstruction:
    insn_str: str
    inames: set 
