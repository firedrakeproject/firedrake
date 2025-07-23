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

import pymbolic as pym
from pyop3.ir.lower import CuPyCodegenContext
from pyop3.lang import (
    Intent,
    INC,
    MAX_RW,
    MAX_WRITE,
    MIN_RW,
    MIN_WRITE,
    READ,
    RW,
)
from pyop3.axtree.tree import UNIT_AXIS_TREE, Add, AxisVar, IndexedAxisTree, Mul, AxisComponent, relabel_path
from pyop3.itree.tree import AffineSliceComponent, LoopIndexVar, Slice, IndexTree
from pyop3.tensor import LinearDatBufferExpression, NonlinearDatBufferExpression, Scalar
from pyop3.tensor.dat import LinearMatBufferExpression, NonlinearMatBufferExpression, MatBufferExpression, BufferExpression
from pyop3.utils import (
    just_one,
)

def lower_expr(expr, iname_maps, loop_indices, ctx : CuPyCodegenContext, *, intent=READ, paths=None, shape=None) -> str:
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
    return pymbolic_to_str(just_one(iname_maps)[axis_var.axis_label])


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
    return f"{name_in_kernel}{indices}"


def maybe_multiindex(buffer_ref, offset_expr, context):
    # hack to handle the facbuffer.t that temporaries can have shape but we want to
    # linearly index it here
    buffer_key = (buffer_ref.buffer.name, buffer_ref.nest_indices)
    if buffer_key in context._temporary_shapes:
        shape = context._temporary_shapes[buffer_key]
        rank = len(shape)
        extra_indices = (0,) * (rank - 1)

        # also has to be a scalar, not an expression
        temp_offset_name = context.add_temporary("j")
        temp_offset_var = pym.var(temp_offset_name)
        context.add_assignment(temp_offset_var, offset_expr)
        indices = extra_indices + (temp_offset_var,)
    else:
        indices = "[offset_expr]"

    return indices


def pymbolic_to_str(pym_expr):
    return _pymbolic_to_str(pym_expr)

@functools.singledispatch
def _pymbolic_to_str(obj: Any, /, *args, **kwargs) -> str:
    raise TypeError(f"No handler defined for {type(obj).__name__}")


@_pymbolic_to_str.register(pym.Variable)
def _pymbolic_to_str_var(obj):
    return obj.name
