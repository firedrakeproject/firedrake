from __future__ import annotations

import abc
import functools
import numbers
from typing import ClassVar

import numpy as np
from immutabledict import immutabledict as idict

import pyop3.axis_tree
import pyop3.buffer
import pyop3.record
from pyop3 import utils
from pyop3.buffer import AbstractBuffer
from pyop3.collections import OrderedFrozenSet
from pyop3.labeled_tree import is_subpath

from .base import Expression, as_str
from .tensor import CompositeDat, Dat, Scalar


# TODO: Should inherit from Terminal (but Terminal has odd attrs)
class BufferExpression(Expression, metaclass=abc.ABCMeta):

    __abstract_record_attrs = ("buffer_view",)

    @property
    def name(self) -> str:
        return self.buffer_view.buffer.name

    @property
    def dtype(self) -> np.dtype:
        return self.buffer_view.buffer.dtype

    def assign(self, other) -> ArrayAssignment:
        from pyop3.insn import Assignment

        return Assignment(self, other, "write")

    def iassign(self, other) -> ArrayAssignment:
        from pyop3.insn import Assignment

        return Assignment(self, other, "inc")

    @property
    def buffer(self) -> pyop3.buffer.AbstractBuffer:
        assert self.buffer_view.nest_indices == (), \
            "Direct access to 'buffer' is ambiguous for nests"
        return self.buffer_view.buffer


@pyop3.record.frozenrecord()
class ScalarBufferExpression(BufferExpression):

    # {{{ instance attrs

    buffer_view: pyop3.buffer.IndexedBuffer

    def collect_buffers(self, visitor):
        return visitor(self.buffer_view)

    def get_disk_cache_key(self, visitor) -> Hashable:
        return (type(self), visitor(self.buffer_view))

    get_instruction_executor_cache_key = get_disk_cache_key

    def __init__(self, buffer_view: AbstractBuffer | pyop3.buffer.IndexedBuffer) -> None:
        if isinstance(buffer_view, pyop3.buffer.AbstractBuffer):
            assert buffer_view.nest_shape() is None
            buffer_view = pyop3.buffer.IndexedBuffer(buffer_view, ())
        object.__setattr__(self, "buffer_view", buffer_view)

    # }}}

    # {{{ interface impls

    child_attrs = ()

    @property
    def local_max(self) -> numbers.Number:
        return self.value

    @property
    def local_min(self) -> numbers.Number:
        return self.value

    @property
    def _full_str(self) -> str:
        return self.name

    # def __add__(self, other: ExpressionT, /) -> ExpressionT:
    #     if self.buffer.constant:
    #         if isinstance(other, numbers.Number):
    #             buffer = ArrayBuffer.from_scalar(self.value+other, constant=True, dtype=self.dtype)
    #             return type(self)(buffer)
    #         elif type(other) is type(self) and other.buffer.constant:
    #             buffer = ArrayBuffer.from_scalar(self.value+other.value, constant=True, dtype=self.dtype)
    #             return type(self)(buffer)
    #     return super().__add__(other)
    #
    # def __sub__(self, other: ExpressionT, /) -> ExpressionT:
    #     if self.buffer.constant:
    #         if isinstance(other, numbers.Number):
    #             buffer = ArrayBuffer.from_scalar(self.value-other, constant=True, dtype=self.dtype)
    #             return type(self)(buffer)
    #         elif type(other) is type(self) and other.buffer.constant:
    #             buffer = ArrayBuffer.from_scalar(self.value-other.value, constant=True, dtype=self.dtype)
    #             return type(self)(buffer)
    #     return super().__sub__(other)
    #
    # def __mul__(self, other: ExpressionT, /) -> ExpressionT:
    #     if self.buffer.constant:
    #         if isinstance(other, numbers.Number):
    #             buffer = ArrayBuffer.from_scalar(self.value*other, constant=True, dtype=self.dtype)
    #             return type(self)(buffer)
    #         elif type(other) is type(self) and other.buffer.constant:
    #             buffer = ArrayBuffer.from_scalar(self.value*other.value, constant=True, dtype=self.dtype)
    #             return type(self)(buffer)
    #     return super().__mul__(other)

    # }}}

    @property
    def value(self) -> numbers.Number:
        return self.buffer_view.buffer.data_ro.item()


# TODO: Does a Dat count as one of these?
class DatBufferExpression(BufferExpression, metaclass=abc.ABCMeta):
    pass


class LinearBufferExpression(BufferExpression, metaclass=abc.ABCMeta):
    pass


class NonlinearBufferExpression(BufferExpression, metaclass=abc.ABCMeta):
    pass


@pyop3.record.frozenrecord()
class LinearDatBufferExpression(DatBufferExpression, LinearBufferExpression):
    """A dat with fixed (?) layout.

    It cannot be indexed.

    This class is useful for describing arrays used in index expressions, at which
    point it has a fixed set of axes.

    """

    # {{{ instance attrs

    buffer_view: pyop3.buffer.IndexedBuffer
    layout: Any

    def collect_buffers(self, visitor):
        return visitor(self.buffer_view) | visitor(self.layout)

    def get_disk_cache_key(self, visitor) -> Hashable:
        return (type(self), visitor(self.buffer_view), visitor(self.layout))

    def get_instruction_executor_cache_key (self, visitor) -> Hashable:
        buffer_key = visitor(self.buffer_view)
        with visitor.strong_hash_buffers():
            layout_key = visitor(self.layout)
        return (type(self), buffer_key, layout_key)

    @property
    def comm(self):
        return self.buffer_view.comm

    def __init__(self, buffer_view, layout):
        if isinstance(buffer_view, pyop3.buffer.AbstractBuffer):
            assert buffer_view.nest_shape() is None
            buffer_view = pyop3.buffer.IndexedBuffer(buffer_view, ())
        object.__setattr__(self, "buffer_view", buffer_view)
        object.__setattr__(self, "layout", layout)

    # }}}

    # {{{ interface impls

    child_attrs = ("layout",)

    @property
    def local_max(self) -> numbers.Number:
        from pyop3.expr.visitors import get_extremum

        return get_extremum(self, "max")

    @property
    def local_min(self) -> numbers.Number:
        from pyop3.expr.visitors import get_extremum

        return get_extremum(self, "min")

    @property
    def _full_str(self) -> str:
        return f"{self.name}[{as_str(self.layout)}]"

    # }}}

    def concretize(self):
        return self


@pyop3.record.frozenrecord()
class NonlinearDatBufferExpression(DatBufferExpression, NonlinearBufferExpression):
    """A dat with fixed layouts.

    This class is useful for describing dats whose layouts have been optimised.

    """
    # {{{ instance attrs

    buffer_view: pyop3.buffer.IndexedBuffer
    layouts: idict

    def get_disk_cache_key(self, visitor) -> Hashable:
        layouts_key = {}
        for path, layout in self.layouts.items():
            layouts_key[visitor.relabel_axis_tree_path(path)] = visitor(layout)
        layouts_key = idict(layouts_key)
        return (type(self), visitor(self.buffer_view), layouts_key)

    def collect_buffers(self, visitor):
        return visitor(self.buffer_view).union(*(map(visitor, self.layouts.values()))) 

    def __init__(self, buffer_view, layouts) -> None:
        if isinstance(buffer_view, pyop3.buffer.AbstractBuffer):
            assert buffer_view.nest_shape() is None
            buffer_view = pyop3.buffer.IndexedBuffer(buffer_view, ())
        object.__setattr__(self, "buffer_view", buffer_view)
        object.__setattr__(self, "layouts", layouts)

    # }}}

    @property
    def comm(self):
        return self.buffer_view.comm


    # {{{ interface impls

    child_attrs = ("layouts",)

    @property
    def local_max(self) -> numbers.Number:
        raise NotImplementedError

    @property
    def local_min(self) -> numbers.Number:
        raise NotImplementedError

    @property
    def _full_str(self) -> str:
        return " :: ".join(
            f"{self.buffer_view.name}[{as_str(layout)}]"
            for layout in self.layouts.values()
        )

    # }}}

    @property
    def leaf_layouts(self) -> idict:
        leaf_layouts_ = {}
        for path, layout in self.layouts.items():
            if not any(
                is_subpath(path, other_path)
                for other_path in self.layouts.keys()
                if other_path != path
            ):
                leaf_layouts_[path] = layout
        return idict(leaf_layouts_)

    def linearize(self, path, *, allow_partial: bool = False) -> LinearDatBufferExpression:
        if allow_partial:
            path = utils.just_one(
                lpath
                for lpath in self.layouts.keys()
                if lpath.keys() <= path.keys()
            )
        return LinearDatBufferExpression(self.buffer_view, self.layouts[path])


class MatBufferExpression(BufferExpression):
    pass


@pyop3.record.frozenrecord()
class MatPetscMatBufferExpression(MatBufferExpression, LinearBufferExpression):

    # {{{ instance attrs

    buffer_view: pyop3.buffer.IndexedBuffer
    row_layout: ExprT
    column_layout: ExprT

    def collect_buffers(self, visitor):
        return visitor(self.buffer_view).union(visitor(self.row_layout), visitor(self.column_layout))

    def get_disk_cache_key(self, visitor) -> Hashable:
        return (
            type(self),
            visitor(self.buffer_view),
            visitor(self.row_layout),
            visitor(self.column_layout),
        )

    # }}}

    # {{{ interface impls

    child_attrs = ("row_layout", "column_layout")

    @property
    def local_max(self) -> numbers.Number:
        raise NotImplementedError

    @property
    def local_min(self) -> numbers.Number:
        raise NotImplementedError

    @property
    def _full_str(self) -> str:
        return f"{self.buffer_view.name}[{as_str(self.row_layout)}, {as_str(self.column_layout)}]"

    # }}}


@pyop3.record.frozenrecord()
class MatArrayBufferExpression(MatBufferExpression, NonlinearBufferExpression):

    # {{{ instance attrs

    buffer_view: pyop3.buffer.IndexedBuffer
    row_layouts: idict
    column_layouts: idict

    def collect_buffers(self, visitor) -> OrderedFrozenSet:
        return visitor(self.buffer_view).union(
            *(map(visitor, self.row_layouts.values())),
            *(map(visitor, self.column_layouts.values())),
        )

    def get_disk_cache_key(self, visitor) -> Hashable:
        row_layouts_key = idict({
            visitor.relabel_axis_tree_path(path): visitor(layout)
            for path, layout in self.row_layouts.items()
        })
        column_layouts_key = idict({
            visitor.relabel_axis_tree_path(path): visitor(layout)
            for path, layout in self.column_layouts.items()
        })
        return (type(self), visitor(self.buffer_view), row_layouts_key, column_layouts_key)

    def __init__(self, buffer_view, row_layouts, column_layouts):
        if isinstance(buffer_view, pyop3.buffer.AbstractBuffer):
            assert buffer_view.nest_shape() is None
            buffer_view = pyop3.buffer.IndexedBuffer(buffer_view, ())
        object.__setattr__(self, "buffer_view", buffer_view)
        object.__setattr__(self, "row_layouts", row_layouts)
        object.__setattr__(self, "column_layouts", column_layouts)

    # }}}

    # {{{ interface impls

    child_attrs = ("row_layouts", "column_layouts")

    @property
    def local_max(self) -> numbers.Number:
        raise NotImplementedError

    @property
    def local_min(self) -> numbers.Number:
        raise NotImplementedError

    @property
    def _full_str(self) -> str:
        return f"{self.buffer_view.buffer.name}[{self.row_layouts}, {self.column_layouts}]"

    # }}}


def as_linear_buffer_expression(obj):
    return _as_linear_buffer_expression(obj)


@functools.singledispatch
def _as_linear_buffer_expression(obj: Any) -> LinearDatBufferExpression:
    raise TypeError


@_as_linear_buffer_expression.register
def _(expr: LinearDatBufferExpression) -> LinearDatBufferExpression:
    return expr


# TODO: This is the same as dat.concretize(linear=True)
@_as_linear_buffer_expression.register
def _(dat: Dat) -> LinearDatBufferExpression:
    assert dat.transform is None
    if not dat.axes.is_linear:
        raise ValueError("The provided dat must be linear")

    axes = dat.axes.regionless()
    # We assume that if we hit an axis forest at this point then any layout
    # expression is valid.
    # This can happen if we use maps with multiple possible matches (e.g. mapping
    # from cells or owned cells).
    if isinstance(axes, pyop3.axis_tree.AxisForest):
        # FIXME, merge?
        axes = axes.trees[-1]

    ibuffer = pyop3.buffer.IndexedBuffer(dat.buffer, ())
    layout = utils.just_one(axes.leaf_subst_layouts.values())
    return LinearDatBufferExpression(ibuffer, layout)


@_as_linear_buffer_expression.register
def _(scalar: Scalar) -> ScalarBufferExpression:
    assert scalar.transform is None
    return ScalarBufferExpression(scalar.buffer)


@_as_linear_buffer_expression.register
def _(array: np.ndarray) -> LinearDatBufferExpression:
    return _as_linear_buffer_expression(Dat.from_array(array))
