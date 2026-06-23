from __future__ import annotations

import abc
import functools
import numbers
from functools import cached_property
from immutabledict import immutabledict as idict
from typing import ClassVar

import pyop3.axis_tree
import pyop3.record
from pyop3 import utils
from pyop3.node import NodeVisitor
from pyop3.labeled_tree import is_subpath
from pyop3.axis_tree import UNIT_AXIS_TREE
from pyop3.buffer import AbstractBuffer, ArrayBuffer
from pyop3.collections import OrderedFrozenSet

from .base import Expression, as_str
from .tensor import Scalar, Dat, CompositeDat


# TODO: Should inherit from Terminal (but Terminal has odd attrs)
class BufferExpression(Expression, metaclass=abc.ABCMeta):

    # {{{ abstract methods

    @property
    @abc.abstractmethod
    def buffer(self) -> AbstractBuffer:
        pass

    # }}}

    # {{{ interface impls

    @property
    def comm(self) -> MPI.Comm:
        return self.buffer.comm

    # }}}

    @property
    def name(self) -> str:
        return self.buffer.name

    @property
    def dtype(self) -> np.dtype:
        return self.buffer.dtype

    @property
    def handle(self) -> Any:
        return self.buffer.handle(nest_indices=self.buffer.nest_indices)

    def assign(self, other) -> ArrayAssignment:
        from pyop3.insn import Assignment

        return Assignment(self, other, "write")

    def iassign(self, other) -> ArrayAssignment:
        from pyop3.insn import Assignment

        return Assignment(self, other, "inc")


@pyop3.record.frozenrecord()
class ScalarBufferExpression(BufferExpression):

    # {{{ instance attrs

    _buffer: AbstractBuffer

    def collect_buffers(self, visitor):
        return visitor(self._buffer)

    def get_disk_cache_key(self, visitor) -> Hashable:
        return (type(self), visitor(self._buffer))

    get_instruction_executor_cache_key = get_disk_cache_key

    def __init__(self, buffer) -> None:
        object.__setattr__(self, "_buffer", buffer)

    # }}}

    # {{{ interface impls

    child_attrs = ()

    buffer = pyop3.record.attr("_buffer")

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
        return self.buffer.data_ro.item()


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

    _buffer: Any  # array buffer type
    layout: Any

    def collect_buffers(self, visitor):
        return visitor(self._buffer) | visitor(self.layout)

    def get_disk_cache_key(self, visitor) -> Hashable:
        return (type(self), visitor(self._buffer), visitor(self.layout))

    def get_instruction_executor_cache_key (self, visitor) -> Hashable:
        return (type(self), visitor(self._buffer), visitor(self.layout, inside=True))

    def __init__(self, buffer, layout):
        object.__setattr__(self, "_buffer", buffer)
        object.__setattr__(self, "layout", layout)
        self.__post_init__()

    def __post_init__(self) -> None:
        pass

    # }}}

    # {{{ interface impls

    child_attrs = ("layout",)

    buffer: ClassVar = pyop3.record.attr("_buffer")

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

    _buffer: AbstractBuffer
    layouts: idict

    def collect_buffers(self, visitor):
        return visitor(self._buffer).union(*(map(visitor, self.layouts.values()))) 

    def get_disk_cache_key(self, visitor) -> Hashable:
        layouts_key = {}
        for path, layout in self.layouts.items():
            layouts_key[visitor.relabel_path(path)] = visitor(layout)
        layouts_key = idict(layouts_key)
        return (type(self), visitor(self._buffer), layouts_key)

    def __post_init__(self) -> None:
        pass

    # }}}

    # {{{ interface impls

    child_attrs = ("layouts",)

    buffer: ClassVar[property] = pyop3.record.attr("_buffer")

    @property
    def local_max(self) -> numbers.Number:
        raise NotImplementedError

    @property
    def local_min(self) -> numbers.Number:
        raise NotImplementedError

    @property
    def _full_str(self) -> str:
        return " :: ".join(
            f"{self.buffer.name}[{as_str(layout)}]"
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

    def linearize(self, path) -> LinearDatBufferExpression:
        return LinearDatBufferExpression(self.buffer, self.layouts[path])


class MatBufferExpression(BufferExpression):
    pass


@pyop3.record.frozenrecord()
class MatPetscMatBufferExpression(MatBufferExpression, LinearBufferExpression):

    # {{{ instance attrs

    _buffer: AbstractBuffer
    row_layout: ExprT
    column_layout: ExprT

    def collect_buffers(self, visitor):
        return visitor(self._buffer).union(visitor(self.row_layout), visitor(self.column_layout))

    def get_disk_cache_key(self, visitor) -> Hashable:
        return (
            type(self),
            visitor(self._buffer),
            visitor(self.row_layout),
            visitor(self.column_layout),
        )

    def __init__(self, buffer, row_layout, column_layout):
        object.__setattr__(self, "_buffer", buffer)
        object.__setattr__(self, "row_layout", row_layout)
        object.__setattr__(self, "column_layout", column_layout)

    # }}}

    # {{{ class constructors

    @classmethod
    def from_axis_trees(cls, buffer_ref, row_axes, column_axes) -> MatPetscMatBufferExpression:
        row_layout, column_layout = (
            CompositeDat(axis_tree.materialize().regionless(), axis_tree.subst_layouts())
            for axis_tree in [row_axes, column_axes]
        )
        return cls(buffer_ref, row_layout, column_layout)

    # }}}

    # {{{ interface impls

    child_attrs = ("row_layout", "column_layout")

    buffer: ClassVar[property] = pyop3.record.attr("_buffer")

    @property
    def local_max(self) -> numbers.Number:
        raise NotImplementedError

    @property
    def local_min(self) -> numbers.Number:
        raise NotImplementedError

    @property
    def _full_str(self) -> str:
        return f"{self.buffer.name}[{as_str(self.row_layout)}, {as_str(self.column_layout)}]"

    # }}}


@pyop3.record.frozenrecord()
class MatArrayBufferExpression(MatBufferExpression, NonlinearBufferExpression):

    # {{{ instance attrs

    _buffer: AbstractBuffer
    row_layouts: idict
    column_layouts: idict

    def collect_buffers(self, visitor) -> OrderedFrozenSet:
        return visitor(self._buffer).union(
            *(map(visitor, self.row_layouts.values())),
            *(map(visitor, self.column_layouts.values())),
        )

    def get_disk_cache_key(self, visitor) -> Hashable:
        row_layouts_key = idict({
            visitor.relabel_path(path): visitor(layout)
            for path, layout in self.row_layouts.items()
        })
        column_layouts_key = idict({
            visitor.relabel_path(path): visitor(layout)
            for path, layout in self.column_layouts.items()
        })
        return (type(self), visitor(self._buffer), row_layouts_key, column_layouts_key)

    def __init__(self, buffer, row_layouts, column_layouts) -> None:
        object.__setattr__(self, "_buffer", buffer)
        object.__setattr__(self, "row_layouts", row_layouts)
        object.__setattr__(self, "column_layouts", column_layouts)

    def __post_init__(self) -> None:
        assert isinstance(self._buffer, AbstractBuffer)
        assert isinstance(self.row_layouts, idict)
        assert isinstance(self.column_layouts, idict)

    # }}}

    # {{{ interface impls

    child_attrs = ("row_layouts", "column_layouts")

    buffer: ClassVar[property] = pyop3.record.attr("_buffer")

    @property
    def local_max(self) -> numbers.Number:
        raise NotImplementedError

    @property
    def local_min(self) -> numbers.Number:
        raise NotImplementedError

    @property
    def _full_str(self) -> str:
        return f"{self.buffer.name}[{self.row_layouts}, {self.column_layouts}]"

    # }}}


def as_linear_buffer_expression(obj):
    return _as_linear_buffer_expression(obj)

    # can't do this as it affects assignees
    # if expr.min_value == expr.max_value:
    #     return expr.min_value


@functools.singledispatch
def _as_linear_buffer_expression(obj: Any) -> LinearDatBufferExpression:
    raise TypeError


@_as_linear_buffer_expression.register
def _(expr: LinearDatBufferExpression) -> LinearDatBufferExpression:
    return expr


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

    layout = utils.just_one(axes.leaf_subst_layouts.values())
    return LinearDatBufferExpression(dat.buffer, layout)


@_as_linear_buffer_expression.register
def _(scalar: Scalar) -> ScalarBufferExpression:
    assert scalar.transform is None
    return ScalarBufferExpression(scalar.buffer)
