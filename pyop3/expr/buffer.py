from __future__ import annotations

import abc
import functools
from functools import cached_property
from typing import ClassVar

from pyop3 import utils
from pyop3.buffer import BufferRef, AbstractBuffer

from .base import Expression, as_str
from .tensor import Dat, NonlinearCompositeDat


# TODO: Should inherit from Terminal (but Terminal has odd attrs)
class BufferExpression(Expression, metaclass=abc.ABCMeta):

    # {{{ abstract methods

    @property
    @abc.abstractmethod
    def buffer(self) -> AbstractBuffer:
        pass

    # }}}

    @property
    def name(self) -> str:
        return self.buffer.buffer.name

    @property
    def handle(self) -> Any:
        return self.buffer.buffer.handle(nest_indices=self.buffer.nest_indices)

    def assign(self, other) -> ArrayAssignment:
        from pyop3.insn import ArrayAssignment

        return ArrayAssignment(self, other, "write")

    def iassign(self, other) -> ArrayAssignment:
        from pyop3.insn import ArrayAssignment

        return ArrayAssignment(self, other, "inc")


# class ArrayBufferExpression(BufferExpression, metaclass=abc.ABCMeta):
#     pass


# class OpaqueBufferExpression(BufferExpression, metaclass=abc.ABCMeta):
#     """A buffer expression that is interfaced with using function calls.
#
#     An example of this is Mat{Get,Set}Values().
#
#     """


# class PetscMatBufferExpression(OpaqueBufferExpression, metaclass=abc.ABCMeta):
#     pass


# class ScalarBufferExpression(BufferExpression, metaclass=abc.ABCMeta):


@utils.record()
class ScalarBufferExpression(BufferExpression):

    # {{{ instance attrs

    _buffer: Any  # array buffer type

    # }}}

    # {{{ interface impls

    buffer: ClassVar[property] = utils.attr("_buffer")

    # }}}

    def __init__(self, buffer) -> None:
        self._buffer = buffer

    @property
    def _full_str(self) -> str:
        return self.name


# TODO: Does a Dat count as one of these?
class DatBufferExpression(BufferExpression, metaclass=abc.ABCMeta):
    pass


class LinearBufferExpression(BufferExpression, metaclass=abc.ABCMeta):
    pass


class NonlinearBufferExpression(BufferExpression, metaclass=abc.ABCMeta):
    pass


@utils.record()
class LinearDatBufferExpression(DatBufferExpression, LinearBufferExpression):
    """A dat with fixed (?) layout.

    It cannot be indexed.

    This class is useful for describing arrays used in index expressions, at which
    point it has a fixed set of axes.

    """

    # {{{ instance attrs

    _buffer: Any  # array buffer type
    layout: Any

    # }}}

    # {{{ interface impls

    buffer: ClassVar = utils.attr("_buffer")

    @property
    def shape(self) -> tuple[AxisTree]:
        from pyop3.expr.visitors import get_shape
        return get_shape(self.layout)

    @cached_property
    def loop_axes(self):
        from pyop3.expr.visitors import get_loop_axes
        return get_loop_axes(self.layout)

    @property
    def _full_str(self) -> str:
        return f"{self.name}[{as_str(self.layout)}]"

    # }}}

    def __init__(self, buffer, layout):
        if isinstance(buffer, AbstractBuffer):
            buffer = BufferRef(buffer)

        self._buffer = buffer
        self.layout = layout

    def __post_init__(self) -> None:
        from pyop3.expr.visitors import get_shape

        assert utils.just_one(get_shape(self.layout)).is_linear

    def concretize(self):
        return self


@utils.record()
class NonlinearDatBufferExpression(DatBufferExpression, NonlinearBufferExpression):
    """A dat with fixed layouts.

    This class is useful for describing dats whose layouts have been optimised.

    Unlike `_ExpressionDat` a `_ConcretizedDat` is permitted to be multi-component.

    """
    # {{{ Instance attrs

    _buffer: BufferRef
    layouts: Any
    # _loop_axes: tuple[Axis]

    # }}}

    # {{{ interface impls

    buffer: ClassVar[property] = utils.attr("_buffer")
    # loop_axes: ClassVar[property] = utils.attr("_loop_axes")

    @property
    def shape(self) -> tuple[AxisTree, ...]:
        shape_ = []
        layout_shapes = (layout.shape for layout in self.layouts.values())
        for layout_shapes_ in zip(*layout_shapes, strict=True):
            shape_.append(merge_axis_trees(layout_shapes_))
        return tuple(shape_)

    @property
    def loop_axes(self):
        breakpoint()
        return utils.unique(map(_extract_loop_axes, self.layouts.values()))

    @property
    def _full_str(self) -> str:
        return " :: ".join(
            f"{self.buffer.buffer.name}[{as_str(layout)}]"
            for layout in self.layouts.values()
        )

    # }}}


class MatBufferExpression(BufferExpression):
    pass


@utils.record()
class MatPetscMatBufferExpression(MatBufferExpression, LinearBufferExpression):

    # {{{ instance attrs

    _buffer: BufferRef
    row_layout: ExprT
    column_layout: ExprT

    def __init__(self, buffer, row_layout, column_layout):
        self._buffer = buffer
        self.row_layout = row_layout
        self.column_layout = column_layout

    # }}}

    # {{{ class constructors

    @classmethod
    def from_axis_trees(cls, buffer_ref, row_axes, column_axes) -> MatPetscMatBufferExpression:
        row_layout, column_layout = (
            NonlinearCompositeDat(axis_tree.materialize(), axis_tree.leaf_subst_layouts, axis_tree.outer_loops)
            for axis_tree in [row_axes, column_axes]
        )
        return cls(buffer_ref, row_layout, column_layout)

    # }}}

    # {{{ interface impls

    buffer: ClassVar[property] = utils.attr("_buffer")

    @property
    def shape(self):
        # NOTE: This doesn't make sense here, need multiple axis trees
        raise NotImplementedError

    @property
    def loop_axes(self):
        raise NotImplementedError

    @property
    def _full_str(self) -> str:
        return f"{self.buffer.buffer.name}[{as_str(self.row_layout)}, {as_str(self.column_layout)}]"

    # }}}


@utils.record()
class MatArrayBufferExpression(MatBufferExpression, NonlinearBufferExpression):

    # {{{ instance attrs

    _buffer: BufferRef
    row_layouts: ExprT
    column_layouts: ExprT

    def __init__(self, buffer, row_layouts, column_layouts):
        self._buffer = buffer
        self.row_layouts = row_layouts
        self.column_layouts = column_layouts

    # }}}

    # {{{ interface impls

    buffer: ClassVar[property] = utils.attr("_buffer")

    @property
    def shape(self):
        # NOTE: This doesn't make sense here, need multiple axis trees
        raise NotImplementedError

    @property
    def loop_axes(self):
        raise NotImplementedError

    @property
    def _full_str(self) -> str:
        return f"{self.buffer.buffer.name}[{as_str(self.row_layout)}, {as_str(self.column_layout)}]"

    # }}}


@functools.singledispatch
def as_linear_buffer_expression(obj: Any) -> LinearDatBufferExpression:
    raise TypeError


@as_linear_buffer_expression.register
def _(expr: LinearDatBufferExpression) -> LinearDatBufferExpression:
    return expr


@as_linear_buffer_expression.register
def _(dat: Dat) -> LinearDatBufferExpression:

    assert dat.parent is None

    if not dat.axes.is_linear:
        raise ValueError("The provided Dat must be linear")

    axes = dat.axes.regionless

    layout = utils.just_one(axes.leaf_subst_layouts.values())
    return LinearDatBufferExpression(BufferRef(dat.buffer), layout)


