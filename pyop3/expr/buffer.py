from __future__ import annotations

import abc
import functools
import numbers
from functools import cached_property
from immutabledict import immutabledict as idict
from typing import ClassVar

from pyop3 import utils
from pyop3.tree.axis_tree import UNIT_AXIS_TREE
from pyop3.buffer import BufferRef, AbstractBuffer, ArrayBuffer
from pyop3.sf import DistributedObject

from .base import Expression, as_str
from .tensor import Scalar, Dat, CompositeDat


# TODO: Should inherit from Terminal (but Terminal has odd attrs)
class BufferExpression(Expression, DistributedObject, metaclass=abc.ABCMeta):

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


@utils.frozenrecord()
class ScalarBufferExpression(BufferExpression):

    # {{{ instance attrs

    _buffer: BufferRef

    def __init__(self, buffer) -> None:
        if isinstance(buffer, AbstractBuffer):
            buffer = BufferRef(buffer)
        object.__setattr__(self, "_buffer", buffer)

    # }}}

    # {{{ interface impls

    child_attrs = ()

    buffer = utils.attr("_buffer")

    @property
    def local_max(self) -> numbers.Number:
        return self.value

    @property
    def local_min(self) -> numbers.Number:
        return self.value

    @property
    def _full_str(self) -> str:
        return self.name

    def __add__(self, other: ExpressionT, /) -> ExpressionT:
        if self.buffer.buffer.constant:
            if isinstance(other, numbers.Number):
                buffer = ArrayBuffer.from_scalar(self.value+other, constant=True)
                return type(self)(buffer)
            elif type(other) is type(self) and other.buffer.buffer.constant:
                buffer = ArrayBuffer.from_scalar(self.value+other.value, constant=True)
                return type(self)(buffer)
        return super().__add__(other)

    def __sub__(self, other: ExpressionT, /) -> ExpressionT:
        if self.buffer.buffer.constant:
            if isinstance(other, numbers.Number):
                buffer = ArrayBuffer.from_scalar(self.value-other, constant=True)
                return type(self)(buffer)
            elif type(other) is type(self) and other.buffer.buffer.constant:
                buffer = ArrayBuffer.from_scalar(self.value-other.value, constant=True)
                return type(self)(buffer)
        return super().__sub__(other)

    def __mul__(self, other: ExpressionT, /) -> ExpressionT:
        if self.buffer.buffer.constant:
            if isinstance(other, numbers.Number):
                buffer = ArrayBuffer.from_scalar(self.value*other, constant=True)
                return type(self)(buffer)
            elif type(other) is type(self) and other.buffer.buffer.constant:
                buffer = ArrayBuffer.from_scalar(self.value*other.value, constant=True)
                return type(self)(buffer)
        return super().__mul__(other)

    # }}}

    @property
    def value(self) -> numbers.Number:
        return self.buffer.buffer.data_ro.item()


# TODO: Does a Dat count as one of these?
class DatBufferExpression(BufferExpression, metaclass=abc.ABCMeta):
    pass


class LinearBufferExpression(BufferExpression, metaclass=abc.ABCMeta):
    pass


class NonlinearBufferExpression(BufferExpression, metaclass=abc.ABCMeta):
    pass


@utils.frozenrecord()
class LinearDatBufferExpression(DatBufferExpression, LinearBufferExpression):
    """A dat with fixed (?) layout.

    It cannot be indexed.

    This class is useful for describing arrays used in index expressions, at which
    point it has a fixed set of axes.

    """

    # {{{ instance attrs

    _buffer: Any  # array buffer type
    layout: Any

    def __init__(self, buffer, layout):
        if isinstance(buffer, AbstractBuffer):
            buffer = BufferRef(buffer)

        object.__setattr__(self, "_buffer", buffer)
        object.__setattr__(self, "layout", layout)
        self.__post_init__()

    def __post_init__(self) -> None:
        from pyop3.expr.visitors import get_shape

        assert utils.just_one(get_shape(self.layout)).is_linear

    # }}}

    # {{{ interface impls

    child_attrs = ("layout",)

    buffer: ClassVar = utils.attr("_buffer")

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


@utils.frozenrecord()
class NonlinearDatBufferExpression(DatBufferExpression, NonlinearBufferExpression):
    """A dat with fixed layouts.

    This class is useful for describing dats whose layouts have been optimised.

    Unlike `_ExpressionDat` a `_ConcretizedDat` is permitted to be multi-component.

    """
    # {{{ instance attrs

    _buffer: BufferRef
    layouts: idict

    def __post_init__(self) -> None:
        assert isinstance(self._buffer, BufferRef)
        assert isinstance(self.layouts, idict)

    # }}}

    # {{{ interface impls

    child_attrs = ("layouts",)

    buffer: ClassVar[property] = utils.attr("_buffer")

    @property
    def local_max(self) -> numbers.Number:
        raise NotImplementedError

    @property
    def local_min(self) -> numbers.Number:
        raise NotImplementedError

    @property
    def _full_str(self) -> str:
        return " :: ".join(
            f"{self.buffer.buffer.name}[{as_str(layout)}]"
            for layout in self.layouts.values()
        )

    # }}}


class MatBufferExpression(BufferExpression):
    pass


@utils.frozenrecord()
class MatPetscMatBufferExpression(MatBufferExpression, LinearBufferExpression):

    # {{{ instance attrs

    _buffer: BufferRef
    row_layout: ExprT
    column_layout: ExprT

    def __init__(self, buffer, row_layout, column_layout):
        object.__setattr__(self, "_buffer", buffer)
        object.__setattr__(self, "row_layout", row_layout)
        object.__setattr__(self, "column_layout", column_layout)

    # }}}

    # {{{ class constructors

    @classmethod
    def from_axis_trees(cls, buffer_ref, row_axes, column_axes) -> MatPetscMatBufferExpression:
        row_layout, column_layout = (
            CompositeDat(axis_tree.materialize().localize(), axis_tree.subst_layouts())
            for axis_tree in [row_axes, column_axes]
        )
        return cls(buffer_ref, row_layout, column_layout)

    # }}}

    # {{{ interface impls

    child_attrs = ("row_layout", "column_layout")

    buffer: ClassVar[property] = utils.attr("_buffer")

    @property
    def local_max(self) -> numbers.Number:
        raise NotImplementedError

    @property
    def local_min(self) -> numbers.Number:
        raise NotImplementedError

    @property
    def _full_str(self) -> str:
        return f"{self.buffer.buffer.name}[{as_str(self.row_layout)}, {as_str(self.column_layout)}]"

    # }}}


@utils.frozenrecord()
class MatArrayBufferExpression(MatBufferExpression, NonlinearBufferExpression):

    # {{{ instance attrs

    _buffer: BufferRef
    row_layouts: idict
    column_layouts: idict

    def __init__(self, buffer, row_layouts, column_layouts) -> None:
        object.__setattr__(self, "_buffer", buffer)
        object.__setattr__(self, "row_layouts", row_layouts)
        object.__setattr__(self, "column_layouts", column_layouts)

    def __post_init__(self) -> None:
        assert isinstance(self._buffer, BufferRef)
        assert isinstance(self.row_layouts, idict)
        assert isinstance(self.column_layouts, idict)

    # }}}

    # {{{ interface impls

    child_attrs = ("row_layouts", "column_layouts")

    buffer: ClassVar[property] = utils.attr("_buffer")

    @property
    def local_max(self) -> numbers.Number:
        raise NotImplementedError

    @property
    def local_min(self) -> numbers.Number:
        raise NotImplementedError

    @property
    def _full_str(self) -> str:
        return f"{self.buffer.buffer.name}[{self.row_layouts}, {self.column_layouts}]"

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
    assert dat.parent is None
    if not dat.axes.is_linear:
        raise ValueError("The provided Dat must be linear")

    axes = dat.axes.localize()
    layout = utils.just_one(axes.leaf_subst_layouts.values())
    return LinearDatBufferExpression(BufferRef(dat.buffer), layout)


@_as_linear_buffer_expression.register
def _(scalar: Scalar) -> ScalarBufferExpression:
    assert scalar.parent is None
    return ScalarBufferExpression(BufferRef(scalar.buffer))
