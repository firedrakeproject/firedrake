from __future__ import annotations

import dataclasses
import numbers
from typing import ClassVar

import numpy as np
from immutabledict import immutabledict as idict
from mpi4py import MPI

from pyop3 import dtypes, exceptions as exc, utils
from pyop3.tree.axis_tree.tree import UNIT_AXIS_TREE
from .base import Tensor
from pyop3.buffer import AbstractArrayBuffer, AbstractBuffer, ArrayBuffer
from pyop3.sf import single_star_sf


@utils.record()
class Scalar(Tensor):

    # {{{ instance attrs

    _name: str
    _buffer: AbstractBuffer

    def __init__(self, value: numbers.Number | None = None, comm: MPI.Comm=MPI.COMM_SELF, *, buffer: AbstractBuffer | None = None, constant: bool | None = None, name: str | None = None, prefix: str | None = None):
        name = utils.maybe_generate_name(name, prefix, self.DEFAULT_PREFIX)

        if buffer is not None:
            # clean me up
            assert constant is None
            if value is not None or comm is not None:
                raise ValueError("Since 'buffer' is given, 'value' and 'comm' should not be passed")
        else:
            sf = single_star_sf(comm)

            buffer_kwargs = {"sf": sf}
            if constant is not None:
                buffer_kwargs["constant"] = constant

            if value is not None:
                data = np.asarray([value])
                buffer = ArrayBuffer(data, **buffer_kwargs)
            else:
                buffer = ArrayBuffer.empty(1, dtype=self.DEFAULT_DTYPE, **buffer_kwargs)

        if buffer.size != 1:
            raise exc.SizeMismatchException("Expected a buffer with unit size")

        self._name = name
        self._buffer = buffer

    # }}}

    # {{{ interface impls

    name: ClassVar[str] = utils.attr("_name")
    buffer: ClassVar[ArrayBuffer] = utils.attr("_buffer")
    dim: ClassVar[int] = 0
    parent: ClassVar[None] = None

    def copy(self) -> Scalar:
        name = f"{self.name}_copy"
        buffer = self._buffer.copy()
        return self.__record_init__(_name=name, _buffer=buffer)

    shape = (UNIT_AXIS_TREE,)
    loop_axes = idict()
    axis_trees = ()

    @property
    def _full_str(self) -> str:
        return f"*{self.name}"

    @property
    def comm(self) -> MPI.Comm:
        return self.buffer.comm

    def concretize(self):
        from pyop3.expr import as_linear_buffer_expression

        return as_linear_buffer_expression(self)

    @property
    def local_max(self) -> numbers.Number:
        return self.value

    @property
    def local_min(self) -> numbers.Number:
        return self.local_max

    # }}}

    # {{{ class attrs

    DEFAULT_PREFIX: ClassVar[str] = "scalar"
    DEFAULT_DTYPE: ClassVar[np.dtype] = dtypes.ScalarType

    # }}}

    @property
    def constant(self) -> bool:
        return self.buffer.constant

    def getitem(self, *, strict=False):
        return self

    def with_context(self, *args, **kwargs):
        return self

    @property
    def alloc_size(self) -> int:
        return 1

    @property
    def leaf_layouts(self):  # or all layouts?
        raise NotImplementedError

    @property
    def value(self):
        return utils.just_one(self.buffer.data_ro)

    # {{{ arithmetic

    # TODO: also think about comm sizes? is this valid for size>1?
    # NOTE: Same impl needed for ScalarExpressions...
    def __add__(self, other: ExpressionT, /) -> ExpressionT:
        if self.constant:
            if isinstance(other, numbers.Number):
                return Scalar(self.value+other, constant=True)
            elif isinstance(other, Scalar) and other.constant:
                return Scalar(self.value+other.value, constant=True)
        return super().__add__(other)

    # }}}
