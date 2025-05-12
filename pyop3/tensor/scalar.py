from __future__ import annotations

import dataclasses
import numbers
from typing import ClassVar

import numpy as np
from mpi4py import MPI

from pyop3 import dtypes, exceptions as exc, utils
from .base import Tensor
from pyop3.buffer import AbstractArrayBuffer, AbstractBuffer, ArrayBuffer
from pyop3.sf import single_star_sf


@utils.record()
class Scalar(Tensor):

    # {{{ instance attrs

    _name: str
    _buffer: AbstractBuffer

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

    # }}}

    # {{{ class attrs

    DEFAULT_PREFIX: ClassVar[str] = "scalar"
    DEFAULT_DTYPE: ClassVar[np.dtype] = dtypes.ScalarType

    # }}}

    def __init__(self, value: numbers.Number | None = None, comm: MPI.Comm | None=None, *, buffer: AbstractBuffer | None = None, name: str | None = None, prefix: str | None = None):
        name = utils.maybe_generate_name(name, prefix, self.DEFAULT_PREFIX)

        if buffer is not None:
            if value is not None or comm is not None:
                raise ValueError("Since 'buffer' is given, 'value' and 'comm' should not be passed")
        else:
            if comm is None:
                comm = MPI.COMM_SELF
            sf = single_star_sf(comm)

            if value is not None:
                data = np.asarray([value])
                buffer = ArrayBuffer(data, sf=sf)
            else:
                buffer = ArrayBuffer.empty(1, sf=sf, dtype=self.DEFAULT_DTYPE)

        if buffer.size != 1:
            raise exc.SizeMismatchException("Expected a buffer with unit size")

        self._name = name
        self._buffer = buffer

    def getitem(self, *, strict=False):
        return self

    def with_context(self, *args, **kwargs):
        return self

    @property
    def context_free(self):
        return self

    @property
    def alloc_size(self) -> int:
        return 1

    @property
    def leaf_layouts(self):  # or all layouts?
        raise NotImplementedError

    @property
    def value(self):
        return utils.just_one(self.buffer._data)
