import dataclasses
import numbers
from typing import ClassVar

import numpy as np
from mpi4py import MPI

from pyop3 import utils
from pyop3.array.base import DistributedArray
from pyop3.buffer import AbstractArrayBuffer, AbstractBuffer, ArrayBuffer
from pyop3.sf import single_star_sf


@utils.record()
class Global(DistributedArray):

    # {{{ instance attrs

    _buffer: AbstractBuffer
    _name: str

    # }}}

    # {{{ interface impls

    buffer: ClassVar[ArrayBuffer] = utils.attr("_buffer")
    name: ClassVar[str] = utils.attr("_name")
    dim: ClassVar[int] = 0
    parent: ClassVar[None] = None

    @property
    def comm(self) -> MPI.Comm:
        return self.buffer.comm

    # }}}

    # {{{ class attrs

    DEFAULT_PREFIX: ClassVar[str] = "global"

    # }}}

    def __init__(self, buffer: AbstractBuffer | None = None, *, value: numbers.Number | None = None, comm=None, name: str | None = None, prefix: str | None = None):
        name = utils.maybe_generate_name(name, prefix, self.DEFAULT_PREFIX)
        if buffer is not None:
            assert value is None and comm is None
        else:
            assert comm is not None
            sf = single_star_sf(comm)
            if value is not None:
                data = np.asarray([value])
                buffer = ArrayBuffer(data, sf=sf)
            else:
                buffer = ArrayBuffer.zeros(1, sf=sf)

        self._buffer = buffer
        self._name = name

    @property
    def dtype(self):
        return self.buffer.dtype

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
