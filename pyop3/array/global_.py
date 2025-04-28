import dataclasses
import numbers
from typing import ClassVar

import numpy as np
from mpi4py import MPI

from pyop3 import utils
from pyop3.array.base import DistributedArray
from pyop3.buffer import AbstractBuffer, ArrayBuffer
from pyop3.sf import single_star_sf


@utils.record(init=False)
class Global(DistributedArray):

    # {{{ Instance attrs

    _buffer: AbstractBuffer

    # }}}

    # {{{ Class attrs

    DEFAULT_PREFIX: ClassVar[str] = "global"

    # }}}

    def __init__(self, buffer: AbstractBuffer | None = None, *, value: numbers.Number | None = None, comm=None, name: str | None = None, prefix: str | None = None):
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

        super().__init__(name, prefix=prefix)
        object.__setattr__(self, "_buffer", buffer)

    # {{{ Array impls

    @property
    def dim(self) -> int:
        return 0

    # }}}

    # {{{ DistributedArray impls

    @property
    def buffer(self) -> AbstractBuffer:
        return self._buffer

    @property
    def comm(self) -> MPI.Comm:
        return self.buffer.comm

    # }}}

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
