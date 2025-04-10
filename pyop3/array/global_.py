import numbers

import numpy as np

from pyop3 import utils
from pyop3.array.base import Array
from pyop3.buffer import AbstractBuffer, Buffer
from pyop3.sf import single_star_sf


class Global(Array):
    DEFAULT_PREFIX = "global"

    def __init__(self, buffer: AbstractBuffer | None = None, *, value: numbers.Number | None = None, comm=None, name: str | None = None, prefix: str | None = None):
        if buffer is not None:
            assert value is None and comm is None
        else:
            assert comm is not None
            sf = single_star_sf(comm)
            if value is not None:
                data = np.asarray([value])
                buffer = Buffer(data, sf=sf)
            else:
                buffer = Buffer.zeros(1, sf=sf)

        self.buffer = buffer
        self.name = utils.maybe_generate_name(name, prefix, self.DEFAULT_PREFIX)

    # {{{ Array impls

    @property
    def dim(self) -> int:
        return 0

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
