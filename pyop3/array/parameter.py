import numbers

import numpy as np

from pyop3 import utils
from pyop3.array.base import Array


class Parameter(Array):
    """Scalar value that can be changed without triggering code generation."""

    DEFAULT_PREFIX = "param"

    def __init__(self, value: numbers.Number, *, name: str | None =None, prefix: str | None=None) -> None:
        self.value = utils.as_numpy_scalar(value)
        self.name = utils.maybe_generate_name(name, prefix, self.DEFAULT_PREFIX)

    # {{{ Array impls

    @property
    def dim(self) -> int:
        return 0

    def getitem(self, strict=False):
        return self

    # }}}

    @property
    def dtype(self) -> np.dtype:
        return self.value.dtype

    def with_context(self):
        raise NotImplementedError

    @property
    def context_free(self):
        raise NotImplementedError

    @property
    def alloc_size(self) -> int:
        return 1

    @property
    def leaf_layouts(self):  # or all layouts?
        raise NotImplementedError
