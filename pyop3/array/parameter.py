from collections.abc import Sequence
import dataclasses
from numbers import Number
from typing import Any, ClassVar

import numpy as np

from pyop3 import utils
from pyop3.array.base import Array


@dataclasses.dataclass(init=False, frozen=True)
class Parameter(Array):
    """Value that can be changed without triggering code generation."""

    # {{{ Instance attrs

    # TODO: can be a scalar or numpy array
    value: Any

    # }}}

    # {{{ Class attrs

    DEFAULT_PREFIX: ClassVar[str] = "param"

    # }}}

    def __init__(self, value: Sequence[Number] | Number, *, name: str | None =None, prefix: str | None=None) -> None:
        if isinstance(value, Number):
            value = utils.as_numpy_scalar(value)
        else:
            raise NotImplementedError("TODO")

        super().__init__(name, prefix=prefix)
        object.__setattr__(self, "value", value)

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
