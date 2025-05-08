from __future__ import annotations

from collections.abc import Sequence
import dataclasses
from numbers import Number
from typing import Any, ClassVar

import numpy as np

from pyop3 import utils
from pyop3.array.base import Array


@utils.record()
class Parameter(Array):
    """Value that can be changed without triggering code generation."""

    # {{{ Instance attrs

    # TODO: can be a scalar or numpy array
    value: Any
    _name: str

    # }}}

    # {{{ Class attrs

    DEFAULT_PREFIX: ClassVar[str] = "param"

    # }}}

    # {{{ Interface impls

    name: ClassVar[property] = property(lambda self: self._name)
    parent: ClassVar[None] = None
    dim: ClassVar[int] = 0

    def getitem(self, strict=False):
        return self

    def copy(self) -> Parameter:
        if not isinstance(value, numbers.Number):
            raise NotImplementedError

        name = f"{self.name}_copy"
        return self.__record_init__(_name=name)

    # }}}

    def __init__(self, value: Sequence[Number] | Number, *, name: str | None =None, prefix: str | None=None) -> None:
        name = utils.maybe_generate_name(name, prefix, self.DEFAULT_PREFIX)

        if isinstance(value, Number):
            value = utils.as_numpy_scalar(value)
        else:
            raise NotImplementedError("TODO")

        self._name = name
        self.value = value

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
