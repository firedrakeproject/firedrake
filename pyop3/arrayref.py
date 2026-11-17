from __future__ import annotations

import math
import numbers
from collections.abc import Sequence
from typing import Any

import numpy as np

import pyop3.dtypes
from pyop3 import utils


# NOTE: This class should be sufficiently generic to work with non-numpy arrays
class ArrayReference:
    """Class representing an array that has been indexed."""
    def __init__(self, base: np.ndarray, indices: Sequence, block_shape: tuple[int, ...] = ()) -> None:
        self.base = base
        self.indices = np.asarray(indices, dtype=pyop3.dtypes.IntType)
        self.block_shape = block_shape

    def __getitem__(self, indices: Any, /) -> Any:
        if indices is Ellipsis:
            return self
        if not isinstance(indices, numbers.Integral):
            raise NotImplementedError("TODO")
        return self.base[self.indices[indices*self._block_size]]

    def __setitem__(self, indices: Any, value: Any, /) -> None:
        if indices is Ellipsis:
            target = self.indices
        elif isinstance(indices, numbers.Integral):
            target = self.indices.reshape((-1, self._block_size))[indices]
        else:
            raise NotImplementedError("Only '...' and integer assignment supported for arrayrefs")

        # 'value' can be a scalar, a block-sized vector, or a vector with the same
        # size as 'target'. Each case is a single write into 'base' because this
        # is called for every eager assignment to an indexed tensor.
        value = np.asarray(value, dtype=self.base.dtype)
        if value.size == 1 or value.size == target.size:
            self.base[target] = value.reshape(-1)
        elif value.size == self._block_size:
            self.base[target.reshape((-1, self._block_size))] = value.reshape(-1)
        else:
            raise NotImplementedError(f"Cannot assign something with shape {value.shape}")

    def __array__(self, dtype: np.dtype | None = None, copy: bool | None = None) -> np.ndarray:
        # Note that the 'dtype' argument is handled by numpy directly
        if copy is False:
            raise ValueError("Casting array references to numpy arrays requires a copy")
        return self.base[self.indices].reshape((-1, *self.block_shape))

    @property
    def size(self):
        return self.indices.size

    @property
    def shape(self) -> tuple[int, ...]:
        ni = utils.strict_floordiv(self.indices.size, self._block_size)
        return (ni, *self.block_shape)

    @property
    def _block_size(self) -> int:
        return math.prod(self.block_shape)
