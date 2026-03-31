from __future__ import annotations

import numbers
from typing import Any

import numpy as np


# NOTE: This class should be sufficiently generic to work with non-numpy arrays
class ArrayReference:
    """Class representing an array that has been indexed."""
    def __init__(self, base: np.ndarray, indices: np.ndarray, block_shape: tuple[int, ...] = ()) -> None:
        self.base = base
        self.indices = indices
        self.block_shape = block_shape

    def __getitem__(self, indices: Any, /) -> Any:
        if not isinstance(indices, numbers.Integral):
            raise NotImplementedError("TODO")
        return self.base[self.indices[indices*self._block_size]]

    def __setitem__(self, indices: Any, value: Any, /) -> Any:
        if not isinstance(indices, numbers.Integral):
            raise NotImplementedError("TODO")
        self.base[self.indices[indices*self._block_size]] = value

    def __array__(self, dtype: np.dtype | None = None, copy: bool | None = None) -> np.ndarray:
        # Note that the 'dtype' argument is handled by numpy directly
        if copy is False:
            raise ValueError("Casting array references to numpy arrays requires a copy")
        return self.base[self.indices].reshape((-1, *self.block_size))

    @property
    def _block_size(self) -> int:
        return np.prod(self.block_shape, dtype=int)
