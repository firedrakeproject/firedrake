from __future__ import annotations

import numbers
from collections.abc import Sequence
from typing import Any

import numpy as np

import pyop3.dtypes
from pyop3 import utils


# NOTE: This class should be sufficiently generic to work with non-numpy arrays
class ArrayReference:
    """Class representing an array that has been indexed."""
    def __init__(
        self,
        base: np.ndarray,
        indices: Sequence[Sequence | slice],
        block_shape: tuple[int, ...] = (),
    ) -> None:
        if len(indices) != len(base.shape):
            raise ValueError("Expect index values for each axis of 'base'")

        self.base = base
        try:
            self.indices = tuple(
                self._as_array_or_slice(idxs) for idxs in indices
            )
        except:
            breakpoint()
        self.block_shape = block_shape

    def __getitem__(self, indices: Any, /) -> Any:
        if len(self.indices) > 1:
            raise NotImplementedError

        if indices is Ellipsis:
            return self
        if not isinstance(indices, numbers.Integral):
            raise NotImplementedError("TODO")
        return self.base[utils.just_one(self.indices)[indices*self._block_size]]

    def __setitem__(self, indices: Any, value: Any, /) -> None:
        print(indices)
        breakpoint()
        if len(self.base.shape) == 1:
            indices = (indices,)

        # if indices is Ellipsis and len(self.base.shape
        if len(self.indices) > 1:
            breakpoint()
            raise NotImplementedError
        else:
            base_indices = utils.just_one(self.indices)

        if not (indices is Ellipsis or isinstance(indices, numbers.Integral)):
            raise NotImplementedError("Only '...' and integer assignment supported for arrayrefs")

        # 'value' can be a scalar, a block-sized vector, or a vector with the same
        # size as 'indices'.
        value = np.asarray(value, dtype=self.base.dtype)

        def get_indices(ib):
            if indices is Ellipsis:
                return slice(ib, None, self._block_size)
            else:
                return indices*self._block_size+ib

        if value.size == 1:
            for ib in range(self._block_size):
                self.base[base_indices[get_indices(ib)]] = value
        elif value.size == self._block_size:
            for ib in range(self._block_size):
                self.base[base_indices[get_indices(ib)]] = value[ib]
        elif (
            indices is Ellipsis and value.size == base_indices.size
            or value.size == len(indices) * self._block_size
        ):
            value = value.reshape((-1, self._block_size))
            for ib in range(self._block_size):
                self.base[base_indices[get_indices(ib)]] = value[:, ib]
        else:
            raise NotImplementedError(f"Cannot assign something with shape {value.shape}")

    def __array__(
        self,
        dtype: np.dtype | None = None,
        copy: bool | None = None,
    ) -> np.ndarray:
        # Note that the 'dtype' argument is handled by numpy directly
        if copy is False:
            raise ValueError("Casting array references to numpy arrays requires a copy")
        return self.base[self.indices].reshape((-1, *self.block_shape))

    @property
    def size(self) -> int:
        return np.prod((idxs.size for idxs in self.indices), dtype=int)

    @property
    def shape(self) -> tuple[int, ...]:
        if len(self.indices) > 1:
            raise NotImplementedError
        ni = utils.strict_floordiv(utils.just_one(self.indices).size, self._block_size)
        return (ni, *self.block_shape)

    @property
    def _block_size(self) -> int:
        return np.prod(self.block_shape, dtype=int)

    def _as_array_or_slice(self, indices):
        if isinstance(indices, slice):
            return indices
        else:
            return np.asarray(indices, dtype=pyop3.dtypes.IntType)
