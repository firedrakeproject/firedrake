from __future__ import annotations

import numbers
from collections.abc import Sequence
from typing import Any

import numpy as np

import pyop3.dtypes
from pyop3 import utils


# NOTE: This class should be sufficiently generic to work with non-numpy arrays
class ArrayReference:
    """Class representing an array that has been indexed.

    Parameters
    ----------
    base : numpy.ndarray
        The underlying array. This must be a flat view.
    indices : collections.abc.Sequence[numbers.Integral]
        The indices from the underlying array. This is also flat.
    shape : tuple[int, ...]
        The shape of the array view.

    """
    def __init__(
        self,
        base: np.ndarray,
        indices: Sequence[numbers.Integral],
        shape: tuple[int, ...],
    ) -> None:
        indices = np.asarray(indices, dtype=pyop3.dtypes.IntType)

        assert indices.size == np.prod(shape, dtype=int)

        self.base = base
        self.indices = indices
        self.shape = shape

    def __getitem__(self, indices: Any, /) -> Any:
        if isinstance(indices, tuple):
            raise NotImplementedError(
                "__getitem__ not implemented for multiple indices"
            )

        if indices is Ellipsis:
            return self
        if not isinstance(indices, numbers.Integral):
            raise NotImplementedError("TODO")
        return self.base[utils.just_one(self.indices)[indices*self._block_size]]

    def __setitem__(self, indices: Any, value: Any, /) -> None:
        if not (indices is Ellipsis or isinstance(indices, numbers.Integral)):
            raise NotImplementedError("Only '...' and integer assignment supported for arrayrefs")

        if not isinstance(indices, tuple):
            indices = (indices,)

        indices_ = []
        for axis in range(self.ndim):
            try:
                idx = indices[axis]
            except IndexError:
                idx = slice(None)
        indices = tuple(indices_)

        # 'value' can be a scalar, a block-sized vector, or a vector with the same
        # size as 'indices'.
        value = np.asarray(value, dtype=self.base.dtype)

        def get_indices():
            return self.indices.reshape(self.shape)[indices].ravel()
            idxs = get_indices_per_axis(0) * self._axis_strides[0]
            for axis in range(1, self.ndim):
                # This magic method turns ([0, 10, 20], [0, 1]) into
                # [0, 1, 10, 11, 20, 21]
                idxs = np.add.outer(
                    idxs,
                    get_indices_per_axis(axis) * self._axis_strides[axis],
                )
            breakpoint()
            return idxs

        def get_indices_per_axis(axis: int):
            sl = indices[axis]
            return self.indices[axis][sl]

        self.base[get_indices()] = value
        return

        iii = get_indices()
        breakpoint()

        if value.size == 1:
            self.base[get_indices()] = value
            # for ib in range(self._block_size):
            #     self.base[base_indices[get_indices(ib)]] = value
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
        return np.prod(self.shape, dtype=int)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def _block_size(self) -> int:
        return np.prod(self.block_shape, dtype=int)

    def _as_array_or_slice(self, indices):
        if isinstance(indices, slice):
            return indices
        else:
            return np.asarray(indices, dtype=pyop3.dtypes.IntType)
