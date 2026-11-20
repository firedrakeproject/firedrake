from __future__ import annotations

from typing import Any

import numpy as np


# NOTE: This class should be sufficiently generic to work with non-numpy arrays
class ArrayReference:
    """Class representing an array that has been indexed.

    This class is necessary to work around numpy indexing behaviour when passed
    integer arrays. In numpy, when one writes

    .. code-block:: python

       y = x[::2]

    the returned value is a *view*. However, when one does

    .. code-block:: python

       y = x[[1, 2, 4]]

    then ``y`` is a *copy*. Subsequent modifications of ``y`` will not propagate
    back to ``x``.

    This is an issue in pyop3 for things like :meth:`pyop3.Dat.data_ro_with_halos`
    and so we use this class to ensure that the write-back behaviour is preserved
    under integer array indexing.

    Parameters
    ----------
    base : numpy.ndarray
        The underlying array. This must be a flat view.
    indices : numpy.ndarray
        The indices that address ``base``. This has shape.

    """
    def __init__(self, base: np.ndarray, indices: np.ndarray) -> None:
        assert base.ndim == 1, "'base' must be flattened"
        assert isinstance(indices, np.ndarray)

        self.base = base
        self.indices = indices

    def __getitem__(self, indices: Any, /) -> Any:
        return self.base[self.indices[indices]]

    def __setitem__(self, indices: Any, value: Any, /) -> None:
        self.base[self.indices[indices]] = value

    def __array__(
        self,
        dtype: np.dtype | None = None,
        copy: bool | None = None,
    ) -> np.ndarray:
        # Note that the 'dtype' argument is handled by numpy directly
        if copy is False:
            raise ValueError("Casting array references to numpy arrays requires a copy")
        return self.base[self.indices]

    @property
    def shape(self) -> tuple[int, ...]:
        return self.indices.shape

    @property
    def size(self) -> int:
        return np.prod(self.shape, dtype=int)

    @property
    def ndim(self) -> int:
        return len(self.shape)
