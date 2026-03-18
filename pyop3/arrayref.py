from __future__ import annotations

import numbers
from typing import Any

import numpy as np


# NOTE: This class should be sufficiently generic to work with non-numpy arrays
class ArrayReference:
    """Class representing an array that has been indexed."""
    def __init__(self, base: np.ndarray, indices: np.ndarray[numbers.Integral]) -> None:
        self.base = base
        self.indices = indices

    def __getitem__(self, indices: Any, /) -> Any:
        raise NotImplementedError

    def __setitem__(self, indices: Any, value: Any, /) -> Any:
        raise NotImplementedError

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        assert copy is not False
        return self.base[self.indices]
