"""Cython extensions for 'pyop3.utils'.

This module should not be imported directly. Instead the functions defined here
should be exposed inside 'pyop3.utils'.

"""
from petsctools cimport cpetsc


def try_cast_slice(indices: cpetsc.PetscInt[::1]):
    """Try to cast an array of integers to a Python `slice`."""
    if len(indices) == 0:
        return slice(0, 0, 1)
    elif len(indices) == 1:
        start = indices[0]
        return slice(start, start+1, 1)
    else:
        step = indices[1] - indices[0]
        for i in range(1, len(indices)-1):
            if indices[i+1] - indices[i] != step:
                # non-const step, cannot be a slice
                return indices
        return slice(indices[0], indices[-1]+1, step)
