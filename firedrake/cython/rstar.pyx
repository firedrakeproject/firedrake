# cython: language_level=3

cimport numpy as np
import numpy as np
import ctypes
import cython
from libc.stddef cimport size_t
from libc.stdint cimport uintptr_t

include "rstarinc.pxi"

cdef class RTree(object):
    """Python class for holding a native spatial index object."""

    cdef RTreeH* tree

    def __cinit__(self, uintptr_t tree_handle):
        self.tree = <RTreeH*>0
        if tree_handle == 0:
            raise RuntimeError("invalid tree handle")
        self.tree = <RTreeH*>tree_handle

    def __dealloc__(self):
        if self.tree != <RTreeH*>0:
            rtree_free(self.tree)

    @property
    def ctypes(self):
        """Returns a ctypes pointer to the native spatial index."""
        return ctypes.c_void_p(<uintptr_t> self.tree)


@cython.boundscheck(False)
@cython.wraparound(False)
def build_from_aabb(np.ndarray[np.float64_t, ndim=2, mode="c"] coords_min,
                    np.ndarray[np.float64_t, ndim=2, mode="c"] coords_max,
                    np.ndarray[np.npy_uintp, ndim=1, mode="c"] ids = None):
    """Builds rtree from two arrays of shape (n, dim) containing the coordinates
    of the lower and upper corners of n axis-aligned bounding boxes, and an
    optional array of shape (n,) containing integer ids for each box.

    Parameters
    ----------
    coords_min : (n, dim) array
        The lower corner coordinates of the bounding boxes.
    regions_hi : (n, dim) array
        The upper corner coordinates of the bounding boxes.
    ids : (n,) array, optional
        Integer ids for each box. If not provided, defaults to 0, 1, ..., n-1.

    Returns
    -------
    RTree
        An RTree object containing the built R*-tree.
    """    
    cdef:
        RTreeH* rtree
        size_t n
        size_t dim
        RTreeError err

    if coords_min.shape[0] != coords_max.shape[0] or coords_min.shape[1] != coords_max.shape[1]:
        raise ValueError("coords_min and coords_max must have the same shape")

    n = <size_t>coords_min.shape[0]
    dim = <size_t>coords_min.shape[1]
    if ids is not None and ids.shape[0] != n:
        raise ValueError("Mismatch between number of boxes and number of ids")
    else:
        ids = np.arange(n, dtype=np.uintp)

    err = rtree_bulk_load(
        &rtree,
        <const double*>coords_min.data,
        <const double*>coords_max.data,
        <const size_t*>ids.data,
        n,
        dim
    )
    if err != Success:
        raise RuntimeError("RTree_FromArray failed")

    return RTree(<uintptr_t>rtree)
