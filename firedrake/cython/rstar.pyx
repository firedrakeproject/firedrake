# cython: language_level=3

cimport numpy as np
import numpy as np
import ctypes
import cython
from libc.stddef cimport size_t
from libc.stdint cimport uintptr_t

include "rstarinc.pxi"

cdef class RStarTree(object):
    """Python class for holding a native spatial index object."""

    cdef RStar_RTree* tree

    def __cinit__(self, uintptr_t tree_handle):
        self.tree = <RStar_RTree*>0
        if tree_handle == 0:
            raise RuntimeError("invalid tree handle")
        self.tree = <RStar_RTree*>tree_handle

    def __dealloc__(self):
        if self.tree != <RStar_RTree*>0:
            RTree_Free(self.tree)

    @property
    def ctypes(self):
        """Returns a ctypes pointer to the native spatial index."""
        return ctypes.c_void_p(<uintptr_t> self.tree)


@cython.boundscheck(False)
@cython.wraparound(False)
def from_regions(np.ndarray[np.float64_t, ndim=2, mode="c"] regions_lo,
                 np.ndarray[np.float64_t, ndim=2, mode="c"] regions_hi):
    """Builds a spatial index from a set of maximum bounding regions (MBRs).

    regions_lo and regions_hi must have the same size.
    regions_lo[i] and regions_hi[i] contain the coordinates of the diagonally
    opposite lower and higher corners of the i-th MBR, respectively.
    """
    cdef:
        RStarTree rstar_tree
        np.ndarray[np.npy_uintp, ndim=1, mode="c"] ids
        RStar_RTree* rtree
        size_t n
        size_t dim
        RTreeError err 

    assert regions_lo.shape[0] == regions_hi.shape[0]
    assert regions_lo.shape[1] == regions_hi.shape[1]
    n = <size_t>regions_lo.shape[0]
    dim = <size_t>regions_lo.shape[1]
    ids = np.arange(n, dtype=np.uintp)

    err = RTree_FromArray(
        <const double*>regions_lo.data,
        <const double*>regions_hi.data,
        <const size_t*>ids.data,
        n,
        dim,
        &rtree
    )
    if err != Ok:
        PrintRTreeError(err)
        raise RuntimeError("RTree_FromArray failed")
    rstar_tree = RStarTree(<uintptr_t>rtree)
    return rstar_tree
