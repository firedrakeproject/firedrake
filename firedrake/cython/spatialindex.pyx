# cython: language_level=3

cimport numpy as np
import numpy as np
import ctypes
import cython
from libc.stdint cimport uintptr_t
from libc.stdlib cimport free

include "spatialindexinc.pxi"

cdef class SpatialIndex(object):
    """Python class for holding a native spatial index object."""

    cdef IndexH index

    def __cinit__(self, uint32_t dim):
        """Initialize a native spatial index.

        :arg dim: spatial (geometric) dimension
        """
        cdef IndexPropertyH ps = NULL
        cdef RTError err = RT_None

        self.index = NULL
        try:
            ps = IndexProperty_Create()
            if ps == NULL:
                raise RuntimeError("failed to create index properties")

            err = IndexProperty_SetIndexType(ps, RT_RTree)
            if err != RT_None:
                raise RuntimeError("failed to set index type")

            err = IndexProperty_SetDimension(ps, dim)
            if err != RT_None:
                raise RuntimeError("failed to set dimension")

            err = IndexProperty_SetIndexStorage(ps, RT_Memory)
            if err != RT_None:
                raise RuntimeError("failed to set index storage")

            self.index = Index_Create(ps)
            if self.index == NULL:
                raise RuntimeError("failed to create index")
        finally:
            IndexProperty_Destroy(ps)

    def __dealloc__(self):
        Index_Destroy(self.index)

    @property
    def ctypes(self):
        """Returns a ctypes pointer to the native spatial index."""
        return ctypes.c_void_p(<uintptr_t> self.index)


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
        SpatialIndex spatial_index
        int64_t i
        uint32_t dim
        RTError err

    assert regions_lo.shape[0] == regions_hi.shape[0]
    assert regions_lo.shape[1] == regions_hi.shape[1]
    dim = regions_lo.shape[1]

    spatial_index = SpatialIndex(dim)
    for i in xrange(len(regions_lo)):
        err = Index_InsertData(spatial_index.index, i, &regions_lo[i, 0], &regions_hi[i, 0], dim, NULL, 0)
        if err != RT_None:
            raise RuntimeError("failed to insert data into spatial index")
    return spatial_index


def bounding_boxes(SpatialIndex sidx not None, np.ndarray[np.float64_t, ndim=1] x):
    """Given a spatial index and a point, return the bounding boxes the point is in.

    :arg sidx: the SpatialIndex
    :arg x: the point
    :returns: a numpy array of candidate bounding boxes."""
    cdef int dim = x.shape[0]
    cdef int64_t *ids = NULL
    cdef uint64_t i
    cdef np.ndarray[np.int64_t, ndim=1, mode="c"] pyids
    cdef uint64_t nids

    err = Index_Intersects_id(sidx.index, &x[0], &x[0], dim, &ids, &nids)
    if err != RT_None:
        raise RuntimeError("intersection failed")

    pyids = np.empty(nids, dtype=np.int64)
    for i in range(nids):
        pyids[i] = ids[i]
    free(ids)
    return pyids
