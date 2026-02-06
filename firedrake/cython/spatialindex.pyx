# cython: language_level=3

cimport numpy as np
import numpy as np
import ctypes
import cython
from libc.stdint cimport uintptr_t
from libc.stdlib cimport free

include "spatialindexinc.pxi"


cdef IndexPropertyH _make_index_properties(uint32_t dim) except *:
    cdef IndexPropertyH ps = NULL
    cdef RTError err = RT_None

    ps = IndexProperty_Create()
    if ps == NULL:
        raise RuntimeError("failed to create index properties")

    err = IndexProperty_SetIndexType(ps, RT_RTree)
    if err != RT_None:
        IndexProperty_Destroy(ps)
        raise RuntimeError("failed to set index type")

    err = IndexProperty_SetDimension(ps, dim)
    if err != RT_None:
        IndexProperty_Destroy(ps)
        raise RuntimeError("failed to set dimension")

    err = IndexProperty_SetIndexStorage(ps, RT_Memory)
    if err != RT_None:
        IndexProperty_Destroy(ps)
        raise RuntimeError("failed to set index storage")

    return ps

cdef class SpatialIndex(object):
    """Python class for holding a native spatial index object."""

    cdef IndexH index

    def __cinit__(self, uintptr_t handle):
        self.index = NULL
        if handle == 0:
            raise ValueError("SpatialIndex handle must be nonzero")
        self.index = <IndexH>handle

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
        np.ndarray[np.int64_t, ndim=1, mode="c"] ids
        IndexPropertyH ps
        IndexH index
        uint64_t n
        uint32_t dim
        uint64_t i_stri
        uint64_t d_i_stri
        uint64_t d_j_stri

    assert regions_lo.shape[0] == regions_hi.shape[0]
    assert regions_lo.shape[1] == regions_hi.shape[1]
    n = <uint64_t>regions_lo.shape[0]
    dim = <uint32_t>regions_lo.shape[1]

    ps = NULL
    index = NULL
    try:
        ps = _make_index_properties(dim)
        if n == 0:
            # Index_CreateWithArray will fail for n=0, so create an empty index instead.
            index = Index_Create(ps)
        else:
            ids = np.arange(n, dtype=np.int64)

            # Calculate the strides
            i_stri = <uint64_t>(ids.strides[0] // ids.itemsize)
            d_i_stri = <uint64_t>(regions_lo.strides[0] // regions_lo.itemsize)
            d_j_stri = <uint64_t>(regions_lo.strides[1] // regions_lo.itemsize)

            index = Index_CreateWithArray(ps, n, dim,
                                          i_stri, d_i_stri, d_j_stri,
                                          <int64_t*>ids.data,
                                          <double*>regions_lo.data,
                                          <double*>regions_hi.data)
        if index == NULL:
            raise RuntimeError("failed to create index")

        spatial_index = SpatialIndex(<uintptr_t>index)
    finally:
        IndexProperty_Destroy(ps)

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
