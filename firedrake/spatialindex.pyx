cimport numpy as np
import ctypes
from libc.stdint cimport uintptr_t

include "spatialindex.pxi"

cdef class SpatialIndex(object):
    cdef IStorageManager *storage_manager
    cdef ISpatialIndex *spatial_index

    def __cinit__(self, uint32_t dim):
        cdef:
            PropertySet ps
            Variant var

        self.storage_manager = NULL
        self.spatial_index = NULL

        var.m_varType = VT_ULONG
        var.m_val.ulVal = dim
        ps.setProperty("Dimension", var)

        self.storage_manager = createNewMemoryStorageManager()
        self.spatial_index = returnRTree(self.storage_manager[0], ps)

    def __dealloc__(self):
        del self.spatial_index
        del self.storage_manager

    @property
    def ctypes(self):
        return ctypes.c_void_p(<uintptr_t> self.spatial_index)


def from_regions(np.ndarray[np.float64_t, ndim=2, mode="c"] regions_lo,
                 np.ndarray[np.float64_t, ndim=2, mode="c"] regions_hi):
    cdef:
        SpatialIndex spatial_index
        Region region
        int i, dim

    assert regions_lo.shape[0] == regions_hi.shape[0]
    assert regions_lo.shape[1] == regions_hi.shape[1]
    dim = regions_lo.shape[1]

    spatial_index = SpatialIndex(dim)
    for i in xrange(len(regions_lo)):
        region = Region(&regions_lo[i, 0], &regions_hi[i, 0], dim)
        spatial_index.spatial_index.insertData(0, NULL, region, i)
    return spatial_index
