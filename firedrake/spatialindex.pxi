from libc.stdint cimport uint8_t, uint32_t, int64_t
from libcpp.string cimport string

ctypedef uint8_t byte
ctypedef int64_t id_type

cdef extern from "spatialindex/SpatialIndex.h" namespace "SpatialIndex":
     cdef cppclass IStorageManager

     cdef cppclass IShape:
         pass

     cdef cppclass Region(IShape):
         Region() except +
         Region(double *pLow, double *pHigh, uint32_t dimension) except +

     cdef cppclass ISpatialIndex:
         void insertData(uint32_t len, byte *pData, IShape& shape, id_type identifier)


cdef extern from "spatialindex/SpatialIndex.h" namespace "SpatialIndex::StorageManager":
     cdef IStorageManager *createNewMemoryStorageManager() except +


cdef extern from "spatialindex/SpatialIndex.h" namespace "SpatialIndex::RTree":
     cdef ISpatialIndex *returnRTree(IStorageManager&, PropertySet&) except +


cdef extern from "spatialindex/SpatialIndex.h" namespace "Tools":
     cdef enum VariantType:
         VT_ULONG

     union VariantValue:
         uint32_t ulVal

     cdef cppclass Variant:
         Variant() except +

         VariantType m_varType
         VariantValue m_val

     cdef cppclass PropertySet:
         PropertySet() except +
         void setProperty(string, Variant&)
