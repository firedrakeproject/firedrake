from libc.stdint cimport int64_t, uint8_t, uint32_t, uint64_t

cdef extern from "spatialindex/capi/sidx_api.h":
    ctypedef enum RTError:
       RT_None = 0,
       RT_Debug = 1,
       RT_Warning = 2,
       RT_Failure = 3,
       RT_Fatal = 4

    ctypedef enum RTIndexType:
       RT_RTree = 0,
       RT_MVRTree = 1,
       RT_TPRTree = 2,
       RT_InvalidIndexType = -99

    ctypedef enum RTStorageType:
       RT_Memory = 0,
       RT_Disk = 1,
       RT_Custom = 2,
       RT_InvalidStorageType = -99

    ctypedef enum RTIndexVariant:
       RT_Linear = 0,
       RT_Quadratic = 1,
       RT_Star = 2,
       RT_InvalidIndexVariant = -99

    struct Index
    struct Tools_PropertySet

    ctypedef Index *IndexH
    ctypedef Tools_PropertySet *IndexPropertyH

    IndexPropertyH IndexProperty_Create()
    RTError IndexProperty_SetIndexType(IndexPropertyH hProp, RTIndexType value)
    RTError IndexProperty_SetDimension(IndexPropertyH hProp, uint32_t value)
    RTError IndexProperty_SetIndexVariant(IndexPropertyH hProp, RTIndexVariant value)
    RTError IndexProperty_SetIndexStorage(IndexPropertyH hProp, RTStorageType value)
    void IndexProperty_Destroy(IndexPropertyH hProp)
    IndexH Index_Create(IndexPropertyH hProp)
    IndexH Index_CreateWithArray(IndexPropertyH hProp, uint64_t n, uint32_t dimension, 
                                 uint64_t i_stri, uint64_t d_i_stri, uint64_t d_j_stri,
                                 int64_t *ids, double *mins, double *maxs)
    RTError Index_Intersects_id(IndexH index, double* pdMin, double* pdMax, uint32_t nDimension,
                                int64_t** ids, uint64_t* nResults)
    void Index_Destroy(IndexH index)
