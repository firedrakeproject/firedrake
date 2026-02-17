from libc.stddef cimport size_t


cdef extern from "rstar_capi.h":
    ctypedef enum RTreeError:
        Ok
        NullPointer
        DimensionNotImplemented
        SizeOverflow
        OutputTooSmall
        Panic

    ctypedef struct RStar_RTree:
        pass

    void PrintRTreeError(RTreeError error)

    RTreeError RTree_FromArray(const double *mins,
                               const double *maxs,
                               const size_t *ids,
                               size_t len,
                               size_t dim,
                               RStar_RTree **out_tree)
    RTreeError RTree_Free(RStar_RTree *tree)
    RTreeError RTree_LocateAllAtPoint(const RStar_RTree *tree,
                                      const double *point,
                                      size_t **ids,
                                      size_t *nids)
