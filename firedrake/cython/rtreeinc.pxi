from libc.stddef cimport size_t
from libc.stdint cimport uint32_t


cdef extern from "rtree-capi.h":
    ctypedef enum RTreeError:
        Success
        NullPointer
        InvalidDimension

    ctypedef struct RTreeH:
        pass

    RTreeError rtree_bulk_load(
        RTreeH **tree,
        const double *mins,
        const double *maxs,
        const size_t *ids,
        size_t n,
        uint32_t dim
    )

    RTreeError rtree_free(RTreeH *tree)

    RTreeError rtree_locate_all_at_point(
        const RTreeH *tree,
        const double *point,
        size_t **ids_out,
        size_t *nids_out
    )
