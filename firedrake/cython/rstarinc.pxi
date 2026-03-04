from libc.stddef cimport size_t
from libc.stdint cimport uint32_t


cdef extern from "rstar-capi.h":
    ctypedef enum RTreeError:
        Success
        NullPointer
        InvalidDimension
        NodeNotLeaf

    ctypedef struct RTreeH:
        pass

    ctypedef struct RTreeNodeH:
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

    RTreeError rtree_root_node(
        const RTreeH *tree,
        RTreeNodeH **node
    )

    RTreeError rtree_node_children(
        const RTreeNodeH *node,
        RTreeNodeH ***children_out,
        size_t *nchildren_out
    )

    RTreeError rtree_node_id(
        const RTreeNodeH *node,
        size_t *id_out
    )

    RTreeError rtree_node_envelope(
        const RTreeNodeH *node,
        double *mins_out,
        double *maxs_out
    )
