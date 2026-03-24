from libc.stddef cimport size_t


<<<<<<< HEAD:firedrake/cython/rstarinc.pxi
cdef extern from "rstar_capi.h":
=======
cdef extern from "rtree-capi.h":
>>>>>>> 12de8f60f (renaming and use `firedrake_rtree` package):firedrake/cython/rtreeinc.pxi
    ctypedef enum RTreeError:
        Ok
        NullPointer
<<<<<<< HEAD:firedrake/cython/rstarinc.pxi
        DimensionNotImplemented
        SizeOverflow
        OutputTooSmall
        Panic
=======
        InvalidDimension
        EmptyNodeEnvelope

    ctypedef struct RStar_RTree:
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

    RTreeError rtree_free_ids(size_t *ids, size_t n)

    RTreeError rtree_locate_all_at_point(
        const RTreeH *tree,
        const double *point,
        size_t **ids_out,
        size_t *nids_out
    )

    RTreeError rtree_depth(const RTreeH *tree, size_t *depth_out)

    RTreeError rtree_root_node(
        const RTreeH *tree,
        RTreeNodeH **node
    )

    RTreeError rtree_node_children(
        const RTreeNodeH *node,
        RTreeNodeH ***children_out,
        size_t *nchildren_out
    )

    RTreeError rtree_node_children_free(RTreeNodeH **children, size_t n)

    RTreeError rtree_node_free(RTreeNodeH *node)

    RTreeError rtree_node_id(
        const RTreeNodeH *node,
        size_t *id_out
    )

    RTreeError rtree_node_envelope(
        const RTreeNodeH *node,
        double *mins_out,
        double *maxs_out
    )
