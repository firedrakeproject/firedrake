# cython: language_level=3

cimport numpy as np
import numpy as np
import ctypes
import cython
from libc.stddef cimport size_t
from libc.stdint cimport uintptr_t, uint32_t, int64_t
from libc.stdlib cimport free, malloc

cimport mpi4py.MPI as MPI
from mpi4py.libmpi cimport (
    MPI_ANY_SOURCE,
    MPI_REQUEST_NULL,
    MPI_STATUS_IGNORE,
    MPI_STATUSES_IGNORE,
    MPI_TYPECLASS_INTEGER,
    MPI_Datatype,
    MPI_Get_count,
    MPI_Ibarrier,
    MPI_Iprobe,
    MPI_Issend,
    MPI_Recv,
    MPI_Request,
    MPI_Status,
    MPI_Test,
    MPI_Testall,
    MPI_Type_match_size,
)

include "petschdr.pxi"

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
        const int64_t *ids,
        size_t n,
        uint32_t dim
    )

    RTreeError rtree_free(RTreeH *tree)

    RTreeError rtree_free_ids(int64_t *ids, size_t n)

    RTreeError rtree_free_offsets(size_t *offsets, size_t n)

    RTreeError rtree_free_point_indices(size_t *point_indices, size_t n)

    RTreeError rtree_locate_points_grouped_by_id_unique(
        const RTreeH *tree,
        const double *points,
        size_t n_points,
        int64_t **ids_out,
        size_t **offsets_out,
        size_t **point_indices_out,
        size_t *n_ids_out
    )

    RTreeError rtree_depth(const RTreeH *tree, size_t *depth_out)

    RTreeError rtree_collect_bounding_boxes(
        const RTreeH *tree,
        size_t level,
        double **mins_out,
        double **maxs_out,
        size_t *nboxes_out
    )

    RTreeError rtree_free_bounding_boxes(
        double *mins,
        double *maxs,
        size_t nboxes,
        uint32_t dim
    )

cdef class RTree(object):
    """Python class for holding an Rtree."""

    cdef RTreeH* tree

    def __cinit__(self, uintptr_t tree_handle):
        self.tree = <RTreeH*>0
        if tree_handle == 0:
            raise RuntimeError("invalid tree handle")
        self.tree = <RTreeH*>tree_handle

    def __dealloc__(self):
        if self.tree != <RTreeH*>0:
            rtree_free(self.tree)

    @property
    def ctypes(self):
        """Returns a ctypes pointer to the rtree."""
        return ctypes.c_void_p(<uintptr_t> self.tree)


@cython.boundscheck(False)
@cython.wraparound(False)
def build_from_aabb(np.ndarray[np.float64_t, ndim=2, mode="c"] coords_min,
                    np.ndarray[np.float64_t, ndim=2, mode="c"] coords_max,
                    np.ndarray[np.int64_t, ndim=1, mode="c"] ids = None):
    """Builds rtree from two arrays of shape (n, dim) containing the coordinates
    of the lower and upper corners of n axis-aligned bounding boxes, and an
    optional array of shape (n,) containing integer ids for each box.

    Parameters
    ----------
    coords_min : numpy.ndarray
        Lower corner coordinates of the bounding boxes, with shape `(n, dim)`.
    coords_max : numpy.ndarray
        Upper corner coordinates of the bounding boxes, with shape `(n, dim)`.
    ids : numpy.ndarray
        Optional integer ids for each box, with shape `(n,)`. If not provided,
        defaults to `0, 1, ..., n-1`.

    Returns
    -------
    RTree
        An RTree object containing the Rtree.
    """    
    cdef:
        RTreeH* rtree
        size_t n
        uint32_t dim
        RTreeError err

    if coords_min.shape[0] != coords_max.shape[0] or coords_min.shape[1] != coords_max.shape[1]:
        raise ValueError("coords_min and coords_max must have the same shape")

    n = coords_min.shape[0]
    dim = coords_min.shape[1]
    if ids is None:
        ids = np.arange(n, dtype=np.int64)
    elif <size_t>ids.shape[0] != n:
        raise ValueError("Mismatch between number of boxes and number of ids")

    err = rtree_bulk_load(
        &rtree,
        <const double*>coords_min.data,
        <const double*>coords_max.data,
        <const int64_t*>ids.data,
        n,
        dim
    )
    if err != Success:
        raise RuntimeError("rtree_bulk_load failed")

    return RTree(<uintptr_t>rtree)

@cython.boundscheck(False)
@cython.wraparound(False)
def discover_remote_roots(
        RTree rtree,
        np.ndarray[np.float64_t, ndim=2, mode="c"] points,
        MPI.Comm comm):
    """Build the remote array for a candidate star forest.

    This implements the non-blocking exchange algorithm from Hoefler et al.
    'Scalable communication protocols for dynamic sparse data exchange'.

    Parameters
    ----------
    rtree : RTree
        The distributed Rtree built by :func:`build_from_aabb` with rank
        numbers as leaf ids.
    points : (n_points, gdim) float64 array
        Local point coordinates.
    comm : mpi4py.MPI.Comm
        The MPI communicator.

    Returns
    -------
    remote : (nleaves, 2) int32 array
        For every local candidate leaf, the MPI rank and local index of its
        remote root point.
    """
    cdef:
        MPI.MPI_Comm mpi_comm = comm.ob_mpi
        MPI_Request *requests = NULL, barrier_request = MPI_REQUEST_NULL
        MPI_Status status
        MPI_Datatype point_index_type
        int count, source_rank, message_ready, sends_complete, barrier_complete = 0
        Py_ssize_t i, nleaves = 0, recv_offset = 0
        int64_t *toranks = NULL
        size_t *point_indices = NULL, *send_offsets = NULL, n_points = points.shape[0], nranks_to = 0
        np.ndarray[np.uintp_t, ndim=1, mode="c"] received
        np.ndarray[np.int32_t, ndim=2, mode="c"] remote
        list recv_messages = []

    # MPI does not have a builtin type for size_t
    CHKERRMPI(MPI_Type_match_size(MPI_TYPECLASS_INTEGER, sizeof(size_t), &point_index_type))

    if rtree_locate_points_grouped_by_id_unique(
        rtree.tree,
        <const double *>points.data,
        n_points,
        &toranks,
        &send_offsets,
        &point_indices,
        &nranks_to,
    ) != Success:
        raise RuntimeError("rtree_locate_points_grouped_by_id_unique failed")

    try:
        if nranks_to:
            requests = <MPI_Request *>malloc(nranks_to * sizeof(MPI_Request))

        for k in range(nranks_to):
            count = send_offsets[k + 1] - send_offsets[k]
            CHKERRMPI(MPI_Issend(
                <const void *>&point_indices[send_offsets[k]],
                count, point_index_type,
                <int>toranks[k],
                0,
                mpi_comm,
                &requests[k],
            ))

        sends_complete = nranks_to == 0
        if sends_complete:
            CHKERRMPI(MPI_Ibarrier(mpi_comm, &barrier_request))

        while not barrier_complete:
            CHKERRMPI(MPI_Iprobe(MPI_ANY_SOURCE, 0, mpi_comm, &message_ready, &status))

            if message_ready:
                CHKERRMPI(MPI_Get_count(&status, point_index_type, &count))
                source_rank = status.MPI_SOURCE
                received = np.empty(count, dtype=np.uintp)
                CHKERRMPI(MPI_Recv(<void *>received.data, count, point_index_type, source_rank, 0, mpi_comm, MPI_STATUS_IGNORE))
                recv_messages.append((source_rank, received))
                nleaves += count

            if not sends_complete:
                CHKERRMPI(MPI_Testall(nranks_to, requests, &sends_complete, MPI_STATUSES_IGNORE))
                if sends_complete:
                    CHKERRMPI(MPI_Ibarrier(mpi_comm, &barrier_request))
            else:
                CHKERRMPI(MPI_Test(&barrier_request, &barrier_complete, MPI_STATUS_IGNORE))
    finally:
        if requests != NULL:
            free(requests)
        if rtree_free_ids(toranks, nranks_to) != Success:
            raise RuntimeError("rtree_free_ids failed")
        if rtree_free_point_indices(point_indices, send_offsets[nranks_to]) != Success:
            raise RuntimeError("rtree_free_point_indices failed")
        if rtree_free_offsets(send_offsets, nranks_to + 1) != Success:
            raise RuntimeError("rtree_free_offsets failed")

    # create remote array for candidate star forest
    remote = np.empty((nleaves, 2), dtype=np.int32)
    for source_rank, received in recv_messages:
        count = received.shape[0]
        remote[recv_offset:recv_offset + count, 0] = source_rank
        remote[recv_offset:recv_offset + count, 1] = received
        recv_offset += count

    return remote


def bounding_boxes_at_level(RTree rtree, size_t level, uint32_t dim):
    """Return all bounding boxes at the specified level of the Rtree."""
    cdef:
        double *mins = NULL
        double *maxs = NULL
        size_t n_boxes = 0
        RTreeError err
        np.ndarray[np.float64_t, ndim=3, mode="c"] boxes

    if rtree_collect_bounding_boxes(rtree.tree, level, &mins, &maxs, &n_boxes) != Success:
        raise RuntimeError("rtree_bounding_boxes failed")

    boxes = np.empty((n_boxes, 2, dim), dtype=np.float64)

    for i in range(n_boxes):
        for j in range(dim):
            boxes[i, 0, j] = mins[i * dim + j]
            boxes[i, 1, j] = maxs[i * dim + j]

    rtree_free_bounding_boxes(mins, maxs, n_boxes, dim)

    return boxes

def tree_depth(RTree rtree):
    """Return the depth of the Rtree."""
    cdef:
        size_t depth = 0
        RTreeError err

    if rtree_depth(rtree.tree, &depth) != Success:
        raise RuntimeError("rtree_depth failed")
    return depth
