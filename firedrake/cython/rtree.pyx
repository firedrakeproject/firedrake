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
    MPI_INT,
    MPI_REQUEST_NULL,
    MPI_STATUS_IGNORE,
    MPI_STATUSES_IGNORE,
    MPI_Get_count,
    MPI_Ibarrier,
    MPI_Iprobe,
    MPI_Issend,
    MPI_Recv,
    MPI_Request,
    MPI_Status,
    MPI_Test,
    MPI_Testall,
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

    RTreeError rtree_locate_all_at_points_unique(
        const RTreeH *tree,
        const double *points,
        size_t n_points,
        int64_t **ids_out,
        size_t **offsets_out
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
cdef tuple _destination_ranks(
        RTree rtree,
        np.ndarray[np.float64_t, ndim=2, mode="c"] points,
        Py_ssize_t comm_size):
    """Group candidate point indices by destination rank.

    Parameters
    ----------
    rtree : RTree
        The distributed Rtree with rank numbers as leaf ids.
    points : (n_points, gdim) float64 array
        The local points to send to remote ranks.
    comm_size : int
        Number of ranks in the MPI communicator.

    Returns
    -------
    toranks : (nranks_to,) int32 array
        Target ranks to send points to.
    point_indices : (total_sends,) int32 array
        Indices into `points` determining which points to send.
    send_offsets : (nranks_to + 1,) int32 array
        Points destined for `toranks[i]` are
        `point_indices[send_offsets[i]:send_offsets[i + 1]]`.
    """
    cdef:
        int64_t *ids_out = NULL
        size_t *offsets_out = NULL
        size_t n_points = points.shape[0]
        size_t i, j
        Py_ssize_t nranks_to = 0
        Py_ssize_t rank, index, offset
        RTreeError err, ids_free_err, offsets_free_err
        np.ndarray[np.int32_t, ndim=1, mode="c"] toranks
        np.ndarray[np.int32_t, ndim=1, mode="c"] rank_counts
        np.ndarray[np.int32_t, ndim=1, mode="c"] rank_write_offsets
        np.ndarray[np.int32_t, ndim=1, mode="c"] send_offsets
        np.ndarray[np.int32_t, ndim=1, mode="c"] point_indices

    # query the partition rtree to find candidate ranks
    # this routine returns the unique IDs for each point
    err = rtree_locate_all_at_points_unique(
        rtree.tree,
        <const double *>points.data,
        n_points,
        &ids_out,
        &offsets_out,
    )
    if err != Success:
        raise RuntimeError("rtree_locate_all_at_points_unique failed")

    try:
        rank_counts = np.zeros(comm_size, dtype=np.int32)

        # Count the points destined for each rank.
        # The candidate ranks for point `i` are
        # `ids_out[offsets_out[i]:offsets_out[i + 1]]`.
        for i in range(n_points):
            for j in range(offsets_out[i], offsets_out[i + 1]):
                rank = <Py_ssize_t>ids_out[j]
                if rank_counts[rank] == 0:
                    nranks_to += 1
                rank_counts[rank] += 1

        toranks = np.empty(nranks_to, dtype=np.int32)
        send_offsets = np.empty(nranks_to + 1, dtype=np.int32)
        rank_write_offsets = np.empty(comm_size, dtype=np.int32)

        index = 0
        offset = 0
        for rank in range(comm_size):
            if rank_counts[rank] == 0:
                # not sending this rank any points
                continue
            toranks[index] = <np.int32_t>rank
            send_offsets[index] = offset
            rank_write_offsets[rank] = offset
            offset += rank_counts[rank]
            index += 1
        send_offsets[nranks_to] = offset

        point_indices = np.empty(offset, dtype=np.int32)
        for i in range(n_points):
            for j in range(offsets_out[i], offsets_out[i + 1]):
                rank = <Py_ssize_t>ids_out[j]
                index = rank_write_offsets[rank]
                point_indices[index] = <np.int32_t>i
                rank_write_offsets[rank] += 1
    finally:
        ids_free_err = rtree_free_ids(ids_out, offsets_out[n_points])
        offsets_free_err = rtree_free_offsets(offsets_out, n_points + 1)

    if ids_free_err != Success:
        raise RuntimeError("rtree_free_ids failed")
    if offsets_free_err != Success:
        raise RuntimeError("rtree_free_offsets failed")

    return toranks, point_indices, send_offsets


@cython.boundscheck(False)
@cython.wraparound(False)
def discover_remote_roots(
        RTree rtree,
        np.ndarray[np.float64_t, ndim=2, mode="c"] points,
        MPI.Comm comm):
    """Build the remote array for a candidate star forest.

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
        MPI_Request *requests = NULL
        MPI_Request barrier_request = MPI_REQUEST_NULL
        MPI_Status status
        PetscMPIInt k, nranks_to
        PetscMPIInt count, source_rank, recv_offset = 0
        int message_ready, sends_complete, barrier_complete = 0
        Py_ssize_t i, nleaves = 0
        np.ndarray[np.int32_t, ndim=1, mode="c"] toranks
        np.ndarray[np.int32_t, ndim=1, mode="c"] send_offsets
        np.ndarray[np.int32_t, ndim=1, mode="c"] point_indices
        np.ndarray[np.int32_t, ndim=1, mode="c"] received
        np.ndarray[np.int32_t, ndim=2, mode="c"] remote
        list recv_ranks = []
        list recv_buffers = []

    toranks, point_indices, send_offsets = _destination_ranks(
        rtree, points, comm.size,
    )
    nranks_to = <PetscMPIInt>toranks.shape[0]

    try:
        if nranks_to:
            requests = <MPI_Request *>malloc(nranks_to * sizeof(MPI_Request))
            if requests == NULL:
                raise MemoryError("failed to allocate MPI requests")

        # Synchronous sends allow completion detection to prove that the
        # matching receive has been posted without first exchanging counts.
        for k in range(nranks_to):
            count = send_offsets[k + 1] - send_offsets[k]
            CHKERRMPI(MPI_Issend(
                <const void *>&point_indices[send_offsets[k]],
                count,
                MPI_INT,
                toranks[k],
                0,
                mpi_comm,
                &requests[k],
            ))

        sends_complete = nranks_to == 0
        if sends_complete:
            CHKERRMPI(MPI_Ibarrier(mpi_comm, &barrier_request))

        # Hoefler's NBX algorithm discovers variable-sized messages with
        # probes while progressing synchronous sends. Once all sends have
        # matched, the nonblocking barrier establishes global completion.
        while not barrier_complete:
            CHKERRMPI(MPI_Iprobe(
                MPI_ANY_SOURCE,
                0,
                mpi_comm,
                &message_ready,
                &status,
            ))
            if message_ready:
                CHKERRMPI(MPI_Get_count(&status, MPI_INT, &count))
                source_rank = status.MPI_SOURCE
                received = np.empty(count, dtype=np.int32)
                CHKERRMPI(MPI_Recv(
                    <void *>received.data,
                    count,
                    MPI_INT,
                    source_rank,
                    0,
                    mpi_comm,
                    MPI_STATUS_IGNORE,
                ))
                recv_ranks.append(source_rank)
                recv_buffers.append(received)
                nleaves += count
            elif not sends_complete:
                CHKERRMPI(MPI_Testall(
                    nranks_to,
                    requests,
                    &sends_complete,
                    MPI_STATUSES_IGNORE,
                ))
                if sends_complete:
                    CHKERRMPI(MPI_Ibarrier(mpi_comm, &barrier_request))
            else:
                CHKERRMPI(MPI_Test(
                    &barrier_request,
                    &barrier_complete,
                    MPI_STATUS_IGNORE,
                ))
    finally:
        if requests != NULL:
            free(requests)

    remote = np.empty((nleaves, 2), dtype=np.int32)
    for k in range(len(recv_buffers)):
        source_rank = recv_ranks[k]
        received = recv_buffers[k]
        count = received.shape[0]
        for i in range(count):
            remote[recv_offset + i, 0] = source_rank
            remote[recv_offset + i, 1] = received[i]
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

    err = rtree_collect_bounding_boxes(rtree.tree, level, &mins, &maxs, &n_boxes)
    if err != Success:
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

    err = rtree_depth(rtree.tree, &depth)
    if err != Success:
        raise RuntimeError("rtree_depth failed")
    return depth
