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
    MPI_Aint,
    MPI_DATATYPE_NULL,
    MPI_INT,
    MPI_STATUSES_IGNORE,
    MPI_Datatype,
    MPI_Irecv,
    MPI_Isend,
    MPI_Request,
    MPI_Type_commit,
    MPI_Type_create_resized,
    MPI_Type_free,
    MPI_Waitall,
)
from petsc4py.PETSc cimport CHKERR

include "petschdr.pxi"

cdef extern from "rtree-capi.h":
    ctypedef enum RTreeError:
        Success
        NullPointer
        InvalidDimension
        EmptyNodeEnvelope

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
    cdef object __weakref__

    def __cinit__(self, uintptr_t tree_handle):
        self.tree = <RTreeH*>0
        if tree_handle == 0:
            raise RuntimeError("invalid tree handle")
        self.tree = <RTreeH*>tree_handle

    def __dealloc__(self):
        if self.tree != <RTreeH*>0:
            rtree_free(self.tree)
            self.tree = <RTreeH*>0

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
    send_counts : (nranks_to,) int32 array
        Number of points to send to each entry of `toranks`.
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
        np.ndarray[np.int32_t, ndim=1, mode="c"] send_counts
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
        send_counts = np.empty(nranks_to, dtype=np.int32)
        rank_write_offsets = np.empty(comm_size, dtype=np.int32)

        index = 0
        offset = 0
        for rank in range(comm_size):
            if rank_counts[rank] == 0:
                # not sending this rank any points
                continue
            toranks[index] = <np.int32_t>rank
            send_counts[index] = rank_counts[rank]
            rank_write_offsets[rank] = offset
            offset += send_counts[index]
            index += 1

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

    return toranks, point_indices, send_counts


@cython.boundscheck(False)
@cython.wraparound(False)
def discover_remote_roots(
        RTree rtree,
        np.ndarray[np.float64_t, ndim=2, mode="c"] points,
        MPI.Comm comm):
    """Build the remote-root array for a point-embedding star forest.

    Parameters
    ----------
    rtree : RTree
        The distributed Rtree built by :func:`build_from_aabb` with rank
        numbers as leaf ids.
    points : (n_points, gdim) float64 array
        Local root-point coordinates.
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
        MPI_Datatype remote_index_type = MPI_DATATYPE_NULL
        MPI_Request *requests = NULL
        PetscMPIInt k, nranks_from, nranks_to, nrequests
        PetscMPIInt count, source_rank, recv_offset = 0
        PetscMPIInt *fromranks = NULL
        void *recv_counts = NULL
        Py_ssize_t i, nleaves = 0, send_offset = 0
        np.ndarray[np.int32_t, ndim=1, mode="c"] toranks
        np.ndarray[np.int32_t, ndim=1, mode="c"] send_counts
        np.ndarray[np.int32_t, ndim=1, mode="c"] point_indices
        np.ndarray[np.int32_t, ndim=2, mode="c"] remote

    toranks, point_indices, send_counts = _destination_ranks(
        rtree, points, comm.size,
    )
    nranks_to = <PetscMPIInt>toranks.shape[0]

    # discover incoming ranks and exchange their point counts
    CHKERR(PetscCommBuildTwoSided(
        mpi_comm,
        1,
        MPI_INT,
        <PetscMPIInt>toranks.shape[0],
        <const PetscMPIInt *>toranks.data,
        <const void *>send_counts.data,
        &nranks_from,
        &fromranks,
        &recv_counts,
    ))

    # Now each rank knows what ranks it is going to receive points from,
    # and how many points. We now proceed and send these points sparsely.

    try:
        for k in range(nranks_from):
            nleaves += (<PetscMPIInt *>recv_counts)[k]
        remote = np.empty((nleaves, 2), dtype=np.int32)

        nrequests = nranks_from + nranks_to
        if nrequests:
            requests = <MPI_Request *>malloc(nrequests * sizeof(MPI_Request))

        # Receive point indices directly into the second column of `remote`.
        # `remote` is a contiguous (nleaves, 2) shaped array, so we create
        # an MPI unit whose payload is MPI_INT, but whose extent is twice that.
        if nranks_from:
            CHKERRMPI(MPI_Type_create_resized(
                MPI_INT,
                <MPI_Aint>0,
                <MPI_Aint>(2 * sizeof(np.int32_t)),
                &remote_index_type,
            ))
            CHKERRMPI(MPI_Type_commit(&remote_index_type))

        # nonblocking receives
        for k in range(nranks_from):
            source_rank = fromranks[k]
            count = (<PetscMPIInt *>recv_counts)[k]
            for i in range(recv_offset, recv_offset + count):
                remote[i, 0] = source_rank
            CHKERRMPI(MPI_Irecv(
                <void *>&remote[recv_offset, 1],
                count,
                remote_index_type,
                source_rank,
                source_rank,
                mpi_comm,
                &requests[k],
            ))
            recv_offset += count

        # nonblocking sends
        for k in range(nranks_to):
            count = send_counts[k]
            CHKERRMPI(MPI_Isend(
                <const void *>&point_indices[send_offset],
                count,
                MPI_INT,
                toranks[k],
                comm.rank,
                mpi_comm,
                &requests[nranks_from + k],
            ))
            send_offset += count

        CHKERRMPI(MPI_Waitall(nrequests, requests, MPI_STATUSES_IGNORE))
    finally:
        if remote_index_type != MPI_DATATYPE_NULL:
            CHKERRMPI(MPI_Type_free(&remote_index_type))
        if requests != NULL:
            free(requests)
        CHKERR(PetscFree(fromranks))
        CHKERR(PetscFree(recv_counts))

    return remote


def bounding_boxes_at_level(RTree rtree, size_t level, uint32_t dim):
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
