# cython: language_level=3

cimport numpy as np
import numpy as np
import ctypes
import cython
from libc.stddef cimport size_t
from libc.stdint cimport uintptr_t, uint32_t, int64_t

cimport mpi4py.MPI as MPI
from mpi4py.libmpi cimport MPI_INT
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

    RTreeError rtree_locate_all_at_point(
        const RTreeH *tree,
        const double *point,
        int64_t **ids_out,
        size_t *nids_out
    )

    RTreeError rtree_locate_all_at_points(
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
def discover_ranks(
        RTree rtree,
        np.ndarray[np.float64_t, ndim=2, mode="c"] points,
        MPI.Comm comm):
    """Query a distributed Rtree and discover which ranks to send points,
    and which ranks are going to send points to us.

    For each point in `points`, we find all candidate ranks whose bounding
    boxes contain that point. We then use `PetscCommBuildTwoSided` to discover 
    which ranks will be sending points to us.

    Parameters
    ----------
    rtree : RTree
        The distributed Rtree built by :func:`build_from_aabb` with
        rank numbers as leaf ids.
    points : (n_points, gdim) float64 array
        The local points to send to remote ranks.
    comm : mpi4py.MPI.Comm
        The MPI communicator.

    Returns
    -------
    toranks : (nto,) int32 array
        Target ranks to send points to.
    send_offsets : (nto + 1,) int32 array
        Points destined for `toranks[i]` are
        `point_indices[send_offsets[i]:send_offsets[i+1]]`.
    point_indices : (total_sends,) int32 array
        Indices into `points` determining which points to send.
    fromranks : (nfrom,) int32 array
        Ranks that will send points to us.
    recv_counts : (nfrom,) int32 array
        Number of points we will receive from each rank in `fromranks`.
    """
    cdef:
        int64_t *ids_out = NULL
        size_t *offsets_out = NULL
        size_t n_points = points.shape[0]
        size_t i, j
        Py_ssize_t nto = 0
        Py_ssize_t rank, index
        RTreeError err, ids_free_err, offsets_free_err
        MPI.MPI_Comm mpi_comm = comm.ob_mpi
        PetscMPIInt k, nfrom
        PetscMPIInt *fromranks = NULL
        void *fromdata = NULL
        np.ndarray[np.intp_t, ndim=1, mode="c"] seen
        np.ndarray[np.intp_t, ndim=1, mode="c"] index_of_rank
        np.ndarray[np.int32_t, ndim=1, mode="c"] toranks
        np.ndarray[np.int32_t, ndim=1, mode="c"] send_counts
        np.ndarray[np.int32_t, ndim=1, mode="c"] point_indices
        np.ndarray[np.int32_t, ndim=1, mode="c"] send_offsets
        np.ndarray[np.int32_t, ndim=1, mode="c"] write_idx
        np.ndarray[np.int32_t, ndim=1, mode="c"] fromranks_out
        np.ndarray[np.int32_t, ndim=1, mode="c"] recv_counts_out

    # the candidate ranks for point `i` are
    # `ids_out[offsets_out[i]:offsets_out[i + 1]]`.
    err = rtree_locate_all_at_points(
        rtree.tree,
        <const double *>points.data,
        n_points,
        &ids_out,
        &offsets_out,
    )
    if err != Success:
        raise RuntimeError("rtree_locate_all_at_points failed")

    try:
        seen = np.full(comm.size, -1, dtype=np.intp)
        # index_of_rank[rank] = index of rank in `toranks` if rank is a candidate, else -1.
        # We build this array on the fly to avoid having to search `toranks` for each candidate rank.
        index_of_rank = np.full(comm.size, -1, dtype=np.intp)
        toranks = np.empty(comm.size, dtype=np.int32)
        send_counts = np.zeros(comm.size, dtype=np.int32)

        # Count how many unique points we will send to each rank.
        # A point may lie in multiple bounding boxes for the same rank,
        # so we need to deduplicate the candidate ranks for each point.
        for i in range(n_points):
            for j in range(offsets_out[i], offsets_out[i + 1]):
                # Loop over candidate ranks for point `i`.
                rank = <Py_ssize_t>ids_out[j]
                if seen[rank] == <np.intp_t>i:
                    # This rank has already been seen for this point, so skip it.
                    continue
                seen[rank] = i
                index = index_of_rank[rank]
                if index == -1:
                    # This rank has not been seen before by any point so add it to `toranks`
                    index = nto
                    index_of_rank[rank] = index
                    toranks[index] = <np.int32_t>rank
                    nto += 1
                send_counts[index] += 1

        # Build `send_offsets`. This is the cumulative sum of `send_counts`.
        send_offsets = np.empty(nto + 1, dtype=np.int32)
        send_offsets[0] = 0
        for index in range(nto):
            send_offsets[index + 1] = send_offsets[index] + send_counts[index]

        # Fill in `point_indices` with the indices of points to send each rank.
        # The points destined for `toranks[i]` are
        # `point_indices[send_offsets[i]:send_offsets[i+1]]`.
        point_indices = np.empty(send_offsets[nto], dtype=np.int32)
        write_idx = send_offsets[:nto].copy()  # Keep track of where to write the next point 
        seen[:] = -1  # reset `seen` 
        for i in range(n_points):
            for j in range(offsets_out[i], offsets_out[i + 1]):
                rank = <Py_ssize_t>ids_out[j]
                if seen[rank] == <np.intp_t>i:
                    continue
                seen[rank] = i
                index = index_of_rank[rank]
                point_indices[write_idx[index]] = i
                write_idx[index] += 1
    finally:
        ids_free_err = rtree_free_ids(ids_out, offsets_out[n_points])
        offsets_free_err = rtree_free_offsets(offsets_out, n_points + 1)

    if ids_free_err != Success:
        raise RuntimeError("rtree_free_ids failed")
    if offsets_free_err != Success:
        raise RuntimeError("rtree_free_offsets failed")

    toranks = toranks[:nto]

    # Routine that discovers communicating ranks given one-sided information
    CHKERR(PetscCommBuildTwoSided(
        mpi_comm,
        1,  # sending/receiving one entry (the number of points)
        MPI_INT,
        <PetscMPIInt>nto,  # number of ranks to send data to
        <const PetscMPIInt *>toranks.data,  # ranks to send to (array of length nto)
        <const void *>send_counts.data,  # data to send to each rank (array of length nto)
        &nfrom,  # number of ranks we're receiving messages from
        &fromranks,  # ranks we're receiving messages from (array of length nfrom)
        &fromdata,  # data we're receiving from each rank (array of length nfrom)
    ))

    # Copy petsc-allocated results into numpy arrays then free them
    fromranks_out = np.empty(nfrom, dtype=np.int32)
    recv_counts_out = np.empty(nfrom, dtype=np.int32)
    for k in range(nfrom):
        fromranks_out[k] = fromranks[k]
        recv_counts_out[k] = (<PetscMPIInt *>fromdata)[k]
    CHKERR(PetscFree(fromranks))
    CHKERR(PetscFree(fromdata))

    return toranks, send_offsets, point_indices, fromranks_out, recv_counts_out

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
