# cython: language_level=3

cimport numpy as np
import numpy as np
import ctypes
import cython
from libc.stddef cimport size_t
from libc.stdint cimport uintptr_t

cimport mpi4py.MPI as MPI
from mpi4py.libmpi cimport MPI_INT
from petsc4py.PETSc cimport CHKERR

include "rstarinc.pxi"
include "petschdr.pxi"

cdef class RTree(object):
    """Python class for holding a spatial index."""

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
        """Returns a ctypes pointer to the native spatial index."""
        return ctypes.c_void_p(<uintptr_t> self.tree)


@cython.boundscheck(False)
@cython.wraparound(False)
def build_from_aabb(np.ndarray[np.float64_t, ndim=2, mode="c"] coords_min,
                    np.ndarray[np.float64_t, ndim=2, mode="c"] coords_max,
                    np.ndarray[np.npy_uintp, ndim=1, mode="c"] ids = None):
    """Builds rtree from two arrays of shape (n, dim) containing the coordinates
    of the lower and upper corners of n axis-aligned bounding boxes, and an
    optional array of shape (n,) containing integer ids for each box.

    Parameters
    ----------
    coords_min : (n, dim) array
        The lower corner coordinates of the bounding boxes.
    regions_hi : (n, dim) array
        The upper corner coordinates of the bounding boxes.
    ids : (n,) array, optional
        Integer ids for each box. If not provided, defaults to 0, 1, ..., n-1.

    Returns
    -------
    RTree
        An RTree object containing the built R*-tree.
    """    
    cdef:
        RTreeH* rtree
        size_t n
        size_t dim
        RTreeError err

    if coords_min.shape[0] != coords_max.shape[0] or coords_min.shape[1] != coords_max.shape[1]:
        raise ValueError("coords_min and coords_max must have the same shape")

    n = <size_t>coords_min.shape[0]
    dim = <size_t>coords_min.shape[1]
    if ids is None:
        ids = np.arange(n, dtype=np.uintp)
    elif ids.shape[0] != n:
        raise ValueError("Mismatch between number of boxes and number of ids")

    err = rtree_bulk_load(
        &rtree,
        <const double*>coords_min.data,
        <const double*>coords_max.data,
        <const size_t*>ids.data,
        n,
        dim
    )
    if err != Success:
        raise RuntimeError("RTree_FromArray failed")

    return RTree(<uintptr_t>rtree)

@cython.boundscheck(False)
@cython.wraparound(False)
def locate_all_at_point(
        RTree rtree,
        np.ndarray[np.float64_t, ndim=1, mode="c"] point):
    """Return the ids of all leaves whose bounding box contains ``point``.

    Parameters
    ----------
    rtree : RTree
        The R*-tree to query.
    point : (dim,) float64 array
        The query point. Must have the same dimensionality as the tree.

    Returns
    -------
    An array of integer ids corresponding to the leaves whose bounding boxes contain the query point.
    The array must be freed by the caller using rtree_locate_all_at_point_free.
    """
    cdef:
        size_t *ids_out = NULL
        size_t nids_out = 0
        RTreeError err
        np.ndarray[np.npy_uintp, ndim=1, mode="c"] result

    err = rtree_locate_all_at_point(
        rtree.tree,
        <const double *>point.data,
        &ids_out,
        &nids_out,
    )
    if err != Success:
        raise RuntimeError("rtree_locate_all_at_point failed")

    result = np.empty(nids_out, dtype=np.uintp)
    for i in range(nids_out):
        result[i] = ids_out[i]
    rtree_free_ids(ids_out, nids_out)
    return result


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
        size_t *ids_out = NULL
        size_t nids_out = 0
        RTreeError err
        size_t i, j
        size_t n_points = points.shape[0]
        MPI.MPI_Comm mpi_comm = comm.ob_mpi
        PetscMPIInt nto
        PetscMPIInt nfrom = 0
        PetscMPIInt *fromranks = NULL
        void *fromdata = NULL
        np.ndarray[np.int32_t, ndim=1, mode="c"] toranks
        np.ndarray[np.int32_t, ndim=1, mode="c"] send_counts
        np.ndarray[np.int32_t, ndim=1, mode="c"] point_indices
        np.ndarray[np.int32_t, ndim=1, mode="c"] send_offsets
        np.ndarray[np.int32_t, ndim=1, mode="c"] fromranks_out
        np.ndarray[np.int32_t, ndim=1, mode="c"] recv_counts_out
        PetscMPIInt k

    # map dest_rank -> list of point indices to send there
    rank_to_indices: dict[int, list[int]] = {}
    for i in range(n_points):
        err = rtree_locate_all_at_point(
            rtree.tree,
            <const double *>&points[i, 0],
            &ids_out,
            &nids_out,
        )
        if err != Success:
            raise RuntimeError("rtree_locate_all_at_point failed")

        # Points may lie in multiple bounding boxes owned by the same rank
        seen_ranks: set[int] = set()
        for j in range(nids_out):
            seen_ranks.add(<int>ids_out[j])
        rtree_free_ids(ids_out, nids_out)

        ids_out = NULL
        for dest_rank in seen_ranks:
            if dest_rank in rank_to_indices:
                rank_to_indices[dest_rank].append(i)
            else:
                rank_to_indices[dest_rank] = [i]

    nto = len(rank_to_indices)
    toranks = np.empty(nto, dtype=np.int32)
    send_counts = np.empty(nto, dtype=np.int32)
    send_offsets = np.empty(nto + 1, dtype=np.int32)
    all_indices: list[int] = []
    send_offsets[0] = 0
    for i, (rank, idx_list) in enumerate(rank_to_indices.items()):
        toranks[i] = rank
        send_counts[i] = len(idx_list)
        send_offsets[i + 1] = send_offsets[i] + len(idx_list)
        all_indices.extend(idx_list)
    point_indices = np.array(all_indices, dtype=np.int32)

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


cdef class RTreeNode(object):
    """Python class for holding a spatial index node."""

    cdef RTreeNodeH* node

    def __cinit__(self, uintptr_t node_handle):
        self.node = <RTreeNodeH*>0
        if node_handle == 0:
            raise RuntimeError("invalid node handle")
        self.node = <RTreeNodeH*>node_handle

    def __dealloc__(self):
        if self.node != <RTreeNodeH*>0:
            rtree_node_free(self.node)
            self.node = <RTreeNodeH*>0


def root_node(RTree rtree):
    """Return the root node of the R*-tree."""
    cdef:
        RTreeNodeH* node
        RTreeError err
    err = rtree_root_node(rtree.tree, &node)
    if err != Success:
        raise RuntimeError("rtree_root_node failed")
    return RTreeNode(<uintptr_t>node)


def node_children(RTreeNode node):
    """Return the children of an R*-tree node as a list of RStarTreeNode."""
    cdef:
        RTreeNodeH** children
        size_t nchildren
        RTreeError err
    err = rtree_node_children(node.node, &children, &nchildren)
    if err != Success:
        raise RuntimeError("rtree_node_children failed")
    result = [RTreeNode(<uintptr_t>children[i]) for i in range(nchildren)]
    rtree_node_children_free(children, nchildren)
    return result


def node_id(RTreeNode node):
    """Return the id of a leaf node."""
    cdef:
        size_t id_out
        RTreeError err
    err = rtree_node_id(node.node, &id_out)
    if err != Success:
        raise RuntimeError("rtree_node_id failed (node may not be a leaf)")
    return id_out


def node_envelope(RTreeNode node, size_t dim):
    """Return the (mins, maxs) bounding envelope of an R*-tree node."""
    cdef:
        np.ndarray[np.float64_t, ndim=1, mode="c"] mins = np.empty(dim, dtype=np.float64)
        np.ndarray[np.float64_t, ndim=1, mode="c"] maxs = np.empty(dim, dtype=np.float64)
        RTreeError err
    err = rtree_node_envelope(node.node, <double*>mins.data, <double*>maxs.data)
    if err != Success:
        raise RuntimeError("rtree_node_envelope failed")
    return mins, maxs

cdef class RStarTreeNode(object):
    """Python class for holding a native spatial index node object."""

    cdef RTreeNodeH* node

    def __cinit__(self, uintptr_t node_handle):
        self.node = <RTreeNodeH*>0
        if node_handle == 0:
            raise RuntimeError("invalid node handle")
        self.node = <RTreeNodeH*>node_handle

    def __dealloc__(self):
        if self.node != <RTreeNodeH*>0:
            rtree_node_free(self.node)
            self.node = <RTreeNodeH*>0


def root_node(RStarTree rtree):
    """Return the root node of the R*-tree."""
    cdef:
        RTreeNodeH* node
        RTreeError err
    err = rtree_root_node(rtree.tree, &node)
    if err != Success:
        raise RuntimeError("rtree_root_node failed")
    return RStarTreeNode(<uintptr_t>node)


def node_children(RStarTreeNode node):
    """Return the children of an R*-tree node as a list of RStarTreeNode."""
    cdef:
        RTreeNodeH** children
        size_t nchildren
        RTreeError err
    err = rtree_node_children(node.node, &children, &nchildren)
    if err != Success:
        raise RuntimeError("rtree_node_children failed")
    result = [RStarTreeNode(<uintptr_t>children[i]) for i in range(nchildren)]
    rtree_node_children_free(children, nchildren)
    return result


def node_id(RStarTreeNode node):
    """Return the id of a leaf node."""
    cdef:
        size_t id_out
        RTreeError err
    err = rtree_node_id(node.node, &id_out)
    if err != Success:
        raise RuntimeError("rtree_node_id failed (node may not be a leaf)")
    return id_out


def node_envelope(RStarTreeNode node, size_t dim):
    """Return the (mins, maxs) bounding envelope of an R*-tree node."""
    cdef:
        np.ndarray[np.float64_t, ndim=1, mode="c"] mins = np.empty(dim, dtype=np.float64)
        np.ndarray[np.float64_t, ndim=1, mode="c"] maxs = np.empty(dim, dtype=np.float64)
        RTreeError err
    err = rtree_node_envelope(node.node, <double*>mins.data, <double*>maxs.data)
    if err != Success:
        raise RuntimeError("rtree_node_envelope failed")
    return mins, maxs
