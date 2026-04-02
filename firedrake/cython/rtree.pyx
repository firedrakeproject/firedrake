# cython: language_level=3

cimport numpy as np
import numpy as np
import ctypes
import cython
from libc.stddef cimport size_t
from libc.stdint cimport uintptr_t, uint32_t

cimport mpi4py.MPI as MPI
from mpi4py.libmpi cimport MPI_INT
from petsc4py.PETSc cimport CHKERR
from firedrake.exceptions import EmptyNodeEnvelopeError

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

    RTreeError rtree_node_envelope(
        const RTreeNodeH *node,
        double *mins_out,
        double *maxs_out
    )

cdef class RTree(object):
    """Python class for holding a spatial index."""

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
            if ids_out != NULL:
                rtree_free_ids(ids_out, nids_out)
            raise RuntimeError("rtree_locate_all_at_point failed")

        # Points may lie in multiple bounding boxes owned by the same rank
        seen_ranks: set[int] = set()
        for j in range(nids_out):
            seen_ranks.add(<int>ids_out[j])
        err = rtree_free_ids(ids_out, nids_out)
        if err != Success:
            raise RuntimeError("rtree_free_ids failed")

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

def tree_depth(RTree rtree):
    """Return the depth of the Rtree."""
    cdef:
        size_t depth = 0
        RTreeError err

    err = rtree_depth(rtree.tree, &depth)
    if err != Success:
        raise RuntimeError("rtree_depth failed")
    return depth


cdef class RTreeNodeChildren(object):
    """Python class for holding the array of child node handles
    returned by ``rtree_node_children``.
    """

    cdef RTreeNodeH** children
    cdef size_t nchildren
    cdef object __weakref__

    def __cinit__(self, uintptr_t children_handle, size_t nchildren):
        self.children = <RTreeNodeH**>children_handle
        self.nchildren = nchildren

    def __dealloc__(self):
        if self.children != <RTreeNodeH**>0:
            rtree_node_children_free(self.children, self.nchildren)
            print(f"Deallocated {self.nchildren} child nodes")
            self.children = <RTreeNodeH**>0
            self.nchildren = 0

cdef class RTreeNode(object):
    """Python class for holding an rtree node."""

    cdef RTreeNodeH* node_handle
    cdef bint is_root
    cdef object owning_tree
    cdef object children
    cdef object __weakref__

    def __cinit__(self, uintptr_t node_handle, bint is_root=False,
                  owning_tree=None, children_owner_ref=None):
        self.is_root = is_root
        # Create a reference to the owning tree to avoid cleaning up the rtree
        # while the node is still alive.
        self.owning_tree = owning_tree
        # We create a reference to the object owning the children since we need
        # to free all the children at the same time with `rtree_node_children_free`.
        self.children = children_owner_ref
        if node_handle == 0:
            raise RuntimeError("invalid node handle")
        self.node_handle = <RTreeNodeH*>node_handle
    
    def get_tree_ref(self):
        return self.owning_tree
    
    def get_is_root(self):
        return self.is_root
    
    def get_children_ref(self):
        return self.children

    def __dealloc__(self):
        if self.node_handle != <RTreeNodeH*>0 and self.is_root:
            # We can only free the node if it is the root. Child nodes
            # must be freed all at once by `rtree_node_children_free`.
            # The RTreeNodeChildren class facilitates this.
            rtree_node_free(self.node_handle)
            self.node_handle = <RTreeNodeH*>0
            print("node deallocated")

def root_node(RTree rtree):
    """Return the root node of the Rtree."""
    cdef:
        RTreeNodeH* node
        RTreeError err
    err = rtree_root_node(rtree.tree, &node)
    if err != Success:
        raise RuntimeError("rtree_root_node failed")
    return RTreeNode(<uintptr_t>node, is_root=True, owning_tree=rtree)


def node_children(RTreeNode node):
    """Return the children of an R-tree node as a list of RTreeNodeHs."""
    cdef:
        RTreeNodeH** children
        size_t nchildren
        RTreeError err
        RTreeNodeChildren child_nodes
        list result
    err = rtree_node_children(node.node_handle, &children, &nchildren)
    if err != Success:
        raise RuntimeError("rtree_node_children failed")

    # We create an RTreeNodeChildren object to own the children array
    # to make sure that it gets freed when we're done with it.
    child_nodes = RTreeNodeChildren(<uintptr_t>children, nchildren)

    result = [
        RTreeNode(
            <uintptr_t>children[i],
            owning_tree=node.owning_tree,
            children_owner_ref=child_nodes,
        ) for i in range(nchildren)
    ]

    return result

def node_envelope(RTreeNode node, size_t dim):
    """Return the (mins, maxs) bounding envelope of an rtree node."""
    cdef:
        np.ndarray[np.float64_t, ndim=1, mode="c"] mins = np.empty(dim, dtype=np.float64)
        np.ndarray[np.float64_t, ndim=1, mode="c"] maxs = np.empty(dim, dtype=np.float64)
        RTreeError err
    err = rtree_node_envelope(node.node_handle, <double*>mins.data, <double*>maxs.data)
    if err == EmptyNodeEnvelope:
        # This only happens if the node is a root of an empty tree.
        raise EmptyNodeEnvelopeError("Node has no envelope (empty node)")
    elif err != Success:
        raise RuntimeError("rtree_node_envelope failed")
    return mins, maxs
