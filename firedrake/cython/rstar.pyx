# cython: language_level=3

cimport numpy as np
import numpy as np
import ctypes
import cython
from libc.stddef cimport size_t
from libc.stdint cimport uintptr_t

include "rstarinc.pxi"

cdef class RStarTree(object):
    """Python class for holding a native spatial index object."""

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
def from_regions(np.ndarray[np.float64_t, ndim=2, mode="c"] regions_lo,
                 np.ndarray[np.float64_t, ndim=2, mode="c"] regions_hi):
    """Builds a spatial index from a set of maximum bounding regions (MBRs).

    regions_lo and regions_hi must have the same size.
    regions_lo[i] and regions_hi[i] contain the coordinates of the diagonally
    opposite lower and higher corners of the i-th MBR, respectively.
    """
    cdef:
        RStarTree rstar_tree
        np.ndarray[np.npy_uintp, ndim=1, mode="c"] ids
        RTreeH* rtree
        size_t n
        size_t dim
        RTreeError err 

    assert regions_lo.shape[0] == regions_hi.shape[0]
    assert regions_lo.shape[1] == regions_hi.shape[1]
    n = <size_t>regions_lo.shape[0]
    dim = <size_t>regions_lo.shape[1]
    ids = np.arange(n, dtype=np.uintp)

    err = rtree_bulk_load(
        &rtree,
        <const double*>regions_lo.data,
        <const double*>regions_hi.data,
        <const size_t*>ids.data,
        n,
        dim
    )
    if err != Success:
        raise RuntimeError("RTree_FromArray failed")
    rstar_tree = RStarTree(<uintptr_t>rtree)
    return rstar_tree

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
