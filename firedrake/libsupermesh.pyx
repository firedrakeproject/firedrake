cimport numpy as np
import numpy

from libc.stdint cimport int64_t, uint64_t
from libc.stdlib cimport free, malloc
cimport spatialindex

cdef extern from "supermesh.h":
    void intersect_tris_c(const double* tri_A, const double* tri_B, double* tris_C, int* n_tris_c);

def intersect_tris(np.ndarray[double, ndim=2, mode="c"] tri_A, np.ndarray[double, ndim=2, mode="c"] tri_B):
    cdef double tris_C[22][3][2]
    cdef int n_tris_C
    intersect_tris_c(&tri_A[0,0], &tri_B[0,0], &tris_C[0][0][0], &n_tris_C)
    np_tris_C = numpy.empty((n_tris_C, 3, 2))
    cdef double[:,:,:] mv_tris_C = tris_C # memory view, allows slicing
    np_tris_C[...] = mv_tris_C[0:n_tris_C, :, :]
    return np_tris_C

# is there a better way to do this?
# I need to somehow declare that mesh.spatial_index is a SpatialIndex
# before I can access its (IndexH) index attribute
cdef spatialindex.SpatialIndex get_index(mesh):
    return mesh.spatial_index

cdef inline get_tri(int [:,:] enlist, int ele, double [:,:] coords, double tri[3][2]):
    cdef int j, n, k
    for j in range(3):
        n = enlist[ele, j]
        for k in range(2):
            tri[j][k] = coords[n, k]

cdef inline get_tri_bbox(double tri[3][2], double tri_min[2], double tri_max[2]):
    cdef int j, k
    for k in range(2):
        tri_min[k] = tri[0][k]
        tri_max[k] = tri[0][k]
    for j in range(1, 3):
        for k in range(2):
            if tri[j][k] < tri_min[k]:
                tri_min[k] = tri[j][k]
            if tri[j][k] > tri_max[k]:
                tri_max[k] = tri[j][k]

def create_supermesh(mesh_A, mesh_B):
    cdef spatialindex.IndexH index_B = get_index(mesh_B).index
    cdef int ele_A
    cdef int ncells_A = mesh_A.num_cells()
    cdef double [:,:] coords_A = mesh_A.coordinates.dat.data
    cdef double [:,:] coords_B = mesh_B.coordinates.dat.data
    cdef int [:,:] enlist_A = mesh_A.coordinates.cell_node_map().values
    cdef int [:,:] enlist_B = mesh_B.coordinates.cell_node_map().values
    cdef double tri_A[3][2], tri_B[3][2]
    cdef double tri_min[2], tri_max[2]
    cdef double tris_C[22][6]
    cdef int n_tris_C
    cdef int j, k
    cdef int64_t *eles_B
    cdef uint64_t n_eles_B

    cdef int *intersection_count = <int *> malloc(ncells_A * sizeof(int)) # n/o intersection per ele_A
    cdef list intersection_list = []  # int[2] for each intersection: ele_B and n/o supermesh triangles
    cdef list tris_C_list = [] # a double[3][2] for each supermesh triangle

    for ele_A in range(ncells_A):
        get_tri(enlist_A, ele_A, coords_A, tri_A)
        get_tri_bbox(tri_A, tri_min, tri_max)
        spatialindex.Index_Intersects_id(index_B, tri_min, tri_max, 2, &eles_B, &n_eles_B)

        intersection_count[ele_A] = 0
        for j in range(n_eles_B):
            get_tri(enlist_B, eles_B[j], coords_B, tri_B)
            intersect_tris_c(&tri_A[0][0], &tri_B[0][0], &tris_C[0][0], &n_tris_C)
            if n_tris_C>0:
                intersection_list.append((eles_B[j], n_tris_C))
                intersection_count[ele_A] += 1
                for k in range(n_tris_C):
                    tris_C_list.append(tris_C[k])

    cdef int all_tris_C = len(tris_C_list)
    cell_map_CA = numpy.empty(all_tris_C, dtype=int)
    cell_map_CB = numpy.empty(all_tris_C, dtype=int)

    cdef int intersection_idx = 0
    cdef int supermesh_idx = 0
    for ele_A in range(ncells_A):
        n_eles_B = intersection_count[ele_A]
        for j in range(n_eles_B):
            ele_B, n_tris_C = intersection_list[intersection_idx]
            intersection_idx += 1
            cell_map_CA[supermesh_idx:supermesh_idx+n_tris_C] = ele_A
            cell_map_CB[supermesh_idx:supermesh_idx+n_tris_C] = ele_B
            supermesh_idx += n_tris_C

    free(intersection_count)
    
    return numpy.asarray(tris_C_list).reshape((all_tris_C, 3, 2)), cell_map_CA, cell_map_CB
