# cython: language_level=3

import numpy
from firedrake.petsc import PETSc
from firedrake.utils import IntType, ScalarType, RealType

cimport numpy
cimport petsc4py.PETSc as PETSc
from libc.stdint cimport uintptr_t

include "petschdr.pxi"


MAGIC = {2: (22, 3, 2),
         3: (81, 4, 3)}


ctypedef int (*compiled_call)(PetscScalar *,PetscScalar *,PetscScalar *,
                                PetscScalar *, PetscScalar *,
                                PetscScalar *, PetscScalar *, int)


cdef extern from "libsupermesh-c.h" nogil:
    void libsupermesh_tree_intersection_finder_set_input(long* nnodes_a, int* dim_a, long* nelements_a, int* loc_a, long* nnodes_b, int* dim_b, long* nelements_b, int* loc_b, double* positions_a, long* enlist_a, double* positions_b, long* enlist_b);
    void libsupermesh_tree_intersection_finder_query_output(long* nindices);
    void libsupermesh_tree_intersection_finder_get_output(long* nelements, long* nindices, long* indices, long* ind_ptr);


# Compute M_AB:
# For cell_A in mesh_A:
#     For cell_B in likely(cell_A):
#         mesh_S = supermesh(cell_A, cell_B)
#         if mesh_S is empty: continue
#         For cell_S in mesh_S:
#             evaluate basis functions of cell_A at dofs(A) of cell_S -> R_AS matrix
#             scale precomputed mass matrix to get M_SS
#                   (or mixed mass matrix if V_A, V_B have different finite elements)
#             evaluate basis functions of cell_B at dofs(B) of cell_S -> R_BS matrix
#             compute out = R_BS^T @ M_SS @ R_AS with dense matrix triple product
#             stuff out into relevant part of M_AB (given by outer(dofs_B, dofs_A))
def assemble_mixed_mass_matrix(V_A, V_B, candidates,
                               numpy.ndarray node_locations_A,
                               numpy.ndarray node_locations_B,
                               numpy.ndarray M_SS,
                               lib, PETSc.Mat mat not None):
    cdef:
        numpy.ndarray[PetscInt, ndim=2, mode="c"] V_A_cell_node_map
        numpy.ndarray[PetscInt, ndim=2, mode="c"] V_B_cell_node_map
        numpy.ndarray[PetscInt, ndim=2, mode="c"] vertex_map_A, vertex_map_B
        numpy.ndarray vertices_A, vertices_B
        numpy.ndarray outmat
        PetscInt cell_A, cell_B, i, gdim, num_dof_A, num_dof_B
        PetscInt num_cell_B, num_cell_A, num_vertices
        PetscInt insert_mode = PETSc.InsertMode.ADD_VALUES
        const PetscInt *V_A_map
        const PetscInt *V_B_map
        numpy.ndarray simplex_A, simplex_B
        numpy.ndarray simplices_C
        compiled_call library_call = (<compiled_call *><uintptr_t>lib)[0]

    num_cell_A = V_A.mesh().cell_set.size
    num_cell_B = V_B.mesh().cell_set.size

    outmat = numpy.empty((V_B.cell_node_map().arity,
                          V_A.cell_node_map().arity), dtype=ScalarType)
    mesh_A = V_A.mesh()
    mesh_B = V_B.mesh()
    vertex_map_A = mesh_A.coordinates.cell_node_map().values_with_halo
    vertex_map_B = mesh_B.coordinates.cell_node_map().values_with_halo

    num_vertices = vertex_map_A.shape[1]
    gdim = mesh_A.geometric_dimension()
    simplex_A = numpy.empty((num_vertices, gdim), dtype=ScalarType)
    simplex_B = numpy.empty_like(simplex_A, dtype=ScalarType)
    simplices_C = numpy.empty(MAGIC[gdim], dtype=ScalarType)

    vertices_A = mesh_A.coordinates.dat.data_ro_with_halos
    vertices_B = mesh_B.coordinates.dat.data_ro_with_halos
    V_A_cell_node_map = V_A.cell_node_map().values_with_halo
    V_B_cell_node_map = V_B.cell_node_map().values_with_halo
    num_dof_A = V_A.cell_node_map().arity
    num_dof_B = V_B.cell_node_map().arity
    for cell_A in range(num_cell_A):
        for cell_B in candidates(cell_A):
            for i in range(num_vertices):
                for j in range(gdim):
                    simplex_A[i, j] = vertices_A[vertex_map_A[cell_A, i], j]
                    simplex_B[i, j] = vertices_B[vertex_map_B[cell_B, i], j]
            library_call(<PetscScalar *>simplex_A.data, <PetscScalar *>simplex_B.data,
                         <PetscScalar *>simplices_C.data,
                         <PetscScalar *>node_locations_A.data,
                         <PetscScalar *>node_locations_B.data,
                         <PetscScalar *>M_SS.data,
                         <PetscScalar *>outmat.data,
                         <int> sum(MAGIC[gdim]))
            V_A_map = <const PetscInt *>(&V_A_cell_node_map[cell_A, 0])
            V_B_map = <const PetscInt *>(&V_B_cell_node_map[cell_B, 0])
            CHKERR(MatSetValuesLocal(mat.mat,
                                     num_dof_B, V_B_map,
                                     num_dof_A, V_A_map,
                                     <PetscScalar *>outmat.data, insert_mode))

    CHKERR(MatAssemblyBegin(mat.mat, MAT_FINAL_ASSEMBLY))
    CHKERR(MatAssemblyEnd(mat.mat, MAT_FINAL_ASSEMBLY))


def intersection_finder(mesh_A, mesh_B):
    # Plan:
    # Call libsupermesh_sort_intersection_finder_set_input
    # Call libsupermesh_sort_intersection_finder_query_output
    # Call libsupermesh_sort_intersection_finder_get_output
    # Return the output

    cdef:
        numpy.ndarray vertex_map_A, vertex_map_B
        numpy.ndarray vertices_A, vertices_B
        long nindices
        numpy.ndarray indices, indptr
        long nnodes_A, nnodes_B, ncells_A, ncells_B
        int dim_A, dim_B, loc_A, loc_B

    dim = mesh_A.geometric_dimension()
    assert dim == mesh_B.geometric_dimension()
    assert dim == mesh_A.topological_dimension()
    assert dim == mesh_B.topological_dimension()

    assert mesh_A.coordinates.function_space().ufl_element().degree() == 1
    assert mesh_B.coordinates.function_space().ufl_element().degree() == 1

    if mesh_A.comm.size > 1:
        compatible = False
        assert mesh_B._parallel_compatible is not None, "Whoever made mesh_B should explicitly mark mesh_A as having a compatible parallel layout."
        for _mesh_A in mesh_B._parallel_compatible:
            if mesh_A is _mesh_A():
                compatible = True
                break

        if not compatible:
            assert ValueError("Whoever made mesh_B should explicitly mark mesh_A as having a compatible parallel layout.")

    vertices_A = numpy.ndarray.astype(mesh_A.coordinates.dat.data_ro_with_halos.real, dtype=RealType)
    vertices_B = numpy.ndarray.astype(mesh_B.coordinates.dat.data_ro_with_halos.real, dtype=RealType)
    vertex_map_A = mesh_A.coordinates.cell_node_map().values_with_halo.astype(int)
    vertex_map_B = mesh_B.coordinates.cell_node_map().values_with_halo.astype(int)
    nnodes_A = mesh_A.coordinates.dof_dset.total_size
    nnodes_B = mesh_B.coordinates.dof_dset.total_size
    dim_A = mesh_A.geometric_dimension()
    dim_B = mesh_B.geometric_dimension()
    ncells_A = mesh_A.num_cells()
    ncells_B = mesh_B.num_cells()
    loc_A = vertex_map_A.shape[1]
    loc_B = vertex_map_B.shape[1]

    libsupermesh_tree_intersection_finder_set_input(&nnodes_A, &dim_A, &ncells_A, &loc_A,
                                                    &nnodes_B, &dim_B, &ncells_B, &loc_B,
                                                    <double*>vertices_A.data,
                                                    <long*>vertex_map_A.data,
                                                    <double*>vertices_B.data,
                                                    <long*>vertex_map_B.data)

    libsupermesh_tree_intersection_finder_query_output(&nindices)

    indices = numpy.empty((nindices,), dtype=int)
    indptr  = numpy.empty((mesh_A.num_cells() + 1,), dtype=int)

    libsupermesh_tree_intersection_finder_get_output(&ncells_A, &nindices, <long*>indices.data, <long*>indptr.data)

    out = {}
    for cell_A in range(ncells_A):
        (start, end) = indptr[cell_A], indptr[cell_A + 1]
        out[cell_A] = indices[start:end]

    return out
