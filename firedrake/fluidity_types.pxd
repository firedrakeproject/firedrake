# These types are the C struct equivalent of the types in Python_Interface_f.F90
ctypedef struct element_t:
    int dimension
    int vertices
    int ndof
    int degree
    int *dofs_per
    # entity_dofs is actualla a 3*ndof array with entries dim, entity, dof.
    int *entity_dofs

ctypedef struct mesh_t:
    int *element_vertex_list
    # naming is T cython_name "c_name"
    int topological_dimension "dim"
    int cell_vertices
    int vertex_count
    int cell_count
    int exterior_facet_count
    int interior_facet_count
    int *cell_classes
    int *vertex_classes
    int *region_ids
    int uid
    int geometric_dimension "space_dimension"
    double *coordinates
    void *fluidity_coordinate
    void *fluidity_mesh
    int *interior_local_facet_number
    int *exterior_local_facet_number
    int *interior_facet_cell
    int *exterior_facet_cell
    int *boundary_ids

ctypedef enum halo_entity:
    VERTEX
    CELL

ctypedef struct halo_t:
    int **sends
    int *nsends
    int **receives
    int *nreceives
    int nowned_nodes
    int nprocs
    halo_entity entity_type
    int *receives_global_to_universal
    int receives_global_to_universal_len
    int universal_offset
    int comm
    void *fluidity_halo

ctypedef struct function_space_t:
    int *element_dof_list
    void *fluidity_mesh
    int element_count
    int dof_count
    int *dof_classes
    int *interior_facet_node_list
    int *exterior_facet_node_list


# Fluidity's old python caching, we want to turn this off
cdef extern bint python_cache
cdef extern function_space_t function_space_f(void *, element_t *)
cdef extern mesh_t read_mesh_f(char *, char *, int)
cdef extern halo_t halo_f(void *, halo_entity)
cdef extern void function_space_destructor_f(void *)
cdef extern void vector_field_destructor_f(void *)
cdef extern function_space_t extruded_mesh_f(void *, element_t *, int *)
