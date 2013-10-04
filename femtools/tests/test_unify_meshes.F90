#include "confdefs.h"
#include "fdebug.h"

subroutine test_unify_meshes

  use unittest_tools
  use read_triangle
  use fields
  use linked_lists
  use intersection_finder_module
  use transform_elements
  use elements
  use supermesh_construction
  use vtk_interfaces
  use unify_meshes_module

  type(vector_field) :: positionsA, positionsB
  type(ilist), dimension(:), allocatable :: map_BA
  real, dimension(:), allocatable :: quad_detwei, tri_detwei
  integer :: ele_A, ele_B, ele_C
  real :: vol_B, vols_C, total_B, total_C
  logical :: fail
  type(element_type), pointer :: shape
  type(inode), pointer :: llnode
  type(vector_field) :: intersection
  type(element_type) :: supermesh_shape
  type(quadrature_type) :: supermesh_quad
  integer :: dim

  type(mesh_type) :: accum_mesh
  type(vector_field) :: accum_positions, accum_positions_tmp

  positionsA = read_triangle_files("data/dg_interpolation_quads_A", quad_degree=1)
  positionsB = read_triangle_files("data/dg_interpolation_quads_B", quad_degree=1)

  dim = positionsA%dim

  allocate(map_BA(ele_count(positionsB)))
  allocate(quad_detwei(ele_ngi(positionsA, 1)))
  shape => ele_shape(positionsA, 1)
  assert(sum(shape%quadrature%weight) == 4)

  supermesh_quad = make_quadrature(vertices = dim+1, dim =dim, degree=5)
  supermesh_shape = make_element_shape(vertices = dim+1, dim =dim, degree=1, quad=supermesh_quad)
  allocate(tri_detwei(supermesh_shape%ngi))

  call allocate(accum_mesh, 0, 0, supermesh_shape, "AccumulatedMesh")
  call allocate(accum_positions, dim, accum_mesh, "AccumulatedPositions")

  total_B = 0.0

  map_BA = intersection_finder(positionsB, positionsA)
  call intersector_set_dimension(dim)

  do ele_B=1,ele_count(positionsB)
    call transform_to_physical(positionsB, ele_B, detwei=quad_detwei)
    vol_B = sum(quad_detwei)

    llnode => map_BA(ele_B)%firstnode
    vols_C = 0.0
    do while(associated(llnode))
      ele_A = llnode%value
      intersection = intersect_elements(positionsA, ele_A, ele_val(positionsB, ele_B), supermesh_shape)
      call unify_meshes_quadratic(accum_positions, intersection, accum_positions_tmp)
      call deallocate(accum_positions)
      accum_positions = accum_positions_tmp

      llnode => llnode%next

      call deallocate(intersection)
    end do

    total_B = total_B + vol_B
  end do

  total_C = 0.0
  do ele_C=1,ele_count(accum_positions)
    call transform_to_physical(accum_positions, ele_C, detwei=tri_detwei)
    vols_C = sum(tri_detwei)
    total_C = total_C + vols_C
  end do

  fail = total_B .fne. total_C
  call report_test("[unify meshes: completeness]", fail, .false., "Need to have the same volume!")

  call vtk_write_fields("unified_mesh", 0, accum_positions, accum_positions%mesh)

end subroutine test_unify_meshes
