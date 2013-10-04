#include "confdefs.h"
#include "fdebug.h"

subroutine test_quad_supermesh

  use unittest_tools
  use read_triangle
  use fields
  use linked_lists
  use intersection_finder_module
  use transform_elements
  use elements
  use supermesh_construction
  use vtk_interfaces

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
  integer :: dump_idx

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

  dump_idx = 0
  total_B = 0.0
  total_C = 0.0

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
#define DUMP_SUPERMESH_INTERSECTIONS
#ifdef DUMP_SUPERMESH_INTERSECTIONS
      if (ele_count(intersection) /= 0) then
        call vtk_write_fields("intersection", dump_idx, intersection, intersection%mesh)
        dump_idx = dump_idx + 1
      end if
#endif
      do ele_C=1,ele_count(intersection)
        call transform_to_physical(intersection, ele_C, detwei=tri_detwei)
        vols_C = vols_C + sum(tri_detwei)
      end do
      llnode => llnode%next
    end do

    total_B = total_B + vol_B
    total_C = total_C + vols_C
    fail = (vol_B .fne. vols_C)
    !call report_test("[quad supermesh: completeness]", fail, .false., "Need to have the same volume!")
    if (fail) then
      write(0,*) "ele_B: ", ele_B
      write(0,*) "vol_B: ", vol_B
      write(0,*) "vols_C: ", vols_C
    end if
  end do

  fail = total_B .fne. total_C
  call report_test("[quad supermesh: completeness]", fail, .false., "Need to have the same volume!")
  !write(0,*) "total_B: ", total_B
  !write(0,*) "total_C: ", total_C

end subroutine test_quad_supermesh
