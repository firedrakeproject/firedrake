#include "confdefs.h"

subroutine test_intersection_finder_completeness

  use unittest_tools
  use read_triangle
  use fields
  use linked_lists
  use intersection_finder_module
  use transform_elements
  use elements
  use supermesh_construction

  type(vector_field) :: positionsA, positionsB
  type(ilist), dimension(:), allocatable :: map_BA
  real, dimension(:), allocatable :: detwei
  integer :: ele_A, ele_B, ele_C
  real :: vol_B, vols_C
  logical :: fail
  type(inode), pointer :: llnode
  type(vector_field) :: intersection

  positionsA = read_triangle_files("data/intersection_finder.1", quad_degree=4)
  positionsB = read_triangle_files("data/intersection_finder.2", quad_degree=4)

  allocate(map_BA(ele_count(positionsB)))
  allocate(detwei(ele_ngi(positionsA, 1)))

  map_BA = advancing_front_intersection_finder(positionsB, positionsA)
  call intersector_set_dimension(positionsA%dim)

  do ele_B=1,ele_count(positionsB)
    call transform_to_physical(positionsB, ele_B, detwei=detwei)
    vol_B = sum(detwei)

    llnode => map_BA(ele_B)%firstnode
    vols_C = 0.0
    do while(associated(llnode))
      ele_A = llnode%value
      intersection = intersect_elements(positionsA, ele_A, ele_val(positionsB, ele_B), ele_shape(positionsB, ele_B))
      do ele_C=1,ele_count(intersection)
        call transform_to_physical(intersection, ele_C, detwei=detwei)
        vols_C = vols_C + sum(detwei)
      end do
      llnode => llnode%next
    end do

    fail = (vol_B .fne. vols_C)
    call report_test("[intersection finder: completeness]", fail, .false., "Need to have the same volume!")
    if (fail) then
      write(0,*) "ele_B: ", ele_B
      write(0,*) "vol_B: ", vol_B
      write(0,*) "vols_C: ", vols_C
    end if
  end do

end subroutine test_intersection_finder_completeness
