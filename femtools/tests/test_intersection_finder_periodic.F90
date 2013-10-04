subroutine test_intersection_finder_periodic

  use intersection_finder_module
  use fields
  use linked_lists
  use read_triangle
  use unittest_tools
  implicit none

  type(vector_field) :: posA, posB
  type(ilist), dimension(1) :: map_AB
  logical :: fail

  ! A has one element
  ! B has two disconnected elements

  posA = read_triangle_files("data/intersection_finder_periodic_A", quad_degree=4)
  posB = read_triangle_files("data/intersection_finder_periodic_B", quad_degree=4)

  map_AB = intersection_finder(posA, posB)

  fail = (map_AB(1)%length /= 2)
  call report_test("[intersection finder periodic]", fail, .false., "")

end subroutine test_intersection_finder_periodic
