subroutine test_intersection_finder_3d

  use unittest_tools
  use read_triangle
  use fields
  use linked_lists
  use intersection_finder_module

  type(vector_field) :: positionsA, positionsB
  type(ilist), dimension(1) :: map_AB

  integer :: i
  logical :: fail

  positionsA = read_triangle_files("data/tet", quad_degree=4)
  positionsB = read_triangle_files("data/tet", quad_degree=4)
  map_AB = advancing_front_intersection_finder(positionsA, positionsB)

  fail = (map_AB(1)%length /= 1)
  call report_test("[intersection finder: length]", fail, .false., "There shall be only one")

  i = fetch(map_AB(1), 1)
  fail = (i /= 1)
  call report_test("[intersection finder: correct]", fail, .false., "The answer should be one")

end subroutine test_intersection_finder_3d
