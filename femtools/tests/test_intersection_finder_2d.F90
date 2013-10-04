subroutine test_intersection_finder_2d

  use unittest_tools
  use read_triangle
  use fields
  use linked_lists
  use intersection_finder_module

  type(vector_field) :: positionsA, positionsB
  type(ilist), dimension(1) :: map_AB
  type(ilist), dimension(3) :: bigger_map_AB
  !type(inode), pointer :: node

  integer :: i
  logical :: fail

  positionsA = read_triangle_files("data/triangle.1", quad_degree=4)
  positionsB = read_triangle_files("data/triangle.1", quad_degree=4)
  map_AB = advancing_front_intersection_finder(positionsA, positionsB)

  fail = (map_AB(1)%length /= 1)
  call report_test("[intersection finder: length]", fail, .false., "There shall be only one")

  i = fetch(map_AB(1), 1)
  fail = (i /= 1)
  call report_test("[intersection finder: correct]", fail, .false., "The answer should be one")

  call deallocate(positionsB)
  positionsB = read_triangle_files("data/triangle.2", quad_degree=4)
  map_AB = advancing_front_intersection_finder(positionsA, positionsB)

  fail = (map_AB(1)%length /= 3)
  call report_test("[intersection finder: length]", fail, .false., "There shall be three elements")
  !node => map_AB(1)%firstnode
  !do while (associated(node))
  !  write(0,*) "node%value: ", node%value
  !  node => node%next
  !end do

  call deallocate(positionsA)
  positionsA = read_triangle_files("data/triangle.2", quad_degree=4)
  bigger_map_AB = advancing_front_intersection_finder(positionsA, positionsB)
  do i=1,ele_count(positionsA)
    fail = (bigger_map_AB(i)%length < 1)
    call report_test("[intersection finder: length]", fail, .false., "There shall be only one")

    fail = (.not. has_value(bigger_map_AB(i), i))
    call report_test("[intersection finder: correct]", fail, .false., "The answer should be correct")
  end do

end subroutine test_intersection_finder_2d
