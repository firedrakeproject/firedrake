subroutine test_linked_edge_list

  use linked_lists
  use unittest_tools
  implicit none

  type(elist) :: edgelist
  logical :: fail
  integer :: i, j

  fail = .false.

  if (edgelist%length /= 0) fail = .true.
  call report_test("[edgelist initialised]", fail, .false., "Initial length &
  & should be zero.")

  fail = .false.
  call insert(edgelist, 1, 2)
  if (edgelist%length /= 1) fail = .true.
  if (.not. associated(edgelist%firstnode)) fail = .true.
  if (.not. associated(edgelist%lastnode)) fail = .true.
  call report_test("[edgelist first insert]", fail, .false., "After &
  & first inserting, the list length should be 1.")

  fail = .false.
  call insert(edgelist, 1, 3)
  if (edgelist%length /= 2) fail = .true.
  call report_test("[edgelist second insert]", fail, .false., "After &
  & a second insert, the state should be correct.")

  fail = .false.
  call spop(edgelist, i, j)
  if (i /= 1) fail = .true.
  if (j /= 2) fail = .true.
  if (edgelist%length /= 1) fail = .true.
  call report_test("[edgelist pop]", fail, .false., "Popping &
  & from a list should give the first value inserted.")

  fail = .false.
  call spop(edgelist, i, j)
  if (i /= 1) fail = .true.
  if (j /= 3) fail = .true.
  if (edgelist%length /= 0) fail = .true.
  call report_test("[edgelist clear]", fail, .false., "Popping &
  & the last element should clear the list.")

end subroutine test_linked_edge_list
