subroutine test_ele_local_num
  use element_numbering
  use unittest_tools
  implicit none

  type(ele_numbering_type), pointer:: ele_num
  integer:: i
  logical:: pass

  ! 4th degree interval
  ele_num => find_element_numbering(2, 1, 4)

  pass = all( ele_local_num( (/1,2/), ele_num)==(/ 1, 2, 3, 4, 5 /) )
  call report_test("[ele_local_num on 4th degree interval]", .not. pass, .false., "wrong output")

  pass = all( ele_local_num( (/2,1/), ele_num)==(/ 5, 4, 3, 2, 1 /) )
  call report_test("[ele_local_num on 4th degree interval reversed]", .not. pass, .false., "wrong output")

  ! 2nd degree triangle
  ele_num => find_element_numbering(3, 2, 2)

  pass = all( ele_local_num( (/1,2,3/), ele_num)==(/ 1, 2, 3, 4, 5, 6 /) )
  call report_test("[ele_local_num on 2nd degree triangle]", .not. pass, .false., "wrong output")

  pass = all( ele_local_num( (/3,2,1/), ele_num)==(/ 6, 5, 3, 4, 2, 1 /) )
  call report_test("[ele_local_num on 2nd degree triangle reordered]", .not. pass, .false., "wrong output")

  ! 2nd degree tet
  ele_num => find_element_numbering(4, 3, 2)

  pass = all( ele_local_num( (/1,2,3,4/), ele_num)==(/ (i, i=1,10) /) )
  call report_test("[ele_local_num on 2nd degree tet]", .not. pass, .false., "wrong output")

  pass = all( ele_local_num( (/3,1,4,2/), ele_num)==(/ 6, 4, 1, 9, 7, 10, 5, 2, 8, 3 /))
  call report_test("[ele_local_num on 2nd degree tet reordered]", .not. pass, .false., "wrong output")

  ! edges of linear quad
  ele_num => find_element_numbering(4, 2, 1)

  pass = all( edge_local_num( (/1,2/), ele_num) == (/ 1,3 /))
  call report_test("[edge_local_num on linear quad vertices 1 and 2]", .not. pass, .false., "wrong output")

  pass = all( edge_local_num( (/3,4/), ele_num) == (/ 2,4 /))
  call report_test("[edge_local_num on linear quad vertices 4 and 3]", .not. pass, .false., "wrong output")

  pass = all( edge_local_num( (/1,3/), ele_num) == (/ 1,2 /))
  call report_test("[edge_local_num on linear quad vertices 1 and 3]", .not. pass, .false., "wrong output")

  pass = all( edge_local_num( (/2,4/), ele_num) == (/ 3,4 /))
  call report_test("[edge_local_num on linear quad vertices 4 and 2]", .not. pass, .false., "wrong output")

end subroutine test_ele_local_num
