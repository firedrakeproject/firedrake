subroutine test_vecset

  use vector_set
  use unittest_tools
  implicit none

  integer :: idx
  logical :: path_taken
  logical :: fail

  call vecset_create(idx)
  call vecset_add(idx, (/float(15210), float(15211)/), path_taken)
  fail = path_taken
  call report_test("[vector_set]", fail, .false., "path_taken should be false")
  call vecset_add(idx, (/float(15210), float(15211)/), path_taken)
  fail = .not. path_taken
  call report_test("[vector_set]", fail, .false., "path_taken should be true")
  call vecset_destroy(idx)

end subroutine test_vecset
