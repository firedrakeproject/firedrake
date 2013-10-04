subroutine test_array_permutation

  use unittest_tools

  implicit none

  integer :: i
  integer, dimension(5) :: integer_data, permutation

  permutation = (/(size(permutation) - i, i = 0, size(permutation) - 1)/)
  integer_data = (/(i, i = 1, size(integer_data))/)

  integer_data = integer_data(permutation)
  call report_test("[array permutation]", any(integer_data /= permutation), .false., "Incorrect permutation")

end subroutine test_array_permutation
