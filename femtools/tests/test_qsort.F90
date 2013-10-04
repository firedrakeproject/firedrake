subroutine test_qsort

  use quicksort
  use unittest_tools

  implicit none

  integer, dimension(5) :: integer_data, permutation
  real, dimension(5) :: real_data

  integer_data = (/4, 2, 1, 5, 3/)
  call qsort(integer_data, permutation)
  call report_test("[Input data unchanged]", any(integer_data /=  (/4, 2, 1, 5, 3/)), .false., "Input data changed")
  call report_test("[Correct permutation]", any(permutation /=  (/3, 2, 5, 1, 4/)), .false., "Incorrect permutation")

  real_data = (/4.0, 2.0, 1.0, 5.0, 3.0/)
  call report_test("[Input data unchanged]", any(abs(real_data -  (/4.0, 2.0, 1.0, 5.0, 3.0/)) > epsilon(0.0)), .false., "Input data changed")
  call report_test("[Correct permutation]", any(permutation /=  (/3, 2, 5, 1, 4/)), .false., "Incorrect permutation")

end subroutine test_qsort
