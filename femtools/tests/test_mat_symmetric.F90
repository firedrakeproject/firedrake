subroutine test_mat_symmetric

  use unittest_tools
  implicit none

  real, dimension(3, 3) :: mat
  logical :: fail, warn

  mat(1, :) = (/5821120296.70721, -288935353.239809, -43439838442.8431/)
  mat(2, :) = (/-288935353.239809, 14341517.9712309, 2156165379.10662/)
  mat(3, :) = (/-43439838442.8430, 2156165379.10662, 324167903664.234/)

  fail = .false.; warn = .false.
  if (.not. mat_is_symmetric(mat)) fail = .true.

  call report_test("[mat_symmetric]", fail, warn, "A symmetric matrix should be regarded as symmetric.")

end subroutine test_mat_symmetric
