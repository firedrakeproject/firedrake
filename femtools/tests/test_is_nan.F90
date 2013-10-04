subroutine test_is_nan

  use unittest_tools
  implicit none

  real :: nan, zero, zero2, rand
  logical :: fail, isnan_output

  call random_number(rand)

  fail = .false.
  zero = 0.0
  zero2 = 0.0
  zero2 = zero2 * zero * rand
  nan = zero2 / zero

  isnan_output = is_nan(nan)

  if (.not. is_nan(nan)) fail = .true.
  call report_test("[is NaN]", fail, .false., "is_nan should report &
                   & true for NaN.")

end subroutine test_is_nan
