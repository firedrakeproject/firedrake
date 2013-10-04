subroutine test_tensormul

  use tensors
  use unittest_tools
  implicit none

  real, dimension(4, 5, 6, 7) :: tensorA
  real, dimension(6) :: vec6
  real, dimension(7) :: vec7

  real, dimension(4, 5, 7) :: resA
  real, dimension(4, 5, 6) :: resB

  real, dimension(2, 2, 2, 2) :: tensorB
  real, dimension(2) :: vec2

  real, dimension(2, 2, 2) :: resC

  logical :: fail

  tensorA = 1.0 ; vec6 = 1.0 ; vec7 = 1.0

  resA = tensormul(tensorA, vec6, 3)

  fail = .false.
  if (any(resA /= 6.0)) fail = .true.

  call report_test("[tensormul_4_1]", fail, .false., "Tensormul should give &
       & known output for known input.")

  resB = tensormul(tensorA, vec7, 4)

  fail = .false.
  if (any(resB /= 7.0)) fail = .true.

  call report_test("[tensormul_4_1]", fail, .false., "Tensormul should give &
       & known output for known input.")

  tensorB = 1.0 ; vec2 = 1.0

  resC = tensormul(tensorB, vec2, 3)

  fail = .false.
  if (any(resC /= 2.0)) fail = .true.

  call report_test("[tensormul_4_1]", fail, .false., "Tensormul should give &
       & known output for known input.")

end subroutine test_tensormul
