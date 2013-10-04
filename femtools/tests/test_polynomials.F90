subroutine test_polynomials

  use polynomials
  use unittest_tools
  implicit none

  type(polynomial) :: poly
  character(len=100) :: buffer

  logical :: fail

  poly=(/3.,2.,1./)

  buffer= poly2string(poly)

  fail=.not.(buffer=="   3.000x^2 +    2.000x +    1.000")

  call report_test("[polynomial2string]", fail, .false., "Wrote the poly wrong.")

end subroutine test_polynomials
