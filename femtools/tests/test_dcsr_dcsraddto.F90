!program to test dcsr_dcsraddto
subroutine test_dcsr_dcsraddto
  use sparse_tools
  use unittest_tools
  implicit none
  !
  logical :: fail
  type(dynamic_csr_matrix) :: m1,m2
  type(csr_matrix) :: m
  real, dimension(3,3) :: mat
  !
  call allocate(m1,3,3)
  call allocate(m2,3,3)

  call set(m1,1,1,-1.0)
  call set(m1,2,2,-1.0)
  call set(m1,3,3,-1.0)
  call set(m2,1,1,-1.0)
  call set(m2,2,2,-1.0)
  call set(m2,3,3,-1.0)
  call set(m1,1,2,1.0)
  call set(m1,2,1,1.0)
  call set(m1,2,3,1.0)
  call set(m1,3,2,1.0)

  call addto(m1,m2)
  m = dcsr2csr(m1)

  mat = 0.
  mat(1,1) = -2.
  mat(2,2) = -2.
  mat(3,3) = -2.
  mat(1,2) = 1.
  mat(2,1) = 1.
  mat(2,3) = 1.
  mat(3,2) = 1.

  fail = maxval(abs(dense(m)-mat))>1.0e-5
  call report_test("[dcsr_dscr_addto]", fail, .false., "dscr_dcsraddto doesnt work")

end subroutine test_dcsr_dcsraddto
