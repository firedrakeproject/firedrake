!! Vector_Tools.solve used to crash if you didn't
!! pass in the optional stat argument.

subroutine test_solve

  use unittest_tools
  use vector_tools

  real, dimension(3, 3) :: id
  real, dimension(3, 1)    :: b, x
  integer :: i
  logical :: fail, warn

  fail = .false.
  warn = .false.

  id = 0.0
  do i=1,3
    id(i, i) = 1.0
    b(i, 1)  = float(i)
  end do

  x = b
  call solve(id, b)

  do i=1,3
    if (.not. fequals(x(i, 1), b(i, 1))) fail = .true.
  end do

  call report_test("[id matrix solve]", fail, warn, "Solving Ix = b should yield x == b.")
end subroutine
