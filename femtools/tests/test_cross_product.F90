subroutine test_cross_product

  use vector_tools
  use unittest_tools
  implicit none

  real, dimension(3) :: a, b, cross
  integer :: i
  logical :: fail
  character(len=20) :: buf

  do i=1,5
    write(buf,'(i0)') i

    a = random_vector(3)
    b = random_vector(3)
    cross = cross_product(a, b)

    fail = .false.
    if (.not. fequals(dot_product(a, cross), 0.0)) fail = .true.
    if (.not. fequals(dot_product(b, cross), 0.0)) fail = .true.
    call report_test("[cross product " // trim(buf) // "]", fail, .false., &
                     "The cross product of two vectors is orthogonal to both.")
  end do

end subroutine test_cross_product
