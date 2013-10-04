subroutine test_blasmul

  use vector_tools
  use unittest_tools
  implicit none

  real, dimension(3, 3) :: A, B, C, D
  real, dimension(3) :: x, e, f, g
  integer :: i
  logical :: fail
  character(len=20) :: buf

  do i=1,5
    write(buf,'(i0)') i
    fail = .false.

    A = random_matrix(3)
    B = random_matrix(3)

    C = blasmul(A, B)
    D = matmul(A, B)

    if (any(C /= D)) fail = .true.

    call report_test("[blasmul_mm " // trim(buf) // "]", fail, .false., "The output of blasmul and matmul should be identical.")
  end do

  do i=1,5
    write(buf,'(i0)') i
    fail = .false.

    A = random_matrix(3)
    e = random_vector(3)

    f = blasmul(A, e)
    g = matmul(A, e)

    if (any(f /= g)) fail = .true.

    call report_test("[blasmul_mv " // trim(buf) // "]", fail, .false., "The output of blasmul and matmul should be identical.")
  end do
end subroutine test_blasmul
