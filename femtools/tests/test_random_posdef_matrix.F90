subroutine test_random_posdef_matrix

  use unittest_tools
  use vector_tools
  implicit none

  real, dimension(3, 3) :: mat, evecs
  real, dimension(3) :: evals
  integer :: i, j, dim = 3
  logical :: fail, warn
  character(len=20) :: buf

  do i=1,5

    fail = .false.
    warn = .false.

    mat = random_posdef_matrix(dim)
    call eigendecomposition_symmetric(mat, evecs, evals)
    do j=1,dim
      if (evals(j) .flt. 0.0) then
        print *, "i == ", i, "; j == ", j, "; evals(j) == ", evals(j)
        fail = .true.
      end if
    end do

    write(buf,'(i0)') i
    call report_test("[positive definite matrix " // trim(buf) // "]", fail, warn, &
    "Positive definite matrices have positive eigenvalues.")
  end do

end subroutine test_random_posdef_matrix
