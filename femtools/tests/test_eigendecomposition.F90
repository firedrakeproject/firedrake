subroutine test_eigendecomposition

  use unittest_tools
  use vector_tools
  implicit none

  real, dimension(3, 3) :: mat, evecs, matout, tmp
  real, dimension(3) :: evals, dot
  logical :: fail, warn
  integer :: i
  character(len=20) :: buf

  do i=1,5
    write(buf,'(i0)') i
    fail = .false.
    warn = .false.
    mat = random_symmetric_matrix(3)
    call eigendecomposition_symmetric(mat, evecs, evals)
    matout = matmul(evecs, matmul(get_mat_diag(evals), transpose(evecs)))
    tmp = mat - matout

    if (.not. mat_zero(tmp)) fail = .true.
    call report_test("[eigendecomposition " // trim(buf) // "]", fail, warn, "M == V * A * V^T")
  end do

  write(buf,'(i0)') 6
  fail = .false.
  mat(1, :) = (/1.0, 0.0, 0.0/)
  mat(2, :) = (/0.0, 1.0, 0.0/)
  mat(3, :) = (/0.0, 0.0, 0.0/)
  call eigendecomposition_symmetric(mat, evecs, evals)
  if (.not. fequals(evals(1), 0.0)) fail = .true.
  if (.not. fequals(evals(2), 1.0)) fail = .true.
  if (.not. fequals(evals(3), 1.0)) fail = .true.
  call report_test("[eigendecomposition " // trim(buf) // "]", fail, warn, "Eigendecomposition should handle degenerate matrices.")

  write(buf, '(i0)') 7
  fail = .false.
  mat(:, 1) = (/1.00967933380561, 5.243225984041118E-003, -5.382979992872267E-005/)
  mat(:, 2) = (/5.243225984041118E-003, 1.00284021806375, -2.915921812107133E-005/)
  mat(:, 3) = (/-5.382979992872245E-005, -2.915921812107133E-005, 1.00000029936434/)
  call eigendecomposition_symmetric(mat, evecs, evals)
  dot(1) = dot_product(evecs(:, 1), evecs(:, 2))
  dot(2) = dot_product(evecs(:, 1), evecs(:, 3))
  dot(3) = dot_product(evecs(:, 2), evecs(:, 3))
  do i=1,3
    if (.not. fequals(dot(i), 0.0)) fail = .true.
  end do
  call report_test("[eigendecomposition " // trim(buf) // "]", fail, warn, "Eigendecomposition of symmetric &
  & matrices should yield orthogonal eigenvectors.")

end subroutine test_eigendecomposition
