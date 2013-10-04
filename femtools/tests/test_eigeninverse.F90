subroutine test_eigeninverse

  use unittest_tools
  use vector_tools
  implicit none

  real, dimension(3, 3) :: mat, evecs
  real, dimension(3) :: evals
  logical :: fail

  evecs(1, :) = (/-6.54163386e-01,  -7.56353267e-01,   5.39898168e-16/)
  evecs(2, :) = (/2.19406960e-16,  -1.91603257e-15,  -1.00000000e-00/)
  evecs(3, :) = (/-7.56353267e-01,   6.54163386e-01,  -6.41747296e-16/)
  evals = (/2.51855956e+07,   2.49437236e+06+0,   1.94481173e-24/)

  call eigenrecomposition(mat, evecs, evals)
  call eigendecomposition(mat, evecs, evals)

  fail = .false.
  if (any(evals < 0.0)) fail = .true.
  call report_test("[inverse relationship of eigenrecomposition and decomposition]", &
                   fail, .false., "These operations should be inverses.")

end subroutine test_eigeninverse
