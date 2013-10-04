subroutine test_eigenrecomposition

  use vector_tools
  use unittest_tools
  implicit none

  real, dimension(3, 3) :: mat, matout, diff, vecs, newvecs
  real, dimension(3)    :: vals, newvals
  real :: norm
  integer :: i, j, k
  logical :: fail = .false.
  logical :: warn = .false.
  character(len=12) :: buf


  do k=1,5
    mat = random_symmetric_matrix(3)

    matout = mat

    call eigendecomposition_symmetric(mat, vecs, vals)
    call eigenrecomposition(matout, vecs, vals)

    diff = matout - mat

    do i=1,3
      do j=1,3
        if (.not. fequals(diff(i, j), 0.0)) fail = .true.
      end do
    end do

    write(buf,'(i0)') k
    call report_test("[eigenrecomposition " // trim(buf) // "]", fail, warn, &
    "Eigenrecomposition and eigendecomposition should be inverses.")
  end do

  do k=6,10
    write(buf,'(i0)') k
    mat = random_symmetric_matrix(3)

    matout = mat

    call eigendecomposition_symmetric(mat, vecs, vals)
    call eigenrecomposition(matout, vecs, vals)
    call eigendecomposition_symmetric(matout, newvecs, newvals)

    norm = 0.0
    do i=1,3
      norm = norm + abs(newvals(i) - vals(i))**2
    end do
    if (.not. fequals(norm, 0.0)) fail = .true.
    call report_test("[eigenrecomposition values  " // trim(buf) // "]", fail, warn, "Eigenvalues should stay the same.")

    fail = .false.
    diff = newvecs - vecs
    do i=1,3
      do j=1,3
        if (.not. fequals(diff(i, j), 0.0)) fail = .true.
      end do
    end do

    call report_test("[eigenrecomposition vectors " // trim(buf) // "]", fail, warn, "Eigenvectors should stay the same.")
  end do

!  vals = (/1.00000000000000, 1.00000000000000, 1.00000000000000/)
!  vecs(:, 1) = (/0.879271809685662, 0.476295258761135, -4.889905291923050E-003/)
!  vecs(:, 2) = (/0.476320359308838,-0.879225474895281,  9.026605290312427E-003/)
!  vecs(:, 3) = (/-0.337266699534584, 0.629799780590775, 0.699716663909657/)
!
!  call eigenrecomposition(matout, vecs, vals)
!  call eigendecomposition_symmetric(matout, newvecs, newvals)
!  write(0,*) "vals == ", vals
!  write(0,*) "newvals == ", newvals
!  write(0,*) "vecs == ", vecs
!  write(0,*) "newvecs == ", newvecs
!  write(0,*) "newvals - vals == ", newvals - vals
!  write(0,*) "newvecs - vecs == ", newvecs - vecs
end subroutine test_eigenrecomposition
