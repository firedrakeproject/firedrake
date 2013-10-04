#include "fdebug.h"

module unittest_tools
!!< This module contains utility functions for the unit testing framework.
  use vector_tools
  use sparse_tools
  use fldebug
  use reference_counting
  use ieee_arithmetic
  implicit none

  private
  public operator(.flt.), operator(.fgt.), operator(.feq.), operator(.fne.), &
    fequals, fnequals, write_vector, report_test,  write_matrix, &
    is_nan, mat_is_symmetric, mat_zero, mat_diag, random_vector, random_matrix, &
    random_symmetric_matrix, random_posdef_matrix, random_sparse_matrix, &
    get_mat_diag, get_matrix_identity, mat_clean, vec_clean, flt, fgt, &
    report_test_no_references

  interface operator(.flt.)
    module procedure flt_op
  end interface

  interface operator(.fgt.)
    module procedure fgt_op
  end interface

  interface operator(.feq.)
    module procedure fequals_scalar_op, fequals_array_op, fequals_array_scalar_op, fequals_matrix_op, fequals_matrix_scalar_op
  end interface

  interface operator(.fne.)
    module procedure fne_scalar_op, fne_array_op, fne_array_scalar_op, fne_matrix_op, fne_matrix_scalar_op
  end interface

  interface fequals
    module procedure fequals_scalar, fequals_array, fequals_array_scalar, fequals_matrix, fequals_matrix_scalar, dcsr_fequals
  end interface

  interface fnequals
    module procedure fne_scalar, fne_array, fne_array_scalar, fne_matrix, fne_matrix_scalar
  end interface

  interface is_nan
    module procedure is_nan_scalar, is_nan_vector, is_nan_tensor2, &
      & is_nan_tensor3
  end interface is_nan

  interface write_vector
    module procedure write_vector_real, write_vector_integer
  end interface

  contains

  subroutine report_test(title, fail, warn, msg)
    !!< This is the subroutine used by unit tests to report the output of
    !!< a test case.

    !! Title: the name of the test case.
    character(len=*), intent(in) :: title
    !! Msg: an explanatory message printed if the test case fails.
    character(len=*), intent(in) :: msg
    !! Has the test case failed, or triggered a warning? Set fail or warn to .true. if so.
    logical, intent(in) :: fail, warn

    if (fail) then
      print "('Fail: ',a,'; error: ',a)", title, msg
    else if (warn) then
      print "('Warn: ',a,'; error: ',a)", title, msg
    else
      print "('Pass: ',a)", title
    end if

  end subroutine report_test

  subroutine report_test_no_references()
    !!< Report the output of a test for the absence of references

    logical :: fail

    fail = associated(refcount_list%next)

    call report_test("[No references]", fail, .false., "Have references remaining")
    if(fail) then
      ewrite(0, *) "References remaining:"
      call print_references(0)
    end if

  end subroutine report_test_no_references

  pure function fequals_scalar_op(float1, float2) result(equals)
    real, intent(in) :: float1
    real, intent(in) :: float2

    logical :: equals

    equals = fequals(float1, float2)

  end function fequals_scalar_op

  pure function fne_scalar_op(float1, float2) result(nequals)
    real, intent(in) :: float1
    real, intent(in) :: float2

    logical :: nequals

    nequals = .not. (float1 .feq. float2)

  end function fne_scalar_op

  pure function fequals_array_op(array1, array2) result(equals)
    real, dimension(:), intent(in) :: array1
    real, dimension(size(array1)), intent(in) :: array2

    logical :: equals

    equals = fequals(array1, array2)

  end function fequals_array_op

  pure function fne_array_op(array1, array2) result(nequals)
    real, intent(in), dimension(:) :: array1, array2

    logical :: nequals

    nequals = .not. fequals(array1, array2)

  end function fne_array_op

  pure function fequals_array_scalar_op(array1, float2) result(equals)
    real, dimension(:), intent(in) :: array1
    real, intent(in) :: float2

    logical :: equals

    equals = fequals(array1, float2)

  end function fequals_array_scalar_op

  pure function fne_array_scalar_op(array1, float2) result(nequals)
    real, dimension(:), intent(in) :: array1
    real, intent(in) :: float2

    logical :: nequals

    nequals = .not. fequals(array1, float2)

  end function fne_array_scalar_op

  pure function fequals_matrix_op(mat1, mat2) result(equals)
    real, dimension(:, :), intent(in) :: mat1
    real, dimension(size(mat1, 1), size(mat1, 2)), intent(in) :: mat2

    logical :: equals

    equals = fequals(mat1, mat2)

  end function fequals_matrix_op

  pure function fne_matrix_op(mat1, mat2) result (nequals)
    real, dimension(:, :), intent(in) :: mat1
    real, dimension(size(mat1, 1), size(mat1, 2)), intent(in) :: mat2

    logical :: nequals

    nequals = fnequals(mat1, mat2)

  end function fne_matrix_op

  pure function fequals_matrix_scalar_op(mat1, float2) result(equals)
    real, dimension(:, :), intent(in) :: mat1
    real, intent(in) :: float2

    logical :: equals

    equals = fequals(mat1, float2)

  end function fequals_matrix_scalar_op

  pure function fne_matrix_scalar_op(mat1, float2) result (nequals)
    real, dimension(:, :), intent(in) :: mat1
    real, intent(in) :: float2

    logical :: nequals

    nequals = fnequals(mat1, float2)

  end function fne_matrix_scalar_op

  pure function fequals_scalar(float1, float2, tol) result(equals)
    !!< This function checks if float1 == float2, to within tol (or
    !!< 100.0 * epsilon(0.0) if tol is not set).
    real, intent(in) :: float1
    real, intent(in) :: float2
    real, intent(in), optional :: tol

    logical :: equals

    real :: eps

    if(present(tol)) then
      eps = tol
    else
      eps = 100.0 * epsilon(0.0)
    end if

    equals = abs(float1 - float2) < max(eps, abs(float1) * eps)

  end function fequals_scalar

  pure function fne_scalar(float1, float2, tol) result(nequals)
    real, intent(in) :: float1
    real, intent(in) :: float2
    real, optional, intent(in) :: tol

    logical :: nequals

    nequals = .not. fequals(float1, float2, tol = tol)

  end function fne_scalar

  pure function fequals_array(array1, array2, tol) result (equals)
    real, dimension(:), intent(in) :: array1
    real, dimension(size(array1)), intent(in) :: array2
    real, intent(in), optional :: tol

    logical :: equals

    integer :: i

    equals = .true.
    do i=1,size(array1)
      if (.not. fequals(array1(i), array2(i), tol)) then
        equals = .false.
        return
      end if
    end do

  end function fequals_array

  pure function fne_array(array1, array2, tol) result(nequals)
    !!< floating point not equals
    real, dimension(:), intent(in) :: array1
    real, dimension(size(array1)), intent(in) :: array2
    real, intent(in), optional :: tol

    logical :: nequals

    nequals = .not. fequals(array1, array2, tol)

  end function fne_array

  pure function fequals_array_scalar(array1, float2, tol) result(equals)
    real, dimension(:), intent(in) :: array1
    real, intent(in) :: float2
    real, intent(in), optional :: tol

    logical :: equals

    integer :: i

    do i = 1, size(array1)
      if(.not. fequals(array1(i), float2, tol = tol)) then
        equals = .false.
        return
      end if
    end do

    equals = .true.

  end function fequals_array_scalar

  pure function fne_array_scalar(array1, float2, tol) result(nequals)
    real, dimension(:), intent(in) :: array1
    real, intent(in) :: float2
    real, intent(in), optional :: tol

    logical :: nequals

    nequals = .not. fequals(array1, float2, tol = tol)

  end function fne_array_scalar

  pure function fequals_matrix(mat1, mat2, tol) result(equals)
    real, dimension(:, :), intent(in) :: mat1
    real, dimension(size(mat1, 1), size(mat1, 2)), intent(in) :: mat2
    real, optional, intent(in) :: tol

    logical :: equals

    integer :: i

    do i = 1, size(mat1, 1)
      if(fnequals(mat1(i, :), mat2(i, :), tol = tol)) then
        equals = .false.
        return
      end if
    end do

    equals = .true.

  end function fequals_matrix

  pure function fne_matrix(mat1, mat2, tol) result (nequals)
    real, dimension(:, :), intent(in) :: mat1
    real, dimension(size(mat1, 1), size(mat1, 2)), intent(in) :: mat2
    real, optional, intent(in) :: tol

    logical :: nequals

    nequals = .not. fequals(mat1, mat2, tol = tol)

  end function fne_matrix

  pure function fequals_matrix_scalar(mat1, float2, tol) result(equals)
    real, dimension(:, :), intent(in) :: mat1
    real, intent(in) :: float2
    real, optional, intent(in) :: tol

    logical :: equals

    integer :: i

    do i = 1, size(mat1, 1)
      if(fnequals(mat1(i, :), float2, tol = tol)) then
        equals = .false.
        return
      end if
    end do

    equals = .true.

  end function fequals_matrix_scalar

  pure function fne_matrix_scalar(mat1, float2, tol) result (nequals)
    real, dimension(:, :), intent(in) :: mat1
    real, intent(in) :: float2
    real, optional, intent(in) :: tol

    logical :: nequals

    nequals = .not. fequals(mat1, float2, tol = tol)

  end function fne_matrix_scalar

  function dcsr_fequals(A, B, tol)
    !!< Checks if the dynamic matrices A and B are the same
    logical dcsr_fequals
    type(dynamic_csr_matrix), intent(in):: A, B
    real, intent(in), optional :: tol

    integer, dimension(:), pointer:: cols
    integer i, j

    if (size(A,1)/=size(B,1) .or. size(A,2)/=size(B,2)) then
      dcsr_fequals=.false.
      return
    end if

    do i=1, size(A,1)
      cols => row_m_ptr(A, i)
      if (size(cols)/=row_length(B,i)) then
        dcsr_fequals=.false.
        return
      end if

      do j=1, size(cols)
        if (.not. &
          fequals( val(A, i, cols(j)), val(B, i, cols(j)) , tol) &
          ) then
          dcsr_fequals=.false.
          return
        end if
      end do
    end do

    dcsr_fequals=.true.

  end function dcsr_fequals

  function flt_op(float1, float2) result(less_than)
    real, intent(in) :: float1, float2
    logical :: less_than

    less_than = flt(float1, float2)
  end function flt_op

  function fgt_op(float1, float2) result(greater_than)
    real, intent(in) :: float1, float2
    logical :: greater_than

    greater_than = fgt(float1, float2)
  end function fgt_op

  function flt(float1, float2, tol) result(less_than)
    !!<  This function checks if float1 < float2, to within tol (or
    !!< 100.0 * epsilon(0.0) if tol is not set).
    real, intent(in) :: float1, float2
    real :: eps
    real, intent(in), optional :: tol
    logical :: less_than

    if (present(tol)) then
      eps = tol
    else
      eps = 100.0 * epsilon(0.0)
    end if

    less_than = .false.
    if (float1 < float2) then
      if (abs(float1 - float2) > eps) less_than = .true.
    end if

  end function flt

  function fgt(float1, float2, tol) result(greater_than)
    !!<  This function checks if float1 > float2, to within tol (or
    !!< 100.0 * epsilon(0.0) if tol is not set).
    real, intent(in) :: float1, float2
    real :: eps
    real, intent(in), optional :: tol
    logical :: greater_than

    if (present(tol)) then
      eps = tol
    else
      eps = 100.0 * epsilon(0.0)
    end if

    greater_than = .false.
    if (float1 > float2) then
      if (abs(float1 - float2) > eps) greater_than = .true.
    end if

  end function fgt

  function is_nan_scalar(float1) result(nan)
    !!< This function checks if float1 is NaN.

    real, intent(in) :: float1

    logical :: nan

    nan = ieee_is_nan(float1)

  end function is_nan_scalar

  function is_nan_vector(float1) result(nan)
    !!< This function checks if float1 is NaN.

    real, dimension(:), intent(in) :: float1

    logical, dimension(size(float1)) :: nan

    integer :: i

    do i = 1, size(float1)
      nan(i) = is_nan(float1(i))
    end do

  end function is_nan_vector

  function is_nan_tensor2(float1) result(nan)
    !!< This function checks if float1 is NaN.

    real, dimension(:, :), intent(in) :: float1

    logical, dimension(size(float1, 1), size(float1, 2)) :: nan

    integer :: i

    do i = 1, size(float1, 1)
      nan(i, :) = is_nan(float1(i, :))
    end do

  end function is_nan_tensor2

  function is_nan_tensor3(float1) result(nan)
    !!< This function checks if float1 is NaN.

    real, dimension(:, :, :), intent(in) :: float1

    logical, dimension(size(float1, 1), size(float1, 2), size(float1, 3)) :: nan

    integer :: i

    do i = 1, size(float1, 1)
      nan(i, :, :) = is_nan(float1(i, :, :))
    end do

  end function is_nan_tensor3

  function mat_is_symmetric(mat) result(symmetric)
    !!< This function checks if mat is a symmetric matrix.
    real, dimension(:, :), intent(in) :: mat
    logical :: symmetric
    real, dimension(size(mat, 1), size(mat, 1)) :: tmp

    symmetric = .false.
!    do i=1,size(mat,1)
!      do j=1,size(mat,2)
!        if (.not. fequals(mat(i, j), mat(j, i))) then
!          symmetric = .false.
!        end if
!      end do
!    end do

    if (.not. symmetric) then
      tmp = mat - transpose(mat)
      if (mat_zero(tmp,1e-3)) symmetric = .true.
    end if
  end function mat_is_symmetric

  function mat_zero(mat, tol) result(zero)
    !!< This function checks if mat is zero.
    !!< It does this by computing the Frobenius norm of the matrix.

    real, dimension(:, :), intent(in) :: mat
    real, optional :: tol
    real :: ltol
    logical :: zero
    real :: frobenius
    integer :: i, j

    if (present(tol)) then
      ltol = tol
    else
      ltol = 100.0 * epsilon(0.0)
    end if

    zero = .false.
    frobenius = 0.0

    do i=1,size(mat,1)
      do j=1,size(mat,2)
        frobenius = frobenius + mat(i, j) * mat(i, j)
      end  do
    end do

    frobenius = sqrt(frobenius)

    if (frobenius .lt. ltol) zero = .true.
  end function mat_zero

  function mat_diag(mat) result(diag)
    !!< This function checks if the matrix is diagonal; that is,
    !!< all non-diagonal entries are zero. (For my purposes the zero
    !!< matrix is diagonal, for example.)

    real, dimension(:, :), intent(in) :: mat
    logical :: diag
    integer :: i, j

    diag = .true.

    do i=1,size(mat, 1)
      do j=1,size(mat, 2)
        if (i == j) cycle
        if (.not. fequals(mat(i, j), 0.0)) then
          diag = .false.
          exit
        end if
      end do
    end do

  end function mat_diag

  function random_vector(dim) result(vec)
    !!< This function generates a random vector of dimension dim.

    integer, intent(in) :: dim
    real, dimension(dim) :: vec
    real :: rand
    integer :: i

    do i=1,dim
      call random_number(rand)
      vec(i) = rand
    end do
  end function random_vector

  function random_matrix(dim) result(mat)
    !!< This function generates a random matrix of dimension dim.

    integer, intent(in) :: dim
    real, dimension(dim, dim) :: mat
    real :: rand
    integer :: i, j

    do i=1,dim
      do j=1,dim
        call random_number(rand)
        mat(i, j) = rand
      end do
    end do
  end function random_matrix

  function random_symmetric_matrix(dim) result(mat)
    !!< This function generates a random symmetric matrix of dimension dim.

    integer, intent(in) :: dim
    real, dimension(dim, dim) :: mat
    real :: rand
    integer :: i, j

    do i=1,dim
      call random_number(rand)
      mat(i, i) = rand
    end do

    do i=1,dim
      do j=i+1,dim
        call random_number(rand)
        mat(i, j) = rand; mat(j, i) = rand
      end do
    end do

  end function random_symmetric_matrix

  function random_posdef_matrix(dim) result(mat)
    !!< This function generates a random symmetric positive definite matrix of dimension dim.

    integer, intent(in) :: dim
    real, dimension(dim, dim) :: mat, evecs
    real, dimension(dim) :: evals
    integer :: i

    mat = random_symmetric_matrix(dim)
    call eigendecomposition_symmetric(mat, evecs, evals)
    do i=1,dim
      evals(i) = max(0.1, abs(evals(i)))
    end do
    call eigenrecomposition(mat, evecs, evals)

    end function random_posdef_matrix

    function random_sparse_matrix(rows, cols, nnz) result(mat)
      !!< This function returns a random rows x cols sparse_matrix
      !!<  with at most nnz entries.
    type(dynamic_csr_matrix) mat
    integer, intent(in):: rows, cols, nnz

      integer i, row, col
      real val, rand

      call allocate(mat, rows, cols)
      do i=1, nnz
        call random_number(rand)
        row=floor(rand*rows)+1
        call random_number(rand)
        col=floor(rand*cols)+1
        call random_number(rand)
        val=(rand-0.5)*1e10
        call set(mat, row, col, val)
      end do

    end function random_sparse_matrix

    function get_mat_diag(vec) result(mat)
      !!< This function returns the matrix whose diagonal is vec.

      real, dimension(:), intent(in) :: vec
      real, dimension(size(vec), size(vec)) :: mat
      integer :: i

      mat = 0.0
      do i=1,size(vec)
        mat(i, i) = vec(i)
      end do
    end function get_mat_diag

    function get_matrix_identity(dim) result(id)
      !!< Return the identity matrix
      integer, intent(in) :: dim
      real, dimension(dim, dim) :: id
      integer :: i

      id = 0.0
      do i=1,dim
        id(i, i) = 1.0
      end do
    end function

  subroutine mat_clean(mat, tol)
    !!< This subroutine goes through the matrix mat
    !!< and replaces any values less than tol with 0.0.
    !!< The spurious numerical errors (on the order of 1e-20)
    !!< can cause problems with the eigenvalue decomposition.
    real, dimension(:, :), intent(inout) :: mat
    real, intent(in) :: tol
    integer :: i, j

    do i=1,size(mat, 1)
      do j=1,size(mat, 2)
        if (abs(mat(i, j)) .lt. tol) mat(i, j) = 0.0
      end do
    end do
  end subroutine mat_clean

  subroutine vec_clean(vec, tol)
    !!< This routine does the same as mat_clean, but for a vector instead.
    real, dimension(:), intent(inout) :: vec
    real, intent(in) :: tol
    integer :: i

    do i=1,size(vec)
      if (abs(vec(i)) .lt. tol) vec(i) = 0.0
    end do
  end subroutine vec_clean

  subroutine write_matrix(mat, namestr)
    real, dimension(:, :), intent(in) :: mat
    character(len=*), intent(in) :: namestr

    integer :: i

    write(0,'(a)') namestr // ":"
    do i=1,size(mat,1)
      write(0,'(3e28.20)') mat(i, :)
    end do
  end subroutine write_matrix

  subroutine write_vector_real(rvec, namestr)
    real, dimension(:), intent(in) :: rvec
    character(len=*), intent(in) :: namestr

    write(0, '(a)') namestr // ":"
    write(0,'(3e28.20)') rvec
  end subroutine write_vector_real

  subroutine write_vector_integer(ivec, namestr)
    integer, dimension(:), intent(in) :: ivec
    character(len=*), intent(in) :: namestr

    write(0, '(a)') namestr // ":"
    write(0,'(i14)') ivec
  end subroutine write_vector_integer

end module unittest_tools
