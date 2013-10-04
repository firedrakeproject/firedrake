#include "fdebug.h"

module vector_tools
  !!< This module contains dense matrix operations such as interfaces to
  !!< LAPACK.

  use fldebug

  implicit none

  interface blasmul
          module procedure blasmul_mm, blasmul_mv
  end interface blasmul

  interface solve
    module procedure solve_single, solve_multiple
  end interface

  interface invert
    module procedure invert_matrix
  end interface invert

!  interface operator(.cross.)
!    module procedure cross_product
!  end interface
!
!  interface operator(.dot.)
!    module procedure dot_product_op
!  end interface

  interface norm2
     module procedure norm2_vector, norm2_tensor
  end interface

  interface cross_product
     module procedure cross_product_array
  end interface

  interface outer_product
     module procedure outer_product
  end interface

  private
  public blasmul, solve, norm2, cross_product, invert, inverse, cholesky_factor, &
     mat_diag_mat, eigendecomposition, eigendecomposition_symmetric, eigenrecomposition, &
     outer_product, det, det_2, det_3, scalar_triple_product, svd, cross_product2

contains

  pure function dot_product_op(vector1, vector2) result(dot)
    !!< Dot product. Need to wrap dot_product to make .dot. operator.
    real, dimension(:), intent(in) :: vector1, vector2
    real :: dot

    dot = dot_product(vector1, vector2)
  end function dot_product_op

  pure function norm2_vector(vector)
    !!< Calculate the 2-norm of vector
    real :: norm2_vector
    real, dimension(:), intent(in) :: vector

    norm2_vector=sqrt(dot_product(vector, vector))

  end function norm2_vector

  pure function norm2_tensor(tensor)
    !!< Calculate the 2-norm of tensor
    real :: norm2_tensor
    real, dimension(:,:), intent(in) :: tensor

    norm2_tensor=sqrt(sum(tensor(:,:)*tensor(:,:)))

  end function norm2_tensor

  pure function cross_product_array(vector1, vector2) result(prod)
    !!< Calculate the cross product of the vectors provided.
    real, dimension(3) :: prod
    real, dimension(3), intent(in) :: vector1, vector2

    prod(1)=vector1(2)*vector2(3) - vector1(3)*vector2(2)
    prod(2)=vector1(3)*vector2(1) - vector1(1)*vector2(3)
    prod(3)=vector1(1)*vector2(2) - vector1(2)*vector2(1)

  end function cross_product_array

  pure function cross_product2(vector1, vector2) result(prod)
    !!< 2-dimensional cross-product analog
    real :: prod
    real, dimension(2), intent(in) :: vector1, vector2

    prod=vector1(1)*vector2(2) - vector1(2)*vector2(1)

  end function cross_product2

  pure function scalar_triple_product(vector1, vector2, vector3) result (prod)
    ! returns a scalar triple product
    real, dimension(3), intent(in) :: vector1, vector2, vector3
    real :: prod

    prod=vector1(1)*(vector2(2)*vector3(3) - vector2(3)*vector3(2)) + &
         vector1(2)*(vector2(3)*vector3(1) - vector2(1)*vector3(3)) + &
         vector1(3)*(vector2(1)*vector3(2) - vector2(2)*vector3(1))

  end function scalar_triple_product

  subroutine solve_single(A, b, info)
    !!< Solve Ax=b for one right hand side b, putting the result in b.
    real, dimension(:, :), intent(in) :: A
    real, dimension(:), intent(inout) :: b
    integer, optional, intent(out) :: info

    real, dimension(size(b), 1) :: b_tmp

    b_tmp(:, 1) = b
    call solve_multiple(A, b_tmp, info)
    b = b_tmp(:, 1)
  end subroutine solve_single

  subroutine solve_multiple(A,B, stat)
    !!< Solve Ax=b for multiple right hand sides B putting the result in B.
    !!<
    !!< This is simply a wrapper for lapack.
    real, dimension(:,:), intent(in) :: A
    real, dimension(:,:), intent(inout) :: B
    integer, optional, intent(out) :: stat

    real, dimension(size(A,1), size(A,2)) :: Atmp
    integer, dimension(size(A,1)) :: ipiv
    integer :: info

    interface
#ifdef DOUBLEP
       SUBROUTINE DGESV( N, NRHS, A, LDA, IPIV, B, LDB, INFO )
         INTEGER :: INFO, LDA, LDB, N, NRHS
         INTEGER :: IPIV( * )
         REAL ::  A( LDA, * ), B( LDB, * )
       END SUBROUTINE DGESV
#else
       SUBROUTINE SGESV( N, NRHS, A, LDA, IPIV, B, LDB, INFO )
         INTEGER :: INFO, LDA, LDB, N, NRHS
         INTEGER :: IPIV( * )
         REAL ::  A( LDA, * ), B( LDB, * )
       END SUBROUTINE SGESV
#endif
    end interface

    if (present(stat)) stat = 0

    ASSERT(size(A,1)==size(A,2))
    ASSERT(size(A,1)==size(B,1))

    Atmp=A

#ifdef DOUBLEP
    call dgesv(size(A,1), size(B,2), Atmp, size(A,1), ipiv, B, size(B,1),&
         & info)
#else
    call sgesv(size(A,1), size(B,2), Atmp, size(A,1), ipiv, B, size(B,1),&
         & info)
#endif

    if (.not. present(stat)) then
      ASSERT(info==0)
    else
      stat = info
    end if

  end subroutine solve_multiple

  subroutine invert_matrix(A, stat)
    !!< Replace the matrix A with its inverse.
    real, dimension(:,:), intent(inout) :: A
    real, dimension(size(A,1),size(A,2)) :: rhs
    integer, intent(out), optional :: stat

    real, dimension(3,3):: a33
    real det, tmp
    integer i

    assert(size(A,1)==size(A,2))

    if(present(stat)) stat=0

    select case (size(A,1))
    case (3) ! I put this one first in the hope the compiler keeps it there
      det=A(1,1)*(A(2,2)*A(3,3)-A(3,2)*A(2,3)) &
        -A(2,1)*(A(1,2)*A(3,3)-A(3,2)*A(1,3)) &
        +A(3,1)*(A(1,2)*A(2,3)-A(2,2)*A(1,3))

      a33(1,1)=A(2,2)*A(3,3)-A(3,2)*A(2,3)
      a33(1,2)=A(3,2)*A(1,3)-A(1,2)*A(3,3)
      a33(1,3)=A(1,2)*A(2,3)-A(2,2)*A(1,3)

      a33(2,1)=A(2,3)*A(3,1)-A(3,3)*A(2,1)
      a33(2,2)=A(3,3)*A(1,1)-A(1,3)*A(3,1)
      a33(2,3)=A(1,3)*A(2,1)-A(2,3)*A(1,1)

      a33(3,1)=A(2,1)*A(3,2)-A(3,1)*A(2,2)
      a33(3,2)=A(3,1)*A(1,2)-A(1,1)*A(3,2)
      a33(3,3)=A(1,1)*A(2,2)-A(2,1)*A(1,2)

      A=a33/det

    case (2)
      det=A(1,1)*A(2,2)-A(1,2)*A(2,1)
      tmp=A(1,1)
      A(1,1)=A(2,2)
      A(2,2)=tmp
      A(1,2)=-A(1,2)
      A(2,1)=-A(2,1)
      A=A/det

    case (1)
      A(1,1)=1.0/A(1,1)

    case default ! otherwise use LAPACK
      rhs=0.0

      forall(i=1:size(A,1))
        rhs(i,i)=1.0
      end forall

      call solve(A, rhs, stat)

      A=rhs
    end select

  end subroutine invert_matrix

  function inverse(A)
    !!< Function version of invert.
    real, dimension(:,:), intent(in) :: A
    real, dimension(size(A,1),size(A,2)) :: inverse

    real det
    integer i

    assert(size(A,1)==size(A,2))

    select case (size(A,1))
    case (3) ! I put this one first in the hope the compiler keeps it there
      det=A(1,1)*(A(2,2)*A(3,3)-A(3,2)*A(2,3)) &
        -A(2,1)*(A(1,2)*A(3,3)-A(3,2)*A(1,3)) &
        +A(3,1)*(A(1,2)*A(2,3)-A(2,2)*A(1,3))

      inverse(1,1)=A(2,2)*A(3,3)-A(3,2)*A(2,3)
      inverse(1,2)=A(3,2)*A(1,3)-A(1,2)*A(3,3)
      inverse(1,3)=A(1,2)*A(2,3)-A(2,2)*A(1,3)

      inverse(2,1)=A(2,3)*A(3,1)-A(3,3)*A(2,1)
      inverse(2,2)=A(3,3)*A(1,1)-A(1,3)*A(3,1)
      inverse(2,3)=A(1,3)*A(2,1)-A(2,3)*A(1,1)

      inverse(3,1)=A(2,1)*A(3,2)-A(3,1)*A(2,2)
      inverse(3,2)=A(3,1)*A(1,2)-A(1,1)*A(3,2)
      inverse(3,3)=A(1,1)*A(2,2)-A(2,1)*A(1,2)

      inverse=inverse/det

    case (2)
      det=A(1,1)*A(2,2)-A(1,2)*A(2,1)
      inverse(1,1)=A(2,2)
      inverse(2,2)=A(1,1)
      inverse(1,2)=-A(1,2)
      inverse(2,1)=-A(2,1)
      inverse=inverse/det

    case (1)
      inverse(1,1)=1.0/A(1,1)

    case default
      inverse=0.0

      forall(i=1:size(A,1))
        inverse(i,i)=1.0
      end forall

      call solve(A, inverse)

    end select

  end function inverse

  subroutine cholesky_factor(A)
    !!< Replace the matrix A with an Upper triangular factor such that
    !!< U^TU=A
    !!<
    !!< This is simply a wrapper for lapack.
    real, dimension(:,:), intent(inout) :: A

    integer :: info

    integer :: i,j

    interface
#ifdef DOUBLEP
       SUBROUTINE DPOTRF( UPLO, N, A, LDA, INFO )
         CHARACTER(len=1) :: UPLO
         INTEGER :: INFO, LDA, N
         REAL ::  A( LDA, * )
       END SUBROUTINE DPOTRF
#else
       SUBROUTINE SPOTRF( UPLO, N, A, LDA, INFO )
         CHARACTER(len=1) :: UPLO
         INTEGER :: INFO, LDA, N
         REAL ::  A( LDA, * )
       END SUBROUTINE SPOTRF
#endif
    end interface

    ASSERT(size(A,1)==size(A,2))

    ! Zero lower triangular entries.
    forall(i=1:size(A,1),j=1:size(a,2),j<i)
       A(i,j)=0.0
    end forall

#ifdef DOUBLEP
    call dpotrf('U', size(A,1), A, size(A,1), info)
#else
    call spotrf('U', size(A,1), A, size(A,1), info)
#endif

    ASSERT(info==0)

  end subroutine cholesky_factor

  function Mat_Diag_Mat(matrix, diag)
    !!< Construct matrix^T * diag(diag) * matrix
    real, dimension(:,:), intent(in) :: matrix
    real, dimension(size(matrix,1)) :: diag

    real, dimension(size(matrix,1), size(matrix,2)) :: mat_diag_mat

    mat_diag_mat=&
         matmul(transpose(matrix)*spread(diag, 2, size(matrix,2)),matrix)

  end function Mat_Diag_Mat

  subroutine eigendecomposition(M, V, A)
    !!<  M == matrix to decompose
    !!<  V == matrix whose columns are normalised eigenvectors
    !!<  A == vector of eigenvalues. Assumed to be real, as they will always be for what I want
    !!<  this routine.

    real, dimension(:, :), intent(in) :: M
    real, dimension(:, :), intent(out) :: V
    real, dimension(:), intent(out) :: A

    integer :: dim, lwork
    real, dimension(size(A), size(A)) :: M_temp
    real, dimension(:), allocatable :: work, A_temp
    real :: rdummy(1,1)
    integer :: info

    interface
#ifdef DOUBLEP
      SUBROUTINE DGEEV( JOBVL, JOBVR, N, A, LDA, WR, WI, VL, LDVL, VR, &
                        LDVR, WORK, LWORK, INFO )
      CHARACTER          JOBVL, JOBVR
      INTEGER            INFO, LDA, LDVL, LDVR, LWORK, N
      REAL               A ( LDA, * ), VL( LDVL, * ), VR( LDVR, * ), &
                         WI( * ), WORK( * ), WR( * )
      END SUBROUTINE DGEEV
#else
      SUBROUTINE SGEEV( JOBVL, JOBVR, N, A, LDA, WR, WI, VL, LDVL, VR, &
                        LDVR, WORK, LWORK, INFO )
      CHARACTER          JOBVL, JOBVR
      INTEGER            INFO, LDA, LDVL, LDVR, LWORK, N
      REAL               A( LDA, * ), VL( LDVL, * ), VR( LDVR, * ), &
                         WI( * ), WORK( * ), WR( * )
      END SUBROUTINE SGEEV
#endif
    end interface

    assert(size(A) .eq. size(M, 1))
    assert(size(A) .eq. size(V, 2))

    dim = size(A)

    lwork =  50 * dim
    allocate(A_temp(dim))
    allocate(work(lwork))

    M_temp = M

#ifdef DOUBLEP
    call DGEEV('N', 'V', dim, M_temp, dim, A, A_temp, rdummy, 1, V, &
               dim, work, lwork, info)
#else
    call SGEEV('N', 'V', dim, M_temp, dim, A, A_temp, rdummy, 1, V, &
               dim, work, lwork, info)
#endif

    assert(info == 0)

    deallocate(work, A_temp)
  end subroutine eigendecomposition

  subroutine eigendecomposition_symmetric(M, V, A, stat)
    !!<  M == matrix to decompose
    !!<  V == matrix whose columns are normalised eigenvectors
    !!<  A == vector of eigenvalues. Assumed to be real, as they will always be for what I want
    !!<  this routine.

    real, dimension(:, :), intent(in) :: M
    real, dimension(:, :), intent(out) :: V
    real, dimension(:), intent(out) :: A
    integer, optional, intent(out) :: stat

    integer :: info, i, j, dim
    real, dimension(3 * size(M,1)) :: work
    real, dimension(size(M,1) * (size(M, 1)+1)/2) :: AP

    interface
#ifdef DOUBLEP
      SUBROUTINE DSPEV( JOBZ, UPLO, N, AP, W, Z, LDZ, WORK, INFO )
#else
      SUBROUTINE SSPEV( JOBZ, UPLO, N, AP, W, Z, LDZ, WORK, INFO )
#endif
        CHARACTER          JOBZ, UPLO
        INTEGER            INFO, LDZ, N
        REAL               AP(N*(N+1)/2), W(N), WORK(3*N), Z(LDZ, N)
      END SUBROUTINE
    end interface

    if(present(stat)) stat = 0

    dim = size(M, 1)
    do j=1,dim
      do i=1,j
        AP(i + (j-1)*j/2) = M(i,j)
      end do
    end do

#ifdef DOUBLEP
    call DSPEV('V', 'U', size(M, 1), AP, A, V, size(V, 1), work, info)
#else
    call SSPEV('V', 'U', size(M, 1), AP, A, V, size(V, 1), work, info)
#endif

    if(info /= 0) then
      if(present(stat)) then
        stat = info
      else
        ewrite(-1, *) "For matrix: ", transpose(M)
        FLAbort("eigendecomposition_symmetric failed")
      end if
    end if

  end subroutine eigendecomposition_symmetric

  subroutine eigenrecomposition(M, V, A)
    !!< Recompose the matrix M from its eigendecomposition.
    real, dimension(:, :), intent(out) :: M
    real, dimension(:, :), intent(in) :: V
    real, dimension(:), intent(in) :: A
    integer :: i

    do i=1,size(A)
      M(:, i) = A(i) * V(:, i)
    end do

    M = matmul(M, transpose(V))
  end subroutine eigenrecomposition

  pure function outer_product(x, y)
    !!< Give two column vectors x, y
    !!< compute the matrix xy*

    real, dimension(:), intent(in) :: x, y
    real, dimension(size(x), size(y)) :: outer_product
    integer :: i, j

    forall (i=1:size(x))
      forall (j=1:size(y))
        outer_product(i, j) = x(i) * y(j)
      end forall
    end forall
  end function outer_product

  function blasmul_mm(A, B) result(C)
    !!< Use DGEMM to multiply A * B and get C.

    real, dimension(:, :), intent(in) :: A, B
    real, dimension(size(A, 1), size(B, 2)) :: C

    interface
#ifdef DOUBLEP
      SUBROUTINE DGEMM ( TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, &
                         BETA, C, LDC )
#else
      SUBROUTINE SGEMM ( TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, &
                         BETA, C, LDC )
#endif
      CHARACTER*1        TRANSA, TRANSB
      INTEGER            M, N, K, LDA, LDB, LDC
      REAL               ALPHA, BETA
      REAL               A( LDA, * ), B( LDB, * ), C( LDC, * )
#ifdef DOUBLEP
      END SUBROUTINE DGEMM
#else
      END SUBROUTINE SGEMM
#endif
    end interface

    assert(size(A, 2) == size(B, 1))

#ifdef DOUBLEP
    call DGEMM('N', 'N', size(A, 1), size(B, 2), size(A, 2), 1.0, A, size(A, 1), B, size(B, 1), &
               0.0, C, size(A, 1))
#else
    call SGEMM('N', 'N', size(A, 1), size(B, 2), size(A, 2), 1.0, A, size(A, 1), B, size(B, 1), &
               0.0, C, size(A, 1))
#endif
  end function blasmul_mm

  function blasmul_mv(A, x) result(b)
    !!< Use DGEMV to multiply A*x to get b.

    real, dimension(:, :), intent(in) :: A
    real, dimension(size(A, 2)), intent(in) :: x
    real, dimension(size(A, 1)) :: b

    interface
#ifdef DOUBLEP
      SUBROUTINE DGEMV ( TRANS, M, N, ALPHA, A, LDA, X, INCX, &
                         BETA, Y, INCY )
#else
      SUBROUTINE SGEMV ( TRANS, M, N, ALPHA, A, LDA, X, INCX, &
                               BETA, Y, INCY )
#endif
      REAL               ALPHA, BETA
      INTEGER            INCX, INCY, LDA, M, N
      CHARACTER*1        TRANS
      REAL               A( LDA, * ), X( * ), Y( * )
#ifdef DOUBLEP
      END SUBROUTINE DGEMV
#else
      END SUBROUTINE SGEMV
#endif
    end interface

#ifdef DOUBLEP
    call DGEMV('N', size(A, 1), size(A, 2), 1.0, A, size(A, 1), x, 1, &
               0.0, b, 1)
#else
    call SGEMV('N', size(A, 1), size(A, 2), 1.0, A, size(A, 1), x, 1, &
                   0.0, b, 1)
#endif

  end function blasmul_mv

  function det_2(mat_2) result(det)
    !!< Determinant of 2x2
    real, dimension(2,2), intent(in) :: mat_2
    real :: det

    det = (mat_2(1,1) * mat_2(2, 2)) - (mat_2(1, 2) * mat_2(2, 1))
  end function det_2

  function det_3(mat_3) result(det)
    !!< Determinant of 3x3
    real, dimension(:, :), intent(in) :: mat_3
    real :: det

    det =   mat_3(1,1) * (mat_3(2,2) * mat_3(3,3) - mat_3(2,3) * mat_3(3,2)) &
          - mat_3(1,2) * (mat_3(2,1) * mat_3(3,3) - mat_3(2,3) * mat_3(3,1)) &
          + mat_3(1,3) * (mat_3(2,1) * mat_3(3,2) - mat_3(2,2) * mat_3(3,1))
  end function det_3

  function det(mat) result(det_out)
    real, dimension(:, :), intent(in) :: mat
    real :: det_out

    det_out = huge(0.0)
    select case (size(mat,1))
    case(1)
       det_out = mat(1,1)
    case(2)
       det_out = det_2(mat)
    case(3)
       det_out = det_3(mat)
    case default
       FLAbort("Determinant not implemented for this dimension")
    end select

  end function det

  subroutine svd(input, U, sigma, VT)
    real, dimension(:, :), intent(in) :: input
    real, dimension(size(input, 1), size(input, 1)), intent(out) :: U
    real, dimension(min(size(input, 1), size(input, 2))), intent(out) :: sigma
    real, dimension(size(input, 2), size(input, 2)), intent(out) :: VT

    real, dimension(size(input, 1), size(input, 2)) :: tmp_input
    real, dimension(:), allocatable :: WORK
    integer :: LWORK, M, N, info

    interface
#ifdef DOUBLEP
      SUBROUTINE DGESVD ( JOBU, JOBVT, M, N, A, LDA, S, U, LDU, VT, LDVT, &
                          WORK, LWORK, INFO )
#else
      SUBROUTINE SGESVD ( JOBU, JOBVT, M, N, A, LDA, S, U, LDU, VT, LDVT, &
                          WORK, LWORK, INFO )
#endif
      CHARACTER          JOBU, JOBVT
      INTEGER            INFO, LDA, LDU, LDVT, LWORK, M, N
      real               A( LDA, * ), S( * ), U( LDU, * ), VT( LDVT, * ), WORK( * )
#ifdef DOUBLEP
      END SUBROUTINE DGESVD
#else
      END SUBROUTINE SGESVD
#endif
   end interface

   tmp_input = input
   M = size(input, 1)
   N = size(input, 2)
   LWORK = 2*MAX(1,3*MIN(M,N)+MAX(M,N),5*MIN(M,N))
   allocate(WORK(LWORK))

#ifdef DOUBLEP
   call DGESVD( &
#else
   call SGESVD( &
#endif
   'A', 'A', M, N, tmp_input, M, &
   sigma, U, M, VT, N, WORK, LWORK, info)

   assert(info == 0)
   deallocate(WORK)
  end subroutine svd

end module vector_tools
