#include "fdebug.h"
subroutine test_petsc_csr_matrix()
#include "petscversion.h"
  use sparse_tools
  use sparse_tools_petsc
  use parallel_tools
  use petsc_tools
  use unittest_tools
  implicit none
#include "finclude/petsc.h"


#if PETSC_VERSION_MINOR>=2
  interface
     subroutine error_handler(comm,line, func, file, dir, n, p, mess, ctx, ierr)
       MPI_Comm:: comm
       PetscInt:: line
       character(len=*):: func, file, dir
       PetscErrorCode:: n
       PetscInt:: p
       character(len=*):: mess
       PetscInt:: ctx
       PetscErrorCode:: ierr
     end subroutine error_handler
  end interface
#else
  interface
     subroutine error_handler(line, func, file, dir, n, p, mess, ctx, ierr)
       PetscInt:: line
       character(len=*):: func, file, dir
       PetscErrorCode:: n
       PetscInt:: p
       character(len=*):: mess
       PetscInt:: ctx
       PetscErrorCode:: ierr

     end subroutine error_handler
  end interface
#endif

  type(petsc_csr_matrix):: A
  type(csr_matrix):: B
  PetscErrorCode:: ierr
  PetscInt:: ctx
  real, dimension(4,4):: vals
  logical:: fail
  integer:: i, j

  ! ---- first check assembly using petsc_csr_vaddto ---

  call allocate(A, 4, 4, &
    dnnz=(/ 4, 4, 4, 4 /), &
    onnz=(/ 0, 0, 0, 0 /), &
    blocks=(/ 1, 1 /), &
    name="TestMatrix")

  call zero(A)

  ! make a trivial 4x4 matrix:
  !  1.  2.  3.  4.
  !  5.  6.  7.  8.
  !  9. 10. 11. 12.
  ! 13. 14. 15. 16.

  vals=transpose(reshape((/ ( real(i), i=1, 16 ) /), (/ 4, 4 /)))

  call addto(A, 1, 1, (/ 1, 2, 3, 4 /), (/ 1, 2, 3, 4/), &
     vals)

  ! assemble and copy into csr_matrix
  call assemble(A)
  B=petsc2csr(A%M)

  ! then check B has the righ values
  fail = any(abs(B%val-(/ ( real(i), i=1, 16 ) /))>1e-12)
  call report_test("[petsc_csr_matrix]", fail, .false., "Correct values in matrix.")

  ! and column indices
  fail = any(B%sparsity%colm/=(/ ( ( i, i=1, 4 ), j=1,4 ) /))
  call report_test("[petsc_csr_matrix]", fail, .false., "Correct column indices.")

  call deallocate(B)

  ! now add the same thing again
  call addto(A, 1, 1, (/ 1, 2, 3, 4 /), (/ 1, 2, 3, 4/), &
    vals )

  ! reassemble and copy into csr_matrix
  call assemble(A)
  B=petsc2csr(A%M)

  ! and check its new values
  fail = any(abs(B%val-2.0*(/ ( real(i), i=1, 16 ) /))>1e-12)
  call report_test("[petsc_csr_matrix]", fail, .false., "Correct values in matrix.")

  ! column indices should remain unchanged
  fail = any(B%sparsity%colm/=(/ ( ( i, i=1, 4 ), j=1,4 ) /))
  call report_test("[petsc_csr_matrix]", fail, .false., "Correct values in matrix.")

  call deallocate(B)
  call deallocate(A)

  ! ---- now check for a fail if we under estimate nnz ---

  ! set error handler to catch the petsc error
  call PetscPushErrorHandler(error_handler, ctx, ierr)

  call allocate(A, 4, 4, &
    dnnz=(/ 1, 1, 1, 1 /), &
    onnz=(/ 0, 0, 0, 0 /), &
    blocks=(/ 1, 1 /), &
    name="TestMatrix")

  call zero(A)

  ! ctx should be set to 1 in the error handler
  ctx=0

  ! addition that over runs preallocated memory
  call addto(A, 1, 1, (/ 1 /), (/ 1, 2 /), &
    reshape( (/ 2.0, 3.0 /), (/ 1, 2 /)) )

  fail= (ctx/=1)
  call report_test("[petsc_csr_matrix]", fail, .false., "PETSc should give an error when overrunning nnz.")

  call deallocate(A)

end subroutine test_petsc_csr_matrix

#if PETSC_VERSION_MINOR>=2
subroutine error_handler(comm,line, func, file, dir, n, p, mess, ctx, ierr)
#include "finclude/petsc.h"
  MPI_Comm:: comm
  PetscInt:: line
  character(len=*):: func, file, dir
  PetscErrorCode:: n
  PetscInt:: p
  character(len=*):: mess
  PetscInt:: ctx
  PetscErrorCode:: ierr


  ctx=1

end subroutine error_handler
#else
    subroutine error_handler(line, func, file, dir, n, p, mess, ctx, ierr)
#include "finclude/petsc.h"
      PetscInt:: line
      character(len=*):: func, file, dir
      PetscErrorCode:: n
      PetscInt:: p
      character(len=*):: mess
      PetscInt:: ctx
      PetscErrorCode:: ierr


      ctx=1

    end subroutine error_handler

#endif
