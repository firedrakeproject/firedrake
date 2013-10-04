#include "fdebug.h"
subroutine test_multigrid
  ! Testing of the "mg" solver using the petsc_solve_setup, petsc_solve_core
  ! and petsc_solve_destroy components of petsc_solve.
  ! This test, tests whether 2 "mg" preconditioner can be set up
  ! simultaneously.
  use global_parameters
  use sparse_tools
  use petsc_tools
  use unittest_tools
  use fldebug
  use solvers
  use fields
  use parallel_tools
#include "petscversion.h"
#ifdef HAVE_PETSC
#ifdef HAVE_PETSC_MODULES
  use petsc
#if PETSC_VERSION_MINOR==0
  use petscvec
  use petscmat
  use petscksp
  use petscpc
  use petscis
  use petscmg
#endif
#endif
#endif
  implicit none
#ifdef HAVE_PETSC
#include "petscversion.h"
#ifdef HAVE_PETSC_MODULES
#if PETSC_VERSION_MINOR==0
#include "finclude/petscvecdef.h"
#include "finclude/petscmatdef.h"
#include "finclude/petsckspdef.h"
#include "finclude/petscpcdef.h"
#include "finclude/petscviewerdef.h"
#include "finclude/petscisdef.h"
#else
#include "finclude/petscdef.h"
#endif
#else
#include "finclude/petsc.h"
#if PETSC_VERSION_MINOR==0
#include "finclude/petscvec.h"
#include "finclude/petscmat.h"
#include "finclude/petscksp.h"
#include "finclude/petscpc.h"
#include "finclude/petscviewer.h"
#include "finclude/petscis.h"
#endif
#endif
#endif

  integer, parameter:: DIM=100, NNZ=1000
  logical fail

#ifdef HAVE_PETSC

  KSP ksp1, ksp2
  Mat A1, A2
  Vec y1, b1, y2, b2
  Vec xex1, xex2
  PetscErrorCode ierr
  PetscScalar norm
  PetscRandom rctx

  type(petsc_numbering_type) petsc_numbering1, petsc_numbering2
  type(dynamic_csr_matrix) dcsr1, dcsr2
  type(csr_matrix) csr1, csr2
  type(scalar_field):: sfield1, sfield2
  character(len=OPTION_PATH_LEN) solver_option_path1, name1
  character(len=OPTION_PATH_LEN) solver_option_path2, name2
  integer literations1, literations2
  logical lstartfromzero1, lstartfromzero2
  integer i

  call allocate(dcsr1, DIM, DIM, name='matrix1')
  call allocate(dcsr2, DIM, DIM, name='matrix2')
  do i=1, DIM
    call set(dcsr1, i, i, 1.0)
    call set(dcsr2, i, i, 2.0)
    call addto(dcsr1, i, min(i+1, DIM), 0.2)
    call addto(dcsr1, i, max(i-1, 1), 0.2)
    call addto(dcsr2, i, min(i+2, DIM), 0.4)
    call addto(dcsr2, i, max(i-2, 1), 0.4)
  end do
  csr1=dcsr2csr(dcsr1)
  csr2=dcsr2csr(dcsr2)

  ! uncomment this to see some solver output:
  call set_debug_level(3)

  call set_solver_options("/scalar_field::Field", ksptype=KSPCG, &
     pctype=PCMG, atol=1e-10, rtol=0.0)
  ! horrible hack - petsc_solve_setup/core only use %name and %option_path
  sfield1%name="Field1"
  sfield1%option_path="/scalar_field::Field"
  sfield2%name="Field2"
  sfield2%option_path="/scalar_field::Field"

  ! setup PETSc objects and petsc_numbering from options and
  ! compute rhs from "exact" solution
  call petsc_solve_setup(y1, A1, b1, ksp1, petsc_numbering1, &
        solver_option_path1, lstartfromzero1, &
        matrix=csr1, sfield=sfield1, &
        option_path="/scalar_field::Field")
  call PetscRandomCreate(MPI_COMM_FEMTOOLS, rctx, ierr)
  call PetscRandomSetFromOptions(rctx, ierr)
  call VecDuplicate(y1, xex1, ierr)
  call VecSetRandom(xex1, rctx, ierr)
  call MatMult(A1, xex1, b1, ierr)

  ! setup PETSc objects and petsc_numbering from options and
  ! compute rhs from "exact" solution
  call petsc_solve_setup(y2, A2, b2, ksp2, petsc_numbering2, &
        solver_option_path2, lstartfromzero2, &
        matrix=csr2, sfield=sfield2, &
        option_path="/scalar_field::Field")
  call VecDuplicate(y2, xex2, ierr)
  call VecSetRandom(xex2, rctx, ierr)
  call MatMult(A2, xex2, b2, ierr)

  call petsc_solve_core(y1, A1, b1, ksp1, petsc_numbering1, &
        solver_option_path1, lstartfromzero1, &
        literations1, sfield=sfield2)
  call petsc_solve_core(y2, A2, b2, ksp2, petsc_numbering2, &
        solver_option_path2, lstartfromzero2, &
        literations2, sfield=sfield2)

  ! check answer of first solve
  call VecAXPY(y1, real(-1.0, kind = PetscScalar_kind), xex1, ierr)
  call VecNorm(y1, NORM_2, norm, ierr)
  fail = (norm > 1e-7)
  call report_test("[test_multigrid1]", fail, .false., "Error too large in multigrid.")

  ! check answer of second solve
  call VecAXPY(y2, real(-1.0, kind = PetscScalar_kind), xex2, ierr)
  call VecNorm(y2, NORM_2, norm, ierr)
  fail = (norm > 1e-7)
  call report_test("[test_multigrid2]", fail, .false., "Error too large in multigrid.")

  ! destroying of PETSc objects, check for remaining references by
  ! running with ./test_multigrid -log_summary
  call petsc_solve_destroy(y1, A1, b1, ksp1, petsc_numbering1, solver_option_path1)
  call petsc_solve_destroy(y2, A2, b2, ksp2, petsc_numbering2, solver_option_path2)
  call PetscRandomDestroy(rctx, ierr)
  call VecDestroy(xex1, ierr)
  call VecDestroy(xex2, ierr)

#else
  ewrite(0,*) "Warning: no PETSc?"
#endif

end subroutine test_multigrid
