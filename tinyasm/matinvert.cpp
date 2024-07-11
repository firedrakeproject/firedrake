#include <petsc.h>
#include <petscblaslapack.h>

PetscErrorCode mymatinvert(PetscInt* n, PetscScalar* mat, PetscBLASInt* piv, PetscInt* info, PetscScalar* work) {
    // Declare BLAS compatible variables
    PetscBLASInt n_blas, info_blas;

    // Convert function arguments to BLAS compatible ints
    PetscCall(PetscBLASIntCast(*n, &n_blas));
    PetscCall(PetscBLASIntCast(*info, &info_blas));

    PetscCallBLAS("LAPACKgetrf", LAPACKgetrf_(&n_blas, &n_blas, mat, &n_blas, piv, &info_blas));
    PetscCallBLAS("LAPACKgetri", LAPACKgetri_(&n_blas, mat, &n_blas, piv, work, &n_blas, &info_blas));

    // Cast info back since it has intent out
    *info = (PetscInt) info_blas;
    return 0;
}

