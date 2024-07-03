#include <petsc.h>
#include <petscblaslapack.h>


PetscErrorCode mymatinvert(PetscInt* n, PetscScalar* mat, PetscInt* piv, PetscInt* info, PetscScalar* work) {
    PetscBLASInt *n_blas, *piv_blas, *info_blas;
    PetscCall(PetscBLASIntCast(*n, n_blas));
    PetscCall(PetscBLASIntCast(*piv, piv_blas));
    PetscCall(PetscBLASIntCast(*info, info_blas));
    PetscCallBLAS("LAPACKgetrf",LAPACKgetrf_(n_blas, n_blas, mat, n_blas, piv_blas, info_blas));
    PetscCallBLAS("LAPACKgetri",LAPACKgetri_(n_blas, mat, n_blas, piv_blas, work, n_blas, info_blas));
    return 0;
}

