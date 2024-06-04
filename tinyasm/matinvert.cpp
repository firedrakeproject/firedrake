#include <petsc.h>
#include <petscblaslapack.h>


PetscErrorCode mymatinvert(PetscInt* n, PetscScalar* mat, PetscInt* piv, PetscInt* info, PetscScalar* work) {
    PetscCallBLAS("LAPACKgetrf",LAPACKgetrf_(n,n,mat,n,piv,info));
    PetscCallBLAS("LAPACKgetri",LAPACKgetri_(n,mat, n, piv,work,n,info));
    return 0;
}

