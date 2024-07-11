#include <petsc.h>
#include <petscblaslapack.h>

PetscErrorCode mymatinvert(PetscBLASInt* n, PetscScalar* mat, PetscBLASInt* piv, PetscBLASInt* info, PetscScalar* work) {
    PetscCallBLAS("LAPACKgetrf", LAPACKgetrf_(n, n, mat, n, piv, info));
    PetscCheck(!(*info), PETSC_COMM_SELF, PETSC_ERR_LIB, "TinyASM error calling ?getrf in mymatinvert");
    PetscCallBLAS("LAPACKgetri", LAPACKgetri_(n, mat, n, piv, work, n, info));
    PetscCheck(!(*info), PETSC_COMM_SELF, PETSC_ERR_LIB, "TinyASM error calling ?getri in mymatinvert");
    return PETSC_SUCCESS;
}

