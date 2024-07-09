#include <petsc.h>
#include <petscblaslapack.h>
#include <assert.h>


PetscErrorCode mymatinvert(PetscInt* n, PetscScalar* mat, PetscInt* piv, PetscInt* info, PetscScalar* work) {
    // Declare BLAS compatible variables
    PetscBLASInt n_blas, info_blas;
    PetscBLASInt *piv_blas;

    // Convert function arguments to BLAS compatible ints
    PetscCall(PetscBLASIntCast(*n, &n_blas));
    // piv is OUT for getrf and IN for getri so we don't need to check
    // whether the cast is safe
    piv_blas = (PetscBLASInt*) piv;
    PetscCall(PetscBLASIntCast(*info, &info_blas));

    assert(*n == n_blas);

    //~ printf("n %" PetscInt_FMT " \n", *n);
    //~ printf("n_blas %" PetscBLASInt_FMT " \n", n_blas);
    //~ printf("piv %" PetscInt_FMT " \n", *piv);
    //~ printf("piv_blas %" PetscBLASInt_FMT " \n", piv_blas);

    PetscCallBLAS("LAPACKgetrf",LAPACKgetrf_(&n_blas, &n_blas, mat, &n_blas, piv_blas, &info_blas));
    PetscCallBLAS("LAPACKgetri",LAPACKgetri_(&n_blas, mat, &n_blas, piv_blas, work, &n_blas, &info_blas));
    return 0;
}

