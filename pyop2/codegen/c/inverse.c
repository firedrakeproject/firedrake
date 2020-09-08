#include <petscsys.h>
#include <petscblaslapack.h>

#ifndef PYOP2_WORK_ARRAYS
#define PYOP2_WORK_ARRAYS
#define BUF_SIZE 30
static PetscBLASInt ipiv_buffer[BUF_SIZE];
static PetscScalar work_buffer[BUF_SIZE*BUF_SIZE];
#endif

static void inverse(PetscScalar* __restrict__ Aout, const PetscScalar* __restrict__ A, PetscBLASInt N)
{
    PetscBLASInt info;
    PetscBLASInt *ipiv = N <= BUF_SIZE ? ipiv_buffer : malloc(N*sizeof(*ipiv));
    PetscScalar *Awork = N <= BUF_SIZE ? work_buffer : malloc(N*N*sizeof(*Awork));
    memcpy(Aout, A, N*N*sizeof(PetscScalar));
    LAPACKgetrf_(&N, &N, Aout, &N, ipiv, &info);
    if(info == 0){
        LAPACKgetri_(&N, Aout, &N, ipiv, Awork, &N, &info);
    }
    if(info != 0){
        fprintf(stderr, "Getri throws nonzero info.");
        abort();
    }
    if ( N > BUF_SIZE ) {
        free(Awork);
        free(ipiv);
    }
}
