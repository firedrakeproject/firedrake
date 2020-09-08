#include <petscsys.h>
#include <petscblaslapack.h>

#ifndef PYOP2_WORK_ARRAYS
#define PYOP2_WORK_ARRAYS
#define BUF_SIZE 30
static PetscBLASInt ipiv_buffer[BUF_SIZE];
static PetscScalar work_buffer[BUF_SIZE*BUF_SIZE];
#endif

static void solve(PetscScalar* __restrict__ out, const PetscScalar* __restrict__ A, const PetscScalar* __restrict__ B, PetscBLASInt N)
{
    PetscBLASInt info;
    PetscBLASInt *ipiv = N <= BUF_SIZE ? ipiv_buffer : malloc(N*sizeof(*ipiv));
    memcpy(out,B,N*sizeof(PetscScalar));
    PetscScalar *Awork = N <= BUF_SIZE ? work_buffer : malloc(N*N*sizeof(*Awork));
    memcpy(Awork,A,N*N*sizeof(PetscScalar));
    PetscBLASInt NRHS = 1;
    const char T = 'T';
    LAPACKgetrf_(&N, &N, Awork, &N, ipiv, &info);
    if(info == 0){
        LAPACKgetrs_(&T, &N, &NRHS, Awork, &N, ipiv, out, &N, &info);
    }
    if(info != 0){
        fprintf(stderr, "Gesv throws nonzero info.");
        abort();
    }

    if ( N > BUF_SIZE ) {
        free(ipiv);
        free(Awork);
    }
}
