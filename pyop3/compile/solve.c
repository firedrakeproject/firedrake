#include <petscsys.h>
#include <petscblaslapack.h>

#ifndef PYOP2_WORK_ARRAYS
#define PYOP2_WORK_ARRAYS
#define BUF_SIZE 30
static PetscBLASInt ipiv_buffer[BUF_SIZE];
static PetscScalar work_buffer[BUF_SIZE*BUF_SIZE];
#endif

#ifndef PYOP2_SOLVE_LOG_EVENTS
#define PYOP2_SOLVE_LOG_EVENTS
PetscLogEvent ID_solve_memcpy = -1;
PetscLogEvent ID_solve_getrf = -1;
PetscLogEvent ID_solve_getrs = -1;
static PetscBool log_active_solve = 0;
#endif

void solve(PetscScalar* __restrict__ out, const PetscScalar* __restrict__ A, const PetscScalar* __restrict__ B, PetscBLASInt N)
{
    PetscLogIsActive(&log_active_solve);
    if (log_active_solve){PetscLogEventBegin(ID_solve_memcpy,0,0,0,0);}
    PetscBLASInt info;
    PetscBLASInt *ipiv = N <= BUF_SIZE ? ipiv_buffer : malloc(N*sizeof(*ipiv));
    memcpy(out,B,N*sizeof(PetscScalar));
    PetscScalar *Awork = N <= BUF_SIZE ? work_buffer : malloc(N*N*sizeof(*Awork));
    memcpy(Awork,A,N*N*sizeof(PetscScalar));
    if (log_active_solve){PetscLogEventEnd(ID_solve_memcpy,0,0,0,0);}

    PetscBLASInt NRHS = 1;
    const char T = 'T';
    if (log_active_solve){PetscLogEventBegin(ID_solve_getrf,0,0,0,0);}
    LAPACKgetrf_(&N, &N, Awork, &N, ipiv, &info);
    if (log_active_solve){PetscLogEventEnd(ID_solve_getrf,0,0,0,0);}

    if(info == 0){
        if (log_active_solve){PetscLogEventBegin(ID_solve_getrs,0,0,0,0);}
        LAPACKgetrs_(&T, &N, &NRHS, Awork, &N, ipiv, out, &N, &info);
        if (log_active_solve){PetscLogEventEnd(ID_solve_getrs,0,0,0,0);}
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
