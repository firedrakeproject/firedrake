#include <petscsys.h>
#include <petscblaslapack.h>

#ifndef PYOP2_WORK_ARRAYS
#define PYOP2_WORK_ARRAYS
#define BUF_SIZE 30
static PetscBLASInt ipiv_buffer[BUF_SIZE];
static PetscScalar work_buffer[BUF_SIZE*BUF_SIZE];
#endif

#ifndef PYOP2_INV_LOG_EVENTS
#define PYOP2_INV_LOG_EVENTS
PetscLogEvent ID_inv_memcpy = -1;
PetscLogEvent ID_inv_getrf = -1;
PetscLogEvent ID_inv_getri = -1;
static PetscBool log_active_inv = 0;
#endif

void inverse(PetscScalar* __restrict__ Aout, const PetscScalar* __restrict__ A, PetscBLASInt N)
{
    PetscLogIsActive(&log_active_inv);
    if (log_active_inv){PetscLogEventBegin(ID_inv_memcpy,0,0,0,0);}
    PetscBLASInt info;
    PetscBLASInt *ipiv = N <= BUF_SIZE ? ipiv_buffer : malloc(N*sizeof(*ipiv));
    PetscScalar *Awork = N <= BUF_SIZE ? work_buffer : malloc(N*N*sizeof(*Awork));
    memcpy(Aout, A, N*N*sizeof(PetscScalar));
    if (log_active_inv){PetscLogEventEnd(ID_inv_memcpy,0,0,0,0);}

    if (log_active_inv){PetscLogEventBegin(ID_inv_getrf,0,0,0,0);}
    LAPACKgetrf_(&N, &N, Aout, &N, ipiv, &info);
    if (log_active_inv){PetscLogEventEnd(ID_inv_getrf,0,0,0,0);}

    if(info == 0){
        if (log_active_inv){PetscLogEventBegin(ID_inv_getri,0,0,0,0);}
        LAPACKgetri_(&N, Aout, &N, ipiv, Awork, &N, &info);
        if (log_active_inv){PetscLogEventEnd(ID_inv_getri,0,0,0,0);}
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
