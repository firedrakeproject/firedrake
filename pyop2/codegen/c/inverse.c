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
static PetscLogEvent USER_EVENT_inv_memcpy;
static PetscLogEvent USER_EVENT_inv_getrf;
static PetscLogEvent USER_EVENT_inv_getri;
#endif

#ifndef BEGIN_LOG
#define BEGIN_LOG
static void beginLog(PetscLogEvent eventId){
    #ifdef PYOP2_PROFILING_ENABLED
    PetscLogEventBegin(eventId,0,0,0,0);
    #endif
}
#endif

#ifndef END_LOG
#define END_LOG
static void endLog(PetscLogEvent eventId){
    #ifdef PYOP2_PROFILING_ENABLED
    PetscLogEventEnd(eventId,0,0,0,0);
    #endif
}
#endif

static void inverse(PetscScalar* __restrict__ Aout, const PetscScalar* __restrict__ A, PetscBLASInt N)
{
    #ifdef PYOP2_PROFILING_ENABLED
    PetscLogEventRegister("PyOP2InverseCallable_memcpy",PETSC_OBJECT_CLASSID,&USER_EVENT_inv_memcpy);
    PetscLogEventRegister("PyOP2InverseCallable_getrf",PETSC_OBJECT_CLASSID,&USER_EVENT_inv_getrf);
    PetscLogEventRegister("PyOP2InverseCallable_getri",PETSC_OBJECT_CLASSID,&USER_EVENT_inv_getri);
    #endif

    beginLog(USER_EVENT_inv_memcpy);
    PetscBLASInt info;
    PetscBLASInt *ipiv = N <= BUF_SIZE ? ipiv_buffer : malloc(N*sizeof(*ipiv));
    PetscScalar *Awork = N <= BUF_SIZE ? work_buffer : malloc(N*N*sizeof(*Awork));
    memcpy(Aout, A, N*N*sizeof(PetscScalar));
    endLog(USER_EVENT_inv_memcpy);

    beginLog(USER_EVENT_inv_getrf);
    LAPACKgetrf_(&N, &N, Aout, &N, ipiv, &info);
    endLog(USER_EVENT_inv_getrf);

    if(info == 0){
        beginLog(USER_EVENT_inv_getri);
        LAPACKgetri_(&N, Aout, &N, ipiv, Awork, &N, &info);
        endLog(USER_EVENT_inv_getri);
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
