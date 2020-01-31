cimport petsc4py.PETSc as PETSc
cimport mpi4py.MPI as MPI

cdef extern from "petsc.h":
   ctypedef long PetscInt
   ctypedef double PetscReal
   ctypedef double PetscScalar
   ctypedef enum PetscBool:
       PETSC_TRUE, PETSC_FALSE
   ctypedef enum PetscCopyMode:
       PETSC_COPY_VALUES,
       PETSC_OWN_POINTER,
       PETSC_USE_POINTER

cdef extern from "petscsys.h" nogil:
   int PetscMalloc1(PetscInt,void*)
   int PetscFree(void*)
   int PetscSortIntWithArray(PetscInt,PetscInt[],PetscInt[])

cdef extern from "petscdmplex.h" nogil:
    int DMPlexGetHeightStratum(PETSc.PetscDM,PetscInt,PetscInt*,PetscInt*)
    int DMPlexGetDepthStratum(PETSc.PetscDM,PetscInt,PetscInt*,PetscInt*)

    int DMPlexGetConeSize(PETSc.PetscDM,PetscInt,PetscInt*)
    int DMPlexGetCone(PETSc.PetscDM,PetscInt,PetscInt*[])
    int DMPlexGetConeOrientation(PETSc.PetscDM,PetscInt,PetscInt*[])
    int DMPlexGetSupportSize(PETSc.PetscDM,PetscInt,PetscInt*)
    int DMPlexGetSupport(PETSc.PetscDM,PetscInt,PetscInt*[])

    int DMPlexGetTransitiveClosure(PETSc.PetscDM,PetscInt,PetscBool,PetscInt *,PetscInt *[])
    int DMPlexRestoreTransitiveClosure(PETSc.PetscDM,PetscInt,PetscBool,PetscInt *,PetscInt *[])
    int DMPlexDistributeData(PETSc.PetscDM,PETSc.PetscSF,PETSc.PetscSection,MPI.MPI_Datatype,void*,PETSc.PetscSection,void**)
    int DMPlexSetAdjacencyUser(PETSc.PetscDM,int(*)(PETSc.PetscDM,PetscInt,PetscInt*,PetscInt[],void*),void*)

cdef extern from "petscdmlabel.h" nogil:
    struct _n_DMLabel
    ctypedef _n_DMLabel* DMLabel "DMLabel"
    int DMLabelCreateIndex(DMLabel, PetscInt, PetscInt)
    int DMLabelDestroyIndex(DMLabel)
    int DMLabelDestroy(DMLabel*)
    int DMLabelHasPoint(DMLabel, PetscInt, PetscBool*)
    int DMLabelSetValue(DMLabel, PetscInt, PetscInt)
    int DMLabelGetValue(DMLabel, PetscInt, PetscInt*)
    int DMLabelClearValue(DMLabel, PetscInt, PetscInt)

cdef extern from "petscdm.h" nogil:
    int DMGetLabel(PETSc.PetscDM,char[],DMLabel*)

cdef extern from "petscis.h" nogil:
    int PetscSectionGetOffset(PETSc.PetscSection,PetscInt,PetscInt*)
    int PetscSectionGetDof(PETSc.PetscSection,PetscInt,PetscInt*)
    int PetscSectionSetDof(PETSc.PetscSection,PetscInt,PetscInt)
    int PetscSectionSetPermutation(PETSc.PetscSection,PETSc.PetscIS)
    int ISGetIndices(PETSc.PetscIS,PetscInt*[])
    int ISRestoreIndices(PETSc.PetscIS,PetscInt*[])
    int ISGeneralSetIndices(PETSc.PetscIS,PetscInt,PetscInt[],PetscCopyMode)

cdef extern from "petscsf.h" nogil:
    struct PetscSFNode_:
        PetscInt rank
        PetscInt index
    ctypedef PetscSFNode_ PetscSFNode "PetscSFNode"

    int PetscSFGetGraph(PETSc.PetscSF,PetscInt*,PetscInt*,PetscInt**,PetscSFNode**)
    int PetscSFSetGraph(PETSc.PetscSF,PetscInt,PetscInt,PetscInt*,PetscCopyMode,PetscSFNode*,PetscCopyMode)
    int PetscSFBcastBegin(PETSc.PetscSF,MPI.MPI_Datatype,const void*, void*,)
    int PetscSFBcastEnd(PETSc.PetscSF,MPI.MPI_Datatype,const void*, void*)
    int PetscSFReduceBegin(PETSc.PetscSF,MPI.MPI_Datatype,const void*, void*,MPI.MPI_Op)
    int PetscSFReduceEnd(PETSc.PetscSF,MPI.MPI_Datatype,const void*, void*,MPI.MPI_Op)

cdef extern from "petscbt.h" nogil:
    ctypedef char * PetscBT
    int PetscBTCreate(PetscInt,PetscBT*)
    int PetscBTDestroy(PetscBT*)
    char PetscBTLookup(PetscBT,PetscInt)
    int PetscBTSet(PetscBT,PetscInt)


# --- Error handling taken from petsc4py (src/PETSc.pyx) -------------

cdef extern from *:
    void PyErr_SetObject(object, object)
    void *PyExc_RuntimeError

cdef object PetscError = <object>PyExc_RuntimeError

cdef inline int SETERR(int ierr) with gil:
    if (<void*>PetscError) != NULL:
        PyErr_SetObject(PetscError, <long>ierr)
    else:
        PyErr_SetObject(<object>PyExc_RuntimeError, <long>ierr)
    return ierr

cdef inline int CHKERR(int ierr) nogil except -1:
    if ierr == 0:
        return 0 # no error
    else:
        SETERR(ierr)
        return -1

# --------------------------------------------------------------------
