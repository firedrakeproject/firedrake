cdef extern from "../mpi-compat.h":
    pass

cdef extern from * nogil:
    int DMPlexCreatePointNumbering(PETSc.PetscDM,PETSc.PetscIS*)
    int ISLocalToGlobalMappingCreateIS(PETSc.PetscIS,PETSc.PetscLGMap*)
    int ISLocalToGlobalMappingGetSize(PETSc.PetscLGMap,PetscInt*)
    int ISLocalToGlobalMappingGetBlockIndices(PETSc.PetscLGMap, const PetscInt**)
    int ISLocalToGlobalMappingRestoreBlockIndices(PETSc.PetscLGMap, const PetscInt**)
