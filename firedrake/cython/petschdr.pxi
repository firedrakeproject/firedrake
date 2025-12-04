cimport petsc4py.PETSc as PETSc
from petsc4py.PETSc cimport CHKERR, CHKERRMPI
cimport mpi4py.MPI as MPI
cimport numpy as np

cdef extern from "mpi-compat.h":
    pass

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
    ctypedef enum PetscDataType:
        PETSC_INT,
        PETSC_REAL,
        PETSC_SCALAR,
        PETSC_COMPLEX,
        PETSC_DATATYPE_UNKNOWN
    ctypedef enum PetscErrorCode:
        PETSC_SUCCESS
        PETSC_ERR_LIB

cdef extern from "petscsys.h" nogil:
    PetscErrorCode PetscMalloc1(PetscInt,void*)
    PetscErrorCode PetscMalloc2(PetscInt,void*,PetscInt,void*)
    PetscErrorCode PetscFree(void*)
    PetscErrorCode PetscFree2(void*,void*)
    PetscErrorCode PetscSortIntWithArray(PetscInt,PetscInt[],PetscInt[])

cdef extern from "petscdmtypes.h" nogil:
    ctypedef enum PetscDMPolytopeType "DMPolytopeType":
        DM_POLYTOPE_POINT
        DM_POLYTOPE_SEGMENT
        DM_POLYTOPE_POINT_PRISM_TENSOR
        DM_POLYTOPE_TRIANGLE
        DM_POLYTOPE_QUADRILATERAL
        DM_POLYTOPE_SEG_PRISM_TENSOR
        DM_POLYTOPE_TETRAHEDRON
        DM_POLYTOPE_HEXAHEDRON
        DM_POLYTOPE_TRI_PRISM
        DM_POLYTOPE_TRI_PRISM_TENSOR
        DM_POLYTOPE_QUAD_PRISM_TENSOR
        DM_POLYTOPE_PYRAMID
        DM_POLYTOPE_FV_GHOST
        DM_POLYTOPE_INTERIOR_GHOST
        DM_POLYTOPE_UNKNOWN
        DM_POLYTOPE_UNKNOWN_CELL
        DM_POLYTOPE_UNKNOWN_FACE
        DM_NUM_POLYTOPES

cdef extern from "petscdmplex.h" nogil:
    PetscErrorCode DMPlexGetHeightStratum(PETSc.PetscDM,PetscInt,PetscInt*,PetscInt*)
    PetscErrorCode DMPlexGetDepthStratum(PETSc.PetscDM,PetscInt,PetscInt*,PetscInt*)
    PetscErrorCode DMPlexGetPointHeight(PETSc.PetscDM,PetscInt,PetscInt*)
    PetscErrorCode DMPlexGetPointDepth(PETSc.PetscDM,PetscInt,PetscInt*)

    PetscErrorCode DMPlexGetChart(PETSc.PetscDM,PetscInt*,PetscInt*)
    PetscErrorCode DMPlexGetConeSize(PETSc.PetscDM,PetscInt,PetscInt*)
    PetscErrorCode DMPlexGetCone(PETSc.PetscDM,PetscInt,PetscInt*[])
    PetscErrorCode DMPlexGetConeOrientation(PETSc.PetscDM,PetscInt,PetscInt*[])
    PetscErrorCode DMPlexGetSupportSize(PETSc.PetscDM,PetscInt,PetscInt*)
    PetscErrorCode DMPlexGetSupport(PETSc.PetscDM,PetscInt,PetscInt*[])
    PetscErrorCode DMPlexGetMaxSizes(PETSc.PetscDM,PetscInt*,PetscInt*)

    PetscErrorCode DMPlexGetTransitiveClosure(PETSc.PetscDM,PetscInt,PetscBool,PetscInt *,PetscInt *[])
    PetscErrorCode DMPlexRestoreTransitiveClosure(PETSc.PetscDM,PetscInt,PetscBool,PetscInt *,PetscInt *[])
    PetscErrorCode DMPlexDistributeData(PETSc.PetscDM,PETSc.PetscSF,PETSc.PetscSection,MPI.MPI_Datatype,void*,PETSc.PetscSection,void**)
    PetscErrorCode DMPlexSetAdjacencyUser(PETSc.PetscDM,int(*)(PETSc.PetscDM,PetscInt,PetscInt*,PetscInt[],void*),void*)
    PetscErrorCode DMPlexCreatePointNumbering(PETSc.PetscDM,PETSc.PetscIS*)
    PetscErrorCode DMPlexLabelComplete(PETSc.PetscDM, PETSc.PetscDMLabel)
    PetscErrorCode DMPlexDistributeOverlap(PETSc.PetscDM,PetscInt,PETSc.PetscSF*,PETSc.PetscDM*)

    PetscErrorCode DMPlexGetSubpointIS(PETSc.PetscDM,PETSc.PetscIS*)
    PetscErrorCode DMPlexGetSubpointMap(PETSc.PetscDM,PETSc.PetscDMLabel*)
    PetscErrorCode DMPlexSetSubpointMap(PETSc.PetscDM,PETSc.PetscDMLabel)

    PetscErrorCode DMPlexSetCellType(PETSc.PetscDM,PetscInt,PetscDMPolytopeType)
    PetscErrorCode DMPlexGetCellType(PETSc.PetscDM,PetscInt,PetscDMPolytopeType*)

cdef extern from "petscdmlabel.h" nogil:
    struct _n_DMLabel
    ctypedef _n_DMLabel* DMLabel "DMLabel"
    PetscErrorCode DMLabelCreateIndex(DMLabel, PetscInt, PetscInt)
    PetscErrorCode DMLabelDestroyIndex(DMLabel)
    PetscErrorCode DMLabelDestroy(DMLabel*)
    PetscErrorCode DMLabelHasPoint(DMLabel, PetscInt, PetscBool*)
    PetscErrorCode DMLabelSetValue(DMLabel, PetscInt, PetscInt)
    PetscErrorCode DMLabelGetValue(DMLabel, PetscInt, PetscInt*)
    PetscErrorCode DMLabelClearValue(DMLabel, PetscInt, PetscInt)
    PetscErrorCode DMLabelGetStratumSize(DMLabel, PetscInt, PetscInt*)
    PetscErrorCode DMLabelGetStratumIS(DMLabel, PetscInt, PETSc.PetscIS*)

cdef extern from "petscdm.h" nogil:
    PetscErrorCode DMCreateLabel(PETSc.PetscDM,char[])
    PetscErrorCode DMGetLabel(PETSc.PetscDM,char[],DMLabel*)
    PetscErrorCode DMGetPointSF(PETSc.PetscDM,PETSc.PetscSF*)
    PetscErrorCode DMSetLabelValue(PETSc.PetscDM,char[],PetscInt,PetscInt)
    PetscErrorCode DMGetLabelValue(PETSc.PetscDM,char[],PetscInt,PetscInt*)

cdef extern from "petscdmswarm.h" nogil:
    PetscErrorCode DMSwarmGetLocalSize(PETSc.PetscDM,PetscInt*)
    PetscErrorCode DMSwarmGetCellDM(PETSc.PetscDM, PETSc.PetscDM*)
    PetscErrorCode DMSwarmGetCellDMActive(PETSc.PetscDM, PETSc.PetscDMSwarmCellDM*)
    PetscErrorCode DMSwarmCellDMGetCellID(PETSc.PetscDMSwarmCellDM, const char *[])
    PetscErrorCode DMSwarmGetField(PETSc.PetscDM,const char[],PetscInt*,PetscDataType*,void**)
    PetscErrorCode DMSwarmRestoreField(PETSc.PetscDM,const char[],PetscInt*,PetscDataType*,void**)

cdef extern from "petscvec.h" nogil:
    PetscErrorCode VecGetArray(PETSc.PetscVec,PetscScalar**)
    PetscErrorCode VecRestoreArray(PETSc.PetscVec,PetscScalar**)
    PetscErrorCode VecGetArrayRead(PETSc.PetscVec,const PetscScalar**)
    PetscErrorCode VecRestoreArrayRead(PETSc.PetscVec,const PetscScalar**)

cdef extern from "petscis.h" nogil:
    PetscErrorCode PetscSectionGetOffset(PETSc.PetscSection,PetscInt,PetscInt*)
    PetscErrorCode PetscSectionGetDof(PETSc.PetscSection,PetscInt,PetscInt*)
    PetscErrorCode PetscSectionSetDof(PETSc.PetscSection,PetscInt,PetscInt)
    PetscErrorCode PetscSectionSetFieldDof(PETSc.PetscSection,PetscInt,PetscInt,PetscInt)
    PetscErrorCode PetscSectionGetFieldDof(PETSc.PetscSection,PetscInt,PetscInt,PetscInt*)
    PetscErrorCode PetscSectionGetConstraintDof(PETSc.PetscSection,PetscInt,PetscInt*)
    PetscErrorCode PetscSectionSetConstraintDof(PETSc.PetscSection,PetscInt,PetscInt)
    PetscErrorCode PetscSectionSetConstraintIndices(PETSc.PetscSection,PetscInt, PetscInt[])
    PetscErrorCode PetscSectionGetConstraintIndices(PETSc.PetscSection,PetscInt, const PetscInt**)
    PetscErrorCode PetscSectionGetMaxDof(PETSc.PetscSection,PetscInt*)
    PetscErrorCode PetscSectionSetPermutation(PETSc.PetscSection,PETSc.PetscIS)
    PetscErrorCode ISGetIndices(PETSc.PetscIS,PetscInt*[])
    PetscErrorCode ISGetSize(PETSc.PetscIS,PetscInt*)
    PetscErrorCode ISRestoreIndices(PETSc.PetscIS,PetscInt*[])
    PetscErrorCode ISGeneralSetIndices(PETSc.PetscIS,PetscInt,PetscInt[],PetscCopyMode)
    PetscErrorCode ISLocalToGlobalMappingCreateIS(PETSc.PetscIS,PETSc.PetscLGMap*)
    PetscErrorCode ISLocalToGlobalMappingGetSize(PETSc.PetscLGMap,PetscInt*)
    PetscErrorCode ISLocalToGlobalMappingGetBlockIndices(PETSc.PetscLGMap, const PetscInt**)
    PetscErrorCode ISLocalToGlobalMappingRestoreBlockIndices(PETSc.PetscLGMap, const PetscInt**)
    PetscErrorCode ISDestroy(PETSc.PetscIS*)

cdef extern from "petscsf.h" nogil:
    struct PetscSFNode_:
        PetscInt rank
        PetscInt index
    ctypedef PetscSFNode_ PetscSFNode "PetscSFNode"

    PetscErrorCode PetscSFGetGraph(PETSc.PetscSF,PetscInt*,PetscInt*,PetscInt**,PetscSFNode**)
    PetscErrorCode PetscSFSetGraph(PETSc.PetscSF,PetscInt,PetscInt,PetscInt*,PetscCopyMode,PetscSFNode*,PetscCopyMode)
    PetscErrorCode PetscSFBcastBegin(PETSc.PetscSF,MPI.MPI_Datatype,const void*, void*,)
    PetscErrorCode PetscSFBcastEnd(PETSc.PetscSF,MPI.MPI_Datatype,const void*, void*)
    PetscErrorCode PetscSFReduceBegin(PETSc.PetscSF,MPI.MPI_Datatype,const void*, void*,MPI.MPI_Op)
    PetscErrorCode PetscSFReduceEnd(PETSc.PetscSF,MPI.MPI_Datatype,const void*, void*,MPI.MPI_Op)

ctypedef PetscErrorCode (*PetscPCPatchComputeFunction)(PETSc.PetscPC,
                                            PetscInt,
                                            PETSc.PetscVec,
                                            PETSc.PetscVec,
                                            PETSc.PetscIS,
                                            PetscInt,
                                            const PetscInt*,
                                            const PetscInt*,
                                            void*)
ctypedef PetscErrorCode (*PetscPCPatchComputeOperator)(PETSc.PetscPC,
                                            PetscInt,
                                            PETSc.PetscVec,
                                            PETSc.PetscMat,
                                            PETSc.PetscIS,
                                            PetscInt,
                                            const PetscInt*,
                                            const PetscInt*,
                                            void*)
cdef extern from "petscsnes.h" nogil:
   PetscErrorCode SNESPatchSetComputeFunction(PETSc.PetscSNES, PetscPCPatchComputeFunction, void *)
   PetscErrorCode SNESPatchSetComputeOperator(PETSc.PetscSNES, PetscPCPatchComputeOperator, void *)

cdef extern from "petscpc.h" nogil:
   PetscErrorCode PCPatchSetComputeFunction(PETSc.PetscPC, PetscPCPatchComputeFunction, void *)
   PetscErrorCode PCPatchSetComputeFunctionInteriorFacets(PETSc.PetscPC, PetscPCPatchComputeFunction, void *)
   PetscErrorCode PCPatchSetComputeOperator(PETSc.PetscPC, PetscPCPatchComputeOperator, void *)
   PetscErrorCode PCPatchSetComputeOperatorInteriorFacets(PETSc.PetscPC, PetscPCPatchComputeOperator, void *)

cdef extern from "petscbt.h" nogil:
    ctypedef char * PetscBT
    PetscErrorCode PetscBTCreate(PetscInt,PetscBT*)
    PetscErrorCode PetscBTDestroy(PetscBT*)
    char PetscBTLookup(PetscBT,PetscInt)
    PetscErrorCode PetscBTSet(PetscBT,PetscInt)

cdef extern from "petscmat.h" nogil:
    PetscErrorCode MatSetValuesLocal(PETSc.PetscMat, PetscInt, const PetscInt[], PetscInt, const PetscInt[],
                          const PetscScalar[], PetscInt)
    PetscErrorCode MatAssemblyBegin(PETSc.PetscMat, PetscInt)
    PetscErrorCode MatAssemblyEnd(PETSc.PetscMat, PetscInt)
    PetscInt MAT_FINAL_ASSEMBLY = 0

cdef extern from * nogil:
    PetscErrorCode PetscObjectTypeCompare(PETSc.PetscObject, char[], PetscBool*)
