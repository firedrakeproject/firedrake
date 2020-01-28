from libc.stdint cimport uintptr_t
from enum import IntEnum

include "petschdr.pxi"

class JacType(IntEnum):
      CELL = 1
      INTERIOR_FACET = 2
      EXTERIOR_FACET = 3

def set_patch_residual(patch, function, ctx, is_snes=False, jac_type = JacType.CELL):
    if is_snes:
        if jac_type == JacType.INTERIOR_FACET:
            raise NotImplementedError
        if jac_type == JacType.EXTERIOR_FACET:
            raise NotImplementedError
        CHKERR(SNESPatchSetComputeFunction((<PETSc.SNES?>patch).snes,
                                           <PetscPCPatchComputeFunction><uintptr_t>function,
                                           <void *><uintptr_t>ctx))
    else:
        if jac_type == JacType.INTERIOR_FACET:
            CHKERR(PCPatchSetComputeFunctionInteriorFacets((<PETSc.PC?>patch).pc,
                                                           <PetscPCPatchComputeFunction><uintptr_t>function,
                                                           <void *><uintptr_t>ctx))
        elif jac_type == JacType.EXTERIOR_FACET:
            CHKERR(PCPatchSetComputeFunctionExteriorFacets((<PETSc.PC?>patch).pc,
                                                           <PetscPCPatchComputeFunction><uintptr_t>function,
                                                           <void *><uintptr_t>ctx))
        else:
            CHKERR(PCPatchSetComputeFunction((<PETSc.PC?>patch).pc,
                                             <PetscPCPatchComputeFunction><uintptr_t>function,
                                             <void *><uintptr_t>ctx))


def set_patch_jacobian(patch, function, ctx, is_snes=False, jac_type = JacType.CELL):
    if is_snes:
        if jac_type == JacType.INTERIOR_FACET:
            raise NotImplementedError
        if jac_type == JacType.EXTERIOR_FACET:
            raise NotImplementedError
        CHKERR(SNESPatchSetComputeOperator((<PETSc.SNES?>patch).snes,
                                               <PetscPCPatchComputeOperator><uintptr_t>function,
                                               <void *><uintptr_t>ctx))
    else:
        if jac_type == JacType.INTERIOR_FACET:
            CHKERR(PCPatchSetComputeOperatorInteriorFacets((<PETSc.PC?>patch).pc,
                                                           <PetscPCPatchComputeOperator><uintptr_t>function,
                                                           <void *><uintptr_t>ctx))
        elif jac_type == JacType.EXTERIOR_FACET:
            CHKERR(PCPatchSetComputeOperatorExteriorFacets((<PETSc.PC?>patch).pc,
                                                           <PetscPCPatchComputeOperator><uintptr_t>function,
                                                           <void *><uintptr_t>ctx))
        else:
            CHKERR(PCPatchSetComputeOperator((<PETSc.PC?>patch).pc,
                                             <PetscPCPatchComputeOperator><uintptr_t>function,
                                             <void *><uintptr_t>ctx))
