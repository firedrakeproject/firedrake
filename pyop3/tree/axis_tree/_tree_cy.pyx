import numpy as np

from petsctools cimport cpetsc
from petsctools.cpetsc cimport CHKERR


def apply_constraints(section: cpetsc.PetscSection_py, sizes: np.ndarray, constrained: np.ndarray):
    cdef:
        cpetsc.PetscInt point
        cpetsc.PetscInt *constrained_idxs_c = NULL

    ptr = 0
    for point, num_dofs in enumerate(sizes):
        constrained_mask = constrained[ptr:ptr+num_dofs]
        num_constrained_dofs = sum(constrained_mask)
        section.setConstraintDof(point, num_constrained_dofs)
        ptr += num_dofs

    # needs to happen before setting constraint indices
    section.setUp()

    CHKERR(cpetsc.PetscMalloc1(sizes.max(), &constrained_idxs_c))

    # preallocate work array
    constrained_idxs = np.empty(sizes.max(), dtype=IntType)

    ptr = 0
    for point, num_dofs in enumerate(sizes):
        constrained_mask = constrained[ptr:ptr+num_dofs]

        constraint_index_ptr = 0
        for dof in range(num_dofs):
            if constrained_mask[dof]:
                constrained_idxs[constraint_index_ptr] = dof
                constraint_index_ptr += 1

        cpetsc.CHKERR(cpetsc.PetscSectionSetConstraintIndices(section.sec, point, constrained_idxs_c))
        ptr += num_dofs
