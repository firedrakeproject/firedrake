# cython: language_level=3

# Utility functions to derive global and local numbering from DMSwarm
import cython
import numpy as np
from firedrake.petsc import PETSc
from mpi4py import MPI
from pyop2.datatypes import IntType
from libc.string cimport memset
from libc.stdlib cimport qsort
cimport numpy as np
cimport mpi4py.MPI as MPI
cimport petsc4py.PETSc as PETSc

np.import_array()

include "petschdr.pxi"

@cython.boundscheck(False)
@cython.wraparound(False)
def remove_ghosts_pic(PETSc.DM swarm, PETSc.DM plex):
    """Remove DMSwarm PICs which are in ghost cells of a distributed
    DMPlex.

    :arg swarm: The DMSWARM which has been associated with the input
        DMPlex `plex` using PETSc `DMSwarmSetCellDM`.
    :arg plex: The DMPlex which is associated with the input DMSWARM
        `swarm`
    """
    cdef:
        PetscInt cStart, cEnd, ncells, i, npics
        PETSc.SF sf
        PetscInt nroots, nleaves
        const PetscInt *ilocal = NULL
        const PetscSFNode *iremote = NULL
        np.ndarray[PetscInt, ndim=1, mode="c"] pic_cell_indices
        np.ndarray[PetscInt, ndim=1, mode="c"] ghost_cell_indices

    assert plex.handle == swarm.getCellDM().handle

    if plex.comm.size > 1:

        cStart, cEnd = plex.getHeightStratum(0)
        ncells = cEnd - cStart

        # Get full list of cell indices for particles
        pic_cell_indices = np.copy(swarm.getField("DMSwarm_cellid"))
        swarm.restoreField("DMSwarm_cellid")
        npics = len(pic_cell_indices)

        # Initialise with zeros since these can't be valid ranks or cell ids
        ghost_cell_indices = np.full(ncells, -1, dtype=IntType)

        # Search for ghost cell indices (spooky!)
        sf = plex.getPointSF()
        CHKERR(PetscSFGetGraph(sf.sf, &nroots, &nleaves, &ilocal, &iremote))
        for i in range(nleaves):
            if cStart <= ilocal[i] < cEnd:
                # NOTE need to check this is correct index. Can I check the labels some how?
                ghost_cell_indices[ilocal[i] - cStart] = ilocal[i]

        # trim -1's and make into set to reduce searching needed
        ghost_cell_indices_set = set(ghost_cell_indices[ghost_cell_indices != -1])

        # remove swarm pic parent cell indices which match ghost cell indices
        for i in range(npics-1, -1, -1):
            if pic_cell_indices[i] in ghost_cell_indices_set:
                # removePointAtIndex shift cell numbers down by 1
                swarm.removePointAtIndex(i)

