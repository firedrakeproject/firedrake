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

    if type(plex) is not PETSc.DMPlex:
        raise ValueError("plex must be a DMPlex")

    if type(swarm) is not PETSc.DMSwarm:
        raise ValueError("swarm must be a DMSwarm")

    if plex.handle != swarm.getCellDM().handle:
        raise ValueError("plex is not the swarm's CellDM")

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


@cython.boundscheck(False)
@cython.wraparound(False)
def label_pic_parent_cell_nums(PETSc.DM swarm, parentmesh):
    """
    For each PIC in the input swarm, label its `parentcellnum` field with
    the relevant cell number from the `parentmesh` in which is it emersed.
    The cell numbering is that given by the `parentmesh.locate_cell`
    method.

    :arg swarm: The DMSWARM which contains the PICs immersed in
        `parentmesh`
    :arg parentmesh: The mesh within with the `swarm` PICs are immersed.

    ..note:: All PICs must be within the parentmesh or this will try to
             assign `None` (returned by `parentmesh.locate_cell`) to a
             `PetscReal`.
    """
    cdef:
        PetscInt num_vertices, i, dim, parent_cell_num
        np.ndarray[PetscReal, ndim=2, mode="c"] swarm_coords
        np.ndarray[PetscInt, ndim=1, mode="c"] parent_cell_nums

    if type(swarm) is not PETSc.DMSwarm:
        raise ValueError("swarm must be a DMSwarm")

    if parentmesh._topology_dm.handle != swarm.getCellDM().handle:
        raise ValueError("parentmesh._topology_dm is not the swarm's CellDM")

    dim = parentmesh.geometric_dimension()

    num_vertices = swarm.getLocalSize()

    # Check size of biggest num_vertices so
    # locate_cell can be called on every processor
    comm = swarm.comm.tompi4py()
    max_num_vertices = comm.allreduce(num_vertices, op=MPI.MAX)

    # Create an out of mesh point to use in locate_cell when needed
    out_of_mesh_point = np.full((1, dim), np.inf)

    # get fields - NOTE this isn't copied so make sure
    # swarm.restoreField is called for each field too!
    swarm_coords = swarm.getField("DMSwarmPIC_coor").reshape((num_vertices, dim))
    parent_cell_nums = swarm.getField("parentcellnum")

    # find parent cell numbers
    for i in range(max_num_vertices):
        if i < num_vertices:
            parent_cell_num = parentmesh.locate_cell(swarm_coords[i])
            parent_cell_nums[i] = parent_cell_num
        else:
            parentmesh.locate_cell(out_of_mesh_point)  # should return None

    # have to restore fields once accessed to allow access again
    swarm.restoreField("parentcellnum")
    swarm.restoreField("DMSwarmPIC_coor")
