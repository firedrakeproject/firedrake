# cython: language_level=3

# Utility functions common to all DMs used in Firedrake
import functools
import cython
import numpy as np
import firedrake
from firedrake.petsc import PETSc
from mpi4py import MPI
from firedrake.utils import IntType, ScalarType
from libc.string cimport memset
from libc.stdlib cimport qsort
from tsfc.finatinterface import as_fiat_cell

cimport numpy as np
cimport mpi4py.MPI as MPI
cimport petsc4py.PETSc as PETSc

np.import_array()

include "petschdr.pxi"


@cython.boundscheck(False)
@cython.wraparound(False)
def temp(mesh,
                                   PETSc.Section global_numbering,
                                   entity_dofs,
                                   entity_permutations,
                                   np.ndarray[PetscInt, ndim=1, mode="c"] offset):
    """
    Builds the DoF mapping.

    :arg mesh: The mesh
    :arg global_numbering: Section describing the global DoF numbering
    :arg entity_dofs: FInAT element entity dofs for the cell
    :arg entity_permutations: FInAT element entity permutations for the cell
    :arg offset: offsets for each entity dof walking up a column.

    Preconditions: This function assumes that cell_closures contains mesh
    entities ordered by dimension, i.e. vertices first, then edges, faces, and
    finally the cell. For quadrilateral meshes, edges corresponding to
    dimension (0, 1) in the FInAT element must precede edges corresponding to
    dimension (1, 0) in the FInAT element.
    """
    pass
